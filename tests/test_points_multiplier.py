"""
points_multiplier, and the one thing about it that is easy to get wrong.

Standalone, like every other test here (the repo has no pytest):
    python tests/test_points_multiplier.py

The load-bearing case is the second block. The CLI has always applied this knob inside
the solver, where the candidate pool is already fixed, so a boost could only re-rank
players who had already got in. fpl-lad always sends `bucket_top_n`, and a pool slot is
exactly what a boosted player is short of — so applying it in the old place would leave
the field silently inert for anyone below the cut line. These checks pin the ordering by
running the real create_watchlist and apply_price_bucket_filter over the multiplied
frame, and asserting the same boost does nothing when applied afterwards.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "")

import pandas as pd  # noqa: E402

import api_server as A  # noqa: E402
from api_server import PointsMultiplierEntry, _apply_points_multipliers  # noqa: E402
from fpl.watchlist import apply_price_bucket_filter, create_watchlist  # noqa: E402

failures = []


def check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got}, want {want}")
    if not ok:
        failures.append(label)


# Seven midfielders in one £5.0m price bucket, ranked 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0
# points per gameweek. The bucket filter keeps the top 5, so ids 6 and 7 are the two
# below the line — 7 is the one a multiplier has to rescue.
GWS = [1, 2, 3, 4, 5]
PER_GW = {1: 6.0, 2: 5.5, 3: 5.0, 4: 4.5, 5: 4.0, 6: 3.5, 7: 3.0}


def predictions_frame():
    return pd.DataFrame([
        {
            "element": pid,
            "event": gw,
            "predicted_points": pts,
            "position": "MID",
            "hist_games": 5,
        }
        for pid, pts in PER_GW.items()
        for gw in GWS
    ])


def gw_data_frame():
    # Five played fixtures each, all full matches, so everyone clears the 60% start
    # filter and create_watchlist's own cut is never what decides these cases.
    return pd.DataFrame([
        {
            "element": pid,
            "GW": gw,
            "value": 50,
            "minutes": 90,
            "kickoff_time": f"2020-08-{gw:02d}T12:00:00Z",
        }
        for pid in PER_GW
        for gw in GWS
    ])


BOOTSTRAP = {
    "elements": [{"id": pid, "element_type": 3, "now_cost": 50} for pid in PER_GW],
    "teams": [],
}


def pool(predictions, must_include=()):
    """The real two-stage narrowing: start filter, then top-5 per price bucket."""
    watchlist = create_watchlist(
        predictions, gw_data_frame(),
        min_hist_pct=0.6, max_hist_window=6, min_minutes=45,
        must_include=list(must_include), must_exclude=[], max_gw=max(GWS),
    )
    return sorted(apply_price_bucket_filter(
        watchlist, predictions, BOOTSTRAP,
        current_gw=1, horizon=5, top_n=5, must_include=list(must_include),
    ))


print("Scaling the numbers")
base = predictions_frame()
out = _apply_points_multipliers(base, [PointsMultiplierEntry(player=7, multiplier=2.0)])
check("player 7 doubled", round(float(out[out["element"] == 7]["predicted_points"].sum()), 2), 30.0)
check("player 1 untouched", round(float(out[out["element"] == 1]["predicted_points"].sum()), 2), 30.0)
check("every gameweek scaled, not just one", len(out[out["element"] == 7]), 5)
check(
    "the input frame is not mutated",
    round(float(base[base["element"] == 7]["predicted_points"].sum()), 2), 15.0,
)
check("no entries is a pass-through", _apply_points_multipliers(base, []) is base, True)
check(
    "an unknown player id is ignored, not an error",
    sorted(_apply_points_multipliers(
        base, [PointsMultiplierEntry(player=999, multiplier=3.0)],
    )["element"].unique().tolist()),
    sorted(PER_GW),
)

print("\nThe ordering: a boost must be able to buy a place in the candidate pool")
check("unboosted, player 7 is below the bucket cut", pool(predictions_frame()), [1, 2, 3, 4, 5])
boosted = _apply_points_multipliers(
    predictions_frame(), [PointsMultiplierEntry(player=7, multiplier=2.0)],
)
# 7 at 2.0x scores 6.0/GW, tying player 1 and beating 2..6, so the top five become
# 1, 7, 2, 3, 4 — he is in and the old fifth man (5) is the one pushed out. The bucket
# still keeps exactly five; a boost buys a slot, it does not widen the pool.
check("boosted BEFORE the pool is built, he is in", pool(boosted), [1, 2, 3, 4, 7])
check(
    "and the man he displaced (5) is out",
    (5 in pool(boosted), len(pool(boosted))),
    (False, 5),
)

print("\nApplied AFTER the pool is built, the same boost changes nothing")
cut = pool(predictions_frame())
check("player 7 still absent from an already-built pool", 7 in cut, False)
check(
    "so the old call site could not have rescued him",
    sorted(set(cut) & {7}),
    [],
)

print("\nFading a player can push him out")
faded = _apply_points_multipliers(
    predictions_frame(), [PointsMultiplierEntry(player=1, multiplier=0.1)],
)
check("player 1 drops below the cut at 0.1x", 1 in pool(faded), False)
check("player 6 takes the freed slot", 6 in pool(faded), True)

print("\nA multiplier cannot get a player past the START filter — that counts appearances")
# Player 8 played one of five fixtures: ceil(5 * 0.6) = 3 required, so he is ineligible
# however many points he is credited with.
sparse_gw = pd.concat([
    gw_data_frame(),
    pd.DataFrame([
        {"element": 8, "GW": gw, "value": 50,
         "minutes": 90 if gw == 1 else 0,
         "kickoff_time": f"2020-08-{gw:02d}T12:00:00Z"}
        for gw in GWS
    ]),
], ignore_index=True)
sparse_pred = pd.concat([
    predictions_frame(),
    pd.DataFrame([
        {"element": 8, "event": gw, "predicted_points": 1.0,
         "position": "MID", "hist_games": 1}
        for gw in GWS
    ]),
], ignore_index=True)
huge = _apply_points_multipliers(sparse_pred, [PointsMultiplierEntry(player=8, multiplier=5.0)])
eligible = create_watchlist(
    huge, sparse_gw, min_hist_pct=0.6, max_hist_window=6, min_minutes=45,
    must_include=[], must_exclude=[], max_gw=max(GWS),
)
check("required starts for player 8", math.ceil(5 * 0.6), 3)
check("even at 5x he is not eligible", 8 in eligible, False)

print("\nReading the standing list out of app_config")


class _Resp:
    def __init__(self, data):
        self.data = data


class _StubSupabase:
    """Just enough of the client for _global_points_multipliers to run."""

    def __init__(self, value):
        self.value = value

    def table(self, _name):
        return self

    def select(self, _cols):
        return self

    def eq(self, _col, _val):
        return self

    def maybe_single(self):
        return self

    def execute(self):
        if isinstance(self.value, Exception):
            raise self.value
        return _Resp({"value": self.value})


def read(value):
    A._supabase = _StubSupabase(value)
    A._MULTIPLIER_CACHE["value"] = None
    A._MULTIPLIER_CACHE["fetched_at"] = 0.0
    return {e.player: e.multiplier for e in A._global_points_multipliers()}


check(
    "a normal row parses, string keys become ints",
    read({"enabled": True, "players": {"432": 0.9, "388": 1.2}}),
    {432: 0.9, 388: 1.2},
)
check("enabled: false switches the list off", read({"enabled": False, "players": {"432": 0.9}}), {})
check("a missing enabled key defaults to on", read({"players": {"432": 0.9}}), {432: 0.9})
check("no row at all", read(None), {})
check("a malformed row", read([1, 2, 3]), {})
check(
    "one bad entry does not take the good ones with it",
    read({"players": {"432": 0.9, "not-an-id": 1.2, "388": "x"}}),
    {432: 0.9},
)
check("an unreadable Supabase yields no adjustments", read(RuntimeError("boom")), {})

print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'All checks passed.'}")
sys.exit(1 if failures else 0)
