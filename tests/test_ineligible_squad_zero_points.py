"""Owned players who fail the recent-starts or absent-streak filter must stay in
the pool (so initial-squad constraints remain feasible) while having their
predicted points zeroed out — unless explicitly promoted via extra_players,
forced_lineup, or GW_POINT_OVERRIDES.

Runnable directly (``python tests/test_ineligible_squad_zero_points.py``) or under pytest.
"""

from datetime import datetime, timedelta, timezone
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from fpl.watchlist import create_watchlist, zero_ineligible_squad_predictions  # noqa: E402

LONG_AGO = datetime.now(timezone.utc) - timedelta(days=3)
GWS = [5, 6, 7, 8, 9, 10]
FUTURE_GWS = [11, 12, 13]

REGULAR_STARTER = 101   # 6/6 starts
BARELY_PLAYING = 327    # 1/6 starts (Bijol shape), high per-appearance xP
SENESI_SHAPE = 498      # 1/6 starts (Senesi shape), high per-appearance xP
STREAK_ABSENT = 173     # 4/6 starts early, then 0 min in last 2 matches


def _rows(element, minutes):
    return [
        {
            "element": element,
            "GW": gw,
            "fixture": 1000 * element + gw,
            "minutes": m,
            "value": 50,
            "kickoff_time": (LONG_AGO - timedelta(days=7 * (12 - gw))).isoformat(),
        }
        for gw, m in zip(GWS, minutes)
    ]


def _gw_data():
    return pd.DataFrame(
        _rows(REGULAR_STARTER, [90, 90, 90, 90, 90, 90])
        + _rows(BARELY_PLAYING, [0, 4, 0, 90, 0, 0])
        + _rows(SENESI_SHAPE, [0, 88, 0, 0, 0, 0])
        + _rows(STREAK_ABSENT, [90, 90, 90, 90, 0, 0])
    )


def _predictions():
    rows = []
    for gw in FUTURE_GWS:
        rows.extend([
            {"element": REGULAR_STARTER, "event": gw, "predicted_points": 4.0, "hist_games": 6, "position": "DEF"},
            {"element": BARELY_PLAYING, "event": gw, "predicted_points": 5.0, "hist_games": 1, "position": "DEF"},
            {"element": SENESI_SHAPE, "event": gw, "predicted_points": 4.8, "hist_games": 1, "position": "DEF"},
            {"element": STREAK_ABSENT, "event": gw, "predicted_points": 4.7, "hist_games": 4, "position": "DEF"},
        ])
    return pd.DataFrame(rows)


def _run_filter(current_squad, explicit_include=()):
    preds = _predictions()
    gw_data = _gw_data()
    must_include = list(dict.fromkeys(list(current_squad) + list(explicit_include)))

    eligible_watchlist = create_watchlist(
        preds, gw_data,
        min_hist_pct=0.6, max_hist_window=6, min_minutes=55, absent_streak_games=2,
        must_include=list(explicit_include), must_exclude=[], max_gw=10,
    )
    watchlist = create_watchlist(
        preds, gw_data,
        min_hist_pct=0.6, max_hist_window=6, min_minutes=55, absent_streak_games=2,
        must_include=must_include, must_exclude=[], max_gw=10,
    )
    adjusted_preds, ineligible_owned = zero_ineligible_squad_predictions(
        preds, current_squad, eligible_watchlist,
    )
    return watchlist, eligible_watchlist, adjusted_preds, ineligible_owned


def test_ineligible_squad_players_stay_in_pool_but_get_zero_points():
    squad = [REGULAR_STARTER, BARELY_PLAYING, SENESI_SHAPE, STREAK_ABSENT]
    watchlist, eligible, preds, ineligible = _run_filter(squad)

    # All owned players remain in the pool so initial-squad constraints are feasible.
    assert set(squad).issubset(set(watchlist))
    # Only the regular starter is genuinely eligible.
    assert eligible == [REGULAR_STARTER]
    assert ineligible == [BARELY_PLAYING, SENESI_SHAPE, STREAK_ABSENT]

    # Regular starter keeps his 4.0/GW; ineligible squad members are zeroed across all GWs.
    assert float(preds[preds["element"] == REGULAR_STARTER]["predicted_points"].sum()) == 12.0
    for pid in (BARELY_PLAYING, SENESI_SHAPE, STREAK_ABSENT):
        assert float(preds[preds["element"] == pid]["predicted_points"].sum()) == 0.0


def test_extra_players_override_preserves_cherry_picked_player_points():
    squad = [REGULAR_STARTER, BARELY_PLAYING, SENESI_SHAPE]
    watchlist, eligible, preds, ineligible = _run_filter(
        squad, explicit_include=[BARELY_PLAYING],
    )

    assert BARELY_PLAYING in eligible
    assert ineligible == [SENESI_SHAPE]
    assert float(preds[preds["element"] == BARELY_PLAYING]["predicted_points"].sum()) == 15.0
    assert float(preds[preds["element"] == SENESI_SHAPE]["predicted_points"].sum()) == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
