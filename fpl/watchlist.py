"""
Efficient player watchlist for the MILP solver.
All data passed as parameters; no hardcoded file paths.
"""

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# How long after kickoff a fixture is assumed to be over: 90 min + halftime + stoppage,
# plus margin. Same value, and the same reason, as fpl-lad's agent/gw-status.ts and its
# fpl-sync history filter.
#
# It is a time test and NOT FPL's `finished` flag on purpose: `finished` means BONUS IS
# FINAL, not "this match was played", and stays false until the whole gameweek closes.
# Measured 2026-09-13, all seven GW4 matches played the day before were still
# `finished: false`. Treating that as "not played" would zero out a club's opportunities
# for its own completed matches — the exact inverse of the bug this guards.
ASSUMED_MATCH_DURATION = pd.Timedelta(minutes=135)


def _played_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows whose fixture has actually been played.

    Falls back to returning everything when there is no usable `kickoff_time` — the
    pre-season proxy path synthesises rows without one, and there a permissive filter is
    correct anyway (the window is empty and eligibility is effectively off).
    """
    if "kickoff_time" not in df.columns or df.empty:
        return df
    kickoff = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True)
    if kickoff.isna().all():
        return df
    cutoff = pd.Timestamp.now(tz="UTC") - ASSUMED_MATCH_DURATION
    # A row we cannot date is kept: dropping it would silently shrink a club's
    # opportunities, which is the direction that wrongly excludes players.
    return df[kickoff.isna() | (kickoff <= cutoff)]


def create_watchlist(
    predictions: pd.DataFrame,
    gw_data: pd.DataFrame,
    min_hist_pct: float = 0.6,
    max_hist_window: int = 6,
    min_minutes: int = 45,
    must_include: Optional[List[int]] = None,
    must_exclude: Optional[List[int]] = None,
    max_gw: Optional[int] = None,
) -> List[int]:
    """
    Build a watchlist of player IDs for the MILP solver.

    Args:
        predictions: Full predictions (element, name, position, predicted_points, hist_games).
        gw_data: Current season GW data (element, value, GW for costs).
        min_hist_pct: Fraction of the recent window a player must have started
            (`min_minutes`+ min) to enter the candidate pool, e.g. 0.6 = 60%.
        max_hist_window: Upper bound on how many recent GWs are considered. Early
            in the season, before this many GWs exist, the window shrinks to
            however many GWs have actually been played — there is no lower bound,
            so a single-GW season still lets qualifying players through.
        min_minutes: Minutes in a fixture that count as an appearance. A parameter
            rather than a literal 60 because fpl-lad's `player_eligibility` view
            reproduces this same filter for Alfie's answers from a shared config row
            (app_config.player_eligibility), and the two must not be able to disagree
            about what an appearance is.
        must_include: Player IDs to always include (e.g. current squad).
        must_exclude: Player IDs to always exclude.
        max_gw: Last gameweek the eligibility window may see. Defaults to whatever
            gw_data holds, which is right only when `predictions` was built from that
            same gw_data. It usually isn't: the API serves predictions from a table
            refreshed an hour AFTER the history sync, so during that hour eligibility
            would be judged on newer data than the points it selects on. Pass the
            predictions' own vintage and the two halves can never disagree.

            This matters because the skew is one-directional. A player becomes newly
            eligible precisely BECAUSE he just started a match, and the prediction that
            predates that match rests on a smaller sample — which is exactly where
            small-sample inflation lives. Observed 2026-09-05: Yalcouyé cleared this
            filter on GW1-3 (2 starts, threshold 2) while priced from GW1-2 at 12.19 for
            GW4. On GW1-3 pricing he is 7.43; on GW1-2 eligibility he has 1 start against
            a threshold of 2 and never enters the pool. Only the mismatch selects him,
            and it put him in a Free Hit squad with the armband.

    Returns:
        List of player IDs.
    """
    must_include = must_include or []
    must_exclude = must_exclude or []

    # Exclusion wins, everywhere and permanently. Without this it only won until step 6b:
    # step 5 drops the excluded players out of `merged`, and step 6b then adds back every
    # must_include player that isn't in `merged` — which step 5 has just guaranteed they
    # aren't. Since the current squad is always in must_include, "never own this player"
    # silently did nothing for anyone the manager already owns.
    if must_exclude:
        excluded = set(must_exclude)
        dropped = [p for p in must_include if p in excluded]
        must_include = [p for p in must_include if p not in excluded]
        if dropped:
            logger.info("Excluded players dropped from must_include: %s", sorted(dropped))

    # 1. Count recent `min_minutes`+ appearances within the last `window_size` GWs,
    #    where window_size is capped at max_hist_window but shrinks to whatever
    #    GWs actually exist early in the season.
    gw_col = "GW" if "GW" in gw_data.columns else None
    if gw_col:
        data_max_gw = int(gw_data[gw_col].max())
        # Never look past the vintage the predictions were built from — and never past the
        # data actually held. min() rather than trusting the caller: a table stamped with a
        # gameweek this service hasn't synced yet must not widen the window.
        window_end = data_max_gw if max_gw is None else min(int(max_gw), data_max_gw)
        window_size = min(window_end, max_hist_window)
        window_start = window_end - window_size + 1
        # The upper bound is new alongside `max_gw`: without it a clamped window still
        # counts appearances from gameweeks the predictions never saw, which is the exact
        # mismatch this exists to close.
        in_window = gw_data[
            (gw_data[gw_col] >= window_start) & (gw_data[gw_col] <= window_end)
        ]

        # Only fixtures that have actually been PLAYED count — as opportunities or as
        # anything else. FPL publishes a player's row for an upcoming fixture on matchday,
        # minutes 0, and it is indistinguishable from a match he was left out of. fpl-lad's
        # sync drops those now, but this path must not depend on that having run: the
        # nightly cron lands at 11PM, so for the whole of a matchday the table still holds
        # them. `kickoff_time` is on every row from both real sources.
        played = _played_only(in_window)

        # The denominator is per player, and it is the number of fixtures his CLUB has
        # played — not the width of the window.
        #
        # It used to be `ceil(window_size * min_hist_pct)`, one number for the whole league,
        # which quietly punishes anyone whose club has played fewer matches than the window
        # is wide. On 2026-09-13, with seven of ten GW4 fixtures played, the window was 4
        # for everybody: a player whose own club had not kicked off yet needed 3 starts from
        # the 2 matches he had had the chance to start. 19 players were dropped from the
        # candidate pool that way — Foden, Cherki, Dalot, Rashford, O'Reilly, Mainoo — hours
        # before their match began. A blank gameweek and a postponement did the same thing,
        # all season, to every player at the affected club.
        #
        # FPL's element-summary gives every player at a club a row for each of that club's
        # fixtures, whether he featured or not, so counting his rows in the window IS his
        # club's played-fixture count — no team column needed, and a mid-season transfer is
        # handled for free (he can only be charged for matches he was there for).
        #
        # This mirrors `fixtures_in_window` in fpl-lad's player_eligibility view, so the two
        # sides keep meaning the same thing by "worth considering". The shared thresholds
        # still come from app_config.player_eligibility.
        #
        # `proxy_predict.ensure_players_present()` synthesises minutes=0 rows for squad
        # members missing from gw_data, and those would read as an extra opportunity here.
        # It is harmless because it only ever covers must_include players, and step 6
        # exempts those from this filter entirely — but it is the reason to keep that
        # exemption in front of this, not behind it.
        opportunities = played.groupby("element").size()
        recent_counts = (
            played[played["minutes"] >= min_minutes]
            .groupby("element")
            .size()
            .reset_index(name="recent_hist_games")
        )
        required_by_element = (
            (opportunities * min_hist_pct).apply(math.ceil).astype(int)
        )
        # Kept only for the log line and for callers that still read it: the league-wide
        # requirement a club playing every gameweek in the window would face.
        min_hist_games = math.ceil(window_size * min_hist_pct)
        if max_gw is not None and window_end < data_max_gw:
            logger.info(
                "Eligibility clamped to GW %d (the predictions' vintage) — gw_data holds GW %d",
                window_end, data_max_gw,
            )
        short = int((opportunities < window_size).sum())
        logger.info(
            "Recent window: GW %d-%d (%d GWs), requiring >= %.0f%% of each club's PLAYED "
            "fixtures (%d for a club that played them all), %d players with %d+ min "
            "appearances, %d players at clubs short of the full window",
            window_start, window_end, window_size, min_hist_pct * 100, min_hist_games,
            len(recent_counts), min_minutes, short,
        )
    else:
        logger.warning("No GW column in gw_data — falling back to all-time hist_games")
        recent_counts = None
        required_by_element = None
        min_hist_games = 0

    # 2. Total expected points per player
    pred_totals = (
        predictions.groupby("element", as_index=False)
        .agg(
            predicted_points=("predicted_points", "sum"),
            hist_games=("hist_games", "first"),
        )
    )

    # 3. Latest player costs from gw_data
    sort_cols = ["element"] + ([gw_col] if gw_col else [])
    gw_sorted = gw_data.sort_values(sort_cols)
    costs = gw_sorted.groupby("element", as_index=False).last()[["element", "value"]]
    costs = costs.rename(columns={"value": "cost"})

    # 4. Merge predictions with costs and recent counts
    merged = pred_totals.merge(costs, on="element", how="left")
    if recent_counts is not None:
        merged = merged.merge(recent_counts, on="element", how="left")
        merged["recent_hist_games"] = merged["recent_hist_games"].fillna(0).astype(int)
    else:
        merged["recent_hist_games"] = merged["hist_games"]

    # 5. Remove must_exclude
    if must_exclude:
        merged = merged[~merged["element"].isin(must_exclude)]

    # 6. Separate must_include (exempt from filtering)
    include_mask = merged["element"].isin(must_include)
    must_include_df = merged[include_mask]
    remaining = merged[~include_mask]

    # 6b. Add must_include players that have no predictions (e.g. bench GKs
    #     who never played 60+ min).  They need to be in the solver's player
    #     pool so initial-squad constraints stay feasible. must_exclude has
    #     already been removed from must_include above, so this cannot undo it.
    missing_ids = set(must_include) - set(merged["element"].tolist())
    if missing_ids:
        gw_col_local = "GW" if "GW" in gw_data.columns else None
        sort_cols_local = ["element"] + ([gw_col_local] if gw_col_local else [])
        latest_gw = gw_data.sort_values(sort_cols_local).groupby("element").last().reset_index()
        missing_rows = latest_gw[latest_gw["element"].isin(missing_ids)][["element", "value"]].copy()
        missing_rows["predicted_points"] = 0.0
        missing_rows["hist_games"] = 0
        missing_rows["recent_hist_games"] = 0
        missing_rows = missing_rows.rename(columns={"value": "cost"})
        must_include_df = pd.concat([must_include_df, missing_rows], ignore_index=True)
        # Report what was actually added, not what was asked for. A player absent from
        # gw_data as well has no row to resurrect and is skipped here — this used to
        # claim it had added him, which is a hard thing to debug when the solve then
        # fails somewhere else entirely.
        added_ids = sorted(missing_rows["element"].tolist())
        logger.info(
            "Added %d must-include players missing from predictions: %s",
            len(added_ids), added_ids,
        )
        skipped_ids = sorted(missing_ids - set(added_ids))
        if skipped_ids:
            logger.warning(
                "Could not add must-include players %s — no gw_data row either, so they "
                "are not in the solver's pool at all",
                skipped_ids,
            )

    # 7. Filter remaining by appearances against that player's OWN requirement. A player
    #    with no row in the window at all has no opportunities, so his requirement is 0 and
    #    he passes on 0 appearances — same as the pre-season case where the window is empty
    #    and the filter is effectively off. He still needs a prediction to be in `merged`.
    if required_by_element is not None:
        required = remaining["element"].map(required_by_element).fillna(0).astype(int)
        filtered = remaining[remaining["recent_hist_games"] >= required]
    else:
        filtered = remaining[remaining["recent_hist_games"] >= min_hist_games]

    # 8. Combine and deduplicate
    combined = pd.concat([filtered, must_include_df], ignore_index=True)
    watchlist_ids = combined["element"].drop_duplicates().astype(int).tolist()

    # 9. Log summary by position
    if "position" in predictions.columns:
        pos_counts = predictions[predictions["element"].isin(watchlist_ids)].groupby(
            "position"
        )["element"].nunique()
        total_before = predictions["element"].nunique()
        total_after = len(watchlist_ids)
        pct = 100 * total_after / total_before if total_before else 0
        logger.info(
            "Watchlist: %d players (%.1f%% retention). By position: %s",
            total_after,
            pct,
            pos_counts.to_dict(),
        )
    else:
        logger.info("Watchlist: %d players", len(watchlist_ids))

    return watchlist_ids


# EXPERIMENTAL — opt-in only via the `bucket_top_n` request field; default
# pipeline behaviour is unchanged when it's left unset.
def apply_price_bucket_filter(
    watchlist_ids: List[int],
    predictions: pd.DataFrame,
    bootstrap: Dict[str, Any],
    current_gw: int,
    horizon: int,
    top_n: int,
    must_include: Optional[List[int]] = None,
) -> List[int]:
    """
    Further restrict an existing watchlist to the top `top_n` players per
    (position, price-bucket) group, ranked by average predicted points over
    the next `horizon` gameweeks starting at `current_gw`.

    Price is bucketed to the nearest quarter-million at or above the player's
    actual price (4.7 -> 4.75, 5.4 -> 5.5), so a group only ever contains
    players a manager could swap for one another within budget.

    must_include players (current squad, forced starts, extra_players) are
    exempt and always kept, matching create_watchlist's own must_include
    semantics — this is a further cut on top of that function's output, not a
    replacement for it.
    """
    must_include_set = set(must_include or [])
    watchlist_set = set(watchlist_ids)

    element_type = {int(e["id"]): e.get("element_type") for e in bootstrap.get("elements", [])}
    now_cost = {int(e["id"]): e.get("now_cost", 0) for e in bootstrap.get("elements", [])}

    window_end = current_gw + horizon - 1
    windowed = predictions[
        (predictions["event"] >= current_gw) & (predictions["event"] <= window_end)
    ]
    avg_points = windowed.groupby("element")["predicted_points"].mean()

    def bucket_key(pid: int) -> tuple:
        return (element_type.get(pid), math.ceil(now_cost.get(pid, 0) / 10 * 4) / 4)

    groups: Dict[tuple, List[tuple]] = defaultdict(list)
    for pid in watchlist_ids:
        if pid in must_include_set:
            continue
        groups[bucket_key(pid)].append((pid, float(avg_points.get(pid, 0.0))))

    kept = watchlist_set & must_include_set
    for members in groups.values():
        members.sort(key=lambda m: -m[1])
        kept.update(pid for pid, _ in members[:top_n])

    filtered_ids = [pid for pid in watchlist_ids if pid in kept]
    logger.info(
        "Price-bucket filter: %d -> %d players (top %d per position/price bucket, "
        "%d-GW horizon GW%d-GW%d)",
        len(watchlist_ids), len(filtered_ids), top_n, horizon, current_gw, window_end,
    )
    return filtered_ids
