"""
Efficient player watchlist for the MILP solver.
All data passed as parameters; no hardcoded file paths.
"""

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def create_watchlist(
    predictions: pd.DataFrame,
    gw_data: pd.DataFrame,
    min_hist_games: int = 3,
    min_hist_window: int = 10,
    must_include: Optional[List[int]] = None,
    must_exclude: Optional[List[int]] = None,
) -> List[int]:
    """
    Build a watchlist of player IDs for the MILP solver.

    Args:
        predictions: Full predictions (element, name, position, predicted_points, hist_games).
        gw_data: Current season GW data (element, value, GW for costs).
        min_hist_games: Minimum 60+ min games threshold.
        min_hist_window: Only count games from the last N GWs when checking min_hist_games.
        must_include: Player IDs to always include (e.g. current squad).
        must_exclude: Player IDs to always exclude.

    Returns:
        List of player IDs.
    """
    must_include = must_include or []
    must_exclude = must_exclude or []

    # 1. Count recent 60+ min appearances within the last min_hist_window GWs
    gw_col = "GW" if "GW" in gw_data.columns else None
    if gw_col:
        max_gw = int(gw_data[gw_col].max())
        window_start = max_gw - min_hist_window + 1
        recent_gw = gw_data[(gw_data[gw_col] >= window_start) & (gw_data["minutes"] >= 60)]
        recent_counts = recent_gw.groupby("element").size().reset_index(name="recent_hist_games")
        logger.info(
            "Recent window: GW %d-%d (%d GWs), %d players with 60+ min appearances",
            window_start, max_gw, min_hist_window, len(recent_counts),
        )
    else:
        logger.warning("No GW column in gw_data — falling back to all-time hist_games")
        recent_counts = None

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
    #     pool so initial-squad constraints stay feasible.
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
        logger.info(
            "Added %d must-include players missing from predictions: %s",
            len(missing_ids), sorted(missing_ids),
        )

    # 7. Filter remaining by recent_hist_games >= min_hist_games
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
