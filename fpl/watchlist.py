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
    must_include: Optional[List[int]] = None,
    must_exclude: Optional[List[int]] = None,
) -> List[int]:
    """
    Build a watchlist of player IDs for the MILP solver.

    Args:
        predictions: Full predictions (element, name, position, predicted_points, hist_games).
        gw_data: Current season GW data (element, value, GW for costs).
        min_hist_games: Minimum 60+ min games threshold.
        must_include: Player IDs to always include (e.g. current squad).
        must_exclude: Player IDs to always exclude.

    Returns:
        List of player IDs.
    """
    must_include = must_include or []
    must_exclude = must_exclude or []

    # 1. Total expected points per player
    pred_totals = (
        predictions.groupby("element", as_index=False)
        .agg(
            predicted_points=("predicted_points", "sum"),
            hist_games=("hist_games", "first"),
        )
    )

    # 2. Latest player costs from gw_data
    sort_cols = ["element"] + (["GW"] if "GW" in gw_data.columns else [])
    gw_sorted = gw_data.sort_values(sort_cols)
    costs = gw_sorted.groupby("element", as_index=False).last()[["element", "value"]]
    costs = costs.rename(columns={"value": "cost"})

    # 3. Merge predictions with costs
    merged = pred_totals.merge(costs, on="element", how="left")

    # 4. Remove must_exclude
    if must_exclude:
        merged = merged[~merged["element"].isin(must_exclude)]

    # 5. Separate must_include (exempt from filtering)
    include_mask = merged["element"].isin(must_include)
    must_include_df = merged[include_mask]
    remaining = merged[~include_mask]

    # 6. Filter remaining by hist_games >= min_hist_games
    filtered = remaining[remaining["hist_games"] >= min_hist_games]

    # 7. Combine and deduplicate
    combined = pd.concat([filtered, must_include_df], ignore_index=True)
    watchlist_ids = combined["element"].drop_duplicates().astype(int).tolist()

    # 8. Log summary by position
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
