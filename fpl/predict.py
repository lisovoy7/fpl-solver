"""
FPL prediction engine: normalized stats and component-based point predictions.

Merges normalization and prediction logic. All data is passed as function parameters;
no hardcoded file paths, season strings, or file I/O in business logic.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Constants
MIN_MINUTES = 60
LAST_N_GAMES = 6

GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}

NORMALIZABLE_COMPONENTS = [
    "assists",
    "bonus",
    "clean_sheets",
    "expected_assists",
    "expected_goals",
    "expected_goals_conceded",
    "goals_conceded",
    "goals_scored",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "saves",
    "yellow_cards",
]

PREDICTION_COMPONENTS = [
    "minutes_played",
    "goals_scored",
    "assists",
    "saves",
    "conceded_goals",
    "yellow_cards",
    "clean_sheet",
    "defensive_contribution",
    "bps",
]

# Population std for defensive contribution probability (when player history insufficient)
POPULATION_STD = {"DEF": 3.5, "MID": 3.9, "FWD": 2.8}

# Raw XGC multipliers for defensive fixture difficulty (tier -> multiplier)
RAW_XGC_MULTIPLIERS = {1.0: 1.126, 2.0: 1.040, 3.0: 1.000, 4.0: 0.735, 5.0: 0.641}


def _normalize_stats(
    gw_data: pd.DataFrame,
    fixtures: pd.DataFrame,
    multipliers: pd.DataFrame,
    team_tiers: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """
    Normalize player stats by removing fixture difficulty bias using multipliers.

    Filters to minutes >= MIN_MINUTES, merges with fixtures for team info,
    adds team tiers, and normalizes each component by dividing by fixture multiplier.
    """
    logger.debug("Normalizing stats for season %s", season)

    filtered = gw_data[gw_data["minutes"] >= MIN_MINUTES].copy()
    filtered["season"] = season
    logger.debug("Filtered to %d records with minutes >= %d", len(filtered), MIN_MINUTES)

    # Merge with fixtures
    gw_data_copy = filtered.copy()
    gw_data_copy["kickoff_time"] = pd.to_datetime(gw_data_copy["kickoff_time"])
    fixtures_copy = fixtures.copy()
    fixtures_copy["kickoff_time"] = pd.to_datetime(fixtures_copy["kickoff_time"])
    fixtures_copy["season"] = season

    fixture_cols = ["id", "season", "team_h", "team_a", "kickoff_time"]
    available_fixture_cols = [c for c in fixture_cols if c in fixtures_copy.columns]
    if "id" not in available_fixture_cols:
        raise ValueError("Fixtures must have 'id' column for merge")

    merged = gw_data_copy.merge(
        fixtures_copy[available_fixture_cols],
        left_on=["fixture", "season"],
        right_on=["id", "season"],
        how="left",
        suffixes=("", "_fixture"),
    )
    if "id" in merged.columns and "id_fixture" not in merged.columns:
        merged = merged.drop(columns=["id"], errors="ignore")

    # Derive player_team_id
    merged["player_team_id"] = merged.apply(
        lambda row: row["team_h"] if row["was_home"] else row["team_a"], axis=1
    )

    # Add team tiers
    player_tier_map = team_tiers[["team_id", "team_tier"]].rename(
        columns={"team_id": "player_team_id", "team_tier": "player_team_tier"}
    )
    merged = merged.merge(player_tier_map, on="player_team_id", how="left")

    opponent_tier_map = team_tiers[["team_id", "team_tier"]].rename(
        columns={"team_id": "opponent_team", "team_tier": "opponent_team_tier"}
    )
    merged = merged.merge(opponent_tier_map, on="opponent_team", how="left")

    merged["is_home"] = merged["was_home"].astype(int)

    # Cast normalizable components to numeric (API sometimes returns strings)
    for col in NORMALIZABLE_COMPONENTS:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    # Normalize each component
    merge_keys = ["position", "player_team_tier", "opponent_team_tier", "is_home"]
    for component in NORMALIZABLE_COMPONENTS:
        if component not in merged.columns:
            logger.warning("Component '%s' not in data, skipping", component)
            continue

        comp_mult = multipliers[multipliers["component_type"] == component]
        if len(comp_mult) == 0:
            logger.warning("No multipliers for '%s', skipping", component)
            continue

        missing = [k for k in merge_keys if k not in merged.columns]
        if missing:
            logger.warning("Missing keys for %s: %s", component, missing)
            continue

        merged = merged.merge(
            comp_mult[merge_keys + ["multiplier"]],
            on=merge_keys,
            how="left",
            suffixes=("", f"_{component}"),
        )
        mult_col = "multiplier"
        if f"multiplier_{component}" in merged.columns:
            mult_col = f"multiplier_{component}"

        norm_col = f"norm_{component}"
        merged[norm_col] = np.where(
            (merged[mult_col].isna()) | (merged[mult_col] == 0),
            merged[component],
            merged[component] / merged[mult_col],
        )
        merged = merged.drop(columns=[mult_col], errors="ignore")

    logger.debug("Normalization complete: %d records", len(merged))
    return merged


def _calculate_player_averages(
    normalized_stats: pd.DataFrame,
    gw_data: pd.DataFrame,
    last_n_games: int = LAST_N_GAMES,
) -> pd.DataFrame:
    """
    Calculate player averages from last N games (normalized and raw components).
    """
    logger.debug("Calculating player averages from last %d games", last_n_games)

    gw_filtered = gw_data[gw_data["minutes"] >= MIN_MINUTES].copy()
    gw_filtered["kickoff_time"] = pd.to_datetime(gw_filtered["kickoff_time"])
    for col in ["defensive_contribution", "bonus", "yellow_cards"]:
        if col in gw_filtered.columns:
            gw_filtered[col] = pd.to_numeric(gw_filtered[col], errors="coerce").fillna(0)
    stats_sorted = normalized_stats.sort_values(["element", "kickoff_time"])

    records = []
    for element, group in stats_sorted.groupby("element"):
        recent = group.tail(last_n_games)
        hist_games = len(recent)

        player_gw = gw_filtered[gw_filtered["element"] == element]
        if len(player_gw) > 0:
            gw_sorted = player_gw.sort_values("kickoff_time")
            recent_gw = gw_sorted.tail(last_n_games)
            avg_def = recent_gw["defensive_contribution"].mean()
            def_history = recent_gw["defensive_contribution"].tolist()
            avg_bonus = (
                recent_gw["bonus"].mean()
                if "bonus" in recent_gw.columns
                else 0.0
            )
        else:
            avg_def = 0.0
            def_history = []
            avg_bonus = 0.0

        records.append(
            {
                "element": element,
                "hist_games": hist_games,
                "avg_norm_expected_goals": (
                    recent["norm_expected_goals"].mean()
                    if "norm_expected_goals" in recent.columns
                    else 0.0
                ),
                "avg_norm_expected_assists": (
                    recent["norm_expected_assists"].mean()
                    if "norm_expected_assists" in recent.columns
                    else 0.0
                ),
                "avg_norm_saves": (
                    recent["norm_saves"].mean()
                    if "norm_saves" in recent.columns
                    else 0.0
                ),
                "avg_norm_expected_goals_conceded": (
                    recent["norm_expected_goals_conceded"].mean()
                    if "norm_expected_goals_conceded" in recent.columns
                    else 0.0
                ),
                "avg_yellow_cards": (
                    recent["yellow_cards"].mean()
                    if "yellow_cards" in recent.columns
                    else 0.0
                ),
                "avg_defensive_contribution": avg_def,
                "avg_bonus_points": avg_bonus,
                "defensive_contribution_history": def_history,
            }
        )

    df = pd.DataFrame(records)
    logger.debug("Calculated averages for %d players", len(df))
    return df


def _get_player_team_assignments(normalized_stats: pd.DataFrame) -> pd.DataFrame:
    """Get latest team assignment per player from normalized stats."""
    logger.debug("Determining player team assignments")
    sorted_stats = normalized_stats.sort_values(["element", "kickoff_time"])
    latest = sorted_stats.groupby("element").last().reset_index()
    assignments = latest[
        ["element", "name", "position", "player_team_id", "player_team_tier"]
    ].copy()
    logger.debug("Team assignments for %d players", len(assignments))
    return assignments


def _generate_player_fixture_combinations(
    player_assignments: pd.DataFrame,
    fixtures: pd.DataFrame,
    team_tiers: pd.DataFrame,
    last_played_gw: int,
) -> pd.DataFrame:
    """
    Generate player-fixture combinations for future fixtures only.
    """
    logger.debug("Generating player-fixture combinations (future fixtures only)")

    fixtures_copy = fixtures.copy()
    fixtures_copy["kickoff_time"] = pd.to_datetime(fixtures_copy["kickoff_time"])

    # Resolve event/gw column
    event_col = "event" if "event" in fixtures_copy.columns else "GW"
    if event_col not in fixtures_copy.columns:
        raise ValueError("Fixtures must have 'event' or 'GW' column")

    future_fixtures = fixtures_copy[
        fixtures_copy[event_col] > last_played_gw
    ].copy()
    logger.debug("Future fixtures: %d (event > %d)", len(future_fixtures), last_played_gw)

    combinations = []
    for _, prow in player_assignments.iterrows():
        pid = prow["player_team_id"]
        for _, frow in future_fixtures.iterrows():
            team_h = frow["team_h"]
            team_a = frow["team_a"]
            if pid == team_h:
                is_home = 1
                opponent = team_a
            elif pid == team_a:
                is_home = 0
                opponent = team_h
            else:
                continue

            opp_tier = team_tiers[team_tiers["team_id"] == opponent]["team_tier"]
            opp_tier_val = opp_tier.iloc[0] if len(opp_tier) > 0 else np.nan

            combinations.append(
                {
                    "element": prow["element"],
                    "name": prow["name"],
                    "position": prow["position"],
                    "player_team_id": pid,
                    "player_team_tier": prow["player_team_tier"],
                    "event": frow[event_col],
                    "kickoff_time": frow["kickoff_time"],
                    "opponent_team": opponent,
                    "opponent_team_tier": opp_tier_val,
                    "is_home": is_home,
                }
            )

    comb_df = pd.DataFrame(combinations)
    logger.debug("Created %d player-fixture combinations", len(comb_df))
    return comb_df


def _get_defensive_fixture_multiplier(
    opponent_team_tier: float, is_home: bool
) -> float:
    """Defensive fixture multiplier from opponent tier and location."""
    raw = RAW_XGC_MULTIPLIERS.get(
        float(opponent_team_tier) if not np.isnan(opponent_team_tier) else 3.0,
        1.0,
    )
    loc_mult = 0.846 if is_home else 1.000
    adjusted = raw * loc_mult
    return 0.269 * adjusted + 0.702


def _calculate_defensive_probability(
    predicted_value: float,
    position: str,
    player_history: Optional[List[float]] = None,
) -> float:
    """
    Probability of reaching defensive contribution threshold using normal distribution.
    Uses player-specific consistency when history available.
    """
    if position == "GK":
        return 0.0

    threshold = 10 if position == "DEF" else 12
    pop_std = POPULATION_STD.get(position, 3.5)

    if player_history and len(player_history) >= 1:
        hist = np.array(player_history)
        success_rate = (hist >= threshold).mean()
        player_std = hist.std() if len(hist) > 1 else 0.0
        mean_val = hist.mean()

        if len(hist) >= 4:
            if success_rate >= 0.8:
                floor = 0.70 if success_rate >= 0.9 else 0.60
                adj = success_rate * (
                    predicted_value / mean_val if mean_val > 0 else 1.0
                )
                return max(floor, min(0.95, adj))
            elif success_rate <= 0.2:
                adj = success_rate * (
                    predicted_value / mean_val if mean_val > 0 else 1.0
                )
                return min(0.30, max(0.0, adj))
            else:
                weight = min(0.7, len(hist) / 10.0)
                std_dev = weight * player_std + (1 - weight) * pop_std
        else:
            std_dev = pop_std

        if std_dev > 0:
            prob = 1 - stats.norm.cdf(threshold, predicted_value, std_dev)
            return max(0.0, min(1.0, prob))
        return 1.0 if predicted_value >= threshold else 0.0

    if pop_std > 0:
        prob = 1 - stats.norm.cdf(threshold, predicted_value, pop_std)
        return max(0.0, min(1.0, prob))
    return 1.0 if predicted_value >= threshold else 0.0


def _create_component_predictions(
    combinations: pd.DataFrame,
    player_averages: pd.DataFrame,
    multipliers: pd.DataFrame,
) -> pd.DataFrame:
    """Create predictions for all PREDICTION_COMPONENTS."""
    merge_keys = ["position", "player_team_tier", "opponent_team_tier", "is_home"]
    all_preds = []

    # 1. minutes_played
    p = combinations.merge(
        player_averages[["element", "hist_games"]], on="element", how="left"
    )
    p["component_type"] = "minutes_played"
    p["predicted_points"] = 2.0
    p["hist_games"] = p["hist_games"].fillna(0)
    all_preds.append(p)

    # 2. goals_scored
    xg_mult = multipliers[multipliers["component_type"] == "expected_goals"]
    p = combinations.merge(
        player_averages[["element", "hist_games", "avg_norm_expected_goals"]],
        on="element",
        how="left",
    )
    p = p.merge(
        xg_mult[merge_keys + ["multiplier"]], on=merge_keys, how="left"
    )
    mult = p["multiplier"].fillna(1.0)
    pred_xg = p["avg_norm_expected_goals"].fillna(0) * mult
    p["predicted_points"] = pred_xg * p["position"].map(GOAL_POINTS).fillna(4)
    p["component_type"] = "goals_scored"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["multiplier"], errors="ignore")
    all_preds.append(p)

    # 3. assists
    xa_mult = multipliers[multipliers["component_type"] == "expected_assists"]
    p = combinations.merge(
        player_averages[["element", "hist_games", "avg_norm_expected_assists"]],
        on="element",
        how="left",
    )
    p = p.merge(
        xa_mult[merge_keys + ["multiplier"]], on=merge_keys, how="left"
    )
    mult = p["multiplier"].fillna(1.0)
    pred_xa = p["avg_norm_expected_assists"].fillna(0) * mult
    p["predicted_points"] = pred_xa * 3.0
    p["component_type"] = "assists"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["multiplier"], errors="ignore")
    all_preds.append(p)

    # 4. saves (GK only)
    saves_mult = multipliers[multipliers["component_type"] == "saves"]
    p = combinations.merge(
        player_averages[["element", "hist_games", "avg_norm_saves"]],
        on="element",
        how="left",
    )
    p = p.merge(
        saves_mult[merge_keys + ["multiplier"]], on=merge_keys, how="left"
    )
    mult = p["multiplier"].fillna(1.0)
    pred_saves = p["avg_norm_saves"].fillna(0) * mult
    p["predicted_points"] = np.where(
        p["position"] == "GK", pred_saves * (1.0 / 3.0), 0.0
    )
    p["component_type"] = "saves"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["multiplier"], errors="ignore")
    all_preds.append(p)

    # 5. conceded_goals (GK/DEF only)
    xgc_mult = multipliers[
        multipliers["component_type"] == "expected_goals_conceded"
    ]
    p = combinations.merge(
        player_averages[
            ["element", "hist_games", "avg_norm_expected_goals_conceded"]
        ],
        on="element",
        how="left",
    )
    p = p.merge(
        xgc_mult[merge_keys + ["multiplier"]], on=merge_keys, how="left"
    )
    mult = p["multiplier"].fillna(1.0)
    pred_xgc = p["avg_norm_expected_goals_conceded"].fillna(0) * mult
    p["predicted_points"] = np.where(
        p["position"].isin(["GK", "DEF"]), pred_xgc * (-0.5), 0.0
    )
    p["component_type"] = "conceded_goals"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["multiplier"], errors="ignore")
    all_preds.append(p)

    # 6. yellow_cards (no multiplier)
    p = combinations.merge(
        player_averages[["element", "hist_games", "avg_yellow_cards"]],
        on="element",
        how="left",
    )
    p["predicted_points"] = p["avg_yellow_cards"].fillna(0) * (-1.0)
    p["component_type"] = "yellow_cards"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["avg_yellow_cards"], errors="ignore")
    all_preds.append(p)

    # 7. clean_sheet (Poisson P(CS) = exp(-xgc))
    p = combinations.merge(
        player_averages[
            ["element", "hist_games", "avg_norm_expected_goals_conceded"]
        ],
        on="element",
        how="left",
    )
    p = p.merge(
        xgc_mult[merge_keys + ["multiplier"]], on=merge_keys, how="left"
    )
    mult = p["multiplier"].fillna(1.0)
    pred_xgc = p["avg_norm_expected_goals_conceded"].fillna(0) * mult
    p["predicted_xgc"] = pred_xgc
    p["clean_sheet_prob"] = np.exp(-pred_xgc)
    cs_pts = np.where(
        p["position"].isin(["GK", "DEF"]),
        p["clean_sheet_prob"] * 4.0,
        np.where(p["position"] == "MID", p["clean_sheet_prob"] * 1.0, 0.0),
    )
    p["predicted_points"] = cs_pts
    p["component_type"] = "clean_sheet"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["multiplier", "predicted_xgc", "clean_sheet_prob"], errors="ignore")
    all_preds.append(p)

    # 8. defensive_contribution
    p = combinations.merge(
        player_averages[
            [
                "element",
                "hist_games",
                "avg_defensive_contribution",
                "defensive_contribution_history",
            ]
        ],
        on="element",
        how="left",
    )
    pos_defaults = {"DEF": 7.15, "MID": 6.98, "FWD": 3.80, "GK": 0.0}
    pred_vals = []
    for _, row in p.iterrows():
        avg = row["avg_defensive_contribution"]
        if pd.isna(avg) or avg == 0:
            avg = pos_defaults.get(row["position"], 5.0)
        mult = _get_defensive_fixture_multiplier(
            row["opponent_team_tier"], bool(row["is_home"])
        )
        pred_val = avg * mult
        hist = row.get("defensive_contribution_history") or []
        prob = _calculate_defensive_probability(
            pred_val, row["position"], hist if isinstance(hist, list) else None
        )
        pred_vals.append(prob * 2.0)
    p["predicted_points"] = pred_vals
    p["component_type"] = "defensive_contribution"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(
        columns=[
            "avg_defensive_contribution",
            "defensive_contribution_history",
        ],
        errors="ignore",
    )
    all_preds.append(p)

    # 9. bps (direct average, no multiplier)
    p = combinations.merge(
        player_averages[["element", "hist_games", "avg_bonus_points"]],
        on="element",
        how="left",
    )
    p["predicted_points"] = p["avg_bonus_points"].fillna(0.0)
    p["component_type"] = "bps"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["avg_bonus_points"], errors="ignore")
    all_preds.append(p)

    # Standardize columns and concat
    base_cols = [
        "element",
        "name",
        "position",
        "player_team_id",
        "player_team_tier",
        "event",
        "kickoff_time",
        "opponent_team",
        "opponent_team_tier",
        "is_home",
        "hist_games",
        "component_type",
        "predicted_points",
    ]
    result = []
    for df in all_preds:
        cols = [c for c in base_cols if c in df.columns]
        result.append(df[cols].copy())

    out = pd.concat(result, ignore_index=True)
    logger.debug("Created %d total predictions across %d components", len(out), len(all_preds))
    return out


def _add_team_names(
    predictions: pd.DataFrame, team_tiers: pd.DataFrame
) -> pd.DataFrame:
    """Add player_team_name and opponent_team_name from team_tiers."""
    if "team_name" not in team_tiers.columns:
        logger.warning("team_tiers has no team_name column, skipping name mapping")
        return predictions

    id_to_name = dict(zip(team_tiers["team_id"], team_tiers["team_name"]))
    predictions = predictions.copy()
    predictions["player_team_name"] = predictions["player_team_id"].map(id_to_name)
    predictions["opponent_team_name"] = predictions["opponent_team"].map(id_to_name)
    return predictions


def generate_predictions(
    gw_data: pd.DataFrame,
    fixtures: pd.DataFrame,
    multipliers: pd.DataFrame,
    team_tiers: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """
    Generate component-based FPL point predictions for all players and future fixtures.

    Args:
        gw_data: Gameweek player data (must include element, name, position, minutes,
                 fixture, was_home, opponent_team, kickoff_time, defensive_contribution,
                 bonus, and NORMALIZABLE_COMPONENTS).
        fixtures: Fixture data with id, event (or GW), team_h, team_a, kickoff_time.
        multipliers: Component multipliers with component_type, position,
                     player_team_tier, opponent_team_tier, is_home, multiplier.
        team_tiers: Team metadata with team_id, team_tier, team_name (optional).
        season: Season string (e.g. '2025-26').

    Returns:
        DataFrame with columns: element, name, position, player_team_id, player_team_tier,
        event, kickoff_time, opponent_team, opponent_team_tier, is_home, hist_games,
        component_type, predicted_points, player_team_name, opponent_team_name.
    """
    logger.info("Generating predictions for season %s", season)

    normalized = _normalize_stats(gw_data, fixtures, multipliers, team_tiers, season)
    player_averages = _calculate_player_averages(normalized, gw_data, LAST_N_GAMES)
    player_assignments = _get_player_team_assignments(normalized)

    gw_col = "GW" if "GW" in gw_data.columns else "round"
    if gw_col not in gw_data.columns:
        raise ValueError("gw_data must have 'GW' or 'round' column")
    last_played_gw = int(gw_data[gw_col].max())

    combinations = _generate_player_fixture_combinations(
        player_assignments, fixtures, team_tiers, last_played_gw
    )

    predictions = _create_component_predictions(
        combinations, player_averages, multipliers
    )
    predictions = _add_team_names(predictions, team_tiers)

    logger.info("Prediction generation complete: %d rows", len(predictions))
    return predictions
