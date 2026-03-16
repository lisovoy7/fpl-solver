"""
Free Hit module: chip scenario generation and Free Hit squad calculator.

Merges chip scenario generation (FH, BB, TC) with the Free Hit squad MILP optimizer.
All data passed as parameters; no hardcoded file paths.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Part 1: Chip Scenario Generation
# -----------------------------------------------------------------------------

CHIP_WINDOWS = {"first_half": (1, 19), "second_half": (20, 38)}


def _half_season_chip_options(
    used_first_half: int,
    used_second_half: int,
    first_half_gws: List[int],
    second_half_gws: List[int],
) -> List[int]:
    """
    Build the list of candidate GWs for a chip that follows the 1-per-half rule.

    Returns:
        List of GWs where the chip can be used (empty if fully exhausted).
    """
    options: List[int] = []
    if used_first_half < 1:
        options.extend(first_half_gws)
    if used_second_half < 1:
        options.extend(second_half_gws)
    return options


def generate_chip_scenarios(
    start_gw: int,
    planning_horizon: int,
    free_hits_used_first_half: int = 0,
    free_hits_used_second_half: int = 0,
    bench_boost_used_first_half: int = 0,
    bench_boost_used_second_half: int = 0,
    triple_captain_used_first_half: int = 0,
    triple_captain_used_second_half: int = 0,
    force_free_hit_gw: Optional[int] = None,
    force_bench_boost_gw: Optional[int] = None,
    force_triple_captain_gw: Optional[int] = None,
    force_wildcard_gw: Optional[int] = None,
) -> List[Dict]:
    """
    Generate all valid chip combinations (FH x BB x TC), respecting the
    1-per-half-season rule for every chip type and the 1-chip-per-GW conflict.

    When a force_*_gw is set, that chip is pinned to the given GW and no other
    options are enumerated for it — dramatically reducing the scenario count.

    Args:
        start_gw: First GW of the planning horizon.
        planning_horizon: Number of GWs to plan.
        *_first_half / *_second_half: How many of each chip have been used in
            GW 1-19 and GW 20-38 respectively (0 or 1).
        force_free_hit_gw: Pin Free Hit to this GW (None = enumerate all options).
        force_bench_boost_gw: Pin Bench Boost to this GW (None = enumerate).
        force_triple_captain_gw: Pin Triple Captain to this GW (None = enumerate).
        force_wildcard_gw: Pin Wildcard to this GW (None = let solver decide).

    Returns:
        List of scenario dicts with: name, free_hit_gws, bench_boost_gw,
        triple_captain_gw, force_wildcard_gw.  -1 means "chip not used this scenario".
    """
    all_gws = list(range(start_gw, start_gw + planning_horizon))
    first_half_gws = [
        gw for gw in all_gws
        if CHIP_WINDOWS["first_half"][0] <= gw <= CHIP_WINDOWS["first_half"][1]
    ]
    second_half_gws = [
        gw for gw in all_gws
        if CHIP_WINDOWS["second_half"][0] <= gw <= CHIP_WINDOWS["second_half"][1]
    ]

    # --- FH scenarios ---
    if force_free_hit_gw is not None:
        fh_scenarios: List[Dict] = [
            {"name": f"FH GW{force_free_hit_gw}", "free_hit_gws": [force_free_hit_gw]},
        ]
        logger.debug("Free Hit forced on GW %d — skipping FH enumeration", force_free_hit_gw)
    else:
        remaining_fh_first = max(0, 1 - free_hits_used_first_half)
        remaining_fh_second = max(0, 1 - free_hits_used_second_half)

        fh_scenarios = [{"name": "No Free Hit", "free_hit_gws": []}]
        if remaining_fh_first > 0 and first_half_gws:
            for gw in first_half_gws:
                fh_scenarios.append({"name": f"FH GW{gw}", "free_hit_gws": [gw]})
        if remaining_fh_second > 0 and second_half_gws:
            for gw in second_half_gws:
                fh_scenarios.append({"name": f"FH GW{gw}", "free_hit_gws": [gw]})
        if remaining_fh_first > 0 and remaining_fh_second > 0 and first_half_gws and second_half_gws:
            for gw1 in first_half_gws:
                for gw2 in second_half_gws:
                    fh_scenarios.append({
                        "name": f"FH GW{gw1}+{gw2}", "free_hit_gws": [gw1, gw2],
                    })

    # --- BB options ---
    if force_bench_boost_gw is not None:
        bb_options: List[int] = [force_bench_boost_gw]
        logger.debug("Bench Boost forced on GW %d — skipping BB enumeration", force_bench_boost_gw)
    else:
        bb_options = [-1]
        bb_options.extend(_half_season_chip_options(
            bench_boost_used_first_half, bench_boost_used_second_half,
            first_half_gws, second_half_gws,
        ))

    # --- TC options ---
    if force_triple_captain_gw is not None:
        tc_options: List[int] = [force_triple_captain_gw]
        logger.debug("Triple Captain forced on GW %d — skipping TC enumeration", force_triple_captain_gw)
    else:
        tc_options = [-1]
        tc_options.extend(_half_season_chip_options(
            triple_captain_used_first_half, triple_captain_used_second_half,
            first_half_gws, second_half_gws,
        ))

    # --- Cartesian product with 1-chip-per-GW conflict filter ---
    scenarios: List[Dict] = []
    for fh in fh_scenarios:
        fh_gws_set = set(fh["free_hit_gws"])
        for bb_gw in bb_options:
            for tc_gw in tc_options:
                if bb_gw != -1 and bb_gw in fh_gws_set:
                    continue
                if tc_gw != -1 and tc_gw in fh_gws_set:
                    continue
                if bb_gw != -1 and tc_gw != -1 and bb_gw == tc_gw:
                    continue

                parts = []
                if fh["free_hit_gws"]:
                    parts.append(" + ".join(f"FH GW{gw}" for gw in fh["free_hit_gws"]))
                if bb_gw != -1:
                    parts.append(f"BB GW{bb_gw}")
                if tc_gw != -1:
                    parts.append(f"TC GW{tc_gw}")
                name = " | ".join(parts) if parts else "No chips"

                scenarios.append({
                    "name": name,
                    "free_hit_gws": list(fh["free_hit_gws"]),
                    "bench_boost_gw": bb_gw,
                    "triple_captain_gw": tc_gw,
                    "force_wildcard_gw": force_wildcard_gw,
                })

    logger.info(
        "Generated %d chip scenarios for GW %d-%d",
        len(scenarios), start_gw, start_gw + planning_horizon - 1,
    )
    return scenarios


# -----------------------------------------------------------------------------
# Part 2: Free Hit Squad Calculator
# -----------------------------------------------------------------------------

SQUAD_COMPOSITION = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
LINEUP_CONSTRAINTS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
TOTAL_SQUAD_SIZE = 15
TOTAL_LINEUP_SIZE = 11
MAX_PLAYERS_PER_CLUB = 3


def calculate_optimal_free_hit_squad(
    gw: int,
    budget: float,
    predictions_df: pd.DataFrame,
    gw_data_df: pd.DataFrame,
    watchlist_players: List[int],
    forced_lineup_players: Optional[List[Tuple[int, List[int]]]] = None,
    points_multiplier_override: Optional[List[Tuple[int, float]]] = None,
    non_playing_players: Optional[List[Tuple[int, List[int]]]] = None,
) -> Dict:
    """
    Calculate optimal Free Hit squad for a gameweek.

    Accepts predictions_df and gw_data_df as parameters (no file reads).
    predictions_df: event, element, name, position, player_team_id, predicted_points.
    gw_data_df: element, value (prices; latest value per element used).

    Returns:
        Dict with: status, squad, lineup, captain, total_points, squad_cost,
        unused_budget, squad_details.
    """
    logger.debug("Calculating optimal Free Hit squad for GW %d", gw)

    # Filter predictions for target GW and watchlist
    gw_predictions = predictions_df[
        (predictions_df["event"] == gw)
        & (predictions_df["element"].isin(watchlist_players))
    ].copy()

    if len(gw_predictions) == 0:
        logger.warning("No predictions found for GW %d in watchlist", gw)
        return {
            "status": "Infeasible",
            "squad": [],
            "lineup": [],
            "captain": None,
            "total_points": 0,
            "squad_cost": 0,
            "unused_budget": budget,
            "squad_details": {},
        }

    # Get latest value per element from gw_data_df
    gw_prices_df = gw_data_df[["element", "value"]].drop_duplicates(
        subset=["element"], keep="last"
    )

    # Merge predictions with prices and aggregate by element
    gw_data = gw_predictions.merge(gw_prices_df, on="element", how="left")
    gw_data = gw_data.groupby("element").agg(
        {
            "name": "first",
            "position": "first",
            "player_team_id": "first",
            "value": "first",
            "predicted_points": "sum",
        }
    ).reset_index()

    logger.debug("Found %d unique players with predictions for GW %d", len(gw_data), gw)

    # Apply points_multiplier_override
    if points_multiplier_override:
        for player_id, multiplier in points_multiplier_override:
            mask = gw_data["element"] == player_id
            if mask.any():
                gw_data.loc[mask, "predicted_points"] *= multiplier

    # Apply non_playing_players override (set to 0)
    if non_playing_players:
        for player_id, gw_list in non_playing_players:
            if gw in gw_list:
                mask = gw_data["element"] == player_id
                if mask.any():
                    gw_data.loc[mask, "predicted_points"] = 0

    # Forced lineup players for this GW
    forced_players: List[int] = []
    if forced_lineup_players:
        for player_id, gw_list in forced_lineup_players:
            if gw in gw_list and player_id in watchlist_players:
                forced_players.append(player_id)

    solution = _solve_free_hit_milp(gw_data, budget, forced_players, gw)

    if solution["status"] == "Optimal":
        logger.debug(
            "Optimal Free Hit squad found: %.1f points", solution["total_points"]
        )
    else:
        logger.warning("Free Hit optimization failed: %s", solution["status"])

    return solution


def _solve_free_hit_milp(
    gw_data: pd.DataFrame,
    budget: float,
    forced_players: List[int],
    gw: int,
) -> Dict:
    """Solve the Free Hit MILP: maximize lineup points + captain bonus."""
    players = gw_data["element"].tolist()
    positions = gw_data.set_index("element")["position"].to_dict()
    costs = gw_data.set_index("element")["value"].to_dict()
    points = gw_data.set_index("element")["predicted_points"].to_dict()
    teams = gw_data.set_index("element")["player_team_id"].to_dict()
    names = gw_data.set_index("element")["name"].to_dict()

    prob = pulp.LpProblem("FreeHit_Squad_Optimization", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("squad", players, cat="Binary")
    y = pulp.LpVariable.dicts("lineup", players, cat="Binary")
    c = pulp.LpVariable.dicts("captain", players, cat="Binary")

    # Objective: lineup points + captain bonus (captain counts double)
    prob += pulp.lpSum(
        [points[p] * y[p] + points[p] * c[p] for p in players]
    )

    # Squad size and composition
    prob += pulp.lpSum([x[p] for p in players]) == TOTAL_SQUAD_SIZE
    for pos, count in SQUAD_COMPOSITION.items():
        pos_players = [p for p in players if positions[p] == pos]
        prob += pulp.lpSum([x[p] for p in pos_players]) == count

    # Budget
    prob += pulp.lpSum([costs[p] * x[p] for p in players]) <= budget

    # Club limit
    unique_teams = set(teams.values())
    for team in unique_teams:
        team_players = [p for p in players if teams[p] == team]
        prob += pulp.lpSum([x[p] for p in team_players]) <= MAX_PLAYERS_PER_CLUB

    # Lineup size and formation
    prob += pulp.lpSum([y[p] for p in players]) == TOTAL_LINEUP_SIZE
    for pos, (min_count, max_count) in LINEUP_CONSTRAINTS.items():
        pos_players = [p for p in players if positions[p] == pos]
        prob += pulp.lpSum([y[p] for p in pos_players]) >= min_count
        prob += pulp.lpSum([y[p] for p in pos_players]) <= max_count

    # Lineup subset of squad
    for p in players:
        prob += y[p] <= x[p]

    # Exactly one captain, captain in lineup
    prob += pulp.lpSum([c[p] for p in players]) == 1
    for p in players:
        prob += c[p] <= y[p]

    # Forced lineup
    for p in forced_players:
        if p in players:
            prob += y[p] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status == pulp.LpStatusOptimal:
        squad = [p for p in players if x[p].varValue == 1]
        lineup = [p for p in players if y[p].varValue == 1]
        captain = next(p for p in players if c[p].varValue == 1)
        total_points = pulp.value(prob.objective)
        squad_cost = sum(costs[p] for p in squad)

        squad_details: Dict[str, List[Dict]] = {}
        for pos in ["GK", "DEF", "MID", "FWD"]:
            pos_players = [p for p in squad if positions[p] == pos]
            squad_details[pos] = [
                {
                    "element": p,
                    "name": names.get(p, f"Player_{p}"),
                    "cost": costs[p],
                    "points": points.get(p, 0),
                    "is_starter": p in lineup,
                    "is_captain": p == captain,
                }
                for p in pos_players
            ]

        return {
            "status": "Optimal",
            "squad": squad,
            "lineup": lineup,
            "captain": captain,
            "total_points": total_points,
            "squad_cost": squad_cost,
            "unused_budget": budget - squad_cost,
            "squad_details": squad_details,
        }
    else:
        return {
            "status": str(pulp.LpStatus[prob.status]),
            "squad": [],
            "lineup": [],
            "captain": None,
            "total_points": 0,
            "squad_cost": 0,
            "unused_budget": budget,
            "squad_details": {},
        }


def calculate_free_hit_benefits_for_horizon(
    start_gw: int,
    planning_horizon: int,
    budget: float,
    predictions_df: pd.DataFrame,
    gw_data_df: pd.DataFrame,
    watchlist_players: List[int],
    forced_lineup_players: Optional[List[Tuple[int, List[int]]]] = None,
    points_multiplier_override: Optional[List[Tuple[int, float]]] = None,
    non_playing_players: Optional[List[Tuple[int, List[int]]]] = None,
) -> Dict[int, Dict]:
    """
    Pre-calculate Free Hit benefits for all GWs in the planning horizon.

    Returns:
        Dict mapping gameweek -> FH result dict.
    """
    logger.debug(
        "Pre-calculating Free Hit benefits for GW %d-%d",
        start_gw,
        start_gw + planning_horizon - 1,
    )

    fh_benefits: Dict[int, Dict] = {}
    for i in range(planning_horizon):
        gw = start_gw + i
        try:
            fh_result = calculate_optimal_free_hit_squad(
                gw=gw,
                budget=budget,
                predictions_df=predictions_df,
                gw_data_df=gw_data_df,
                watchlist_players=watchlist_players,
                forced_lineup_players=forced_lineup_players,
                points_multiplier_override=points_multiplier_override,
                non_playing_players=non_playing_players,
            )
            fh_benefits[gw] = fh_result
            if fh_result["total_points"] > 0:
                logger.debug(
                    "  GW %d: %.1f points (%.1fM squad)",
                    gw,
                    fh_result["total_points"],
                    fh_result["squad_cost"] / 10,
                )
            else:
                logger.warning("  GW %d: Failed to calculate Free Hit benefit", gw)
        except Exception as e:
            logger.exception("Error calculating Free Hit for GW %d: %s", gw, e)
            fh_benefits[gw] = {
                "status": "Error",
                "squad": [],
                "lineup": [],
                "captain": None,
                "total_points": 0,
                "squad_cost": 0,
                "unused_budget": budget,
                "squad_details": {},
            }

    logger.debug("Pre-calculated Free Hit benefits for %d gameweeks", len(fh_benefits))
    return fh_benefits
