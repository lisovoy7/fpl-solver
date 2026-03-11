"""
Single CLI entry point for fpl-solver.

Orchestrates the full pipeline: config loading, API data fetching,
prediction generation, watchlist building, chip scenario enumeration,
and MILP solving. Outputs the optimal strategy to console and to
output/strategy_gw{N}.json.

Usage:
    python run.py
    python run.py --skip-predictions
    python run.py --horizon 5
    python run.py --no-chips
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from fpl import api, config as cfg
from fpl.predict import generate_predictions
from fpl.solver import FPLSolver, TRANSFER_PENALTY_POINTS
from fpl.free_hit import generate_chip_scenarios, calculate_free_hit_benefits_for_horizon
from fpl.watchlist import create_watchlist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="fpl-solver: holistic FPL optimization")
    parser.add_argument("--skip-predictions", action="store_true",
                        help="Skip prediction generation, use cached predictions.csv")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Override planning horizon (number of GWs)")
    parser.add_argument("--no-chips", action="store_true",
                        help="Disable chip optimization (no FH/BB/TC enumeration)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    return parser.parse_args()


def load_bundled_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load bundled static data: multipliers and team tiers.

    Returns:
        Tuple of (multipliers DataFrame, team_tiers DataFrame).
    """
    multipliers = pd.read_csv(DATA_DIR / "multipliers.csv")
    team_tiers = pd.read_csv(DATA_DIR / "team_tiers.csv")
    logger.info("Loaded bundled data: %d multipliers, %d team tier records",
                len(multipliers), len(team_tiers))
    return multipliers, team_tiers


def display_strategy(solution: Dict, solver: FPLSolver, players: pd.DataFrame,
                     start_gw: int, non_playing: List[Tuple[int, List[int]]],
                     free_hit_gws: List[int], fh_benefits: Dict,
                     initial_bank: int, initial_selling_prices: Dict[int, int]) -> None:
    """Display the optimal strategy in a user-friendly way."""
    non_playing = non_playing or []
    initial_selling_prices = initial_selling_prices or {}

    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTIMAL FPL STRATEGY")
    logger.info("=" * 80)
    logger.info("Total Expected Points: %.1f", solution["objective_value"])
    logger.info("Average per GW: %.1f", solution["objective_value"] / solver.T)

    predictions = solver.predictions
    expected_points: Dict[Tuple[int, int], float] = {}
    for _, row in predictions.iterrows():
        key = (int(row["element"]), int(row["event"]))
        expected_points[key] = expected_points.get(key, 0) + row["predicted_points"]

    player_market_prices = dict(zip(players["element"], players["value"]))

    cumulative_bank = initial_bank
    selling_prices = dict(initial_selling_prices)

    for i, gw in enumerate(range(start_gw, start_gw + solver.T)):
        t = i + 1
        transfers = solution["transfers"][t]
        chips = solution["chips"][t]
        captain_id = solution["captains"].get(t)
        squad_ids = solution["squads"].get(t, [])
        lineup_data = solution["lineups"].get(t, {})
        lineup_ids = lineup_data.get("starters", []) if lineup_data else []

        is_free_hit_gw = free_hit_gws and gw in free_hit_gws
        is_bench_boost_gw = "bench_boost" in chips
        is_triple_captain_gw = "triple_captain" in chips

        gw_label = f"\nGAMEWEEK {gw}"
        if is_free_hit_gw:
            gw_label += " [FREE HIT]"
        if is_bench_boost_gw:
            gw_label += " [BENCH BOOST]"
        if is_triple_captain_gw:
            gw_label += " [TRIPLE CAPTAIN]"
        logger.info(gw_label)

        available_transfers = transfers.get("available_transfers", 0)
        logger.info("  Free Transfers: %d", int(available_transfers))

        if transfers["out"]:
            for pid in transfers["out"]:
                sell_price = selling_prices.get(pid, player_market_prices.get(pid, 0))
                cumulative_bank += sell_price
                selling_prices.pop(pid, None)
        if transfers["in"]:
            for pid in transfers["in"]:
                buy_price = player_market_prices.get(pid, 0)
                cumulative_bank -= buy_price
                selling_prices[pid] = buy_price

        if is_free_hit_gw:
            logger.info("  FREE HIT: Using optimal squad for this GW")
            fh_data = fh_benefits.get(gw, {})
            if fh_data:
                logger.info("    Expected points: %.1f", fh_data.get("total_points", 0))
        elif transfers["count"] > 0:
            real_in = [p for p in transfers["in"] if p not in transfers["out"]]
            real_out = [p for p in transfers["out"] if p not in transfers["in"]]
            real_count = len(real_in)

            if real_count > 0:
                wildcard_active = transfers.get("wildcard_active", False)
                paid = transfers.get("paid_transfers", 0)
                free = transfers.get("free_transfers", 0)

                info = f"  Transfers ({real_count}"
                if wildcard_active:
                    info += " - WILDCARD, all free"
                elif paid > 0:
                    info += f" - {int(free)} free, {int(paid)} paid [{int(paid * TRANSFER_PENALTY_POINTS)} pts]"
                else:
                    info += " - all free"
                info += "):"
                logger.info(info)

                for pid in real_in:
                    row = players[players["element"] == pid]
                    name = row["name"].iloc[0] if len(row) else str(pid)
                    cost = row["value"].iloc[0] / 10 if len(row) else 0
                    pts = expected_points.get((pid, gw), 0)
                    if any(pid == p and gw in gws for p, gws in non_playing):
                        pts = 0
                    logger.info("    IN:  %s (%.1fM | %.1f pts)", name, cost, pts)

                for pid in real_out:
                    row = players[players["element"] == pid]
                    name = row["name"].iloc[0] if len(row) else str(pid)
                    cost = row["value"].iloc[0] / 10 if len(row) else 0
                    pts = expected_points.get((pid, gw), 0)
                    logger.info("    OUT: %s (%.1fM | %.1f pts)", name, cost, pts)
            else:
                logger.info("  No transfers")
        else:
            logger.info("  No transfers")

        if captain_id:
            if is_free_hit_gw and fh_benefits and gw in fh_benefits:
                fh_data = fh_benefits[gw]
                if "squad_details" in fh_data:
                    cap_name = None
                    for pos_players in fh_data["squad_details"].values():
                        for p in pos_players:
                            if p.get("is_captain"):
                                cap_name = p["name"]
                                break
                        if cap_name:
                            break
                    cap_name = cap_name or "Unknown"
                else:
                    row = players[players["element"] == captain_id]
                    cap_name = row["name"].iloc[0] if len(row) else str(captain_id)
            else:
                row = players[players["element"] == captain_id]
                cap_name = row["name"].iloc[0] if len(row) else str(captain_id)

            multiplier = "3x" if is_triple_captain_gw else "2x"
            logger.info("  Captain: %s (%s points)", cap_name, multiplier)

        if squad_ids and not is_free_hit_gw:
            _display_squad(squad_ids, players, expected_points, gw, captain_id,
                           lineup_ids, chips, cumulative_bank, non_playing)

        if is_free_hit_gw and fh_benefits and gw in fh_benefits:
            fh = fh_benefits[gw]
            if "squad_details" in fh:
                _display_fh_squad(fh["squad_details"])

    logger.info("")
    logger.info("=" * 80)


def _display_squad(squad_ids, players, expected_points, gw, captain_id,
                   lineup_ids, chips, bank, non_playing):
    """Display squad composition for a gameweek."""
    is_bb = "bench_boost" in chips
    positions = {"GK": [], "DEF": [], "MID": [], "FWD": []}

    for pid in squad_ids:
        row = players[players["element"] == pid]
        if len(row) == 0:
            continue
        info = row.iloc[0]
        pts = expected_points.get((pid, gw), 0)
        if any(pid == p and gw in gws for p, gws in non_playing):
            pts = 0
        is_starter = pid in lineup_ids
        is_captain = pid == captain_id
        positions[info["position"]].append({
            "name": info["name"], "cost": info["value"] / 10,
            "pts": pts, "starter": is_starter, "captain": is_captain,
        })

    for pos in ["GK", "DEF", "MID", "FWD"]:
        if not positions[pos]:
            continue
        logger.info("    %s:", pos)
        for p in sorted(positions[pos], key=lambda x: x["cost"], reverse=True):
            mark = "*" if p["starter"] else " "
            cap = " (C)" if p["captain"] else ""
            logger.info("      %s %-25s %5.1fM | %5.1f pts%s",
                        mark, p["name"], p["cost"], p["pts"], cap)

    logger.info("    Bank: %.1fM", bank / 10)


def _display_fh_squad(squad_details):
    """Display Free Hit squad details."""
    for pos in ["GK", "DEF", "MID", "FWD"]:
        if pos not in squad_details:
            continue
        logger.info("    %s:", pos)
        for p in squad_details[pos]:
            mark = "*" if p["is_starter"] else " "
            cap = " (C)" if p.get("is_captain") else ""
            logger.info("      %s %-25s %5.1fM | %5.1f pts%s",
                        mark, p["name"], p["cost"] / 10, p["points"], cap)


def save_strategy(solution: Dict, scenario_name: str, total_points: float,
                  current_gw: int) -> None:
    """Save strategy to JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = {
        "scenario": scenario_name,
        "total_expected_points": round(total_points, 1),
        "start_gw": current_gw,
        "objective_value": round(solution["objective_value"], 1),
        "transfers": {str(k): v for k, v in solution["transfers"].items()},
        "captains": {str(k): v for k, v in solution["captains"].items()},
        "chips": {str(k): v for k, v in solution["chips"].items()},
    }
    path = OUTPUT_DIR / f"strategy_gw{current_gw}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Saved strategy to %s", path)


def main() -> None:
    """Main pipeline."""
    args = parse_args()

    # 1. Load config
    config = cfg.load_config(args.config)
    team_id = config["team_id"]
    free_transfers = config["free_transfers"]

    # 2. Auto-detect from API
    logger.info("Fetching bootstrap data from FPL API...")
    bootstrap = api.fetch_bootstrap_data()
    detected_gw = api.detect_current_gw(bootstrap)
    season = api.detect_current_season(bootstrap)
    logger.info("Detected season: %s, current GW: %d", season, detected_gw)

    detected_chips = api.detect_chips_used(team_id)
    logger.info("Detected chips used: %s", detected_chips)

    cfg.merge_api_values(config, current_gw=detected_gw, chips=detected_chips)

    current_gw = config.get("current_gw", detected_gw)
    solver_params = cfg.get_solver_params(config)
    overrides = cfg.get_player_overrides(config)
    chips_cfg = config.get("chips", {})

    # Resolve planning horizon
    horizon = solver_params.get("planning_horizon", "rest_of_season")
    if args.horizon is not None:
        horizon = args.horizon
    if horizon == "rest_of_season" or horizon is None:
        horizon = 38 - current_gw + 1
    horizon = int(horizon)

    # 3. Fetch team data + selling prices
    last_played_gw = current_gw - 1
    logger.info("Fetching team data for team %d, GW %d...", team_id, last_played_gw)
    team_data = api.fetch_team_data(team_id, last_played_gw)
    current_squad = team_data["squad"]

    selling_info, selling_summary = api.get_squad_selling_prices(team_id, last_played_gw)
    total_budget = selling_summary["correct_budget"]
    initial_bank = selling_summary["bank"]
    initial_selling_prices = selling_summary["selling_prices"]

    logger.info("Budget: %.1fM (bank: %.1fM), squad: %d players",
                total_budget / 10, initial_bank / 10, len(current_squad))

    # 4. Fetch GW data + fixtures
    multipliers, team_tiers = load_bundled_data()
    current_season_tiers = team_tiers[team_tiers["season"] == season].copy()

    fixtures = api.fetch_current_fixtures(bootstrap)
    logger.info("Fetched %d fixtures", len(fixtures))

    predictions_path = OUTPUT_DIR / "predictions.csv"

    if args.skip_predictions and predictions_path.exists():
        logger.info("Loading cached predictions from %s", predictions_path)
        predictions = pd.read_csv(predictions_path)
        gw_data = api.fetch_gameweek_data(bootstrap)
    else:
        logger.info("Fetching gameweek data (this takes a few minutes)...")
        gw_data = api.fetch_gameweek_data(bootstrap)
        logger.info("GW data: %d records", len(gw_data))

        logger.info("Generating predictions...")
        predictions = generate_predictions(gw_data, fixtures, multipliers,
                                           current_season_tiers, season)

        OUTPUT_DIR.mkdir(exist_ok=True)
        predictions.to_csv(predictions_path, index=False)
        logger.info("Saved predictions to %s (%d rows)", predictions_path, len(predictions))

    # 5. Build watchlist
    must_include = list(current_squad)
    must_include.extend(overrides.get("extra_players", []))
    must_exclude = overrides.get("excluded_players", [])
    min_hist_games = solver_params.get("min_hist_games", 7)

    watchlist = create_watchlist(predictions, gw_data, min_hist_games=min_hist_games,
                                must_include=must_include, must_exclude=must_exclude)
    logger.info("Watchlist: %d players", len(watchlist))

    # 6. Chip scenarios
    wildcards_used = chips_cfg.get("wildcards_used", 0)
    free_hits_used = chips_cfg.get("free_hits_used", 0)
    bench_boost_used = chips_cfg.get("bench_boost_used", False)
    triple_captain_used = chips_cfg.get("triple_captain_used", False)
    chip_opt = config.get("chip_optimization", {})

    if args.no_chips:
        chip_scenarios = [{"name": "No chips", "free_hit_gws": [],
                           "bench_boost_gw": -1, "triple_captain_gw": -1}]
    else:
        chip_scenarios = generate_chip_scenarios(
            start_gw=current_gw,
            planning_horizon=horizon,
            enable_free_hit=chip_opt.get("enable_free_hit", True),
            enable_bench_boost=chip_opt.get("enable_bench_boost", True),
            enable_triple_captain=chip_opt.get("enable_triple_captain", True),
            free_hits_used_first_half=min(free_hits_used, 1),
            free_hits_used_second_half=max(0, free_hits_used - 1),
            bench_boost_used=bench_boost_used,
            triple_captain_used=triple_captain_used,
        )

    max_scenarios = solver_params.get("max_scenarios", 100)
    if len(chip_scenarios) > max_scenarios:
        logger.warning("Limiting scenarios from %d to %d", len(chip_scenarios), max_scenarios)
        chip_scenarios = chip_scenarios[:max_scenarios]

    # Pre-calculate Free Hit benefits
    fh_benefits: Dict = {}
    any_fh = any(s["free_hit_gws"] for s in chip_scenarios)
    if any_fh:
        logger.info("Pre-calculating Free Hit benefits...")
        fh_benefits = calculate_free_hit_benefits_for_horizon(
            start_gw=current_gw, planning_horizon=horizon, budget=total_budget,
            predictions_df=predictions, gw_data_df=gw_data,
            watchlist_players=watchlist,
            forced_lineup_players=overrides.get("forced_lineup"),
            points_multiplier_override=overrides.get("points_multiplier"),
            non_playing_players=overrides.get("non_playing"),
        )

    # 7. Solve each scenario
    transfer_topup = config.get("transfer_topup", {})
    sub_probability = solver_params.get("sub_probability", 0.10)
    first_gw_penalty = solver_params.get("first_gw_transfer_penalty", -1)
    time_limit = solver_params.get("time_limit_per_scenario", 15)

    logger.info("Testing %d chip scenarios...", len(chip_scenarios))
    start_time = time.time()
    best_result = None
    best_total = -float("inf")
    scenario_results = []

    for idx, scenario in enumerate(chip_scenarios, 1):
        logger.info("--- Scenario %d/%d: %s ---", idx, len(chip_scenarios), scenario["name"])

        solver = FPLSolver(
            planning_horizon=horizon, budget=total_budget, start_gw=current_gw,
            afcon_enabled=transfer_topup.get("enabled", True),
            afcon_trigger_gw=transfer_topup.get("trigger_gw", 15),
            afcon_transfer_count=transfer_topup.get("transfer_count", 5),
            points_multiplier_override=overrides.get("points_multiplier"),
            forced_lineup_players=overrides.get("forced_lineup"),
            non_playing_players=overrides.get("non_playing"),
            first_gw_transfer_penalty=first_gw_penalty,
            sub_probability=sub_probability,
            bench_boost_gw=scenario["bench_boost_gw"],
            triple_captain_gw=scenario["triple_captain_gw"],
            free_hit_gws=scenario["free_hit_gws"],
        )

        solver.load_predictions(predictions)
        if len(solver.predictions) == 0:
            logger.error("No predictions found for GW %d+", current_gw)
            continue

        solver.load_player_data(gw_data, predictions, player_subset=watchlist)
        solver.set_initial_squad(current_squad, available_transfers=free_transfers)
        solver.set_chip_state(
            wildcard_first_half=min(wildcards_used, 1),
            wildcard_second_half=max(0, wildcards_used - 1),
        )

        solver.build_model()
        success = solver.solve(time_limit=time_limit)
        if not success:
            logger.warning("Solver failed for: %s", scenario["name"])
            continue

        solution = solver.extract_solution()
        base_points = solution["objective_value"]

        fh_total = sum(
            fh_benefits.get(gw, {}).get("total_points", 0)
            for gw in scenario["free_hit_gws"]
        )
        total_points = base_points + fh_total

        result = {
            "scenario_name": scenario["name"],
            "free_hit_gws": scenario["free_hit_gws"],
            "bench_boost_gw": scenario["bench_boost_gw"],
            "triple_captain_gw": scenario["triple_captain_gw"],
            "base_points": base_points,
            "free_hit_benefit": fh_total,
            "total_points": total_points,
            "solution": solution,
            "solver": solver,
            "players": solver.players,
        }
        scenario_results.append(result)

        if total_points > best_total:
            best_result = result
            best_total = total_points
            logger.info("New best: %.1f pts (%s)", total_points, scenario["name"])
        else:
            logger.info("Result: %.1f pts", total_points)

    elapsed = time.time() - start_time
    logger.info("Chip enumeration: %d/%d solved in %.0fs",
                len(scenario_results), len(chip_scenarios), elapsed)

    if not best_result:
        logger.error("No optimal solution found in any scenario")
        return

    logger.info("")
    logger.info("BEST STRATEGY: %s (%.1f pts)", best_result["scenario_name"], best_total)

    display_strategy(
        best_result["solution"], best_result["solver"], best_result["players"],
        current_gw, overrides.get("non_playing", []),
        best_result["free_hit_gws"], fh_benefits,
        initial_bank, initial_selling_prices,
    )

    save_strategy(best_result["solution"], best_result["scenario_name"],
                  best_total, current_gw)


if __name__ == "__main__":
    main()
