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


def _player_name(players: pd.DataFrame, pid: int) -> str:
    row = players[players["element"] == pid]
    return row["name"].iloc[0] if len(row) else str(pid)


def _player_cost(players: pd.DataFrame, pid: int) -> float:
    row = players[players["element"] == pid]
    return row["value"].iloc[0] / 10 if len(row) else 0.0


def _build_strategy_text(
    solution: Dict, solver: FPLSolver, players: pd.DataFrame,
    start_gw: int, total_points: float, scenario_name: str,
    non_playing: List[Tuple[int, List[int]]],
    free_hit_gws: List[int], fh_benefits: Dict,
    initial_bank: int, initial_selling_prices: Dict[int, int],
) -> str:
    """
    Build the full visual strategy text.

    Returns:
        Multi-line string with the formatted strategy.
    """
    non_playing = non_playing or []
    initial_selling_prices = initial_selling_prices or {}
    lines: List[str] = []
    W = 78

    predictions = solver.predictions
    expected_points: Dict[Tuple[int, int], float] = {}
    for _, row in predictions.iterrows():
        key = (int(row["element"]), int(row["event"]))
        expected_points[key] = expected_points.get(key, 0) + row["predicted_points"]
    player_market_prices = dict(zip(players["element"], players["value"]))

    # Header
    lines.append("=" * W)
    lines.append(f"  FPL SOLVER  -  OPTIMAL STRATEGY")
    lines.append(f"  GW {start_gw}-{start_gw + solver.T - 1}  |  {total_points:.1f} expected pts  |  {total_points / solver.T:.1f} per GW")
    lines.append(f"  Scenario: {scenario_name}")
    lines.append("=" * W)

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

        is_fh = free_hit_gws and gw in free_hit_gws
        is_bb = "bench_boost" in chips
        is_tc = "triple_captain" in chips
        is_wc = "wildcard" in chips

        # GW header
        chip_badges = []
        if is_wc:
            chip_badges.append("[WILDCARD]")
        if is_fh:
            chip_badges.append("[FREE HIT]")
        if is_bb:
            chip_badges.append("[BENCH BOOST]")
        if is_tc:
            chip_badges.append("[TRIPLE CAPTAIN]")
        badge_str = "  ".join(chip_badges)

        lines.append("")
        lines.append("-" * W)
        gw_header = f"  GW {gw}"
        if badge_str:
            gw_header += f"  {badge_str}"
        lines.append(gw_header)
        lines.append("-" * W)

        available_ft = int(transfers.get("available_transfers", 0))
        lines.append(f"  Free Transfers: {available_ft}")

        # Track bank
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

        # Transfers
        if is_fh:
            fh_data = fh_benefits.get(gw, {})
            fh_pts = fh_data.get("total_points", 0)
            lines.append(f"  FREE HIT: Optimal squad selected from entire pool ({fh_pts:.1f} pts)")
        elif transfers["count"] > 0:
            real_in = [p for p in transfers["in"] if p not in transfers["out"]]
            real_out = [p for p in transfers["out"] if p not in transfers["in"]]
            real_count = len(real_in)
            if real_count > 0:
                wildcard_active = transfers.get("wildcard_active", False)
                paid = int(transfers.get("paid_transfers", 0))
                free = int(transfers.get("free_transfers", 0))
                if wildcard_active:
                    desc = "WILDCARD - all free"
                elif paid > 0:
                    desc = f"{free} free, {paid} paid ({paid * TRANSFER_PENALTY_POINTS} pts)"
                else:
                    desc = "all free"
                lines.append(f"  Transfers ({real_count} - {desc}):")
                for pid in real_in:
                    name = _player_name(players, pid)
                    cost = _player_cost(players, pid)
                    pts = expected_points.get((pid, gw), 0)
                    if any(pid == p and gw in gws for p, gws in non_playing):
                        pts = 0
                    lines.append(f"    >>> IN:  {name:<25s} {cost:5.1f}M | {pts:5.1f} pts")
                for pid in real_out:
                    name = _player_name(players, pid)
                    cost = _player_cost(players, pid)
                    pts = expected_points.get((pid, gw), 0)
                    lines.append(f"    <<< OUT: {name:<25s} {cost:5.1f}M | {pts:5.1f} pts")
            else:
                lines.append("  No transfers")
        else:
            lines.append("  No transfers")

        # Captain
        if captain_id:
            if is_fh and fh_benefits and gw in fh_benefits:
                fh_data = fh_benefits[gw]
                cap_name = None
                if "squad_details" in fh_data:
                    for pos_players in fh_data["squad_details"].values():
                        for p in pos_players:
                            if p.get("is_captain"):
                                cap_name = p["name"]
                                break
                        if cap_name:
                            break
                cap_name = cap_name or _player_name(players, captain_id)
            else:
                cap_name = _player_name(players, captain_id)
            mult = "3x" if is_tc else "2x"
            lines.append(f"  Captain: {cap_name} ({mult} points)")

        # Squad display
        if is_fh and fh_benefits and gw in fh_benefits:
            fh = fh_benefits[gw]
            if "squad_details" in fh:
                lines.append("")
                for pos in ["GK", "DEF", "MID", "FWD"]:
                    if pos not in fh["squad_details"]:
                        continue
                    lines.append(f"    {pos}:")
                    for p in fh["squad_details"][pos]:
                        starter = "*" if p["is_starter"] else " "
                        cap = " (C)" if p.get("is_captain") else ""
                        lines.append(f"      {starter} {p['name']:<25s} {p['cost']/10:5.1f}M | {p['points']:5.1f} pts{cap}")
        elif squad_ids:
            positions: Dict[str, List[Dict]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
            for pid in squad_ids:
                row = players[players["element"] == pid]
                if len(row) == 0:
                    continue
                info = row.iloc[0]
                pts = expected_points.get((pid, gw), 0)
                if any(pid == p and gw in gws for p, gws in non_playing):
                    pts = 0
                positions[info["position"]].append({
                    "name": info["name"], "cost": info["value"] / 10,
                    "pts": pts, "starter": pid in lineup_ids,
                    "captain": pid == captain_id,
                })
            lines.append("")
            for pos in ["GK", "DEF", "MID", "FWD"]:
                if not positions[pos]:
                    continue
                lines.append(f"    {pos}:")
                for p in sorted(positions[pos], key=lambda x: (-x["starter"], -x["pts"])):
                    starter = "*" if p["starter"] else " "
                    cap = " (C)" if p["captain"] else ""
                    lines.append(f"      {starter} {p['name']:<25s} {p['cost']:5.1f}M | {p['pts']:5.1f} pts{cap}")
            lines.append(f"    Bank: {cumulative_bank / 10:.1f}M")

    lines.append("")
    lines.append("=" * W)
    return "\n".join(lines)


def display_strategy(solution: Dict, solver: FPLSolver, players: pd.DataFrame,
                     start_gw: int, total_points: float, scenario_name: str,
                     non_playing: List[Tuple[int, List[int]]],
                     free_hit_gws: List[int], fh_benefits: Dict,
                     initial_bank: int, initial_selling_prices: Dict[int, int]) -> str:
    """
    Display the optimal strategy to console and return the text.

    Returns:
        Formatted strategy text (also printed to console).
    """
    text = _build_strategy_text(
        solution, solver, players, start_gw, total_points, scenario_name,
        non_playing, free_hit_gws, fh_benefits, initial_bank, initial_selling_prices,
    )
    print(text)
    return text


def save_strategy(solution: Dict, scenario_name: str, total_points: float,
                  current_gw: int, strategy_text: str = "") -> None:
    """Save strategy to JSON and formatted text file."""
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
    json_path = OUTPUT_DIR / f"strategy_gw{current_gw}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Saved JSON strategy to %s", json_path)

    if strategy_text:
        txt_path = OUTPUT_DIR / f"strategy_gw{current_gw}.txt"
        with open(txt_path, "w") as f:
            f.write(strategy_text)
        logger.info("Saved visual strategy to %s", txt_path)


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

    # Apply fixture overrides (DGW/BGW schedule changes)
    fixture_overrides = config.get("fixture_overrides", [])
    for override in fixture_overrides:
        home = override.get("home_team")
        away = override.get("away_team")
        new_gw = override.get("gameweek")
        if home is None or away is None or new_gw is None:
            continue
        mask = (fixtures["team_h"] == home) & (fixtures["team_a"] == away)
        matched = mask.sum()
        if matched > 0:
            old_gw = fixtures.loc[mask, "event"].iloc[0]
            fixtures.loc[mask, "event"] = new_gw
            logger.info("Fixture override: team %d vs %d moved from GW %s to GW %d",
                        home, away, old_gw, new_gw)
        else:
            logger.warning("Fixture override: no match for team %d vs %d", home, away)

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
    bench_boost_used = chips_cfg.get("bench_boost_used", 0)
    triple_captain_used = chips_cfg.get("triple_captain_used", 0)

    if args.no_chips:
        chip_scenarios = [{"name": "No chips", "free_hit_gws": [],
                           "bench_boost_gw": -1, "triple_captain_gw": -1}]
    else:
        chip_scenarios = generate_chip_scenarios(
            start_gw=current_gw,
            planning_horizon=horizon,
            free_hits_used_first_half=min(free_hits_used, 1),
            free_hits_used_second_half=max(0, free_hits_used - 1),
            bench_boost_used_first_half=min(bench_boost_used, 1),
            bench_boost_used_second_half=max(0, bench_boost_used - 1),
            triple_captain_used_first_half=min(triple_captain_used, 1),
            triple_captain_used_second_half=max(0, triple_captain_used - 1),
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

    strategy_text = display_strategy(
        best_result["solution"], best_result["solver"], best_result["players"],
        current_gw, best_total, best_result["scenario_name"],
        overrides.get("non_playing", []),
        best_result["free_hit_gws"], fh_benefits,
        initial_bank, initial_selling_prices,
    )

    save_strategy(best_result["solution"], best_result["scenario_name"],
                  best_total, current_gw, strategy_text=strategy_text)


if __name__ == "__main__":
    main()
