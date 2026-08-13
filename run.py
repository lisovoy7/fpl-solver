"""
Single CLI entry point for fpl-solver.

Orchestrates the full pipeline: config loading, API data fetching,
prediction generation, watchlist building, chip scenario enumeration,
and MILP solving. Outputs the optimal strategy to console and to
output/strategy_gw{N}.json.

Usage:
    python run.py
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

from fpl import api, config as cfg, proxy_predict
from fpl.predict import generate_predictions
from fpl.solver import FPLSolver, TRANSFER_PENALTY_POINTS
from fpl.free_hit import (
    generate_chip_scenarios, calculate_free_hit_benefits_for_horizon,
    triple_captain_candidate_gws, find_best_triple_captain_gw,
)
from fpl.watchlist import create_watchlist
from fpl.scenario_runner import build_solver, default_worker_count, solve_scenarios

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="fpl-solver: holistic FPL optimization")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Override planning horizon (number of GWs)")
    parser.add_argument("--no-chips", action="store_true",
                        help="Disable chip optimization (no FH/BB/TC enumeration)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix appended to output filenames (e.g. 'ars_liv')")
    parser.add_argument("--extra-override", action="append", default=[],
                        metavar="HOME:AWAY:GW",
                        help="Additional fixture override (repeatable). Format: home_id:away_id:new_gw")
    parser.add_argument("--force-wildcard-gw", type=int, default=None,
                        help="Force wildcard on a specific GW (CLI override)")
    parser.add_argument("--force-free-hit-gw", type=int, default=None,
                        help="Force free hit on a specific GW (CLI override)")
    parser.add_argument("--force-bench-boost-gw", type=int, default=None,
                        help="Force bench boost on a specific GW (CLI override)")
    parser.add_argument("--time-limit", type=int, default=None,
                        help="Override solver time limit per scenario (seconds)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Scenario solver processes (default: CPU count - 1; "
                             "1 disables parallelism)")
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

        # Calculate GW total expected points
        gw_total_pts = 0.0
        if is_fh and fh_benefits and gw in fh_benefits:
            gw_total_pts = fh_benefits[gw].get("total_points", 0)
        elif squad_ids:
            for pid in lineup_ids:
                pts = expected_points.get((pid, gw), 0)
                if any(pid == p and gw in gws for p, gws in non_playing):
                    pts = 0
                gw_total_pts += pts
                if pid == captain_id:
                    mult_factor = 2 if is_tc else 1
                    gw_total_pts += pts * mult_factor
            if is_bb:
                bench_ids = [p for p in squad_ids if p not in lineup_ids]
                for pid in bench_ids:
                    pts = expected_points.get((pid, gw), 0)
                    if any(pid == p and gw in gws for p, gws in non_playing):
                        pts = 0
                    gw_total_pts += pts

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
        gw_header = f"  GW {gw}  ({gw_total_pts:.1f} pts)"
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
                  current_gw: int, strategy_text: str = "",
                  output_suffix: str = "") -> None:
    """Save strategy to JSON and formatted text file."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""
    output = {
        "scenario": scenario_name,
        "total_expected_points": round(total_points, 1),
        "start_gw": current_gw,
        "objective_value": round(solution["objective_value"], 1),
        "transfers": {str(k): v for k, v in solution["transfers"].items()},
        "captains": {str(k): v for k, v in solution["captains"].items()},
        "chips": {str(k): v for k, v in solution["chips"].items()},
    }
    json_path = OUTPUT_DIR / f"strategy_gw{current_gw}{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Saved JSON strategy to %s", json_path)

    if strategy_text:
        txt_path = OUTPUT_DIR / f"strategy_gw{current_gw}{suffix}.txt"
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
    # If last played GW was a Free Hit, the picks on that GW are the temporary FH squad
    # which reverts after the GW. Fetch the squad from the GW before the Free Hit instead.
    free_hit_gws = detected_chips.get("free_hit_gws", [])
    squad_gw = last_played_gw
    if squad_gw in free_hit_gws and squad_gw > 1:
        squad_gw = last_played_gw - 1
        logger.info("Last played GW %d was a Free Hit — using GW %d squad instead",
                    last_played_gw, squad_gw)
    logger.info("Fetching team data for team %d, GW %d...", team_id, squad_gw)
    team_data = api.fetch_team_data(team_id, squad_gw)
    current_squad = team_data["squad"]

    selling_info, selling_summary = api.get_squad_selling_prices(team_id, squad_gw)
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

    # Apply CLI extra overrides (--extra-override HOME:AWAY:GW)
    for extra in args.extra_override:
        parts = extra.split(":")
        if len(parts) != 3:
            logger.warning("Invalid --extra-override format '%s' (expected H:A:GW)", extra)
            continue
        home, away, new_gw = int(parts[0]), int(parts[1]), int(parts[2])
        mask = (fixtures["team_h"] == home) & (fixtures["team_a"] == away)
        matched = mask.sum()
        if matched > 0:
            old_gw = fixtures.loc[mask, "event"].iloc[0]
            fixtures.loc[mask, "event"] = new_gw
            logger.info("CLI fixture override: team %d vs %d moved from GW %s to GW %d",
                        home, away, old_gw, new_gw)
        else:
            logger.warning("CLI fixture override: no match for team %d vs %d", home, away)

    predictions_path = OUTPUT_DIR / "predictions.csv"
    gw_data_path = OUTPUT_DIR / "gw_data.csv"
    gw_data_max_age_hours = 4

    def _load_cached_gw_data() -> Optional[pd.DataFrame]:
        """Load gw_data from cache if it exists and is fresh enough."""
        if not gw_data_path.exists():
            return None
        age_hours = (time.time() - gw_data_path.stat().st_mtime) / 3600
        if age_hours > gw_data_max_age_hours:
            logger.info("Cached gw_data is %.1f hours old (limit: %d) — will re-fetch",
                        age_hours, gw_data_max_age_hours)
            return None
        logger.info("Loading cached gw_data from %s (%.1f hours old)", gw_data_path, age_hours)
        return pd.read_csv(gw_data_path)

    # Early in the season there aren't enough 60+ minute appearances for
    # generate_predictions() to build player averages from, so fall back to the
    # pre-computed proxy predictions until GW `points_pred_gw_threshold`'s
    # deadline passes.
    use_proxy = proxy_predict.should_use_proxy(
        bootstrap, solver_params.get("points_pred_gw_threshold"), DATA_DIR
    )

    if use_proxy:
        # Skip the (slow) gameweek-data fetch entirely — there is no history to
        # fetch yet. Costs and player metadata come from bootstrap instead.
        gw_data = proxy_predict.synthesize_gw_data(bootstrap)
        predictions = proxy_predict.load_proxy_predictions(DATA_DIR)
    else:
        # gw_data is fixture-independent (player history), so always try cache first
        gw_data = _load_cached_gw_data()
        if gw_data is None:
            logger.info("Fetching gameweek data (this takes a few minutes)...")
            gw_data = api.fetch_gameweek_data(bootstrap)
            OUTPUT_DIR.mkdir(exist_ok=True)
            gw_data.to_csv(gw_data_path, index=False)
            logger.info("Cached gw_data to %s (%d rows)", gw_data_path, len(gw_data))
        else:
            logger.info("GW data: %d records (from cache)", len(gw_data))

        logger.info("Generating predictions...")
        predictions = generate_predictions(gw_data, fixtures, multipliers,
                                           current_season_tiers, season)

    OUTPUT_DIR.mkdir(exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    logger.info("Saved predictions to %s (%d rows)", predictions_path, len(predictions))

    # 5. Build watchlist
    must_include = list(current_squad)
    must_include.extend(overrides.get("extra_players", []))
    # Auto-promote forced_lineup players into the candidate pool so the
    # solver can actually start them (otherwise the constraint is silently
    # skipped if the player didn't pass the recent-minutes threshold).
    must_include.extend(pid for pid, _ in overrides.get("forced_lineup", []))
    must_exclude = overrides.get("excluded_players", [])
    min_hist_pct = solver_params.get("min_hist_pct", 0.6)
    max_hist_window = solver_params.get("max_hist_window", 6)

    if use_proxy:
        # The appearance-count filter is meaningless against synthesized gw_data
        # — nobody has played yet, so every player would be filtered out.
        # WARNING: this leaves the candidate pool unfiltered, and the proxy model
        # assumes every player starts, so cheap non-playing players are ranked
        # like nailed starters. Sanity-check the chosen squad in these GWs.
        logger.warning("Proxy predictions in use — min_hist_pct filter disabled "
                       "(candidate pool is unfiltered; verify the squad manually)")
        min_hist_pct = 0.0

    watchlist = create_watchlist(predictions, gw_data, min_hist_pct=min_hist_pct,
                                max_hist_window=max_hist_window,
                                must_include=must_include, must_exclude=must_exclude)

    # 6. Chip scenarios
    wildcards_used = chips_cfg.get("wildcards_used", 0)
    free_hits_used = chips_cfg.get("free_hits_used", 0)
    bench_boost_used = chips_cfg.get("bench_boost_used", 0)
    triple_captain_used = chips_cfg.get("triple_captain_used", 0)

    force_free_hit_gw = chips_cfg.get("force_free_hit_gw")
    force_bench_boost_gw = chips_cfg.get("force_bench_boost_gw")
    force_triple_captain_gw = chips_cfg.get("force_triple_captain_gw")
    force_wildcard_gw = chips_cfg.get("force_wildcard_gw")

    # CLI overrides take precedence over config
    if args.force_wildcard_gw is not None:
        force_wildcard_gw = args.force_wildcard_gw
    if args.force_free_hit_gw is not None:
        force_free_hit_gw = args.force_free_hit_gw
    if args.force_bench_boost_gw is not None:
        force_bench_boost_gw = args.force_bench_boost_gw

    triple_captain_used_first_half = min(triple_captain_used, 1)
    triple_captain_used_second_half = max(0, triple_captain_used - 1)

    # Triple Captain only ever adds a fixed bonus on the plan's own best
    # captain GW — it can't change who's captained, since the multiplier is
    # uniform across players (see find_best_triple_captain_gw). So instead of
    # multiplying it into the FH x BB scenario count, solve FH x BB with TC
    # off, then score+place TC post-hoc. Skipped when TC is forced (already
    # pinned) or fully spent (nothing left to place).
    defer_triple_captain = (
        not args.no_chips
        and force_triple_captain_gw is None
        and (triple_captain_used_first_half < 1 or triple_captain_used_second_half < 1)
    )

    if args.no_chips:
        chip_scenarios = [{"name": "No chips", "free_hit_gws": [],
                           "bench_boost_gw": -1, "triple_captain_gw": -1,
                           "force_wildcard_gw": None}]
    else:
        chip_scenarios = generate_chip_scenarios(
            start_gw=current_gw,
            planning_horizon=horizon,
            free_hits_used_first_half=min(free_hits_used, 1),
            free_hits_used_second_half=max(0, free_hits_used - 1),
            bench_boost_used_first_half=min(bench_boost_used, 1),
            bench_boost_used_second_half=max(0, bench_boost_used - 1),
            triple_captain_used_first_half=1 if defer_triple_captain else triple_captain_used_first_half,
            triple_captain_used_second_half=1 if defer_triple_captain else triple_captain_used_second_half,
            force_free_hit_gw=force_free_hit_gw,
            force_bench_boost_gw=force_bench_boost_gw,
            force_triple_captain_gw=force_triple_captain_gw,
            force_wildcard_gw=force_wildcard_gw,
        )

    max_scenarios = solver_params.get("max_scenarios", 100)
    if len(chip_scenarios) > max_scenarios:
        logger.warning("Limiting scenarios from %d to %d", len(chip_scenarios), max_scenarios)
        chip_scenarios = chip_scenarios[:max_scenarios]

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
    if args.time_limit is not None:
        time_limit = args.time_limit

    total_scenarios = len(chip_scenarios)
    start_time = time.time()

    # Everything a worker process needs to rebuild a solver on its own side.
    # The prediction/price frames travel once per worker, not once per scenario.
    solver_ctx = {
        "horizon": horizon,
        "budget": total_budget,
        "start_gw": current_gw,
        "afcon_enabled": transfer_topup.get("enabled", True),
        "afcon_trigger_gw": transfer_topup.get("trigger_gw", 15),
        "afcon_transfer_count": transfer_topup.get("transfer_count", 5),
        "points_multiplier": overrides.get("points_multiplier"),
        "forced_lineup": overrides.get("forced_lineup"),
        "non_playing": overrides.get("non_playing"),
        "first_gw_penalty": first_gw_penalty,
        "sub_probability": sub_probability,
        "predictions": predictions,
        "gw_data": gw_data,
        "watchlist": watchlist,
        "current_squad": current_squad,
        "free_transfers": free_transfers,
        "wildcard_first_half": min(wildcards_used, 1),
        "wildcard_second_half": max(0, wildcards_used - 1),
        "time_limit": time_limit,
        "mip_gap": solver_params.get("mip_gap"),
    }

    workers = args.workers if args.workers is not None else default_worker_count()
    raw_results = solve_scenarios(chip_scenarios, solver_ctx, workers=workers)

    best_result = None
    best_total = -float("inf")
    scenario_results = []
    failed_count = 0

    for raw in raw_results:
        scenario = raw["scenario"]
        if raw["status"] != "solved":
            failed_count += 1
            continue

        fh_total = sum(
            fh_benefits.get(gw, {}).get("total_points", 0)
            for gw in scenario["free_hit_gws"]
        )
        total_points = raw["base_points"] + fh_total

        result = {
            "scenario_name": scenario["name"],
            "free_hit_gws": scenario["free_hit_gws"],
            "bench_boost_gw": scenario["bench_boost_gw"],
            "triple_captain_gw": scenario["triple_captain_gw"],
            "base_points": raw["base_points"],
            "free_hit_benefit": fh_total,
            "total_points": total_points,
            "solution": raw["solution"],
            "captain_points": raw["captain_points"],
            "scenario": scenario,
        }
        scenario_results.append(result)

        if total_points > best_total:
            best_result = result
            best_total = total_points

    if best_result:
        logger.info("Best of %d solved scenarios: %s (%.1f pts)",
                    len(scenario_results), best_result["scenario_name"], best_total)

    # Score every solved FH x BB plan's best legal Triple Captain GW, then
    # re-solve only the top candidates as full MILPs (TC's placement can shift
    # the wildcard GW the solver picks, so the heuristic score alone isn't
    # trustworthy enough to skip the re-solve).
    if defer_triple_captain and scenario_results:
        tc_gws = triple_captain_candidate_gws(
            start_gw=current_gw, planning_horizon=horizon,
            used_first_half=triple_captain_used_first_half,
            used_second_half=triple_captain_used_second_half,
        )
        tc_ranked = []
        for result in scenario_results:
            wildcard_gws = [
                current_gw + t - 1
                for t, tr in result["solution"]["transfers"].items()
                if tr.get("wildcard_active")
            ]
            tc_gw, tc_bonus = find_best_triple_captain_gw(
                captains_by_t=result["solution"]["captains"],
                expected_points=result["captain_points"],
                wildcard_gws=wildcard_gws,
                start_gw=current_gw, horizon=horizon,
                free_hit_gws=result["free_hit_gws"],
                bench_boost_gw=result["bench_boost_gw"],
                candidate_gws=tc_gws,
                sub_probability=sub_probability,
            )
            if tc_gw is not None:
                tc_ranked.append((result, tc_gw, tc_bonus))

        tc_ranked.sort(key=lambda item: item[0]["total_points"] + item[2], reverse=True)
        tc_top_k = solver_params.get("triple_captain_candidates", 5)
        logger.info("Re-solving top %d Triple Captain candidate(s) (out of %d scored)...",
                    min(tc_top_k, len(tc_ranked)), len(tc_ranked))

        tc_candidates = tc_ranked[:tc_top_k]
        tc_scenarios = [
            {
                "name": f"{result['scenario_name']} | TC GW{tc_gw}",
                "free_hit_gws": result["free_hit_gws"],
                "bench_boost_gw": result["bench_boost_gw"],
                "triple_captain_gw": tc_gw,
                "force_wildcard_gw": force_wildcard_gw,
            }
            for result, tc_gw, _ in tc_candidates
        ]
        tc_results = solve_scenarios(tc_scenarios, solver_ctx, workers=workers)

        for raw in tc_results:
            if raw["status"] != "solved":
                continue
            scenario = raw["scenario"]
            fh_total = sum(
                fh_benefits.get(gw, {}).get("total_points", 0)
                for gw in scenario["free_hit_gws"]
            )
            total_points = raw["base_points"] + fh_total
            if total_points > best_total:
                best_result = {
                    "scenario_name": scenario["name"],
                    "free_hit_gws": scenario["free_hit_gws"],
                    "bench_boost_gw": scenario["bench_boost_gw"],
                    "triple_captain_gw": scenario["triple_captain_gw"],
                    "base_points": raw["base_points"],
                    "free_hit_benefit": fh_total,
                    "total_points": total_points,
                    "solution": raw["solution"],
                    "captain_points": raw["captain_points"],
                    "scenario": scenario,
                }
                best_total = total_points

    elapsed = time.time() - start_time
    logger.info("Done: %d/%d solved, %d failed in %.0fs",
                len(scenario_results), total_scenarios, failed_count, elapsed)

    if not best_result:
        logger.error("No optimal solution found in any scenario")
        return

    logger.info("")
    logger.info("BEST STRATEGY: %s (%.1f pts)", best_result["scenario_name"], best_total)

    # Workers return plain data, not solver objects (a solved PuLP model is
    # expensive to ship between processes). Rebuilding the winner's solver is
    # ~0.2s and gives display_strategy the predictions/horizon it reads.
    best_solver = build_solver(best_result["scenario"], solver_ctx)

    strategy_text = display_strategy(
        best_result["solution"], best_solver, best_solver.players,
        current_gw, best_total, best_result["scenario_name"],
        overrides.get("non_playing", []),
        best_result["free_hit_gws"], fh_benefits,
        initial_bank, initial_selling_prices,
    )

    save_strategy(best_result["solution"], best_result["scenario_name"],
                  best_total, current_gw, strategy_text=strategy_text,
                  output_suffix=args.output_suffix)


if __name__ == "__main__":
    main()
