"""
FastAPI wrapper for fpl-solver — serves the optimization pipeline as a REST API.

Deployed on GCP Cloud Run. Accepts JSON config payloads, runs the solver
pipeline (data fetch → predictions → MILP optimization), and returns
structured results for the fpl-lad frontend.

Endpoints:
    POST /api/optimize  — run solver with given config
    GET  /api/squad     — fetch and return a user's current squad
    GET  /api/health    — health check
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from supabase import create_client as _create_supabase
    _SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    _SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
    _supabase = _create_supabase(_SUPABASE_URL, _SUPABASE_KEY) if _SUPABASE_URL and _SUPABASE_KEY else None
except Exception:
    _supabase = None

_CRON_SECRET = os.environ.get("CRON_SECRET", "")

from fpl import api, config as cfg, proxy_predict
from fpl.predict import generate_predictions
from fpl.solver import FPLSolver, TRANSFER_PENALTY_POINTS
from fpl.free_hit import (
    generate_chip_scenarios, calculate_free_hit_benefits_for_horizon,
    triple_captain_candidate_gws, find_best_triple_captain_gw,
)
from fpl.watchlist import create_watchlist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# How many top-scored (plan, TC GW) candidates get re-solved as full MILPs
# during the post-hoc Triple Captain placement — see run.py's mirror of this
# for the rationale (find_best_triple_captain_gw in fpl/free_hit.py).
TC_RESELECT_TOP_K = 5

app = FastAPI(title="fpl-solver API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory cache for expensive FPL API data
# ---------------------------------------------------------------------------
_gw_data_cache: dict[str, Any] = {"data": None, "fetched_at": 0.0}
GW_DATA_MAX_AGE_HOURS = 4


def _get_gw_data(bootstrap: dict) -> pd.DataFrame:
    """Return cached gw_data, or Supabase's daily-synced copy, or fetch fresh from FPL API."""
    age_hours = (time.time() - _gw_data_cache["fetched_at"]) / 3600
    if _gw_data_cache["data"] is not None and age_hours < GW_DATA_MAX_AGE_HOURS:
        logger.info("Using cached gw_data (%.1f hours old)", age_hours)
        return _gw_data_cache["data"]

    gw_data = api.fetch_gameweek_data_from_supabase(_supabase, bootstrap)
    if gw_data is not None:
        logger.info("Using gw_data from Supabase: %d rows", len(gw_data))
    else:
        logger.info("Fetching gameweek data from FPL API (this may take a few minutes)...")
        gw_data = api.fetch_gameweek_data(bootstrap)

    _gw_data_cache["data"] = gw_data
    _gw_data_cache["fetched_at"] = time.time()
    logger.info("Cached gw_data: %d rows", len(gw_data))
    return gw_data


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class NonPlayingEntry(BaseModel):
    """One player who won't feature in a set of gameweeks. Mirrors the `non_playing`
    shape in config.yaml so the API contract matches the documented YAML."""
    player: int
    gameweeks: list[int] = Field(default_factory=list)


class OptimizeRequest(BaseModel):
    """Input payload for the /api/optimize endpoint."""
    team_id: int
    free_transfers: int = Field(ge=1, le=5)
    planning_horizon: int = Field(default=3, ge=1, le=10)
    use_chips: bool = True
    forced_lineup: list[int] = Field(default_factory=list)
    excluded_players: list[int] = Field(default_factory=list)
    # Zero out these players' points for the listed GWs (injury/suspension/rotation).
    # Note: `excluded_players` still wins over `extra_players` — create_watchlist drops
    # must_exclude before it exempts must_include.
    non_playing: list[NonPlayingEntry] = Field(default_factory=list)
    # Force into the candidate pool even if they fail the min_hist_pct filter.
    extra_players: list[int] = Field(default_factory=list)
    time_limit_per_scenario: int = Field(default=10, ge=5, le=30)
    max_scenarios: int = Field(default=50, ge=1, le=200)
    force_wildcard_gw: Optional[int] = None
    force_free_hit_gw: Optional[int] = None
    force_bench_boost_gw: Optional[int] = None
    force_triple_captain_gw: Optional[int] = None
    # Chip-usage state. Auto-detected from the FPL API when left as None; set a value
    # to override, e.g. 2 to hide a chip from the solver and save it for later.
    wildcards_used: Optional[int] = Field(default=None, ge=0, le=2)
    free_hits_used: Optional[int] = Field(default=None, ge=0, le=2)
    bench_boost_used: Optional[int] = Field(default=None, ge=0, le=2)
    triple_captain_used: Optional[int] = Field(default=None, ge=0, le=2)
    # ── Squad/budget overrides ────────────────────────────────────────────────
    # FPL's public API only reports a squad as of the last deadline that has passed, so
    # it is stale for anyone who has since made a transfer, and pre-season it reports no
    # squad at all. When the caller has a squad the user has explicitly confirmed, that
    # beats what we can fetch — plan from theirs.
    #
    # `squad` is 15 element IDs. `total_budget` is squad selling value + bank in £m
    # (e.g. 100.0), matching the shape /api/squad returns; internally the pipeline works
    # in tenths, so it's converted on the way in. Send both together: a squad with a
    # budget derived from a different squad is worse than either alone.
    squad: Optional[list[int]] = Field(default=None, min_length=15, max_length=15)
    total_budget: Optional[float] = Field(default=None, gt=0, le=200)


class SquadRequest(BaseModel):
    """Query params for the /api/squad endpoint."""
    team_id: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player_name(players: pd.DataFrame, pid: int) -> str:
    row = players[players["element"] == pid]
    return row["name"].iloc[0] if len(row) else str(pid)


def _player_info(bootstrap: dict, pid: int) -> dict:
    """Get player name, position, and price from bootstrap data."""
    for el in bootstrap.get("elements", []):
        if el["id"] == pid:
            pos = POSITION_MAP.get(el.get("element_type", 0), "UNKNOWN")
            return {
                "id": pid,
                "name": f"{el.get('first_name', '')} {el.get('second_name', '')}".strip(),
                "position": pos,
                "price": el.get("now_cost", 0) / 10,
            }
    return {"id": pid, "name": str(pid), "position": "UNKNOWN", "price": 0}


def _format_solution(
    solution: dict,
    solver: FPLSolver,
    players: pd.DataFrame,
    start_gw: int,
    total_points: float,
    scenario_name: str,
    fh_benefits: dict,
    bootstrap: dict,
) -> dict:
    """Convert solver solution into the frontend-friendly JSON format."""
    predictions = solver.predictions
    expected_points: dict[tuple[int, int], float] = {}
    for _, row in predictions.iterrows():
        key = (int(row["element"]), int(row["event"]))
        expected_points[key] = expected_points.get(key, 0) + row["predicted_points"]

    gameweeks_output = []
    for i in range(solver.T):
        t = i + 1
        gw = start_gw + i
        transfers = solution["transfers"][t]
        chips = solution["chips"][t]
        captain_id = solution["captains"].get(t)
        lineup_data = solution["lineups"].get(t, {})
        lineup_ids = lineup_data.get("starters", []) if lineup_data else []
        bench_ids = lineup_data.get("bench", []) if lineup_data else []

        real_in = [p for p in transfers["in"] if p not in transfers["out"]]
        real_out = [p for p in transfers["out"] if p not in transfers["in"]]

        chip = None
        if "wildcard" in chips:
            chip = "wildcard"
        elif "free_hit" in chips:
            chip = "free_hit"
        elif "bench_boost" in chips:
            chip = "bench_boost"
        elif "triple_captain" in chips:
            chip = "triple_captain"

        gw_pts = 0.0
        for pid in lineup_ids:
            pts = expected_points.get((pid, gw), 0)
            gw_pts += pts
            if pid == captain_id:
                mult = 2 if chip == "triple_captain" else 1
                gw_pts += pts * mult
        if chip == "bench_boost":
            for pid in bench_ids:
                gw_pts += expected_points.get((pid, gw), 0)

        if chip == "free_hit" and fh_benefits and gw in fh_benefits:
            fh = fh_benefits[gw]
            gw_pts = fh.get("total_points", 0)

        starters = [
            {
                **_player_info(bootstrap, pid),
                "expected_points": round(expected_points.get((pid, gw), 0), 1),
                "is_captain": pid == captain_id,
                "is_vice_captain": False,
            }
            for pid in lineup_ids
        ]
        bench = [
            {
                **_player_info(bootstrap, pid),
                "expected_points": round(expected_points.get((pid, gw), 0), 1),
                "bench_order": idx + 1,
            }
            for idx, pid in enumerate(bench_ids)
        ]

        gw_entry = {
            "gw": gw,
            "transfers_in": [_player_info(bootstrap, pid) for pid in real_in],
            "transfers_out": [_player_info(bootstrap, pid) for pid in real_out],
            "chip": chip,
            "starters": starters,
            "bench": bench,
            "captain": _player_info(bootstrap, captain_id) if captain_id else None,
            "expected_points": round(gw_pts, 1),
            "free_transfers_available": int(transfers.get("available_transfers", 0)),
            "paid_transfers": int(transfers.get("paid_transfers", 0)),
        }
        gameweeks_output.append(gw_entry)

    return {
        "objective": round(total_points, 1),
        "scenario": scenario_name,
        "start_gw": start_gw,
        "end_gw": start_gw + solver.T - 1,
        "gameweeks": gameweeks_output,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "fpl-solver"}


def _last_known_gw(started_event: Optional[int], bootstrap_data: dict) -> Optional[int]:
    """
    Latest GW that's both (a) past its deadline and (b) on or after the team's
    `started_event` — i.e. the most recent GW this team could actually have
    public picks for.

    Handles pre-GW1 (started_event == 1, but GW1's deadline hasn't passed) and
    managers who joined mid-season (started_event == some later GW whose
    deadline hasn't passed yet) the same way: both return None.

    Returns:
        The GW number, or None if this team has no public picks yet.
    """
    now = datetime.now(timezone.utc)
    known: Optional[int] = None
    for event in bootstrap_data.get("events", []):
        eid = int(event.get("id", 0))
        deadline_str = event.get("deadline_time")
        if not deadline_str:
            continue
        try:
            deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if deadline > now:
            continue
        if started_event is not None and eid < started_event:
            continue
        if known is None or eid > known:
            known = eid
    return known


@app.get("/api/squad")
async def get_squad(team_id: int):
    """Fetch a user's current squad from the FPL API."""
    try:
        bootstrap = api.fetch_bootstrap_data()
        current_gw = api.detect_current_gw(bootstrap)

        entry = api.fetch_entry_summary(team_id)
        if entry is None:
            # FPL's entry endpoint 404s (or is unreachable) — don't mistake a bad
            # team ID for a team whose picks simply aren't public yet.
            raise HTTPException(status_code=404, detail=f"No FPL team found with ID {team_id}")

        last_gw = _last_known_gw(entry.get("started_event"), bootstrap)

        if last_gw is None:
            return {
                "team_id": team_id,
                "squad": None,
                "reason": "picks_not_public_yet",
            }

        team_data = api.fetch_team_data(team_id, last_gw)
        selling_info, selling_summary = api.get_squad_selling_prices(team_id, last_gw)
        chips_used = api.detect_chips_used(team_id)

        squad_details = []
        for pid in team_data["squad"]:
            info = _player_info(bootstrap, pid)
            sell = next((s for s in selling_info if s["element"] == pid), None)
            info["selling_value"] = sell["selling_value"] / 10 if sell else info["price"]
            squad_details.append(info)

        return {
            "team_id": team_id,
            "current_gw": current_gw,
            "squad": squad_details,
            "bank": selling_summary["bank"] / 10,
            "total_budget": selling_summary["correct_budget"] / 10,
            "chips_used": {
                "wildcards_used": chips_used.get("wildcards_used", 0),
                "free_hits_used": chips_used.get("free_hits_used", 0),
                "bench_boost_used": chips_used.get("bench_boost_used", 0),
                "triple_captain_used": chips_used.get("triple_captain_used", 0),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fetching squad for team %d", team_id)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/optimize")
async def optimize(req: OptimizeRequest):
    """Run the fpl-solver pipeline and return the optimal strategy."""
    try:
        return await _optimize_inner(req)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Optimization failed for team %d", req.team_id)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Async endpoint — fires optimization in background, writes result to Supabase
# ---------------------------------------------------------------------------

def _update_job(job_id: str, status: str, result: Any = None, error: str | None = None) -> None:
    if not _supabase:
        logger.warning("Supabase not configured — cannot update job %s", job_id)
        return
    payload: dict[str, Any] = {"status": status, "updated_at": "now()"}
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error
    try:
        _supabase.table("solver_jobs").update(payload).eq("id", job_id).execute()
    except Exception as exc:
        logger.exception("Failed to update solver_jobs for job %s: %s", job_id, exc)


def _run_and_store(req: OptimizeRequest, job_id: str) -> None:
    """Background task: run optimize pipeline and persist result to Supabase."""
    _update_job(job_id, "running")
    try:
        # Reuse the synchronous optimize logic by calling it inline
        import asyncio
        result = asyncio.run(_optimize_inner(req))
        _update_job(job_id, "complete", result=result)
        logger.info("Async job %s complete — %.1f pts", job_id, result.get("objective", 0))
    except Exception as exc:
        logger.exception("Async job %s failed", job_id)
        _update_job(job_id, "failed", error=str(exc))


async def _optimize_inner(req: OptimizeRequest) -> dict:
    """Pure optimization logic shared by /api/optimize and /api/optimize-async."""
    start_time = time.time()

    bootstrap = api.fetch_bootstrap_data()
    detected_gw = api.detect_current_gw(bootstrap)
    season = api.detect_current_season(bootstrap)
    detected_chips = api.detect_chips_used(req.team_id)

    current_gw = detected_gw
    horizon = min(req.planning_horizon, 38 - current_gw + 1)

    free_hit_gws = detected_chips.get("free_hit_gws", [])
    squad_gw = current_gw - 1
    if squad_gw in free_hit_gws and squad_gw > 1:
        squad_gw -= 1

    # A caller-supplied squad is one the user has confirmed, so it outranks whatever the
    # FPL API reports — which is only ever as of the last deadline, and is empty
    # pre-season. Skipping the fetches also saves two API round trips.
    if req.squad:
        current_squad = list(req.squad)
        logger.info("Using caller-supplied squad for team %d", req.team_id)
    else:
        team_data = api.fetch_team_data(req.team_id, squad_gw)
        current_squad = team_data["squad"]

    if req.total_budget is not None:
        # Pipeline works in tenths of a million throughout; the API takes £m.
        total_budget = int(round(req.total_budget * 10))
    else:
        _, selling_summary = api.get_squad_selling_prices(req.team_id, squad_gw)
        total_budget = selling_summary["correct_budget"]

    multipliers = pd.read_csv(DATA_DIR / "multipliers.csv")
    team_tiers = pd.read_csv(DATA_DIR / "team_tiers.csv")
    current_season_tiers = team_tiers[team_tiers["season"] == season].copy()

    fixtures = api.fetch_current_fixtures(bootstrap)

    # Early in the season there aren't enough 60+ minute appearances for
    # generate_predictions() to build player averages from (it needs THIS season's
    # per-fixture history, which is empty pre-season and for the first few GWs).
    # Mirrors the same fallback run.py uses — see fpl/proxy_predict.py.
    solver_config = cfg.load_config()
    points_pred_gw_threshold = cfg.get_solver_params(solver_config).get("points_pred_gw_threshold")
    use_proxy = proxy_predict.should_use_proxy(bootstrap, points_pred_gw_threshold, DATA_DIR)

    if use_proxy:
        gw_data = proxy_predict.synthesize_gw_data(bootstrap)
        predictions = proxy_predict.load_proxy_predictions(DATA_DIR)
    else:
        gw_data = _get_gw_data(bootstrap)
        predictions = generate_predictions(gw_data, fixtures, multipliers, current_season_tiers, season)

    # extra_players bypasses the min_hist_pct filter — new signings, players just
    # back from injury, anyone with too little recent game time to qualify on merit.
    must_include = list(dict.fromkeys(list(current_squad) + list(req.extra_players)))
    # Synthesized gw_data has nobody with real appearances — the filter would
    # exclude everyone, so it's disabled while proxy predictions are in use.
    min_hist_pct = 0.0 if use_proxy else 0.6
    watchlist = create_watchlist(
        predictions, gw_data,
        min_hist_pct=min_hist_pct, max_hist_window=6,
        must_include=must_include, must_exclude=req.excluded_players,
    )

    non_playing_tuples = [(e.player, list(e.gameweeks)) for e in req.non_playing if e.gameweeks] or None

    # Explicit request values win over API detection; None means "use what we detected".
    def _chip_state(override: Optional[int], key: str) -> int:
        return override if override is not None else detected_chips.get(key, 0)

    wildcards_used = _chip_state(req.wildcards_used, "wildcards_used")
    free_hits_used = _chip_state(req.free_hits_used, "free_hits_used")
    bench_boost_used = _chip_state(req.bench_boost_used, "bench_boost_used")
    triple_captain_used = _chip_state(req.triple_captain_used, "triple_captain_used")

    triple_captain_used_first_half = min(triple_captain_used, 1)
    triple_captain_used_second_half = max(0, triple_captain_used - 1)

    # Triple Captain only ever adds a fixed bonus on the plan's own best
    # captain GW — see find_best_triple_captain_gw. Solve FH x BB with TC off,
    # then score+place TC post-hoc instead of multiplying it into the
    # scenario count. Skipped when TC is forced or fully spent.
    defer_triple_captain = (
        req.use_chips
        and req.force_triple_captain_gw is None
        and (triple_captain_used_first_half < 1 or triple_captain_used_second_half < 1)
    )

    if not req.use_chips:
        chip_scenarios = [{"name": "No chips", "free_hit_gws": [], "bench_boost_gw": -1, "triple_captain_gw": -1, "force_wildcard_gw": None}]
    else:
        chip_scenarios = generate_chip_scenarios(
            start_gw=current_gw, planning_horizon=horizon,
            free_hits_used_first_half=min(free_hits_used, 1),
            free_hits_used_second_half=max(0, free_hits_used - 1),
            bench_boost_used_first_half=min(bench_boost_used, 1),
            bench_boost_used_second_half=max(0, bench_boost_used - 1),
            triple_captain_used_first_half=1 if defer_triple_captain else triple_captain_used_first_half,
            triple_captain_used_second_half=1 if defer_triple_captain else triple_captain_used_second_half,
            force_free_hit_gw=req.force_free_hit_gw,
            force_bench_boost_gw=req.force_bench_boost_gw,
            force_triple_captain_gw=req.force_triple_captain_gw,
            force_wildcard_gw=req.force_wildcard_gw,
        )

    if len(chip_scenarios) > req.max_scenarios:
        chip_scenarios = chip_scenarios[:req.max_scenarios]

    forced_lineup_tuples = [(pid, list(range(current_gw, current_gw + horizon))) for pid in req.forced_lineup]
    fh_benefits: dict = {}
    if any(s["free_hit_gws"] for s in chip_scenarios):
        fh_benefits = calculate_free_hit_benefits_for_horizon(
            start_gw=current_gw, planning_horizon=horizon, budget=total_budget,
            predictions_df=predictions, gw_data_df=gw_data,
            watchlist_players=watchlist,
            forced_lineup_players=forced_lineup_tuples if req.forced_lineup else None,
            non_playing_players=non_playing_tuples,
        )

    best_result = None
    best_total = -float("inf")
    scenario_results = []

    for idx, scenario in enumerate(chip_scenarios, 1):
        solver = FPLSolver(
            planning_horizon=horizon, budget=total_budget, start_gw=current_gw,
            points_multiplier_override=None,
            forced_lineup_players=forced_lineup_tuples if req.forced_lineup else None,
            non_playing_players=non_playing_tuples,
            first_gw_transfer_penalty=-1,
            sub_probability=0.10,
            bench_boost_gw=scenario["bench_boost_gw"],
            triple_captain_gw=scenario["triple_captain_gw"],
            free_hit_gws=scenario["free_hit_gws"],
            force_wildcard_gw=scenario.get("force_wildcard_gw"),
        )
        solver.load_predictions(predictions)
        if len(solver.predictions) == 0:
            continue
        solver.load_player_data(gw_data, predictions, player_subset=watchlist)
        solver.set_initial_squad(current_squad, available_transfers=req.free_transfers)
        solver.set_chip_state(
            wildcard_first_half=min(wildcards_used, 1),
            wildcard_second_half=max(0, wildcards_used - 1),
        )
        solver.build_model()
        if not solver.solve(time_limit=req.time_limit_per_scenario):
            continue

        solution = solver.extract_solution()
        base_points = solution["objective_value"]
        fh_total = sum(fh_benefits.get(gw, {}).get("total_points", 0) for gw in scenario["free_hit_gws"])
        total_points = base_points + fh_total

        result = {"scenario_name": scenario["name"], "free_hit_gws": scenario["free_hit_gws"],
                  "bench_boost_gw": scenario["bench_boost_gw"], "solution": solution,
                  "solver": solver, "players": solver.players, "total_points": total_points}
        scenario_results.append(result)

        if total_points > best_total:
            best_total = total_points
            best_result = result

        logger.info("[%d/%d] %-30s  %.1f pts", idx, len(chip_scenarios), scenario["name"], total_points)

    # Score every solved FH x BB plan's best legal Triple Captain GW, then
    # re-solve only the top candidates as full MILPs — see run.py's mirror of
    # this stage for the rationale.
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
                expected_points=result["solver"].expected_points,
                wildcard_gws=wildcard_gws,
                start_gw=current_gw, horizon=horizon,
                free_hit_gws=result["free_hit_gws"],
                bench_boost_gw=result["bench_boost_gw"],
                candidate_gws=tc_gws,
                sub_probability=0.10,
            )
            if tc_gw is not None:
                tc_ranked.append((result, tc_gw, tc_bonus))

        tc_ranked.sort(key=lambda item: item[0]["total_points"] + item[2], reverse=True)

        for result, tc_gw, _tc_bonus in tc_ranked[:TC_RESELECT_TOP_K]:
            tc_scenario_name = f"{result['scenario_name']} | TC GW{tc_gw}"
            solver = FPLSolver(
                planning_horizon=horizon, budget=total_budget, start_gw=current_gw,
                points_multiplier_override=None,
                forced_lineup_players=forced_lineup_tuples if req.forced_lineup else None,
                non_playing_players=non_playing_tuples,
                first_gw_transfer_penalty=-1,
                sub_probability=0.10,
                bench_boost_gw=result["bench_boost_gw"],
                triple_captain_gw=tc_gw,
                free_hit_gws=result["free_hit_gws"],
                force_wildcard_gw=req.force_wildcard_gw,
            )
            solver.load_predictions(predictions)
            if len(solver.predictions) == 0:
                continue
            solver.load_player_data(gw_data, predictions, player_subset=watchlist)
            solver.set_initial_squad(current_squad, available_transfers=req.free_transfers)
            solver.set_chip_state(
                wildcard_first_half=min(wildcards_used, 1),
                wildcard_second_half=max(0, wildcards_used - 1),
            )
            solver.build_model()
            if not solver.solve(time_limit=req.time_limit_per_scenario):
                continue

            solution = solver.extract_solution()
            base_points = solution["objective_value"]
            fh_total = sum(fh_benefits.get(gw, {}).get("total_points", 0) for gw in result["free_hit_gws"])
            total_points = base_points + fh_total

            if total_points > best_total:
                best_total = total_points
                best_result = {"scenario_name": tc_scenario_name, "free_hit_gws": result["free_hit_gws"],
                               "bench_boost_gw": result["bench_boost_gw"], "solution": solution,
                               "solver": solver, "players": solver.players, "total_points": total_points}

            logger.info("  %-30s  %.1f pts", tc_scenario_name, total_points)

    if not best_result:
        raise ValueError("No feasible solution found")

    result = _format_solution(
        best_result["solution"], best_result["solver"],
        best_result["players"], current_gw, best_total,
        best_result["scenario_name"], fh_benefits, bootstrap,
    )
    result["elapsed_seconds"] = round(time.time() - start_time, 1)
    result["scenarios_evaluated"] = len(chip_scenarios)
    return result


@app.post("/api/optimize-async", status_code=202)
async def optimize_async(
    req: OptimizeRequest,
    background_tasks: BackgroundTasks,
    job_id: str = Query(..., description="UUID of the solver_jobs row to update"),
):
    """Start optimization in background; returns 202 immediately. Result written to Supabase."""
    background_tasks.add_task(_run_and_store, req, job_id)
    return Response(status_code=202)


# ---------------------------------------------------------------------------
# Daily predictions job — meant to be hit by GCP Cloud Scheduler, ~1 hour
# after fpl-lad's fpl-sync cron populates player_gw_history for the day.
# Generates the same component-based predictions /api/optimize uses, but
# once for everyone, and writes them to Supabase so both Alfie and the
# solver can read them back instead of recomputing per request.
# ---------------------------------------------------------------------------

def _predictions_sync_enabled(default: bool = True) -> bool:
    """
    Read app_config.predictions_sync_enabled.

    Defaults to enabled when the row is absent or unreadable, so a missing key
    never silently stops predictions from being generated — the flag has to be
    set to false deliberately to take effect.
    """
    try:
        resp = (
            _supabase.table("app_config")
            .select("value")
            .eq("key", "predictions_sync_enabled")
            .maybe_single()
            .execute()
        )
    except Exception:
        logger.exception("Could not read predictions_sync_enabled — assuming %s", default)
        return default

    if not resp or not getattr(resp, "data", None):
        return default

    value = resp.data.get("value")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "off", "no")
    return default


@app.post("/api/cron/generate-predictions")
async def generate_predictions_cron(secret: str = Query(...)):
    if not _CRON_SECRET or secret != _CRON_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if _supabase is None:
        raise HTTPException(status_code=503, detail="Supabase not configured on this service")

    # Kill switch. Early in the season player_predictions is populated by hand
    # with proxy predictions (see fpl-lad's proxy-points-predictions skill) and
    # must stay static — regenerating would overwrite it with the empty result
    # the real pipeline produces before any fixtures have been played. Flip
    # app_config.predictions_sync_enabled back to true to resume normal syncing.
    if not _predictions_sync_enabled():
        logger.info("generate-predictions cron: skipped (predictions_sync_enabled = false)")
        return {"ok": True, "rows_written": 0, "note": "predictions sync disabled via app_config"}

    start_time = time.time()

    bootstrap = api.fetch_bootstrap_data()
    season = api.detect_current_season(bootstrap)

    gw_data = _get_gw_data(bootstrap)
    if gw_data.empty or "minutes" not in gw_data.columns:
        # No fixture has been played yet this season (e.g. pre-season, before GW1's
        # deadline) — generate_predictions() needs real minutes-played history to
        # build player averages from. Nothing to do yet; try again on the next run.
        return {"ok": True, "rows_written": 0, "note": "no gameweek data yet this season"}

    fixtures = api.fetch_current_fixtures(bootstrap)
    multipliers = pd.read_csv(DATA_DIR / "multipliers.csv")
    team_tiers = pd.read_csv(DATA_DIR / "team_tiers.csv")
    current_season_tiers = team_tiers[team_tiers["season"] == season].copy()

    predictions = generate_predictions(gw_data, fixtures, multipliers, current_season_tiers, season)
    if len(predictions) == 0:
        return {"ok": True, "rows_written": 0, "note": "generate_predictions returned no rows"}

    # Sum the 9 scoring components down to one predicted_points total per
    # player per gameweek — simplest shape for both Alfie's SQL and the solver
    # (which already sums per (element, event) itself when it loads predictions).
    totals = predictions.groupby(["element", "event"], as_index=False)["predicted_points"].sum()

    generated_at = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "player_id": int(r.element),
            "event": int(r.event),
            "predicted_points": round(float(r.predicted_points), 2),
            "generated_at": generated_at,
        }
        for r in totals.itertuples()
    ]

    BATCH_SIZE = 500
    written = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        _supabase.table("player_predictions").upsert(batch, on_conflict="player_id,event").execute()
        written += len(batch)

    elapsed = round(time.time() - start_time, 1)
    logger.info("generate-predictions cron: wrote %d rows in %.1fs", written, elapsed)
    return {"ok": True, "rows_written": written, "elapsed_seconds": elapsed}
