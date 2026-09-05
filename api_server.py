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
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    bench_boost_candidate_gws, find_best_bench_boost_gw,
)
from fpl.watchlist import create_watchlist, apply_price_bucket_filter
from fpl.scenario_runner import build_solver, solve_scenarios

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# How many top-scored (FH plan, BB GW, TC GW) candidates get re-solved as full
# MILPs during the post-hoc chip placement — see run.py's mirror of this for
# the rationale (find_best_bench_boost_gw / find_best_triple_captain_gw in
# fpl/free_hit.py). Cut from 5 to 3 on 2026-09-05 as part of the solve-time
# work: the re-solve exists to catch BB's bench approximation, and the top-3
# candidates were the only ones ever observed winning.
CHIP_RESELECT_TOP_K = 3

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


class ForcedLineupEntry(BaseModel):
    """One player who must be in the starting XI for a set of gameweeks. Mirrors the
    `forced_lineup` shape in config.yaml, same as NonPlayingEntry does for non_playing.

    An entry with no gameweeks is dropped rather than treated as "every gameweek" —
    forcing a start across a whole horizon is a constraint nobody asks for by accident,
    so it has to be spelled out gameweek by gameweek."""
    player: int
    gameweeks: list[int] = Field(default_factory=list)


class SellingPriceEntry(BaseModel):
    """What one player the manager already owns would actually raise if sold.

    FPL never publishes this: a player who has risen sells for his purchase price plus
    half the rise, so it depends on what this particular manager paid. The caller
    normally has it already (it is on the squad card the user confirmed), and sending it
    saves re-deriving it from transfer history here."""
    player: int
    selling_value: float = Field(gt=0, le=30)


class OptimizeRequest(BaseModel):
    """Input payload for the /api/optimize endpoint."""
    team_id: int
    # 0 is legitimate, not a degenerate input: a manager who has already spent this
    # week's transfer is planning from zero, and any transfer the plan makes in the first
    # gameweek should cost 4 points. The MILP already models it — `A` (transfers_available)
    # is declared lowBound=0 in fpl/solver.py — so only this validator was rejecting it.
    free_transfers: int = Field(ge=0, le=5)
    # 19 = the chip half-season boundary (GW1-19 vs GW20-38, see CHIP_WINDOWS in
    # fpl/free_hit.py) — the default and ceiling both reach exactly that far from
    # a GW1 start. horizon = min(planning_horizon, 38 - current_gw + 1) below still
    # clamps to whatever's actually left in the season later on.
    planning_horizon: int = Field(default=19, ge=1, le=19)
    use_chips: bool = True
    # Per-gameweek, matching config.yaml. This was a flat list[int] that got expanded to
    # `range(current_gw, current_gw + horizon)` — i.e. the API could only ever say "start
    # this player in EVERY gameweek of the plan", which at the default horizon of 19 is a
    # near-certain infeasibility and never what "start Palmer in GW27" meant. The solver
    # itself always took (player, [gws]) tuples; only this layer flattened them away.
    forced_lineup: list[ForcedLineupEntry] = Field(default_factory=list)
    excluded_players: list[int] = Field(default_factory=list)
    # Zero out these players' points for the listed GWs (injury/suspension/rotation).
    # Note: `excluded_players` still wins over `extra_players` — create_watchlist drops
    # must_exclude before it exempts must_include.
    non_playing: list[NonPlayingEntry] = Field(default_factory=list)
    # Force into the candidate pool even if they fail the min_hist_pct filter.
    extra_players: list[int] = Field(default_factory=list)
    time_limit_per_scenario: int = Field(default=10, ge=5, le=90)
    max_scenarios: int = Field(default=50, ge=1, le=500)
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
    # Cash in hand in £m, and the per-player sale prices behind `total_budget`. Both are
    # optional and derived from the FPL API when absent, but sending them is what makes
    # the plan's money exact: without the sale prices every owned player is assumed to
    # sell for his market price, which over-states what a risen squad can raise.
    bank: Optional[float] = Field(default=None, ge=0, le=200)
    selling_prices: Optional[list[SellingPriceEntry]] = Field(default=None)
    # ── Experimental: price-bucket watchlist filter (A/B test, not for prod use) ──
    # When set, further restricts create_watchlist's output to the top
    # `bucket_top_n` players per (position, price rounded up to the nearest
    # quarter-million) group, ranked by average predicted points over the next
    # `bucket_horizon` gameweeks. None (default) leaves the pipeline unchanged.
    bucket_top_n: Optional[int] = Field(default=None, ge=1, le=50)
    bucket_horizon: int = Field(default=5, ge=1, le=19)
    # Prune the Free Hit placement scenarios to the top K, ranked by the
    # precomputed per-week FH benefit, before the expensive full solves — the
    # same score-then-reselect treatment BB/TC placements already get. Measured
    # 2026-09-05 across 4 teams: pruning to top-4 costs at most 0.2 points,
    # because the candidate weeks are near-ties. None disables pruning (full
    # enumeration); ignored when a specific FH week is forced.
    fh_top_k: Optional[int] = Field(default=5, ge=1, le=50)


class SquadRequest(BaseModel):
    """Query params for the /api/squad endpoint."""
    team_id: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player_name(players: pd.DataFrame, pid: int) -> str:
    row = players[players["element"] == pid]
    return row["name"].iloc[0] if len(row) else str(pid)


# A table read is trusted only this long past its generated_at before the
# request falls back to regenerating in-process. The nightly cron rewrites the
# table daily, and the fpl-lad freshness alert fires at 26h — so 30h means
# "stale enough that the alert is already ringing", not a second opinion on it.
PREDICTIONS_TABLE_MAX_AGE_HOURS = 30


def _predictions_from_supabase(bootstrap: dict, gw_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Load the nightly cron's predictions from player_predictions instead of
    regenerating them in-process (~25s of pandas per request for numbers that
    only change once a day, and that Alfie is already reading from this table).

    Returns None whenever the table can't serve the request — missing client,
    empty table, stale rows, or any read error — and the caller falls back to
    generate_predictions(), which is always current. The table stores summed
    points per (player, gameweek); name/position/club come from bootstrap and
    hist_games (60+ minute appearances, which create_watchlist aggregates) is
    recounted from gw_data, so consumers see the same columns either way.
    """
    if _supabase is None:
        return None
    try:
        rows: list[dict] = []
        page = 1000
        offset = 0
        while True:
            resp = (
                _supabase.table("player_predictions")
                .select("player_id,event,predicted_points,generated_at")
                .range(offset, offset + page - 1)
                .execute()
            )
            batch = resp.data or []
            rows.extend(batch)
            if len(batch) < page:
                break
            offset += page
        if not rows:
            logger.info("player_predictions table is empty — regenerating in-process")
            return None

        newest = max(r["generated_at"] for r in rows if r.get("generated_at"))
        newest_dt = datetime.fromisoformat(newest.replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - newest_dt).total_seconds() / 3600
        if age_hours > PREDICTIONS_TABLE_MAX_AGE_HOURS:
            logger.warning(
                "player_predictions is %.1fh old (max %dh) — regenerating in-process",
                age_hours, PREDICTIONS_TABLE_MAX_AGE_HOURS,
            )
            return None

        df = pd.DataFrame(rows).rename(columns={"player_id": "element"})
        df = df[["element", "event", "predicted_points"]]
        df["element"] = df["element"].astype(int)
        df["event"] = df["event"].astype(int)

        elements = bootstrap.get("elements", [])
        name_map = {int(e["id"]): f"{e.get('first_name', '')} {e.get('second_name', '')}".strip() for e in elements}
        pos_map = {int(e["id"]): POSITION_MAP.get(e.get("element_type", 0), "UNKNOWN") for e in elements}
        club_map = {int(e["id"]): e.get("team") for e in elements}
        df["name"] = df["element"].map(name_map)
        df["position"] = df["element"].map(pos_map)
        df["player_team_id"] = df["element"].map(club_map)

        if "minutes" in gw_data.columns:
            counts = gw_data[gw_data["minutes"] >= 60].groupby("element").size()
            df["hist_games"] = df["element"].map(counts).fillna(0).astype(int)
        else:
            df["hist_games"] = 0

        logger.info(
            "Predictions from Supabase: %d rows, %d players, %.1fh old",
            len(df), df["element"].nunique(), age_hours,
        )
        return df
    except Exception:
        logger.exception("player_predictions read failed — regenerating in-process")
        return None


def _non_free_hit_squad_gw(last_gw: int, free_hit_gws: List[int], first_gw: int = 1) -> int:
    """
    The gameweek whose picks are the squad the manager actually owns.

    A Free Hit squad exists for exactly one gameweek and then reverts, so picks on a
    Free Hit GW are a one-week loan, not a squad. Step back one gameweek when that is
    what `last_gw` landed on — but never past `first_gw`, the manager's earliest
    gameweek, where there are no picks to read at all. In that corner the Free Hit
    squad is the only thing on record and is returned as-is.
    """
    if last_gw in free_hit_gws and last_gw - 1 >= first_gw:
        return last_gw - 1
    return last_gw


def _chip_halves_from(
    detected_chips: Dict[str, Any], key: str, total_override: Optional[int]
) -> Tuple[int, int]:
    """
    One chip's spent halves as (first_half, second_half), each 0 or 1.

    Two sources, unioned, because neither alone is trustworthy in both directions:

    - Detection knows WHICH half a use belongs to (the history payload carries the
      gameweek — see api.detect_chips_used), and is authoritative about the past.
    - An override is only a total, so it cannot say which half it means. A caller in
      the second half encodes "still available" as 1, not 0, since an unplayed
      first-half chip expires rather than carrying over. Splitting that total as
      min/max is therefore right for the caller's intent and blind to a chip actually
      played after GW20.

    Unioning them means a half detection has *proved* spent can never be talked back
    into being available, while a caller can still say a chip is gone that FPL has not
    recorded yet — the "plan as if my Free Hit were already gone" case. The reverse,
    asking to plan with a chip the record shows was played, is not a use case: it can
    only produce a plan that cannot be executed.

    Worked example, the bug this fixes. Manager let the first-half Bench Boost expire
    and played it in GW22; a solve at GW25 sees a detected total of 1. The old split
    read that as "first half gone, second half free" and the winning plan scheduled a
    second Bench Boost. Now detection reports second_half=1 and no override undoes it.
    """
    detected_first = min(1, detected_chips.get(f"{key}_first_half", 0))
    detected_second = min(1, detected_chips.get(f"{key}_second_half", 0))
    if total_override is None:
        return detected_first, detected_second

    override_first = min(total_override, 1)
    override_second = max(0, total_override - 1)
    return max(override_first, detected_first), max(override_second, detected_second)


def _wildcard_halves(use_chips: bool, halves: Tuple[int, int]) -> Tuple[int, int]:
    """
    Which wildcard halves to treat as already spent, as (first_half, second_half).

    `use_chips: false` means "no chips", and that has to include the wildcard. The other
    three are enumerated into scenarios, so collapsing to the single "No chips" scenario
    rules them out on its own — but the wildcard is a decision *variable* inside the MILP,
    the same in every scenario, so nothing about scenario selection touches it. The only
    thing that takes it off the table is chip state saying both halves are gone, which is
    what this returns.

    Left unsaid, a "no chips" plan would come back playing a wildcard — and looking
    entirely reasonable while doing it, since a wildcard week shows no points hit.

    Wins over an explicit `wildcards_used` for the same reason: `use_chips: false` is the
    broader instruction, and honouring the narrower one would mean silently planning with
    a chip the caller ruled out.
    """
    if not use_chips:
        return 1, 1
    return halves


def _resolve_money(
    req: "OptimizeRequest",
    current_squad: List[int],
    bootstrap: dict,
    squad_gw: int,
) -> Tuple[int, int, Dict[int, int]]:
    """
    Work out the three money inputs the pipeline needs, all in tenths of a million.

    Returns:
        (total_budget, bank, selling_discounts) — the squad's sale value plus cash, the
        cash on its own, and per-player {market price - sale price} for the players the
        manager already owns.

    The caller's own figures win where given: they come from a squad the user confirmed,
    which outranks anything the FPL API reports (it is only ever current as of the last
    deadline, and pre-season it reports nothing). Falling back to the API costs three
    requests, which is why it is skipped whenever the caller has supplied the answer.
    """
    market = {int(e["id"]): int(e.get("now_cost", 0)) for e in bootstrap.get("elements", [])}

    selling: Dict[int, int] = {}
    fetched: Optional[dict] = None
    if req.selling_prices:
        selling = {e.player: int(round(e.selling_value * 10)) for e in req.selling_prices}
    else:
        try:
            _, fetched = api.get_squad_selling_prices(req.team_id, squad_gw)
            selling = dict(fetched["selling_prices"])
        except Exception as exc:
            if req.total_budget is None:
                # Nothing to fall back on — the pipeline cannot price a squad at all.
                raise
            # A confirmed budget is enough to plan with. Every owned player is then
            # assumed to sell for his market price, which is what this did before sale
            # prices existed: no worse, just less precise for a squad that has risen.
            logger.warning(
                "Selling prices unavailable for team %d (%s); assuming market prices",
                req.team_id, exc,
            )

    def sale_price(pid: int) -> int:
        return selling.get(pid, market.get(pid, 0))

    squad_sale_value = sum(sale_price(pid) for pid in current_squad)

    if req.total_budget is not None:
        # Pipeline works in tenths of a million throughout; the API takes £m.
        total_budget = int(round(req.total_budget * 10))
    else:
        total_budget = int(fetched["correct_budget"])

    if req.bank is not None:
        bank = int(round(req.bank * 10))
    elif req.total_budget is None and fetched is not None:
        bank = int(fetched["bank"])
    else:
        # `total_budget` is squad sale value + bank by definition, so this inverts it.
        bank = max(0, total_budget - squad_sale_value)

    discounts = {
        pid: market[pid] - sale_price(pid)
        for pid in current_squad
        if pid in market and market[pid] > sale_price(pid)
    }

    logger.info(
        "Money: budget %.1fM, bank %.1fM, %d players below market (%.1fM total)",
        total_budget / 10, bank / 10, len(discounts), sum(discounts.values()) / 10,
    )
    return total_budget, bank, discounts


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

        bank_units = solution.get("bank", {}).get(t)

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

        fh_squad = None
        if chip == "free_hit" and fh_benefits and gw in fh_benefits:
            fh = fh_benefits[gw]
            gw_pts = fh.get("total_points", 0)
            if fh.get("squad_details"):
                fh_squad = fh["squad_details"]

        if fh_squad is not None:
            # The main MILP doesn't model Free Hit weeks: add_lineup_constraints()
            # skips lineup size / captaincy for them and every prediction is zeroed,
            # so solution["lineups"][t] is an empty starting XI, a 15-man bench and
            # no captain. The real FH squad comes from the sub-MILP instead.
            # squad_details is keyed by position, which also gives the bench a
            # stable GK->DEF->MID->FWD order (the sub-solver has no bench ordering).
            fh_players = [
                p
                for pos in ["GK", "DEF", "MID", "FWD"]
                for p in fh_squad.get(pos, [])
            ]
            fh_captain = next(
                (p["element"] for p in fh_players if p.get("is_captain")), None
            )
            captain_id = fh_captain if fh_captain is not None else captain_id
            starters = [
                {
                    **_player_info(bootstrap, p["element"]),
                    "expected_points": round(p.get("points", 0), 1),
                    "is_captain": bool(p.get("is_captain")),
                    "is_vice_captain": False,
                }
                for p in fh_players
                if p.get("is_starter")
            ]
            bench = [
                {
                    **_player_info(bootstrap, p["element"]),
                    "expected_points": round(p.get("points", 0), 1),
                    "bench_order": idx + 1,
                }
                for idx, p in enumerate(
                    [p for p in fh_players if not p.get("is_starter")]
                )
            ]
        else:
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
            # Cash left after this gameweek's moves, in £m. A planning figure, not a
            # forecast: the pipeline holds prices still for the whole horizon, so this is
            # what the manager would have left if nobody's price moved between now and
            # then. On a Free Hit gameweek no money changes hands, so it carries across
            # from the previous gameweek unchanged.
            "bank": round(bank_units / 10, 1) if bank_units is not None else None,
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

        chips_used = api.detect_chips_used(team_id)

        # A Free Hit squad exists for exactly one gameweek and then reverts, so the
        # picks on a Free Hit GW are not the squad this manager owns — mirroring the
        # same step-back in _optimize_inner and run.py.
        #
        # Without this, for the whole week between a Free Hit deadline and the next
        # one, /api/squad hands back 15 players the manager will not have. The
        # frontend then asks them to confirm that squad on the verification card, and
        # a confirmed card is sent to the optimizer as the authoritative `squad`
        # override — so the entire plan gets built on the wrong team. Selling prices
        # are wrong too: _build_purchase_prices skips Free Hit week transfers, so
        # those players fall back to season-start prices.
        first_gw = max(1, int(entry.get("started_event") or 1))
        squad_gw = _non_free_hit_squad_gw(
            last_gw, chips_used.get("free_hit_gws", []), first_gw
        )
        if squad_gw == last_gw and last_gw in chips_used.get("free_hit_gws", []):
            logger.warning(
                "Team %d played a Free Hit in GW%d, its earliest gameweek — returning "
                "the Free Hit squad, which will revert",
                team_id, last_gw,
            )

        team_data = api.fetch_team_data(team_id, squad_gw)
        selling_info, selling_summary = api.get_squad_selling_prices(team_id, squad_gw)

        squad_details = []
        for pid in team_data["squad"]:
            info = _player_info(bootstrap, pid)
            sell = next((s for s in selling_info if s["element"] == pid), None)
            info["selling_value"] = sell["selling_value"] / 10 if sell else info["price"]
            squad_details.append(info)

        return {
            "team_id": team_id,
            "current_gw": current_gw,
            # Which gameweek's picks this squad actually came from. Normally the last
            # gameweek with public picks; one earlier when that was a Free Hit.
            "squad_gw": squad_gw,
            "squad": squad_details,
            "bank": selling_summary["bank"] / 10,
            "total_budget": selling_summary["correct_budget"] / 10,
            "chips_used": {
                "wildcards_used": chips_used.get("wildcards_used", 0),
                "free_hits_used": chips_used.get("free_hits_used", 0),
                "bench_boost_used": chips_used.get("bench_boost_used", 0),
                "triple_captain_used": chips_used.get("triple_captain_used", 0),
            },
            # The same four chips, split by the half each use was actually played in.
            # A total cannot answer "has this manager spent the chip for the half being
            # planned?": one use in GW22 is a second-half use, and a consumer deriving
            # the half from the count alone has to assume the first use was GW1-19, so
            # it reads that manager as still holding a Bench Boost. Additive to the
            # totals above rather than replacing them, so existing callers are
            # unaffected.
            "chips_used_by_half": {
                key: {
                    "first_half": min(1, chips_used.get(f"{key}_first_half", 0)),
                    "second_half": min(1, chips_used.get(f"{key}_second_half", 0)),
                }
                for key in (
                    "wildcards_used",
                    "free_hits_used",
                    "bench_boost_used",
                    "triple_captain_used",
                )
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


# Floor on the gap between two progress writes. A 25-scenario run finishing in a burst
# would otherwise fire 25 updates in a few seconds, and the frontend polls every 5s —
# so it could not see them anyway. Phase changes ignore this and always write: they are
# rare, and they are the part of the bar that tells the user what is happening.
_PROGRESS_MIN_INTERVAL_S = 2.0


def _progress_writer(job_id: str):
    """
    Build the callback that turns solve progress into a `solver_jobs.progress` row.

    Deliberately fire-and-forget and swallowing: a solve is minutes long and a failed
    progress write is not a reason to lose it. The frontend treats a missing or stale
    progress value as "no signal", not as an error, and falls back to easing the bar on
    elapsed time — so the worst case of this never working is the old spinner.
    """
    state = {"at": 0.0, "phase": ""}

    def report(phase: str, done: int | None = None, total: int | None = None) -> None:
        if not _supabase:
            return
        now = time.monotonic()
        if phase == state["phase"] and now - state["at"] < _PROGRESS_MIN_INTERVAL_S:
            return
        state["at"] = now
        state["phase"] = phase
        payload: dict[str, Any] = {"phase": phase}
        if done is not None and total is not None:
            payload.update({"done": done, "total": total})
        try:
            # "now()" is Postgres's own clock, same as every other write here — which is
            # also what the frontend's staleness check compares against.
            _supabase.table("solver_jobs").update(
                {"progress": payload, "updated_at": "now()"}
            ).eq("id", job_id).execute()
        except Exception:
            logger.debug("Progress write failed for job %s", job_id, exc_info=True)

    return report


def _run_and_store(req: OptimizeRequest, job_id: str) -> None:
    """Background task: run optimize pipeline and persist result to Supabase."""
    _update_job(job_id, "running")
    try:
        # Reuse the synchronous optimize logic by calling it inline
        import asyncio
        result = asyncio.run(_optimize_inner(req, on_progress=_progress_writer(job_id)))
        _update_job(job_id, "complete", result=result)
        logger.info("Async job %s complete — %.1f pts", job_id, result.get("objective", 0))
    except Exception as exc:
        logger.exception("Async job %s failed", job_id)
        _update_job(job_id, "failed", error=str(exc))


# What a run reports as it goes. Only two of the four can count anything — the others
# are setup and teardown with no natural unit of work — which is why the frontend pairs
# these checkpoints with time-based easing rather than driving the bar from them alone.
ProgressFn = Callable[..., None]


def _noop_progress(phase: str, done: int | None = None, total: int | None = None) -> None:
    """Progress sink for the synchronous endpoint, which has nobody to report to."""


async def _optimize_inner(req: OptimizeRequest, on_progress: ProgressFn = _noop_progress) -> dict:
    """Pure optimization logic shared by /api/optimize and /api/optimize-async."""
    start_time = time.time()
    on_progress("preparing")

    bootstrap = api.fetch_bootstrap_data()
    detected_gw = api.detect_current_gw(bootstrap)
    season = api.detect_current_season(bootstrap)
    detected_chips = api.detect_chips_used(req.team_id)

    current_gw = detected_gw
    horizon = min(req.planning_horizon, 38 - current_gw + 1)

    squad_gw = _non_free_hit_squad_gw(current_gw - 1, detected_chips.get("free_hit_gws", []))

    # A caller-supplied squad is one the user has confirmed, so it outranks whatever the
    # FPL API reports — which is only ever as of the last deadline, and is empty
    # pre-season. Skipping the fetches also saves two API round trips.
    if req.squad:
        current_squad = list(req.squad)
        logger.info("Using caller-supplied squad for team %d", req.team_id)
    else:
        team_data = api.fetch_team_data(req.team_id, squad_gw)
        current_squad = team_data["squad"]

    total_budget, bank, selling_discounts = _resolve_money(
        req, current_squad, bootstrap, squad_gw
    )

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
        predictions = _predictions_from_supabase(bootstrap, gw_data)
        if predictions is None:
            predictions = generate_predictions(gw_data, fixtures, multipliers, current_season_tiers, season)

    # Every owned player must exist in the pool or the model is infeasible rather
    # than merely suboptimal — see ensure_players_present. Applied after
    # generate_predictions so the synthesized rows never reach the averages.
    gw_data, unknown_squad_players = proxy_predict.ensure_players_present(
        gw_data, current_squad, bootstrap
    )
    if unknown_squad_players:
        raise ValueError(
            "FPL has no player data for squad member(s) "
            f"{unknown_squad_players} — cannot build a plan around them"
        )

    # extra_players bypasses the min_hist_pct filter — new signings, players just
    # back from injury, anyone with too little recent game time to qualify on merit.
    # forced_lineup players are promoted too, mirroring run.py: a forced start is only
    # enforceable if the player is in the candidate pool at all, so without this the
    # constraint is silently dropped for anyone who didn't clear min_hist_pct — the
    # caller's instruction disappears with no error and a plausible plan comes back.
    must_include = list(dict.fromkeys(
        list(current_squad)
        + list(req.extra_players)
        + [e.player for e in req.forced_lineup]
    ))
    # Synthesized gw_data has nobody with real appearances — the filter would
    # exclude everyone, so it's disabled while proxy predictions are in use.
    min_hist_pct = 0.0 if use_proxy else 0.6
    watchlist = create_watchlist(
        predictions, gw_data,
        min_hist_pct=min_hist_pct, max_hist_window=6,
        must_include=must_include, must_exclude=req.excluded_players,
    )

    if req.bucket_top_n:
        watchlist = apply_price_bucket_filter(
            watchlist, predictions, bootstrap,
            current_gw=current_gw, horizon=req.bucket_horizon,
            top_n=req.bucket_top_n, must_include=must_include,
        )

    non_playing_tuples = [(e.player, list(e.gameweeks)) for e in req.non_playing if e.gameweeks] or None

    # Explicit request values win over API detection; None means "use what we detected".
    def _chip_halves(total_override: Optional[int], key: str) -> Tuple[int, int]:
        return _chip_halves_from(detected_chips, key, total_override)

    wildcard_first_half, wildcard_second_half = _wildcard_halves(
        req.use_chips, _chip_halves(req.wildcards_used, "wildcards_used")
    )
    free_hits_used_first_half, free_hits_used_second_half = _chip_halves(
        req.free_hits_used, "free_hits_used"
    )
    bench_boost_used_first_half, bench_boost_used_second_half = _chip_halves(
        req.bench_boost_used, "bench_boost_used"
    )
    triple_captain_used_first_half, triple_captain_used_second_half = _chip_halves(
        req.triple_captain_used, "triple_captain_used"
    )

    # Neither chip changes anything about the plan except which GW it lands
    # on relative to its own already-solved lineup (Triple Captain exactly;
    # Bench Boost approximately — see find_best_bench_boost_gw). So solve
    # FH-only with both off, then score+place both post-hoc instead of
    # multiplying either into the scenario count. Skipped per-chip when that
    # chip is forced or fully spent.
    defer_bench_boost = (
        req.use_chips
        and req.force_bench_boost_gw is None
        and (bench_boost_used_first_half < 1 or bench_boost_used_second_half < 1)
    )
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
            free_hits_used_first_half=free_hits_used_first_half,
            free_hits_used_second_half=free_hits_used_second_half,
            bench_boost_used_first_half=1 if defer_bench_boost else bench_boost_used_first_half,
            bench_boost_used_second_half=1 if defer_bench_boost else bench_boost_used_second_half,
            triple_captain_used_first_half=1 if defer_triple_captain else triple_captain_used_first_half,
            triple_captain_used_second_half=1 if defer_triple_captain else triple_captain_used_second_half,
            force_free_hit_gw=req.force_free_hit_gw,
            force_bench_boost_gw=req.force_bench_boost_gw,
            force_triple_captain_gw=req.force_triple_captain_gw,
            force_wildcard_gw=req.force_wildcard_gw,
        )

    if len(chip_scenarios) > req.max_scenarios:
        chip_scenarios = chip_scenarios[:req.max_scenarios]

    forced_lineup_tuples = [(e.player, list(e.gameweeks)) for e in req.forced_lineup if e.gameweeks] or None
    fh_benefits: dict = {}
    if any(s["free_hit_gws"] for s in chip_scenarios):
        fh_benefits = calculate_free_hit_benefits_for_horizon(
            start_gw=current_gw, planning_horizon=horizon, budget=total_budget,
            predictions_df=predictions, gw_data_df=gw_data,
            watchlist_players=watchlist,
            forced_lineup_players=forced_lineup_tuples,
            non_playing_players=non_playing_tuples,
        )

    # Prune FH placements to the top fh_top_k by precomputed benefit. A
    # scenario's final score is base_points + fh_benefits[gw] and the second
    # term is already exact here, so the cheap rank only has to identify the
    # cluster of good weeks — the full solves below still pick the winner
    # among them. Skipped when a week is forced (the caller has already
    # decided) and for scenarios without a Free Hit (always kept).
    if req.fh_top_k is not None and fh_benefits and req.force_free_hit_gw is None:
        fh_scens = [s for s in chip_scenarios if s["free_hit_gws"]]
        if len(fh_scens) > req.fh_top_k:
            def _fh_score(s: dict) -> float:
                return sum(fh_benefits.get(g, {}).get("total_points", 0.0) for g in s["free_hit_gws"])
            kept_ids = {id(s) for s in sorted(fh_scens, key=_fh_score, reverse=True)[:req.fh_top_k]}
            dropped = [s["name"] for s in fh_scens if id(s) not in kept_ids]
            chip_scenarios = [
                s for s in chip_scenarios if not s["free_hit_gws"] or id(s) in kept_ids
            ]
            logger.info(
                "FH pruning: kept top %d of %d FH placements by precomputed benefit "
                "(%d scenarios remain); dropped: %s",
                req.fh_top_k, len(fh_scens), len(chip_scenarios), dropped,
            )

    best_result = None
    best_total = -float("inf")
    scenario_results = []

    # Everything a worker process needs to rebuild a solver on its own side.
    solver_ctx = {
        "horizon": horizon,
        "budget": total_budget,
        "start_gw": current_gw,
        "points_multiplier": None,
        "forced_lineup": forced_lineup_tuples,
        "non_playing": non_playing_tuples,
        "first_gw_penalty": -1,
        "sub_probability": 0.10,
        "predictions": predictions,
        "gw_data": gw_data,
        "watchlist": watchlist,
        "current_squad": current_squad,
        "free_transfers": req.free_transfers,
        "wildcard_first_half": wildcard_first_half,
        "wildcard_second_half": wildcard_second_half,
        "time_limit": req.time_limit_per_scenario,
        "mip_gap": None,
        "bank": bank,
        "selling_discounts": selling_discounts,
        # How the solver tells a real blank gameweek from a player it simply has no
        # forecast for. Without these it falls back to "no forecast means no fixture",
        # which two gameweeks into a season is wrong for most of the game.
        "player_clubs": api.player_club_map(bootstrap),
        "club_gameweeks": api.club_gameweek_map(fixtures),
    }

    # Announced before the first solve returns, not after: everything above this point
    # (bootstrap, predictions, watchlist, free-hit benefits) is a silent minute or so,
    # and the phase change is what tells the user it ended.
    on_progress("scenarios", 0, len(chip_scenarios))
    raw_results = solve_scenarios(
        chip_scenarios, solver_ctx,
        on_progress=lambda done, total: on_progress("scenarios", done, total),
    )

    for raw in raw_results:
        if raw["status"] != "solved":
            continue
        scenario = raw["scenario"]
        fh_total = sum(fh_benefits.get(gw, {}).get("total_points", 0) for gw in scenario["free_hit_gws"])
        total_points = raw["base_points"] + fh_total

        # The built PuLP model (every decision variable + constraint) used to stay
        # resident here once per scenario and OOM'd long runs — see 0437e12, which
        # fixed that by dropping .prob after extract_solution(). That patch is
        # unnecessary now: solving happens in worker processes that return plain
        # data and then exit, so no MILP model ever reaches this process.
        result = {"scenario_name": scenario["name"], "free_hit_gws": scenario["free_hit_gws"],
                  "bench_boost_gw": scenario["bench_boost_gw"], "solution": raw["solution"],
                  "squad_points": raw["squad_points"], "scenario": scenario,
                  "total_points": total_points}
        scenario_results.append(result)

        if total_points > best_total:
            best_total = total_points
            best_result = result

    # ── EXPERIMENTAL shadow evaluation of Free Hit pruning — logging only. ──
    # Question under test: could the 16 "FH in GW g" scenarios be pruned to a
    # top-K before solving, the way BB/TC placements already are? A scenario's
    # total is base_points + fh_benefits[g] and the second term is exact and
    # precomputed, so a cheap rank only has to predict the base term's variation.
    # Two candidate rankings, logged against the truth the full enumeration just
    # produced: the FH week's own precomputed points ("benefit"), and that minus
    # what the no-chip plan scored the same week ("uplift" — the opportunity cost
    # proxy). If the true winner consistently ranks top-3 in either, pruning to
    # K=6 is safe; if it ever ranks below K, this idea dies here.
    try:
        _fh_rows = [r for r in scenario_results if len(r["free_hit_gws"]) == 1]
        _nochip = next((r for r in scenario_results if not r["free_hit_gws"]), None)
        if len(_fh_rows) >= 4 and _nochip is not None and fh_benefits:
            _nochip_week_pts: Dict[int, float] = {}
            for _t, _lineup in _nochip["solution"]["lineups"].items():
                _gw = current_gw + _t - 1
                _pts = sum(
                    _nochip["squad_points"].get((_p, _gw), 0.0)
                    for _p in _lineup.get("starters", [])
                )
                _cap = _nochip["solution"]["captains"].get(_t)
                if _cap is not None:
                    _pts += _nochip["squad_points"].get((_cap, _gw), 0.0)
                _nochip_week_pts[_gw] = _pts
            _shadow = []
            for _r in _fh_rows:
                _g = _r["free_hit_gws"][0]
                _benefit = fh_benefits.get(_g, {}).get("total_points", 0.0)
                _shadow.append({
                    "gw": _g,
                    "true": round(_r["total_points"], 1),
                    "benefit": round(_benefit, 1),
                    "uplift": round(_benefit - _nochip_week_pts.get(_g, 0.0), 1),
                })
            _winner_gw = max(_shadow, key=lambda x: x["true"])["gw"]
            def _winner_rank(key: str) -> int:
                _order = sorted(_shadow, key=lambda x: -x[key])
                return next(i + 1 for i, x in enumerate(_order) if x["gw"] == _winner_gw)
            logger.info(
                "FH-prune shadow: true winner GW%d ranks %d/%d by benefit-only, "
                "%d/%d by uplift. rows=%s",
                _winner_gw, _winner_rank("benefit"), len(_shadow),
                _winner_rank("uplift"), len(_shadow),
                sorted(_shadow, key=lambda x: -x["true"]),
            )
    except Exception:
        logger.exception("FH-prune shadow evaluation failed (harmless)")

    # Score every solved FH-only plan's best legal Bench Boost GW and/or Triple
    # Captain GW (whichever is deferred — see the comment above defer_bench_boost),
    # then re-solve only the top-scored (FH, BB, TC) combinations as full MILPs.
    # BB is picked first and excluded from TC's candidates, not jointly optimized
    # with it: cheap to score either way, but scoring every (BB gw, TC gw) pair
    # together buys little given both get a full re-solve below anyway.
    if (defer_bench_boost or defer_triple_captain) and scenario_results:
        bb_gws = bench_boost_candidate_gws(
            start_gw=current_gw, planning_horizon=horizon,
            used_first_half=bench_boost_used_first_half,
            used_second_half=bench_boost_used_second_half,
        ) if defer_bench_boost else []
        tc_gws = triple_captain_candidate_gws(
            start_gw=current_gw, planning_horizon=horizon,
            used_first_half=triple_captain_used_first_half,
            used_second_half=triple_captain_used_second_half,
        ) if defer_triple_captain else []

        ranked = []
        for result in scenario_results:
            wildcard_gws = [
                current_gw + t - 1
                for t, tr in result["solution"]["transfers"].items()
                if tr.get("wildcard_active")
            ]

            if defer_bench_boost:
                bb_gw, bb_bonus = find_best_bench_boost_gw(
                    lineups_by_t=result["solution"]["lineups"],
                    expected_points=result["squad_points"],
                    wildcard_gws=wildcard_gws,
                    start_gw=current_gw, horizon=horizon,
                    free_hit_gws=result["free_hit_gws"],
                    triple_captain_gw=-1,
                    candidate_gws=bb_gws,
                    sub_probability=0.10,
                )
            else:
                # Already baked into this scenario's own full solve (forced, or
                # not available to defer in the first place) — nothing to add.
                bb_gw = result["bench_boost_gw"] if result["bench_boost_gw"] != -1 else None
                bb_bonus = 0.0

            if defer_triple_captain:
                tc_gw, tc_bonus = find_best_triple_captain_gw(
                    captains_by_t=result["solution"]["captains"],
                    expected_points=result["squad_points"],
                    wildcard_gws=wildcard_gws,
                    start_gw=current_gw, horizon=horizon,
                    free_hit_gws=result["free_hit_gws"],
                    bench_boost_gw=bb_gw if bb_gw is not None else -1,
                    candidate_gws=tc_gws,
                    sub_probability=0.10,
                )
            else:
                tc_gw = result["scenario"]["triple_captain_gw"]
                tc_gw = tc_gw if tc_gw != -1 else None
                tc_bonus = 0.0

            if bb_gw is None and tc_gw is None:
                continue
            ranked.append((result, bb_gw, tc_gw, bb_bonus + tc_bonus))

        # Dedup identical (FH, BB, TC) allocations — different FH-only scenarios
        # can independently land on the same BB/TC placement, and re-solving the
        # same combination twice buys nothing.
        seen = set()
        deduped = []
        for item in ranked:
            result, bb_gw, tc_gw, _ = item
            key = (tuple(result["free_hit_gws"]), bb_gw, tc_gw)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        deduped.sort(key=lambda item: item[0]["total_points"] + item[3], reverse=True)

        final_scenarios = []
        for result, bb_gw, tc_gw, _ in deduped[:CHIP_RESELECT_TOP_K]:
            chip_parts = [f"BB GW{bb_gw}"] if bb_gw is not None else []
            chip_parts += [f"TC GW{tc_gw}"] if tc_gw is not None else []
            name = " | ".join([result["scenario_name"], *chip_parts]) if chip_parts else result["scenario_name"]
            final_scenarios.append({
                "name": name,
                "free_hit_gws": result["free_hit_gws"],
                "bench_boost_gw": bb_gw if bb_gw is not None else -1,
                "triple_captain_gw": tc_gw if tc_gw is not None else -1,
                "force_wildcard_gw": req.force_wildcard_gw,
            })

        on_progress("chip_resolve", 0, len(final_scenarios))
        for raw in solve_scenarios(
            final_scenarios, solver_ctx,
            on_progress=lambda done, total: on_progress("chip_resolve", done, total),
        ):
            if raw["status"] != "solved":
                continue
            scenario = raw["scenario"]
            fh_total = sum(fh_benefits.get(gw, {}).get("total_points", 0) for gw in scenario["free_hit_gws"])
            total_points = raw["base_points"] + fh_total

            if total_points > best_total:
                best_total = total_points
                best_result = {"scenario_name": scenario["name"], "free_hit_gws": scenario["free_hit_gws"],
                               "bench_boost_gw": scenario["bench_boost_gw"], "solution": raw["solution"],
                               "squad_points": raw["squad_points"], "scenario": scenario,
                               "total_points": total_points}

    if not best_result:
        raise ValueError("No feasible solution found")

    on_progress("finalising")

    # Workers return plain data, not solver objects; rebuild the winner's solver
    # (~0.2s, no solve) for the fields _format_solution reads.
    best_solver = build_solver(best_result["scenario"], solver_ctx)

    result = _format_solution(
        best_result["solution"], best_solver,
        best_solver.players, current_gw, best_total,
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

    # Drop predictions this run did not rewrite. Upserting alone never removes
    # anything, so every prediction the model can no longer produce survives
    # forever, timestamped but otherwise indistinguishable from a fresh one —
    # and Alfie reads this table with no idea which is which. Observed
    # 2026-08-23: the first real run wrote 4,662 rows for the 126 players with a
    # 60+ minute appearance and left 17,260 pre-season proxy rows in place,
    # leaving the table a silent mix of two different models.
    #
    # A player with no prediction is the honest state and self-heals as fixtures
    # are played. Only runs when rows were actually written, so a bad run cannot
    # empty the table.
    stale_deleted = 0
    if written:
        try:
            deleted = (
                _supabase.table("player_predictions")
                .delete()
                .lt("generated_at", generated_at)
                .execute()
            )
            stale_deleted = len(deleted.data or [])
        except Exception:
            logger.exception("generate-predictions cron: stale row purge failed")

    elapsed = round(time.time() - start_time, 1)
    logger.info(
        "generate-predictions cron: wrote %d rows, purged %d stale, in %.1fs",
        written, stale_deleted, elapsed,
    )
    return {
        "ok": True,
        "rows_written": written,
        "stale_rows_deleted": stale_deleted,
        "elapsed_seconds": elapsed,
    }
