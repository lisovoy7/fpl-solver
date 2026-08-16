"""
Parallel chip-scenario solving.

Each chip scenario is an independent MILP, so the scenario loop is embarrassingly
parallel. Model *building* is cheap (~0.2s); CBC's branch-and-bound is effectively
all of the runtime, so the only way to make a run meaningfully faster is to run
several solves at once.

Shared by run.py (CLI) and api_server.py (HTTP) so the two cannot drift.

## Why processes, and why threads=1 inside them

CBC can use multiple threads per solve, but spending the cores on *scenarios*
beats spending them on one scenario's search tree, and the two compete for the
same CPUs. So: N worker processes, one CBC thread each. Setting both oversubscribes
and makes everything slower.

## Time limits interact with parallelism

`time_limit` is wall-clock inside CBC. Running W solves on W cores gives each one
roughly a full core, so quality per scenario holds — but the limit should not be
left at a value tuned for sequential runs, because the wall-clock budget freed by
parallelism is better spent on a longer per-scenario limit. Measured on a 10-core
machine, 20 scenarios at horizon 6: sequential at 15s took 189s; 8 workers at 60s
took 41s and produced solutions that were better on 2 scenarios and worse on none.
"""

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, Dict, List, Optional

from fpl.solver import FPLSolver

logger = logging.getLogger(__name__)

# Populated once per worker process by _init_worker, so the prediction/price
# frames are shipped across the process boundary once instead of per scenario.
_CTX: Dict = {}


def default_worker_count() -> int:
    """
    Worker processes to use by default.

    FPL_SOLVER_WORKERS wins when set. This matters on Cloud Run, where the
    container sees the host's core count rather than the `--cpu` limit, so
    autodetection would badly oversubscribe a 2-vCPU service.
    """
    env = os.environ.get("FPL_SOLVER_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            logger.warning("Ignoring non-integer FPL_SOLVER_WORKERS=%r", env)

    try:
        # Respects cpuset/affinity on Linux; absent on macOS.
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    # Leave a core for the parent process and the OS.
    return max(1, n - 1)


def build_solver(scenario: Dict, ctx: Dict) -> Optional[FPLSolver]:
    """
    Construct and populate an FPLSolver for one scenario, without solving.

    Also used by the parent process to rebuild the winning scenario's solver for
    display, which is why it is separate from the solve step.

    Returns None when the scenario has no predictions in range.
    """
    solver = FPLSolver(
        planning_horizon=ctx["horizon"],
        budget=ctx["budget"],
        start_gw=ctx["start_gw"],
        points_multiplier_override=ctx["points_multiplier"],
        forced_lineup_players=ctx["forced_lineup"],
        non_playing_players=ctx["non_playing"],
        first_gw_transfer_penalty=ctx["first_gw_penalty"],
        sub_probability=ctx["sub_probability"],
        bench_boost_gw=scenario["bench_boost_gw"],
        triple_captain_gw=scenario["triple_captain_gw"],
        free_hit_gws=scenario["free_hit_gws"],
        force_wildcard_gw=scenario.get("force_wildcard_gw"),
    )
    solver.load_predictions(ctx["predictions"])
    if len(solver.predictions) == 0:
        return None
    solver.load_player_data(ctx["gw_data"], ctx["predictions"], player_subset=ctx["watchlist"])
    solver.set_initial_squad(ctx["current_squad"], available_transfers=ctx["free_transfers"])
    solver.set_chip_state(
        wildcard_first_half=ctx["wildcard_first_half"],
        wildcard_second_half=ctx["wildcard_second_half"],
    )
    return solver


def _solve_scenario(scenario: Dict, ctx: Dict) -> Dict:
    """
    Solve one scenario and return a picklable result.

    Deliberately returns plain data rather than the FPLSolver: the solved PuLP
    model holds thousands of LpVariable objects and is expensive and fragile to
    send back across a process boundary. `squad_points` is included because the
    post-hoc Bench Boost and Triple Captain scoring in the parent needs it.
    """
    solver = build_solver(scenario, ctx)
    if solver is None:
        return {"scenario": scenario, "status": "no_predictions"}

    solver.build_model()
    if not solver.solve(time_limit=ctx["time_limit"], mip_gap=ctx.get("mip_gap")):
        return {"scenario": scenario, "status": "infeasible"}

    solution = solver.extract_solution()
    return {
        "scenario": scenario,
        "status": "solved",
        "solution": solution,
        "base_points": solution["objective_value"],
        "squad_points": _squad_points(solver, solution),
        "proven_optimal": solver.proven_optimal,
    }


def _squad_points(solver: FPLSolver, solution: Dict) -> Dict:
    """
    The only slice of `expected_points` that survives this scenario.

    find_best_triple_captain_gw and find_best_bench_boost_gw read entries for
    whichever players are actually in the squad (starters + bench) at each GW
    — so returning the full dict would ship (and then retain, for every
    scenario, until the request ends) roughly `players x GWs` entries to serve
    at most `15 x horizon` lookups. At 581 players over a 10-GW horizon that is
    ~5.8k entries per scenario against ~150 that are read; with 100+ scenarios
    resident the difference is the kind of thing that OOMs a 2Gi container
    (see 0437e12, which this narrowing supersedes — it used to be captain-only,
    widened to the full squad once Bench Boost started needing bench players'
    points too). Keyed `(player, gw)` to match what both functions expect.
    """
    points: Dict = {}
    for t, lineup in solution["lineups"].items():
        gw = solver.start_gw + t - 1
        for p in lineup.get("starters", []) + lineup.get("bench", []):
            points[(p, gw)] = solver.expected_points.get((p, gw), 0.0)
    return points


def _init_worker(ctx: Dict) -> None:
    """Seed the per-process context and quieten worker logging."""
    _CTX.clear()
    _CTX.update(ctx)
    # Workers have no console of their own; their records would interleave
    # unreadably with the parent's progress lines.
    logging.getLogger("fpl").setLevel(logging.WARNING)


def _worker_entry(scenario: Dict) -> Dict:
    """Top-level (picklable) worker callable. Never raises — failures come back as data."""
    try:
        return _solve_scenario(scenario, _CTX)
    except Exception as exc:  # noqa: BLE001 - one bad scenario must not kill the pool
        logger.exception("Scenario %s failed", scenario.get("name"))
        return {"scenario": scenario, "status": "error", "error": str(exc)}


def _report(on_progress: Optional[Callable[[int, int], None]], done: int, total: int) -> None:
    """
    Hand a completion count to the caller's progress hook, and never let it matter.

    The hook writes over the network (a Supabase update), so it can be slow, fail, or
    throw. None of that is worth losing a solve over — progress is decoration, and a
    run that reports nothing still finishes and still returns a plan.
    """
    if on_progress is None:
        return
    try:
        on_progress(done, total)
    except Exception:  # noqa: BLE001 - progress reporting must never break a solve
        logger.debug("Progress callback failed at %d/%d", done, total, exc_info=True)


def solve_scenarios(
    scenarios: List[Dict],
    ctx: Dict,
    workers: Optional[int] = None,
    progress: bool = True,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict]:
    """
    Solve every scenario, in parallel when it is worth it.

    Args:
        scenarios: Chip scenario dicts from generate_chip_scenarios().
        ctx: Everything build_solver() needs, plus 'time_limit' and optional 'mip_gap'.
        workers: Process count. None uses default_worker_count(); 1 runs in-process.
        progress: Log a line per solved scenario.
        on_progress: Called (done, total) each time a scenario finishes, in completion
            order. Exists so a caller can surface a live progress bar; a solve is
            minutes long and otherwise reports nothing until it is over. Exceptions
            from it are swallowed.

    Returns:
        Results in the same order as `scenarios`. Order is preserved so that the
        caller's choice of best scenario stays deterministic across runs —
        completion order under a pool is not reproducible.
    """
    if not scenarios:
        return []

    if workers is None:
        workers = default_worker_count()
    # No point paying spawn cost to parallelise fewer solves than we have workers.
    workers = max(1, min(workers, len(scenarios)))

    total = len(scenarios)
    if workers > 1:
        try:
            results = _solve_pooled(scenarios, ctx, workers, on_progress)
        except (BrokenProcessPool, OSError, RuntimeError) as exc:
            # A pool can fail to start for reasons that have nothing to do with
            # the model: too little memory for N interpreters, a restricted
            # container, or a caller that invoked us at import time (spawn
            # re-imports the parent module, so that self-destructs). None of
            # those should cost the user the whole run.
            logger.warning("Parallel solve unavailable (%s) — falling back to sequential", exc)
        else:
            for idx, res in enumerate(results, 1):
                _log_progress(res, idx, total, progress)
            return results

    logger.info("Solving %d scenarios sequentially (time limit: %ss each)",
                total, ctx["time_limit"])
    results = []
    _CTX.clear()
    _CTX.update(ctx)
    for idx, scenario in enumerate(scenarios, 1):
        res = _worker_entry(scenario)
        _log_progress(res, idx, total, progress)
        _report(on_progress, idx, total)
        results.append(res)
    return results


def _solve_pooled(
    scenarios: List[Dict],
    ctx: Dict,
    workers: int,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict]:
    """Run the scenarios across a process pool. Raises if the pool cannot run."""
    logger.info("Solving %d scenarios across %d worker processes (time limit: %ss each)",
                len(scenarios), workers, ctx["time_limit"])
    # "spawn", not the Linux default "fork": api_server.py calls this from a
    # FastAPI threadpool worker, and forking a multithreaded process can deadlock
    # a child that inherits a lock held by a thread that does not exist in it.
    # Spawn costs ~1s of interpreter startup per worker, which is noise next to
    # a solve.
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(ctx,),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        # submit + as_completed rather than executor.map: map yields in *input* order,
        # so a slow first scenario hides the fact that ten others already finished and
        # there is nothing to report progress from until the run is nearly over.
        #
        # One scenario per submit, no batching: solve times vary by an order of
        # magnitude, so a chunk would leave workers idle behind its slowest member.
        index_of = {executor.submit(_worker_entry, s): i for i, s in enumerate(scenarios)}
        results: List[Optional[Dict]] = [None] * len(scenarios)
        for done, future in enumerate(as_completed(index_of), 1):
            # Placed back at its input index — the caller picks the best scenario by
            # iterating this list, and completion order under a pool is not
            # reproducible, so returning it that way would make the chosen plan
            # non-deterministic across identical runs.
            results[index_of[future]] = future.result()
            _report(on_progress, done, len(scenarios))
        return results  # type: ignore[return-value]


def _log_progress(res: Dict, idx: int, total: int, progress: bool) -> None:
    if not progress:
        return
    name = res["scenario"]["name"]
    if res["status"] == "solved":
        flag = "" if res.get("proven_optimal") else "  (time-limited)"
        logger.info("[%d/%d] %-40s  %.1f pts%s", idx, total, name, res["base_points"], flag)
    else:
        logger.info("[%d/%d] %-40s  %s", idx, total, name, res["status"].upper())
