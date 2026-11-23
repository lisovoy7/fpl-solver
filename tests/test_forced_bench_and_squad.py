"""Tests for per-gameweek `forced_bench` and `forced_squad` constraints in FPLSolver
and Free Hit optimization.

Runnable directly (``python tests/test_forced_bench_and_squad.py``) or under pytest.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.free_hit import calculate_optimal_free_hit_squad  # noqa: E402
from fpl.solver import FPLSolver  # noqa: E402

HORIZON = 3
START_GW = 4

GKS = [101, 102]
DEFS = [103, 104, 105, 106, 107]
MIDS = [108, 109, 110, 111, 112]
FWDS = [113, 114, 115]
SQUAD = GKS + DEFS + MIDS + FWDS

# Unowned candidates
UNOWNED_ENABLER = 201  # DEF
UNOWNED_SWING = 202    # MID: high points in GW4, low points in GW5

POSITIONS = (
    {p: "GK" for p in GKS}
    | {p: "DEF" for p in DEFS + [UNOWNED_ENABLER]}
    | {p: "MID" for p in MIDS + [UNOWNED_SWING]}
    | {p: "FWD" for p in FWDS}
)
POOL = SQUAD + [UNOWNED_ENABLER, UNOWNED_SWING]


def _predictions():
    rows = []
    for gw in range(START_GW, START_GW + HORIZON):
        for p in POOL:
            if p == DEFS[0]:
                # Star owned defender (would always start unless forced onto bench)
                pts = 8.0
            elif p == UNOWNED_ENABLER:
                pts = 1.0
            elif p == UNOWNED_SWING:
                # High in START_GW (should start), low in START_GW+1 (should be benched)
                pts = 9.0 if gw == START_GW else 1.0
            else:
                pts = 4.0
            rows.append({"element": p, "event": gw, "predicted_points": pts})
    return pd.DataFrame(rows)


def _solver(
    banned=None,
    free_hit_gws=None,
    forced_lineup=None,
    forced_bench=None,
    forced_squad=None,
    forced_captain=None,
    forced_vice_captain=None,
):
    solver = FPLSolver(
        planning_horizon=HORIZON,
        budget=1000,
        start_gw=START_GW,
        banned_players=banned,
        free_hit_gws=free_hit_gws,
        forced_lineup_players=forced_lineup,
        forced_bench_players=forced_bench,
        forced_squad_players=forced_squad,
        forced_captain_players=forced_captain,
        forced_vice_captain_players=forced_vice_captain,
    )
    solver.load_predictions(_predictions())
    solver.players = pd.DataFrame(
        {
            "element": POOL,
            "name": [f"Player_{p}" for p in POOL],
            "position": [POSITIONS[p] for p in POOL],
            "value": [50] * len(POOL),
            "team": [(i % 10) + 1 for i in range(len(POOL))],
        }
    )
    solver.set_initial_squad(list(SQUAD), available_transfers=2)
    solver.build_model()
    return solver


def _owned(solution, gw):
    return set(solution["squads"][gw - START_GW + 1])


def _starters(solution, gw):
    return set(solution["lineups"][gw - START_GW + 1]["starters"])


def _bench(solution, gw):
    return set(solution["lineups"][gw - START_GW + 1]["bench"])


def test_forced_bench_owned_star_benched_only_in_specified_gw():
    star = DEFS[0]
    solver = _solver(forced_bench=[(star, [START_GW])])
    assert solver.solve(time_limit=30)
    sol = solver.extract_solution()
    assert star in _owned(sol, START_GW)
    assert star not in _starters(sol, START_GW)
    assert star in _bench(sol, START_GW)
    # In the next GW, not forced to bench -> starts again on merit (8.0 pts)
    assert star in _starters(sol, START_GW + 1)


def test_forced_bench_unowned_enabler_is_bought_and_benched():
    solver = _solver(forced_bench=[(UNOWNED_ENABLER, [START_GW])])
    assert solver.solve(time_limit=30)
    sol = solver.extract_solution()
    assert UNOWNED_ENABLER in _owned(sol, START_GW)
    assert UNOWNED_ENABLER in _bench(sol, START_GW)
    assert UNOWNED_ENABLER not in _starters(sol, START_GW)


def test_forced_squad_lets_solver_decide_start_vs_bench_by_gw():
    solver = _solver(
        forced_squad=[(UNOWNED_SWING, [START_GW, START_GW + 1])]
    )
    assert solver.solve(time_limit=30)
    sol = solver.extract_solution()
    # Must be owned in both GWs
    assert UNOWNED_SWING in _owned(sol, START_GW)
    assert UNOWNED_SWING in _owned(sol, START_GW + 1)
    # 9.0 pts in START_GW -> solver starts him; 1.0 pt in START_GW+1 -> solver benches him
    assert UNOWNED_SWING in _starters(sol, START_GW)
    assert UNOWNED_SWING in _bench(sol, START_GW + 1)


def test_Precedence_exclusion_and_forced_lineup_do_not_break_feasibility():
    star = DEFS[0]
    # 1. Banned beats forced_bench and forced_squad
    solver1 = _solver(
        banned=[star],
        forced_bench=[(star, [START_GW])],
        forced_squad=[(star, [START_GW + 1])],
    )
    assert solver1.solve(time_limit=30)
    sol1 = solver1.extract_solution()
    assert star not in _owned(sol1, START_GW)

    # 2. Forced lineup beats forced bench on the same (player, gw)
    solver2 = _solver(
        forced_lineup=[(UNOWNED_ENABLER, [START_GW])],
        forced_bench=[(UNOWNED_ENABLER, [START_GW])],
        forced_squad=[(UNOWNED_ENABLER, [START_GW])],
    )
    assert solver2.solve(time_limit=30)
    sol2 = solver2.extract_solution()
    assert UNOWNED_ENABLER in _starters(sol2, START_GW)


def test_free_hit_handles_forced_bench_and_forced_squad():
    # Main solver skips FH GW so an unowned forced_bench/forced_squad in FH GW
    # does not conflict with the frozen permanent squad.
    solver = _solver(
        free_hit_gws=[START_GW],
        forced_bench=[(UNOWNED_ENABLER, [START_GW])],
        forced_squad=[(UNOWNED_SWING, [START_GW])],
    )
    assert solver.solve(time_limit=30)

    # And the Free Hit subproblem enforces both on the FH squad itself
    pred_df = _predictions().copy()
    pred_df["name"] = pred_df["element"].map(lambda p: f"Player_{p}")
    pred_df["position"] = pred_df["element"].map(POSITIONS)
    pred_df["player_team_id"] = pred_df["element"].map(lambda p: (p % 10) + 1)
    gw_data_df = pd.DataFrame({"element": POOL, "value": [50] * len(POOL)})

    fh = calculate_optimal_free_hit_squad(
        gw=START_GW,
        budget=1000,
        predictions_df=pred_df,
        gw_data_df=gw_data_df,
        watchlist_players=POOL,
        forced_bench_players=[(DEFS[0], [START_GW])],
        forced_squad_players=[(UNOWNED_ENABLER, [START_GW])],
    )
    assert fh["status"] == "Optimal"
    assert DEFS[0] in fh["squad"]
    assert DEFS[0] not in fh["lineup"]
    assert UNOWNED_ENABLER in fh["squad"]


def test_forced_captain_and_vice_captain():
    # Without forcing, DEFS[0] (8.0 pts) is captain in START_GW
    # Force MIDS[0] (4.0 pts) as captain and MIDS[1] (4.0 pts) as vice-captain in START_GW
    solver = _solver(
        forced_captain=[(MIDS[0], [START_GW])],
        forced_vice_captain=[(MIDS[1], [START_GW])],
    )
    assert solver.solve(time_limit=30)
    sol = solver.extract_solution()
    assert MIDS[0] in _starters(sol, START_GW)
    assert MIDS[1] in _starters(sol, START_GW)
    assert sol["captains"][1] == MIDS[0]
    assert sol["vice_captains"][1] == MIDS[1]
    # In START_GW+1, not forced -> DEFS[0] is captain again
    assert sol["captains"][2] == DEFS[0]


def test_forced_captain_unowned_player_bought_started_and_captained():
    solver = _solver(
        forced_captain=[(UNOWNED_ENABLER, [START_GW])],
        forced_vice_captain=[(DEFS[1], [START_GW])],
    )
    assert solver.solve(time_limit=30)
    sol = solver.extract_solution()
    assert UNOWNED_ENABLER in _owned(sol, START_GW)
    assert UNOWNED_ENABLER in _starters(sol, START_GW)
    assert sol["captains"][1] == UNOWNED_ENABLER
    assert sol["vice_captains"][1] == DEFS[1]


def test_free_hit_handles_forced_captain_and_vice_captain():
    pred_df = _predictions().copy()
    pred_df["name"] = pred_df["element"].map(lambda p: f"Player_{p}")
    pred_df["position"] = pred_df["element"].map(POSITIONS)
    pred_df["player_team_id"] = pred_df["element"].map(lambda p: (p % 10) + 1)
    gw_data_df = pd.DataFrame({"element": POOL, "value": [50] * len(POOL)})

    fh = calculate_optimal_free_hit_squad(
        gw=START_GW,
        budget=1000,
        predictions_df=pred_df,
        gw_data_df=gw_data_df,
        watchlist_players=POOL,
        forced_captain_players=[(UNOWNED_ENABLER, [START_GW])],
        forced_vice_captain_players=[(MIDS[0], [START_GW])],
    )
    assert fh["status"] == "Optimal"
    assert fh["captain"] == UNOWNED_ENABLER
    assert fh["vice_captain"] == MIDS[0]
    assert UNOWNED_ENABLER in fh["lineup"]
    assert MIDS[0] in fh["lineup"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
