"""Non-playing players must score 0 in squad_points, post-hoc Bench Boost, and formatted output.

Regression test for prod job f11b2d81 (team 2437541, GW11): Bobby Thomas (173) and
Anton Stach (335) were in `non_playing` for GW11. The MILP objective zeroed their
local weight `E_pt` and benched both in GW11, but `solver.expected_points` and
`solver.predictions` kept their healthy projections (4.8 and 4.1). Post-hoc Bench
Boost (`find_best_bench_boost_gw`) therefore scored GW11's bench at 16.3 instead of
7.4 and placed Bench Boost on the exact week two bench players were injured.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server import _format_solution  # noqa: E402
from fpl.free_hit import find_best_bench_boost_gw  # noqa: E402
from fpl.scenario_runner import _squad_points, build_solver  # noqa: E402


def _fixture_ctx():
    # 15 players: 2 GK (1..2), 5 DEF (3..7), 5 MID (8..12), 3 FWD (13..15)
    positions = (
        [(1, "GK"), (2, "GK")]
        + [(i, "DEF") for i in range(3, 8)]
        + [(i, "MID") for i in range(8, 13)]
        + [(i, "FWD") for i in range(13, 16)]
    )
    gw_rows = []
    pred_rows = []
    for pid, pos in positions:
        gw_rows.append({
            "element": pid,
            "GW": 10,
            "name": f"P{pid}",
            "position": pos,
            "value": 50,
            "team": (pid - 1) % 10 + 1,
        })
        for gw in (11, 12):
            # Starters score 5.0; bench fodder (2, 7, 12, 15) score 3.5 in both GWs,
            # except players 6 and 11 who normally score 4.8 / 4.5 (starters when fit)
            # but are non_playing in GW11.
            if pid in (6, 11):
                pts = 4.8 if pid == 6 else 4.5
            elif pid in (2, 7, 12, 15):
                pts = 3.5
            else:
                pts = 5.0
            pred_rows.append({
                "element": pid,
                "event": gw,
                "predicted_points": pts,
                "name": f"P{pid}",
                "position": pos,
                "player_team_id": (pid - 1) % 10 + 1,
            })

    squad = [p for p, _ in positions]
    return {
        "horizon": 2,
        "budget": 1000,
        "start_gw": 11,
        "points_multiplier": None,
        "forced_lineup": None,
        "non_playing": [(6, [11]), (11, [11])],
        "banned_players": [],
        "no_transfer_gws": [11, 12],
        "first_gw_penalty": -1,
        "sub_probability": 0.0,
        "predictions": pd.DataFrame(pred_rows),
        "gw_data": pd.DataFrame(gw_rows),
        "watchlist": squad,
        "current_squad": squad,
        "free_transfers": 1,
        "wildcard_first_half": 1,
        "wildcard_second_half": 1,
        "time_limit": 10,
        "solver_name": "HIGHS",
        "mip_gap": 0.0,
        "bank": 250,
        "selling_discounts": {},
        "player_clubs": {p: (p - 1) % 10 + 1 for p, _ in positions},
        "club_gameweeks": {c: {11, 12} for c in range(1, 11)},
    }


def test_non_playing_zeroed_in_squad_points_and_bench_boost_selection():
    ctx = _fixture_ctx()
    scenario = {
        "name": "No chips",
        "free_hit_gws": [],
        "bench_boost_gw": -1,
        "triple_captain_gw": -1,
        "force_wildcard_gw": None,
    }
    solver = build_solver(scenario, ctx)
    solver.build_model()
    assert solver.solve(time_limit=10)
    solution = solver.extract_solution()
    squad_pts = _squad_points(solver, solution)

    # Players 6 and 11 are benched in GW11 because they are non_playing, and their
    # exported squad_points in GW11 must be 0.0 (not 4.8 / 4.5).
    assert 6 in solution["lineups"][1]["bench"]
    assert 11 in solution["lineups"][1]["bench"]
    assert squad_pts[(6, 11)] == 0.0
    assert squad_pts[(11, 11)] == 0.0
    # In GW12 they are fit again and carry their real projections.
    assert squad_pts[(6, 12)] == 4.8
    assert squad_pts[(11, 12)] == 4.5

    bb_gw, bb_bonus = find_best_bench_boost_gw(
        lineups_by_t=solution["lineups"],
        expected_points=squad_pts,
        wildcard_gws=[],
        start_gw=11,
        horizon=2,
        free_hit_gws=[],
        triple_captain_gw=-1,
        candidate_gws=[11, 12],
        sub_probability=0.0,
    )
    # GW11 has only 2 playing bench members (2 * 3.5 = 7.0); GW12 has all 4 (4 * 3.5 = 14.0).
    assert bb_gw == 12, f"Expected Bench Boost in GW12, got GW{bb_gw} (bonus {bb_bonus})"
    assert abs(bb_bonus - 14.0) < 1e-6


def test_format_solution_reports_zero_for_non_playing_bench_players():
    ctx = _fixture_ctx()
    scenario = {
        "name": "No chips",
        "free_hit_gws": [],
        "bench_boost_gw": -1,
        "triple_captain_gw": -1,
        "force_wildcard_gw": None,
    }
    solver = build_solver(scenario, ctx)
    solver.build_model()
    assert solver.solve(time_limit=10)
    solution = solver.extract_solution()
    # Mirror _with_placed_chips: post-hoc BB written onto an already-solved plan.
    solution["chips"][1] = ["bench_boost"]

    # Rebuild solver the same way api_server.py does (build_solver without build_model).
    rebuilt = build_solver({**scenario, "bench_boost_gw": 11}, ctx)
    bootstrap = {
        "elements": [
            {
                "id": int(r.element),
                "first_name": f"P{int(r.element)}",
                "second_name": "",
                "element_type": {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}[r.position],
                "now_cost": int(r.value),
            }
            for r in ctx["gw_data"].itertuples()
        ]
    }
    formatted = _format_solution(
        solution, rebuilt, rebuilt.players, 11, solution["objective_value"],
        "No chips | BB GW11", {}, bootstrap,
    )
    gw11_bench = {p["id"]: p["expected_points"] for p in formatted["gameweeks"][0]["bench"]}
    assert gw11_bench[6] == 0.0
    assert gw11_bench[11] == 0.0
    # All 13 fit players (9 * 5.0 + 4 * 3.5) + captain 5.0 = 64.0 (not + 4.8 + 4.5).
    assert abs(formatted["gameweeks"][0]["expected_points"] - 64.0) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
