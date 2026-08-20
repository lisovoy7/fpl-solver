"""Reproduce the phantom-hit bug at real scale, with real dev data.

Uses the exact squad and chip scenario from the dev job that surfaced the bug
(team 555, job 6b66353a-7fd9-4a4d-ae91-c0af9e31fb78, GW1-19: Free Hit GW1,
Wildcard GW2, Triple Captain GW3, Bench Boost GW7 — the scenario the full
solver picked as best). Player metadata and predictions are a snapshot from
the dev Supabase project (fixture_players_dev.json,
fixture_predictions_dev_gw1-19.json — 2026-08-20), so this runs offline.

This is a single scenario solve, not the full chip-enumeration pipeline the
API runs, so the plan itself won't exactly match the original job — the point
is checking the transfer/hit pattern the bug produced, not reproducing the
narrated plan bit-for-bit.

Run: python tests/verify_transfer_penalty_fix.py
"""

import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.solver import FPLSolver  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SQUAD = [140, 497, 4, 356, 201, 473, 329, 154, 398, 105, 121, 70, 411, 490, 464]
TOTAL_BUDGET_M = 100  # £m, as sent to the API — converted to tenths below
HORIZON = 19
POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def load_players():
    rows = json.load(open(os.path.join(HERE, "fixture_players_dev.json")))
    df = pd.DataFrame(rows)
    df["position"] = df["element_type"].map(POSITION_BY_ELEMENT_TYPE)
    df = df.rename(columns={"id": "element", "now_cost": "value"})
    return df[["element", "position", "value", "team"]]


def load_predictions():
    rows = json.load(open(os.path.join(HERE, "fixture_predictions_dev_gw1-19.json")))
    df = pd.DataFrame(rows)
    return df.rename(columns={"player_id": "element", "event": "event", "predicted_points": "predicted_points"})


def main():
    players = load_players()
    predictions = load_predictions()

    solver = FPLSolver(
        planning_horizon=HORIZON,
        budget=TOTAL_BUDGET_M * 10,
        start_gw=1,
        free_hit_gws=[1],
        force_wildcard_gw=2,
        triple_captain_gw=3,
        bench_boost_gw=7,
    )
    solver.players = players[players["element"].isin(
        set(players["element"]) & (set(predictions["element"]) | set(SQUAD))
    )].copy()
    solver.load_predictions(predictions)
    solver.set_initial_squad(SQUAD, available_transfers=1)
    solver.set_chip_state()
    solver.build_model()
    print(f"model: {len(solver.prob.variables())} vars, {len(solver.prob.constraints)} constraints")

    t0 = time.time()
    ok = solver.solve(time_limit=90)
    print(f"solve: ok={ok} in {time.time() - t0:.1f}s  proven_optimal={solver.proven_optimal}")
    if not ok:
        print("No feasible solution — nothing to check.")
        return 1

    solution = solver.extract_solution()
    print(f"\n{'GW':<4}{'used':<6}{'available':<11}{'paid':<6}{'wildcard'}")
    phantom = {}
    for t in range(1, HORIZON + 1):
        tr = solution["transfers"][t]
        print(f"{t:<4}{tr['count']:<6}{tr['available_transfers']:<11}{tr['paid_transfers']:<6}{tr['wildcard_active']}")
        if tr["wildcard_active"] or (solver.start_gw + t - 1) in solver.free_hit_gws:
            continue
        excess = tr["paid_transfers"] - max(0, tr["count"] - tr["available_transfers"])
        if excess > 0:
            phantom[t] = excess

    print()
    if phantom:
        print(f"FAIL — phantom hits (charged despite a free transfer available) in GW: {phantom}")
        return 1
    print("PASS — no gameweek charges a hit while a free transfer is available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
