"""Solve a real horizon-19 model with a risen squad, and check the money adds up.

The cash-flow budget model can't be exercised against live FPL data yet: as of the
2026/27 pre-season not a single price has moved, so every selling price equals its
market price and the change is a guaranteed no-op. This injects the price rises the
season will produce — the squad from the dev fixture, marked up so it is worth more
to buy than it would raise to sell — and checks three things at full scale:

  1. the model stays feasible (the old rule read a risen squad as unaffordable to keep)
  2. every reported bank balance matches the transfers that produced it, and none is
     negative
  3. the extra constraints don't blow up the solve

Player metadata and predictions are the same offline snapshot the transfer-penalty
verification uses (2026-08-20, dev Supabase).

Run: python tests/verify_cash_flow_at_scale.py
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
HORIZON = 19
POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Mid-season shape: the squad has risen, and the bank is thin. Every owned player is
# marked up 0.4 above what he was bought for, so he sells for 0.2 less than he now
# costs — a 3.0 gap across the squad, against 0.3 of cash.
RISE_PER_PLAYER = 4  # tenths
BANK = 3  # tenths


def load_players():
    rows = json.load(open(os.path.join(HERE, "fixture_players_dev.json")))
    df = pd.DataFrame(rows)
    df["position"] = df["element_type"].map(POSITION_BY_ELEMENT_TYPE)
    df = df.rename(columns={"id": "element", "now_cost": "value"})
    return df[["element", "position", "value", "team"]]


def load_predictions():
    rows = json.load(open(os.path.join(HERE, "fixture_predictions_dev_gw1-19.json")))
    return pd.DataFrame(rows).rename(columns={"player_id": "element"})


def main():
    players = load_players()
    predictions = load_predictions()

    # Mark the owned squad up. Market price is what it now costs to buy them; the
    # selling price is the purchase price plus half the rise, so the discount is half.
    owned = players["element"].isin(SQUAD)
    players.loc[owned, "value"] = players.loc[owned, "value"] + RISE_PER_PLAYER
    discounts = {int(pid): RISE_PER_PLAYER // 2 for pid in SQUAD}

    market = dict(zip(players["element"], players["value"]))
    squad_market = sum(market[p] for p in SQUAD)
    squad_sale = sum(market[p] - discounts[p] for p in SQUAD)
    total_budget = squad_sale + BANK

    print(f"squad at market: {squad_market / 10:.1f}M")
    print(f"squad if sold:   {squad_sale / 10:.1f}M   (+ bank {BANK / 10:.1f}M = {total_budget / 10:.1f}M)")
    print(f"the old rule was short by {(squad_market - total_budget) / 10:.1f}M before making a single move\n")

    solver = FPLSolver(
        planning_horizon=HORIZON,
        budget=total_budget,
        start_gw=1,
        free_hit_gws=[1],
        force_wildcard_gw=2,
        triple_captain_gw=3,
        bench_boost_gw=7,
        bank=BANK,
        selling_discounts=discounts,
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
    elapsed = time.time() - t0
    print(f"solve: ok={ok} in {elapsed:.1f}s  proven_optimal={solver.proven_optimal}")
    if not ok:
        print("FAIL — no feasible solution. A risen squad must still be plannable.")
        return 1

    solution = solver.extract_solution()
    sale = solver.sale_prices()

    print(f"\n{'GW':<4}{'in':<5}{'out':<6}{'spent':<9}{'raised':<9}{'bank':<8}{'expected'}")
    running = float(BANK)
    failures = []
    for t in range(1, HORIZON + 1):
        tr = solution["transfers"][t]
        spent = sum(market[p] for p in tr["in"])
        raised = sum(sale[p] for p in tr["out"])
        running = running + raised - spent
        reported = solution["bank"][t]
        flag = ""
        if abs(reported - running) > 0.5:
            failures.append(f"GW{t}: reported {reported} vs {running:.0f}")
            flag = "  <-- MISMATCH"
        if reported < 0:
            failures.append(f"GW{t}: negative bank {reported}")
            flag = "  <-- NEGATIVE"
        print(f"{t:<4}{len(tr['in']):<5}{len(tr['out']):<6}"
              f"{spent / 10:<9.1f}{raised / 10:<9.1f}{reported / 10:<8.1f}{running / 10:.1f}{flag}")

    print()
    if failures:
        for f in failures:
            print(f"FAIL — {f}")
        return 1
    print("PASS — every gameweek's bank matches its transfers, and none went negative.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
