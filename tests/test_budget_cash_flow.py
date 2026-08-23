"""Money is a running bank, not a squad-value ceiling.

The old budget rule compared the squad's *market* value against the manager's
*selling* value plus bank. Those are different measures of the same squad, and the
gap grows all season: a squad that has risen 1.5M with 0.2M in the bank read as
unaffordable to simply keep, and the model had no way to say so — it sold a player
to balance an arithmetic error and returned a plan that looked like football.

`add_budget_constraints()` tracks cash instead: sales raise the FPL selling price,
purchases cost the market price, and the balance may never go negative. That is what
FPL actually enforces, and holding an appreciated player is correctly free.

Runnable directly (``python tests/test_budget_cash_flow.py``) or under pytest.
"""

import os
import sys

import pandas as pd
import pulp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.solver import FPLSolver  # noqa: E402

HORIZON = 2

# Four players, prices in tenths. The manager owns OWNED and has appreciated: both
# cost more to buy now than they would raise if sold.
OWNED = [1, 2]
PRICES = {1: 100, 2: 50, 3: 60, 4: 200}
DISCOUNTS = {1: 10, 2: 4}  # so p1 sells for 90, p2 for 46
BANK = 10

# What the squad card would report: sale value of the squad plus the bank.
TOTAL_BUDGET = (PRICES[1] - DISCOUNTS[1]) + (PRICES[2] - DISCOUNTS[2]) + BANK  # 146


def _model(bank=BANK, budget=TOTAL_BUDGET, pins=()):
    """Build the flow + cash-flow model on its own, with transfers pinned.

    Squad composition, lineups and points are all left out: none of them touch the
    cash constraint, and leaving them out keeps the test free of position, club and
    prediction fixtures. The constraints under test are the real ones.
    """
    solver = FPLSolver(
        planning_horizon=HORIZON,
        budget=budget,
        start_gw=1,
        bank=bank,
        selling_discounts=dict(DISCOUNTS),
    )
    solver.players = pd.DataFrame({
        'element': list(PRICES),
        'value': [PRICES[p] for p in PRICES],
    })
    solver.initial_squad = list(OWNED)
    solver.initial_transfers = 1
    solver.create_decision_variables()

    solver.prob = pulp.LpProblem("cash_flow", pulp.LpMaximize)
    solver.add_squad_flow_constraints()
    solver.add_budget_constraints()

    # Nothing in this model prices a wildcard, so pin it off — otherwise CBC is free to
    # flip it and extract_solution()'s transfer ledger reads differently run to run.
    for t in range(1, HORIZON + 1):
        solver.prob += (solver.variables['wildcard'][t] == 0, f"No_Wildcard_{t}")

    for name, expr in pins:
        solver.prob += (expr(solver), name)

    # Feasibility only, but extract_solution() reads the objective's value, and PuLP
    # reports None for an objective that holds no live variable (a literal 0, or 0 * x,
    # both collapse to a constant). A variable pinned to zero is inert and still real.
    solver.prob += pulp.LpVariable("objective_stub", lowBound=0, upBound=0)
    return solver


def _solve(solver):
    solver.prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.LpStatus[solver.prob.status]


def test_appreciated_squad_can_be_held():
    """The bug this replaced: keeping your own risen squad must not cost anything.

    Market value of the squad is 15.0 against a budget of 14.6, which the old
    `squad value <= budget` rule read as 0.4M short of standing still.
    """
    solver = _model()
    assert sum(PRICES[p] for p in OWNED) > TOTAL_BUDGET, "fixture must be appreciated"
    assert _solve(solver) == "Optimal"

    solution = solver.extract_solution()
    for t in range(1, HORIZON + 1):
        assert solution['transfers'][t]['count'] == 0, f"GW{t} was forced into a transfer"
        assert solution['bank'][t] == BANK, f"GW{t} bank moved without a transfer"


def test_sale_raises_selling_price_not_market_price():
    """Selling p1 raises 9.0, not his 10.0 market price; buying p3 costs 6.0."""
    solver = _model(pins=[
        ("Sell_1", lambda s: s.variables['r'][(1, 1)] == 1),
        ("Buy_3", lambda s: s.variables['s'][(3, 1)] == 1),
    ])
    assert _solve(solver) == "Optimal"

    solution = solver.extract_solution()
    expected = BANK + (PRICES[1] - DISCOUNTS[1]) - PRICES[3]
    assert solution['bank'][1] == expected, solution['bank']
    # Nothing moves in GW2, so the balance carries.
    assert solution['bank'][2] == expected


def test_purchase_beyond_the_bank_is_refused():
    """Owning p4 (20.0) is out of reach: 1.0 in the bank plus 13.6 of sellable squad.

    Pinned on ownership rather than on the transfer, because a purchase paired with a
    sale of the same player in the same gameweek nets to zero and satisfies a pinned
    `transfer_in` without ever actually holding him.
    """
    solver = _model(pins=[
        ("Own_4", lambda s: s.variables['x'][(4, 1)] == 1),
    ])
    sellable = BANK + sum(PRICES[p] - DISCOUNTS.get(p, 0) for p in OWNED)
    assert sellable < PRICES[4], "fixture must put p4 out of reach"
    assert _solve(solver) == "Infeasible"


def test_bank_is_derived_from_budget_when_not_supplied():
    """Callers that only know the total: budget minus the squad's sale value is the bank."""
    solver = _model(bank=None)
    assert solver.opening_bank() == BANK

    assert _solve(solver) == "Optimal"
    assert solver.extract_solution()['bank'][1] == BANK


def test_players_the_manager_does_not_own_sell_for_market_price():
    """A player bought inside the plan has no purchase history to discount against.

    Buy p3 in GW1 and sell him again in GW2: prices are held still across the horizon,
    so the round trip must be exactly break-even.
    """
    solver = _model(pins=[
        ("Sell_1", lambda s: s.variables['r'][(1, 1)] == 1),
        ("Buy_3", lambda s: s.variables['s'][(3, 1)] == 1),
        ("Sell_3", lambda s: s.variables['r'][(3, 2)] == 1),
        ("Buy_1_back", lambda s: s.variables['s'][(1, 2)] == 1),
    ])
    assert _solve(solver) == "Optimal"

    solution = solver.extract_solution()
    after_gw1 = BANK + (PRICES[1] - DISCOUNTS[1]) - PRICES[3]
    # p3 raises his full 6.0 back, having been bought at 6.0. Buying p1 back costs his
    # market 10.0, not the 9.0 he was sold for, so the round trip has cost exactly his
    # 1.0 discount — which is what FPL charges for selling and re-buying a risen player.
    assert solution['bank'][2] == after_gw1 + PRICES[3] - PRICES[1]
    assert solution['bank'][2] == BANK - DISCOUNTS[1]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    sys.exit(1 if failures else 0)
