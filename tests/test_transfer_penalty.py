"""Free transfers must be spent before hits are taken.

FPL gives you no way to decline a free transfer, so a plan that charges -4 in a
gameweek with a free transfer spare is not executable: follow it and FPL applies
the free transfer anyway, leaving you with fewer banked transfers than the rest
of the plan assumes.

The shape costs no points (a -4 buys a banked transfer worth at most one future
-4), so it cannot be ruled out by an objective tie-break — the phantom plan
carries the same hit count as the honest one, just placed differently. The
Penalty_Exact_When_Needed / Penalty_Zero_When_Covered constraints make it
infeasible instead, and that is what is under test here.

Runnable directly (``python tests/test_transfer_penalty.py``) or under pytest.
"""

import os
import sys

import pandas as pd
import pulp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.solver import TRANSFER_PENALTY_POINTS, FPLSolver  # noqa: E402

# Per-gameweek transfer counts and chip placement from dev solver job
# 6b66353a-7fd9-4a4d-ae91-c0af9e31fb78 (team 555, GW1-19), the run that came back
# charging -4 in GW12 and GW13 with a free transfer available in each.
TRANSFERS_USED = [0, 7, 1, 3, 2, 0, 2, 0, 2, 2, 1, 1, 1, 0, 1, 4, 1, 0, 3]
FREE_HIT_GW = 1
WILDCARD_GW = 2
INITIAL_TRANSFERS = 1


def _build():
    """Build the real banking/penalty constraint set with transfer counts pinned.

    Scoring is stubbed: the squad-value half of the objective is irrelevant to how
    free transfers are accounted for, and stubbing it keeps the test free of player
    and prediction fixtures. The constraints and penalty constant are the real ones.
    """
    solver = FPLSolver(
        planning_horizon=len(TRANSFERS_USED),
        budget=1000,  # unused: no squad-composition or budget constraints are added
        start_gw=1,
        free_hit_gws=[FREE_HIT_GW],
        force_wildcard_gw=WILDCARD_GW,
    )
    # create_decision_variables() only reads the 'element' column; one player is
    # enough because no squad-composition constraints are added here.
    solver.players = pd.DataFrame({'element': [1]})
    solver.initial_transfers = INITIAL_TRANSFERS
    solver.create_decision_variables()

    solver.prob = pulp.LpProblem("free_transfers_before_hits", pulp.LpMaximize)
    solver.add_transfer_banking_constraints()
    solver.add_chip_constraints()

    for t, used in enumerate(TRANSFERS_USED, start=1):
        solver.prob += (solver.variables['u'][t] == used, f"Pin_Transfers_{t}")

    solver.prob += pulp.lpSum(
        TRANSFER_PENALTY_POINTS * solver.variables['penalty_transfers'][t]
        for t in range(1, len(TRANSFERS_USED) + 1)
    )
    return solver


def _solve(solver):
    solver.prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.LpStatus[solver.prob.status]


def _phantom_hits(solver):
    """Hits charged beyond what the free-transfer balance made unavoidable."""
    phantom = {}
    for t, used in enumerate(TRANSFERS_USED, start=1):
        if round(solver.variables['wildcard'][t].varValue) == 1:
            continue
        if (solver.start_gw + t - 1) in solver.free_hit_gws:
            continue
        available = round(solver.variables['A'][t].varValue)
        charged = round(solver.variables['penalty_transfers'][t].varValue)
        excess = charged - max(0, used - available)
        if excess > 0:
            phantom[solver.start_gw + t - 1] = excess
    return phantom


def test_free_transfers_are_spent_before_hits():
    """The optimum never charges a hit while a free transfer is available."""
    solver = _build()
    status = _solve(solver)
    assert status == "Optimal", status
    phantom = _phantom_hits(solver)
    assert not phantom, f"hits charged with a free transfer spare in GW(s): {phantom}"


def test_phantom_hit_is_infeasible():
    """Charging a hit with a free transfer in hand is ruled out, not just costed.

    The phantom plan ties with the honest one on points, so this has to be
    infeasibility — a cheaper objective tie-break would leave CBC free to return
    either shape.
    """
    solver = _build()
    # GW12 in the real run: one transfer made, one free transfer available, and a
    # -4 charged anyway. Pin that situation and the model must reject it.
    solver.prob += (solver.variables['A'][12] == 1, "Free_Transfer_Available_GW12")
    solver.prob += (solver.variables['penalty_transfers'][12] >= 1, "Force_Phantom_GW12")
    status = _solve(solver)

    assert status == "Infeasible", f"expected Infeasible, got {status}"


def test_honest_plan_still_costs_the_same_hits():
    """The fix relocates hits, it does not add them.

    The phantom and honest shapes were break-even, so the run this regressed from
    should still total the same number of paid transfers.
    """
    solver = _build()
    status = _solve(solver)
    assert status == "Optimal", status
    total = sum(
        round(solver.variables['penalty_transfers'][t].varValue)
        for t in range(1, len(TRANSFERS_USED) + 1)
    )
    assert total == 7, f"expected 7 paid transfers over GW1-19, got {total}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"PASS {name}")
    sys.exit(1 if failures else 0)
