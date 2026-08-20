"""Free transfers must be spent before hits are taken.

FPL gives you no way to decline a free transfer, so a plan that charges -4 in a
gameweek with a free transfer spare is not executable: follow it and FPL applies
the free transfer anyway, leaving you with fewer banked transfers than the rest
of the plan assumes.

The MILP can still produce that shape internally — `penalty_transfers` is bounded
below by (u - A) but not pinned to it, and over-declaring is exactly break-even, so
no objective tie-break sees it. Pinning it in the model costs a binary per gameweek
and pushes horizon-19 solves past CBC's feasibility-pump cliff (TECHNICAL.md), so
`extract_solution()` rebuilds the ledger instead. These tests cover that rebuild.

Runnable directly (``python tests/test_transfer_penalty.py``) or under pytest.
"""

import os
import sys

import pandas as pd
import pulp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.solver import MAX_FREE_TRANSFERS, FPLSolver  # noqa: E402

# Per-gameweek transfer counts and chip placement from dev solver job
# 6b66353a-7fd9-4a4d-ae91-c0af9e31fb78 (team 555, GW1-19), the run that came back
# charging -4 in GW12 and GW13 with a free transfer available in each.
TRANSFERS_USED = [0, 7, 1, 3, 2, 0, 2, 0, 2, 2, 1, 1, 1, 0, 1, 4, 1, 0, 3]
FREE_HIT_GW = 1
WILDCARD_GW = 2
INITIAL_TRANSFERS = 1


def _honest_ledger(transfers_used, initial, free_hit_gw, wildcard_gw):
    """Independent reimplementation of the rule, to check the solver's against."""
    available = initial
    rows = []
    for gw, used in enumerate(transfers_used, start=1):
        if gw in (free_hit_gw, wildcard_gw):
            rows.append((gw, used, available, 0))
            continue
        paid = max(0, used - available)
        rows.append((gw, used, available, paid))
        available = min(MAX_FREE_TRANSFERS, available - min(used, available) + 1)
    return rows


def _solved_solver(phantom_gw=None):
    """Solve the real banking/penalty model with transfer counts pinned.

    Scoring is stubbed: the squad-value half of the objective is irrelevant to how
    free transfers are accounted for, and stubbing it keeps the test free of player
    and prediction fixtures. The constraints are the real ones, and so is
    extract_solution()'s ledger rebuild.

    phantom_gw forces an over-declared hit in that gameweek — the shape the MILP is
    free to produce — so the rebuild can be checked against it.
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
    solver.initial_squad = [1]
    solver.initial_transfers = INITIAL_TRANSFERS
    solver.create_decision_variables()

    solver.prob = pulp.LpProblem("free_transfers_before_hits", pulp.LpMaximize)
    solver.add_transfer_banking_constraints()
    solver.add_chip_constraints()

    for t, used in enumerate(TRANSFERS_USED, start=1):
        solver.prob += (solver.variables['u'][t] == used, f"Pin_Transfers_{t}")

    if phantom_gw is not None:
        solver.prob += (
            solver.variables['penalty_transfers'][phantom_gw] >= 1,
            f"Force_Phantom_{phantom_gw}",
        )

    solver.prob += pulp.lpSum(
        -4 * solver.variables['penalty_transfers'][t]
        for t in range(1, len(TRANSFERS_USED) + 1)
    )
    solver.prob.solve(pulp.PULP_CBC_CMD(msg=0))
    assert pulp.LpStatus[solver.prob.status] == "Optimal", pulp.LpStatus[solver.prob.status]
    return solver


def _reported(solver):
    """The per-gameweek ledger as extract_solution() reports it."""
    solution = solver.extract_solution()
    return [
        (t, solution['transfers'][t]['count'],
         solution['transfers'][t]['available_transfers'],
         solution['transfers'][t]['paid_transfers'])
        for t in range(1, len(TRANSFERS_USED) + 1)
    ]


def test_reported_ledger_spends_free_transfers_first():
    """No gameweek is reported as paying a hit while holding a free transfer."""
    reported = _reported(_solved_solver())
    offenders = {
        gw: (used, available, paid)
        for gw, used, available, paid in reported
        if gw not in (FREE_HIT_GW, WILDCARD_GW) and paid > max(0, used - available)
    }
    assert not offenders, f"hits reported with a free transfer spare: {offenders}"


def test_reported_ledger_matches_the_rule():
    """The rebuild agrees with an independent implementation of FPL's rule."""
    assert _reported(_solved_solver()) == _honest_ledger(
        TRANSFERS_USED, INITIAL_TRANSFERS, FREE_HIT_GW, WILDCARD_GW
    )


def test_rebuild_survives_a_phantom_solve():
    """An over-declared hit inside the MILP is not carried into the report.

    GW12 in the real run: one transfer made, one free transfer available, a -4
    charged anyway. The model still permits that internally, so this forces it and
    checks the reported ledger comes out honest regardless.
    """
    solver = _solved_solver(phantom_gw=12)
    assert int(solver.variables['penalty_transfers'][12].varValue) >= 1, (
        "expected the forced phantom to be present in the raw solution"
    )
    reported = dict((gw, (used, avail, paid)) for gw, used, avail, paid in _reported(solver))
    used, available, paid = reported[12]
    assert paid == max(0, used - available) == 0, (
        f"phantom leaked into the report for GW12: used={used}, available={available}, paid={paid}"
    )


def test_total_hits_are_unchanged_by_the_rebuild():
    """The rebuild relocates hits, it does not add or remove them."""
    reported = _reported(_solved_solver())
    total = sum(paid for _, _, _, paid in reported)
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
