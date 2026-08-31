""""No chips" has to include the wildcard.

Free Hit, Bench Boost and Triple Captain are enumerated into scenarios, so
`use_chips: false` rules them out just by collapsing to the single "No chips"
scenario. The wildcard is not: it is a decision variable inside the MILP, identical
in every scenario, so scenario selection never touches it. Only chip state — both
halves reported spent — takes it off the table.

Without that, a "no chips" request came back playing a wildcard, and looked
reasonable doing it: a wildcard gameweek makes every transfer free, so 10 moves
show up with no points hit and nothing on the plan says why.

Runnable directly (``python tests/test_use_chips.py``) or under pytest.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api_server as A  # noqa: E402


def _halves(first: int = 0, second: int = 0) -> dict:
    """A detect_chips_used-shaped dict for the wildcard alone."""
    return {"wildcards_used_first_half": first, "wildcards_used_second_half": second}


def _wildcard(use_chips: bool, override, detected: dict) -> tuple:
    """Resolve wildcard halves the way _optimize_inner does."""
    return A._wildcard_halves(
        use_chips, A._chip_halves_from(detected, "wildcards_used", override)
    )


def test_no_chips_spends_both_wildcards():
    assert _wildcard(False, None, _halves()) == (1, 1)


def test_no_chips_beats_an_explicit_wildcard_count():
    """`use_chips: false` is the broader instruction, so it wins.

    Honouring the narrower one would mean quietly planning with a chip the caller
    ruled out — the worse of the two ways to resolve the contradiction.
    """
    assert _wildcard(False, 0, _halves()) == (1, 1)


def test_an_explicit_count_is_used_when_chips_are_on():
    assert _wildcard(True, 1, _halves()) == (1, 0)
    # 0 is a real value, not a missing one: it must not fall through to detection.
    assert _wildcard(True, 0, _halves()) == (0, 0)


def test_detection_fills_in_when_no_count_is_sent():
    assert _wildcard(True, None, _halves(first=1)) == (1, 0)
    assert _wildcard(True, None, _halves(second=1)) == (0, 1)


def test_both_halves_spent_pins_every_wildcard_variable_to_zero():
    """The other half of the rule: that a count of 2 actually disables the chip.

    `add_chip_constraints()` caps wildcards per half-season at `1 - used`, so 2 (one
    per half) leaves no room in either window. Checked here rather than assumed,
    because the whole fix above rests on it.
    """
    import pandas as pd
    import pulp

    from fpl.solver import FPLSolver

    solver = FPLSolver(planning_horizon=4, budget=1000, start_gw=1)
    solver.players = pd.DataFrame({'element': [1], 'value': [50]})
    solver.initial_squad = [1]
    solver.create_decision_variables()
    solver.prob = pulp.LpProblem("wildcard_off", pulp.LpMaximize)
    solver.set_chip_state(wildcard_first_half=1, wildcard_second_half=1)
    solver.add_chip_constraints()

    # Ask for a wildcard as loudly as possible: if any gameweek can still take one,
    # maximising their sum finds it.
    solver.prob += pulp.lpSum(solver.variables['wildcard'][t] for t in range(1, 5))
    solver.prob.solve(pulp.PULP_CBC_CMD(msg=0))

    assert pulp.LpStatus[solver.prob.status] == "Optimal"
    played = [t for t in range(1, 5) if solver.variables['wildcard'][t].varValue == 1]
    assert played == [], f"wildcard still available in {played}"


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
