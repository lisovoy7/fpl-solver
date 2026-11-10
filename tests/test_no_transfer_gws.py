"""`no_transfer_gws` — "roll my transfer in GW7": no regular transfer that week,
the free transfer banks forward, and a wildcard there is still allowed.

The constraint is `u[t] <= 15 * wildcard[t]` (fpl/solver.py,
_add_no_transfer_constraints). Three things have to hold, each pinned below:

1. With the wildcard unavailable, a gameweek the free optimum WOULD transfer in
   makes zero transfers instead, and the next gameweek has one more free transfer
   than it otherwise would — the roll is what the user asked for, not the ban.
2. The bank is capped at MAX_FREE_TRANSFERS, same as FPL: rolling at 5 stays 5.
3. A wildcard pinned to that same gameweek is honoured — the plan rebuilds the
   squad there — because "don't transfer" has never meant "don't wildcard".

Runnable directly (``python tests/test_no_transfer_gws.py``) or under pytest.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.solver import FPLSolver, MAX_FREE_TRANSFERS  # noqa: E402

HORIZON = 3
START_GW = 7

GKS = [101, 102]
DEFS = [103, 104, 105, 106, 107]
MIDS = [108, 109, 110, 111, 112]
FWDS = [113, 114, 115]
SQUAD = GKS + DEFS + MIDS + FWDS
WEAK = DEFS[0]
REPLACEMENT = 201  # a clearly better defender, so a free optimum swaps in GW1

POSITIONS = (
    {p: 'GK' for p in GKS}
    | {p: 'DEF' for p in DEFS + [REPLACEMENT]}
    | {p: 'MID' for p in MIDS}
    | {p: 'FWD' for p in FWDS}
)
POOL = SQUAD + [REPLACEMENT]


def _predictions():
    rows = []
    for gw in range(START_GW, START_GW + HORIZON):
        for p in POOL:
            pts = 1.0 if p == WEAK else (8.0 if p == REPLACEMENT else 4.0)
            rows.append({'element': p, 'event': gw, 'predicted_points': pts})
    return pd.DataFrame(rows)


def _solver(no_transfer_gws=None, rolled_transfers=None, free_transfers=1,
            wildcard_available=False, force_wildcard_gw=None, extra_replacements=0):
    extra_ids = [202 + i for i in range(extra_replacements)]
    pool = POOL + extra_ids
    positions = POSITIONS | {p: 'DEF' for p in extra_ids}
    weak_set = {DEFS[i] for i in range(1 + extra_replacements)}
    strong_set = {REPLACEMENT, *extra_ids}

    rows = []
    for gw in range(START_GW, START_GW + HORIZON):
        for p in pool:
            pts = 1.0 if p in weak_set else (8.0 if p in strong_set else 4.0)
            rows.append({'element': p, 'event': gw, 'predicted_points': pts})

    solver = FPLSolver(
        planning_horizon=HORIZON,
        budget=1000,
        start_gw=START_GW,
        no_transfer_gws=no_transfer_gws,
        rolled_transfers=rolled_transfers,
        force_wildcard_gw=force_wildcard_gw,
    )
    solver.load_predictions(pd.DataFrame(rows))
    solver.players = pd.DataFrame({
        'element': pool,
        'name': [str(p) for p in pool],
        'position': [positions[p] for p in pool],
        'value': [50] * len(pool),
        'team': [(i % 10) + 1 for i in range(len(pool))],
    })
    solver.set_initial_squad(list(SQUAD), available_transfers=free_transfers)
    solver.set_chip_state(
        wildcard_first_half=0 if wildcard_available else 1,
        wildcard_second_half=1,
    )
    solver.build_model()
    return solver


def _solve(solver):
    assert solver.solve(time_limit=30), "model must stay solvable"
    return solver.extract_solution()


def test_free_optimum_transfers_immediately():
    """Control: without the constraint the swap happens in the first gameweek."""
    solution = _solve(_solver())
    assert solution['transfers'][1]['count'] == 1
    assert REPLACEMENT in solution['transfers'][1]['in']


def test_no_transfer_gw_rolls_the_free_transfer():
    solver = _solver(no_transfer_gws=[START_GW])
    assert f"No_Transfer_GW{START_GW}" in solver.prob.constraints
    solution = _solve(solver)
    t1, t2 = solution['transfers'][1], solution['transfers'][2]
    assert t1['count'] == 0, "a rolled gameweek makes no transfer"
    assert t1['available_transfers'] == 1
    assert t2['available_transfers'] == 2, "the free transfer banked forward"
    # The swap is still worth making — it just waited a week.
    assert REPLACEMENT in t2['in']


def test_two_rolled_gameweeks_bank_two():
    solution = _solve(_solver(no_transfer_gws=[START_GW, START_GW + 1]))
    assert solution['transfers'][1]['count'] == 0
    assert solution['transfers'][2]['count'] == 0
    assert solution['transfers'][3]['available_transfers'] == 3


def test_bank_is_capped_at_the_fpl_maximum():
    solution = _solve(_solver(no_transfer_gws=[START_GW], free_transfers=MAX_FREE_TRANSFERS))
    assert solution['transfers'][1]['count'] == 0
    assert solution['transfers'][2]['available_transfers'] == MAX_FREE_TRANSFERS


def test_wildcard_in_a_rolled_gameweek_is_still_allowed():
    """The exemption: a wildcard's moves are not transfers, so pinning one to the
    rolled gameweek must be honoured rather than fought."""
    solver = _solver(no_transfer_gws=[START_GW], wildcard_available=True,
                     force_wildcard_gw=START_GW)
    solution = _solve(solver)
    t1 = solution['transfers'][1]
    assert t1['wildcard_active']
    assert t1['count'] >= 1, "the wildcard rebuild went ahead"
    assert REPLACEMENT in t1['in']
    assert t1['paid_transfers'] == 0


def test_gameweek_outside_the_horizon_is_ignored():
    solver = _solver(no_transfer_gws=[START_GW - 1, START_GW + HORIZON])
    assert not any(n.startswith("No_Transfer_GW") for n in solver.prob.constraints)
    solution = _solve(solver)
    assert solution['transfers'][1]['count'] == 1


def test_partial_roll_spends_one_and_rolls_two():
    """With 3 free transfers and 3 upgrades worth making, asking to roll 2 in GW1
    spends 1 transfer in GW1 and rolls the other 2 (entering GW2 with 3)."""
    free_sol = _solve(_solver(free_transfers=3, extra_replacements=2))
    assert free_sol['transfers'][1]['count'] == 3
    assert free_sol['transfers'][2]['available_transfers'] == 1

    rolled_sol = _solve(_solver(
        rolled_transfers=[(START_GW, 2)],
        free_transfers=3,
        extra_replacements=2,
    ))
    assert rolled_sol['transfers'][1]['count'] == 1
    assert rolled_sol['transfers'][1]['available_transfers'] == 3
    assert rolled_sol['transfers'][2]['available_transfers'] == 3
    assert rolled_sol['transfers'][2]['count'] == 2


def test_rolling_two_in_second_gw_banks_first_gw_honestly():
    """Starting with 1 FT, rolling 2 in GW2 requires making 0 moves in GW1 and 0 in GW2 —
    and the solver must not fake a -4 hit in GW1 to manufacture a second FT."""
    sol = _solve(_solver(
        rolled_transfers=[(START_GW + 1, 2)],
        free_transfers=1,
        extra_replacements=2,
    ))
    assert sol['transfers'][1]['count'] == 0
    assert sol['transfers'][1]['paid_transfers'] == 0
    assert sol['transfers'][2]['available_transfers'] == 2
    assert sol['transfers'][2]['count'] == 0
    assert sol['transfers'][3]['available_transfers'] == 3


def test_wildcard_in_partially_rolled_gw_is_still_allowed():
    solver = _solver(
        rolled_transfers=[(START_GW, 2)],
        free_transfers=3,
        wildcard_available=True,
        force_wildcard_gw=START_GW,
        extra_replacements=2,
    )
    sol = _solve(solver)
    assert sol['transfers'][1]['wildcard_active']
    assert sol['transfers'][1]['count'] == 3
    assert sol['transfers'][1]['paid_transfers'] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
