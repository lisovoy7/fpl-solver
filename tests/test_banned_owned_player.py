"""Excluding a player the manager OWNS must produce a plan that sells him, not
"No feasible solution found".

The old chain: exclusion removed the player from the candidate pool entirely
(create_watchlist, f696dfc — correct for players you don't own), so an owned
player lost his x variable. The squad-size constraint then opened on 14 of 15
owned players with transfers-in forced equal to transfers-out, i.e. the model
could never reach 15 and EVERY scenario was infeasible. Observed on dev
2026-09-06, job f3d1faef ("plan squad replacing Gabriel with VDV",
excluded_players=[4]) — and in prod before 2026-09-05 the same request appeared
to work only because exclusion was silently ignored for owned players.

The fix keeps an owned-and-excluded player in the pool and bans ownership
inside the model instead (banned_players), which forces the sale at the first
legal transfer and credits his selling price.

Runnable directly (``python tests/test_banned_owned_player.py``) or under pytest.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.solver import FPLSolver  # noqa: E402

HORIZON = 3
START_GW = 4

# 15 owned players: 2 GK, 5 DEF, 5 MID, 3 FWD (ids 101..115), plus one
# replacement defender (201) to buy with the freed cash.
GKS = [101, 102]
DEFS = [103, 104, 105, 106, 107]
MIDS = [108, 109, 110, 111, 112]
FWDS = [113, 114, 115]
SQUAD = GKS + DEFS + MIDS + FWDS
REPLACEMENT = 201
BANNED = DEFS[0]  # an owned defender, the "get rid of Gabriel" case

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
            # The replacement is clearly better than the banned player so the
            # swap is also what a free optimum wants; everyone else is even.
            pts = 2.0 if p == BANNED else (6.0 if p == REPLACEMENT else 4.0)
            rows.append({'element': p, 'event': gw, 'predicted_points': pts})
    return pd.DataFrame(rows)


def _solver(banned=None, free_hit_gws=None, forced_lineup=None):
    solver = FPLSolver(
        planning_horizon=HORIZON,
        budget=1000,  # 15 x 50 units + 250 bank — money is not what's under test
        start_gw=START_GW,
        banned_players=banned,
        free_hit_gws=free_hit_gws,
        forced_lineup_players=forced_lineup,
    )
    solver.load_predictions(_predictions())
    solver.players = pd.DataFrame({
        'element': POOL,
        'name': [str(p) for p in POOL],
        'position': [POSITIONS[p] for p in POOL],
        'value': [50] * len(POOL),
        'team': [(i % 10) + 1 for i in range(len(POOL))],  # spread clubs, no 3-limit issue
    })
    solver.set_initial_squad(list(SQUAD), available_transfers=1)
    solver.build_model()
    return solver


def _owned(solution, gw):
    # solution dicts are keyed by internal gameweek index 1..T
    return set(solution['squads'][gw - START_GW + 1])


def test_banning_an_owned_player_is_feasible_and_sells_him():
    solver = _solver(banned=[BANNED])
    assert solver.solve(time_limit=30), "banning an owned player must stay solvable"
    solution = solver.extract_solution()
    for gw in range(START_GW, START_GW + HORIZON):
        assert BANNED not in _owned(solution, gw), f"banned player still owned in GW{gw}"
        assert len(_owned(solution, gw)) == 15, f"squad not full in GW{gw}"
    # The sale happened in the first gameweek: he is out immediately, and the
    # transfer shows up as such.
    assert BANNED in solution['transfers'][1]['out']


def test_banned_player_cannot_be_bought_back():
    """A banned player who is NOT owned can simply never be bought."""
    solver = _solver(banned=[REPLACEMENT])
    assert solver.solve(time_limit=30)
    solution = solver.extract_solution()
    for gw in range(START_GW, START_GW + HORIZON):
        assert REPLACEMENT not in _owned(solution, gw)


def test_free_hit_first_gw_defers_the_sale_not_the_plan():
    """FH gameweeks freeze all transfers, so a ban starting under a Free Hit
    must skip the frozen week (sale lands the week after) instead of demanding
    a transfer the scenario forbids."""
    solver = _solver(banned=[BANNED], free_hit_gws=[START_GW])
    names = set(solver.prob.constraints)
    assert f"Banned_Player_{BANNED}_GW1" not in names, "FH-frozen week must be skipped"
    assert f"Banned_Player_{BANNED}_GW2" in names
    assert solver.solve(time_limit=30), "FH-in-first-GW scenario must stay solvable"
    solution = solver.extract_solution()
    for gw in range(START_GW + 1, START_GW + HORIZON):
        assert BANNED not in _owned(solution, gw)


def test_exclusion_wins_over_a_forced_start():
    """Prod job of 2026-08-24 (team 33695) sent excluded [448] + forced_lineup
    [448]. Contradictory — exclusion wins and the forced start is dropped, so
    the model stays feasible."""
    solver = _solver(banned=[BANNED], forced_lineup=[(BANNED, [START_GW + 1])])
    names = set(solver.prob.constraints)
    assert f"Forced_Lineup_{BANNED}_GW{START_GW + 1}" not in names
    assert solver.solve(time_limit=30)
    solution = solver.extract_solution()
    for gw in range(START_GW, START_GW + HORIZON):
        assert BANNED not in _owned(solution, gw)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
