"""The three-per-club limit binds at transfer time, not continuously.

FPL only validates the limit when you confirm a transfer. If a player you already own
moves to a club you hold three of, you keep all four — indefinitely, for as long as you
leave the squad alone. The solver used to apply `<= 3` to every gameweek
unconditionally, so a legitimately grandfathered squad was rejected outright: dev job
76711f57 (team 124578, four Arsenal players) died with `No feasible solution found`.

That job failed specifically because a Free Hit was forced on the horizon's first
gameweek. Normally GW1-of-horizon can quietly transfer its way to a legal squad before
the constraint bites, which hid the bug everywhere else; a Free Hit pins s and r to zero
for that gameweek and removes the escape hatch.

The rule now: carry the excess for as long as the squad is untouched, but any transfer
must land on a fully compliant squad.

Runnable directly (``python tests/test_club_limit.py``) or under pytest.
"""

import os
import sys

import pandas as pd
import pulp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.solver import MAX_PLAYERS_PER_CLUB, FPLSolver  # noqa: E402

HORIZON = 2
OVER_CLUB = 1  # the club the initial squad holds four of

# A pool laid out by position, wide enough that a legal repair always exists.
POOL = {
    'GK': [101, 102, 103, 104],
    'DEF': [201, 202, 203, 204, 205, 206, 207, 208],
    'MID': [301, 302, 303, 304, 305, 306, 307, 308],
    'FWD': [401, 402, 403, 404, 405],
}
ALL_PLAYERS = [p for pos_players in POOL.values() for p in pos_players]
SQUAD = [101, 102, 201, 202, 203, 204, 205, 301, 302, 303, 304, 305, 401, 402, 403]

# Four squad members from OVER_CLUB, spread across positions so that no single
# positional constraint dictates which one has to go.
OVER_CLUB_MEMBERS = [201, 202, 301, 401]

UNRELATED_DEF = 203  # in the squad, in a club of its own


def _clubs():
    """OVER_CLUB holds four squad members; everyone else gets a private club.

    Private clubs keep OVER_CLUB the only breach in the fixture, so buying any spare
    can never create a second one and a failing assertion has only one possible cause.
    """
    clubs = {p: OVER_CLUB for p in OVER_CLUB_MEMBERS}
    next_club = 10
    for pos_players in POOL.values():
        for p in pos_players:
            if p not in clubs:
                clubs[p] = next_club
                next_club += 1
    return clubs


CLUBS = _clubs()


def _model(squad=None, free_hit_gws=(), pins=()):
    """Flow + composition constraints only, with transfers pinned by the caller.

    Points, money and lineups are all left out: none of them touch the club limit, and
    leaving them out keeps the test free of prediction and price fixtures. The
    constraints under test are the real ones.
    """
    elements = [p for pos_players in POOL.values() for p in pos_players]
    positions = {p: pos for pos, pos_players in POOL.items() for p in pos_players}

    solver = FPLSolver(planning_horizon=HORIZON, budget=1000, start_gw=1)
    solver.players = pd.DataFrame({
        'element': elements,
        'position': [positions[p] for p in elements],
        'team': [CLUBS[p] for p in elements],
    })
    solver.initial_squad = list(SQUAD if squad is None else squad)
    solver.initial_transfers = 1
    solver.free_hit_gws = list(free_hit_gws)
    solver.create_decision_variables()

    solver.prob = pulp.LpProblem("club_limit", pulp.LpMaximize)
    solver.add_squad_flow_constraints()
    solver.add_squad_composition_constraints()

    for name, expr in pins:
        solver.prob += (expr(solver), name)

    # Feasibility only, but PuLP reports None for an objective holding no live variable.
    # A variable pinned to zero is inert and still real.
    solver.prob += pulp.LpVariable("objective_stub", lowBound=0, upBound=0)
    return solver


def _solve(solver):
    solver.prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.LpStatus[solver.prob.status]


def _club_count(solver, t, club=OVER_CLUB):
    return sum(
        1 for p in solver.players['element']
        if CLUBS[p] == club and solver.variables['x'][(p, t)].varValue == 1
    )


def _no_transfers(*gameweeks):
    return [
        (f"No_Transfers_{t}", lambda s, t=t: s.variables['u'][t] == 0)
        for t in gameweeks
    ]


def _club_pinned_at(count, t, club=OVER_CLUB):
    """Pin the club to an exact size.

    The constraint under test is an upper bound, so a forced-transfer plan is free to
    land *below* the limit — with no real objective, CBC often does. Asserting on the
    count CBC happened to pick tests nothing; pinning the count and asking whether a
    plan exists is the actual question.
    """
    members = [p for p in ALL_PLAYERS if CLUBS[p] == club]
    return (
        f"Club_Pinned_{club}_{t}_{count}",
        lambda s: pulp.lpSum([s.variables['x'][(p, t)] for p in members]) == count,
    )


def test_grandfathered_squad_survives_being_left_alone():
    """Four from one club is legal to hold, and holding it is not a plan the solver may
    refuse. This is the shape that killed job 76711f57."""
    solver = _model(pins=_no_transfers(1, 2))
    assert _solve(solver) == "Optimal"
    assert _club_count(solver, 1) == 4
    assert _club_count(solver, 2) == 4


def test_free_hit_on_the_first_gameweek_is_feasible():
    """The exact prod failure. A Free Hit pins s and r to zero, so the held squad cannot
    be repaired in that gameweek — and does not need to be: a Free Hit never touches the
    real squad, and the temp FH squad is a separate MILP (fpl/free_hit.py) that builds a
    compliant squad from scratch. Before the fix this was INFEASIBLE."""
    solver = _model(free_hit_gws=[1])
    assert _solve(solver) == "Optimal"
    assert _club_count(solver, 1) == 4


def test_one_transfer_forces_compliance():
    solver = _model(pins=[
        ("One_Transfer", lambda s: s.variables['u'][1] == 1),
        *_no_transfers(2),
    ])
    assert _solve(solver) == "Optimal"
    assert _club_count(solver, 1) == MAX_PLAYERS_PER_CLUB


def test_an_unrelated_transfer_does_not_get_a_pass():
    """The strict reading of the rule: compliance is required after *any* transfer, not
    only one that touches the offending club. Spending a single transfer on an unrelated
    player leaves the club on four, so there is no legal plan — the solver has to spend
    a transfer on the breach instead."""
    solver = _model(pins=[
        ("One_Transfer", lambda s: s.variables['u'][1] == 1),
        ("Sell_Unrelated", lambda s: s.variables['r'][(UNRELATED_DEF, 1)] == 1),
    ])
    assert _solve(solver) == "Infeasible"


def test_two_transfers_do_not_overshoot():
    """A double transfer caps the club at three, not two.

    This is the trap in the cheaper formulation: a single aggregate row written against
    the transfer count (`held + extra * u[t] <= 3 + extra`) reads a second transfer as a
    second unit of tightening and drives the club below the real limit.
    """
    # Three is still reachable with two transfers spent, one of them on a player who
    # has nothing to do with the breach.
    solver = _model(pins=[
        ("Two_Transfers", lambda s: s.variables['u'][1] == 2),
        ("Sell_Unrelated", lambda s: s.variables['r'][(UNRELATED_DEF, 1)] == 1),
        _club_pinned_at(MAX_PLAYERS_PER_CLUB, 1),
        *_no_transfers(2),
    ])
    assert _solve(solver) == "Optimal"

    # Four is not, so the cap is genuinely enforced rather than merely permitted.
    still_breached = _model(pins=[
        ("Two_Transfers", lambda s: s.variables['u'][1] == 2),
        _club_pinned_at(MAX_PLAYERS_PER_CLUB + 1, 1),
    ])
    assert _solve(still_breached) == "Infeasible"


def test_a_wildcard_sized_rebuild_does_not_overshoot():
    """Same overshoot check at four transfers, where the aggregate form would drive the
    club to zero. A wildcard is just a large number of transfers, so it needs no
    separate handling."""
    solver = _model(pins=[
        ("Wildcard", lambda s: s.variables['wildcard'][1] == 1),
        ("Four_Transfers", lambda s: s.variables['u'][1] == 4),
        _club_pinned_at(MAX_PLAYERS_PER_CLUB, 1),
        *_no_transfers(2),
    ])
    assert _solve(solver) == "Optimal"


def test_excess_cannot_reappear_after_a_repair():
    """Once compliance is restored it holds, with no monotonicity constraint needed:
    climbing back to four takes a transfer in, and that transfer's own row caps the club
    at three."""
    solver = _model(pins=[
        ("One_Transfer_1", lambda s: s.variables['u'][1] == 1),
        ("One_Transfer_2", lambda s: s.variables['u'][2] == 1),
    ])
    assert _solve(solver) == "Optimal"
    assert _club_count(solver, 1) == MAX_PLAYERS_PER_CLUB
    assert _club_count(solver, 2) <= MAX_PLAYERS_PER_CLUB


def test_a_legal_squad_leaves_the_model_untouched():
    """No grandfathering rows for a compliant squad, so the model keeps the exact shape
    it has been solving in prod all season. This is the safety property that matters
    most: the new formulation is unreachable for the overwhelming majority of solves."""
    legal = [p for p in SQUAD if p != 401] + [404]  # drops OVER_CLUB to three
    solver = _model(squad=legal, pins=_no_transfers(1, 2))

    player_club = dict(zip(solver.players['element'], solver.players['team']))
    assert solver._grandfathered_club_excess(player_club) == {}
    assert not [n for n in solver.prob.constraints if 'Grandfathered' in n]
    assert _solve(solver) == "Optimal"
    assert _club_count(solver, 1) == MAX_PLAYERS_PER_CLUB


def test_excess_is_measured_against_the_solver_own_club_map():
    """The excess and the constraint must read club membership from one source, or the
    relaxation lands on a different club than the breach."""
    solver = _model()
    player_club = dict(zip(solver.players['element'], solver.players['team']))
    assert solver._grandfathered_club_excess(player_club) == {OVER_CLUB: 1}


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

    # Not an assertion — a cost report. Rows scale with pool x horizon per breached
    # club, but add no integer variables, which is the part that matters near CBC's
    # feasibility-pump cliff at long horizons.
    grandfathered = _model()
    baseline = _model(squad=[p for p in SQUAD if p != 401] + [404])
    extra = len(grandfathered.prob.constraints) - len(baseline.prob.constraints)
    print(
        f"\nrows: {len(baseline.prob.constraints)} legal squad, "
        f"{len(grandfathered.prob.constraints)} grandfathered (+{extra}) "
        f"for {len(POOL['GK'] + POOL['DEF'] + POOL['MID'] + POOL['FWD'])} players "
        f"x {HORIZON} GWs x 1 breached club"
    )
    sys.exit(1 if failures else 0)
