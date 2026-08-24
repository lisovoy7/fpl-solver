"""A blank gameweek is a club with no fixture, not a player with no forecast.

Point forecasts are only built for players who have had a 60+ minute appearance in the
current season (`fpl/predict.py` filters on `MIN_MINUTES`). Two gameweeks into a season
that is a small minority of the game — 188 of 609 players on 2026-08-24. Every other
player has no forecast rows at all.

The blank-gameweek rule used to read "no forecast for this player in this gameweek" as
"his club has no fixture that gameweek" and ban him from starting. Two things followed:

  1. Fit players a manager actually owned were silently unable to start for the whole
     horizon, so the plan sold them. Prod team 33695 was told to dump Joao Pedro, Konsa
     and Caicedo in GW2 for exactly this reason.
  2. `forced_lineup` on any of them produced "must start" and "cannot start" for the same
     player and gameweek. That is not a soft preference the MILP can trade off — it is a
     contradiction, and it failed the entire job. Prod job 0220fa29 died in 16 seconds
     that way, on nothing worse than "start Dan Burn in GW7" while Burn was injured and
     had not yet played a minute.

The rule now asks the club, via the fixture map handed to the solver. A player nobody can
forecast is worth 0 points, which the objective already handles; he is not unavailable.

Runnable directly (``python tests/test_blank_gameweek_detection.py``) or under pytest.
"""

import os
import sys

import pandas as pd
import pulp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.solver import FPLSolver  # noqa: E402

HORIZON = 3
START_GW = 1

FORECAST_PLAYER = 301   # ordinary: has forecasts for every gameweek
UNFORECAST_PLAYER = 302  # the Dan Burn case: in the pool, no forecasts anywhere
BLANKING_PLAYER = 303   # has forecasts, but his club sits out GW2

CLUB_NORMAL = 1
CLUB_BLANKING = 2

CLUBS = {
    FORECAST_PLAYER: CLUB_NORMAL,
    UNFORECAST_PLAYER: CLUB_NORMAL,
    BLANKING_PLAYER: CLUB_BLANKING,
}
BLANK_GW = 2

CLUB_GAMEWEEKS = {
    CLUB_NORMAL: {1, 2, 3},
    CLUB_BLANKING: {1, 3},  # no fixture in GW2
}


def _predictions():
    """Forecast rows for everyone except UNFORECAST_PLAYER, who has none at all.

    BLANKING_PLAYER is missing GW2 the way a real blank looks in this data: his club has
    no fixture, so no (player, gameweek) row is ever generated for him.
    """
    rows = []
    for gw in range(START_GW, START_GW + HORIZON):
        rows.append({'element': FORECAST_PLAYER, 'event': gw, 'predicted_points': 5.0})
        if gw != BLANK_GW:
            rows.append({'element': BLANKING_PLAYER, 'event': gw, 'predicted_points': 5.0})
    return pd.DataFrame(rows)


def _solver(with_club_map=True, forced_lineup=None):
    solver = FPLSolver(
        planning_horizon=HORIZON,
        budget=1000,
        start_gw=START_GW,
        forced_lineup_players=forced_lineup,
        player_clubs=CLUBS if with_club_map else None,
        club_gameweeks=CLUB_GAMEWEEKS if with_club_map else None,
    )
    solver.load_predictions(_predictions())
    solver.players = pd.DataFrame({
        'element': list(CLUBS),
        'name': [str(p) for p in CLUBS],
        'position': ['MID'] * len(CLUBS),
        'value': [50] * len(CLUBS),
        'team': [CLUBS[p] for p in CLUBS],
    })
    solver.prob = pulp.LpProblem("bgw", pulp.LpMaximize)
    solver.create_decision_variables()
    # Also what populates expected_points, which both rules under test read.
    solver.create_objective()
    # Squad, money and lineup rules are left out: none of them touch blank-gameweek
    # detection, and leaving them out keeps the fixture free of prices and a 15-man squad.
    solver.add_advanced_constraints()
    return solver


def _constraint_names(solver):
    return set(solver.prob.constraints)


def _internal(gw):
    return gw - START_GW + 1


def test_unforecast_player_is_not_banned_from_starting():
    """The core of it. Burn has no forecasts, but Newcastle play every week."""
    names = _constraint_names(_solver())
    for gw in range(START_GW, START_GW + HORIZON):
        assert f"BGW_No_Start_{UNFORECAST_PLAYER}_GW{gw}" not in names, (
            f"player with no forecast was banned from starting in GW{gw}"
        )


def test_real_blank_gameweek_is_still_enforced():
    """The rule still has to do its actual job: a club that isn't playing can't field
    anyone. Only GW2 is banned, and only for the club that sits it out."""
    names = _constraint_names(_solver())
    assert f"BGW_No_Start_{BLANKING_PLAYER}_GW{BLANK_GW}" in names
    assert f"BGW_No_Captain_{BLANKING_PLAYER}_GW{BLANK_GW}" in names
    for gw in (1, 3):
        assert f"BGW_No_Start_{BLANKING_PLAYER}_GW{gw}" not in names
    assert f"BGW_No_Start_{FORECAST_PLAYER}_GW{BLANK_GW}" not in names


def test_forcing_an_unforecast_player_is_feasible():
    """Prod job 0220fa29, reduced. 'Start Burn in GW7' must produce a plan, not a
    contradiction."""
    solver = _solver(forced_lineup=[(UNFORECAST_PLAYER, [START_GW + 1])])
    solver.prob.solve(pulp.PULP_CBC_CMD(msg=0))
    assert pulp.LpStatus[solver.prob.status] == "Optimal"
    assert solver.variables['y'][(UNFORECAST_PLAYER, _internal(START_GW + 1))].varValue == 1


def test_forcing_a_start_in_a_real_blank_is_dropped_not_fatal():
    """One impossible instruction should cost the caller that instruction, not the plan.
    Callers that can talk back (api_server, and trigger_solver above it) should refuse
    before this point; this is the backstop that keeps a stray one from failing the run."""
    solver = _solver(forced_lineup=[(BLANKING_PLAYER, [BLANK_GW])])
    names = _constraint_names(solver)
    assert f"Forced_Lineup_{BLANKING_PLAYER}_GW{BLANK_GW}" not in names
    solver.prob.solve(pulp.PULP_CBC_CMD(msg=0))
    assert pulp.LpStatus[solver.prob.status] == "Optimal"


def test_without_a_club_map_nothing_changes():
    """No fixture map supplied means the old forecast-presence test, so a caller that
    hasn't been updated behaves exactly as it did before."""
    names = _constraint_names(_solver(with_club_map=False))
    for gw in range(START_GW, START_GW + HORIZON):
        assert f"BGW_No_Start_{UNFORECAST_PLAYER}_GW{gw}" in names


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
