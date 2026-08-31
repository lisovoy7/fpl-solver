"""Every owned player must be in the solver's pool, or the model is INFEASIBLE.

Not "suboptimal" — infeasible. `load_player_data()` builds the pool from gw_data;
`Squad_Size` pins ownership at exactly 15 in every gameweek; and the transfer-count
constraints force transfers-in to equal transfers-out. So 14 owned players can never
become 15: every scenario returns INFEASIBLE and the run dies with the opaque "No
feasible solution found".

A squad member with no history is ordinary, not corrupt: a mid-season signing bought
the day FPL registers him has no per-fixture rows until his club next plays, and a sync
gap does the same for anyone. `create_watchlist()` already resurrects must-include
players missing from PREDICTIONS, but it reads their price out of gw_data, so it could
never help someone missing from gw_data too.

Runnable directly (``python tests/test_squad_pool_completeness.py``) or under pytest.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import pulp  # noqa: E402

from fpl import proxy_predict  # noqa: E402
from fpl.solver import FPLSolver  # noqa: E402

NEW_SIGNING = 99


def _bootstrap(include_new_signing: bool = True) -> dict:
    elements = [
        {"id": 1, "first_name": "A", "second_name": "One", "element_type": 3, "team": 4, "now_cost": 70},
        {"id": 2, "first_name": "B", "second_name": "Two", "element_type": 4, "team": 7, "now_cost": 95},
    ]
    if include_new_signing:
        elements.append({
            "id": NEW_SIGNING, "first_name": "New", "second_name": "Signing",
            "element_type": 3, "team": 11, "now_cost": 55,
        })
    return {"elements": elements}


def _gw_data() -> pd.DataFrame:
    """Two players with history. The new signing has none — his club has not played."""
    return pd.DataFrame([
        {"element": 1, "name": "A One", "position": "MID", "value": 70, "GW": 1, "minutes": 90},
        {"element": 2, "name": "B Two", "position": "FWD", "value": 95, "GW": 1, "minutes": 61},
    ])


# --- the guarantee ------------------------------------------------------------


def test_a_squad_member_with_no_history_is_added_to_gw_data():
    filled, still_missing = proxy_predict.ensure_players_present(
        _gw_data(), [1, 2, NEW_SIGNING], _bootstrap()
    )
    assert still_missing == []
    assert NEW_SIGNING in set(filled["element"])

    row = filled[filled["element"] == NEW_SIGNING].iloc[0]
    assert row["position"] == "MID"
    assert row["value"] == 55
    assert row["team"] == 11
    assert row["name"] == "New Signing"


def test_the_synthesized_row_is_marked_as_no_appearance():
    """`minutes=0` so nothing mistakes it for a real game.

    It is also why this must run AFTER generate_predictions: folded in earlier, a
    zero-minute row would drag down the per-player averages predictions are built from.
    """
    filled, _ = proxy_predict.ensure_players_present(
        _gw_data(), [NEW_SIGNING], _bootstrap()
    )
    assert filled[filled["element"] == NEW_SIGNING].iloc[0]["minutes"] == 0


def test_it_is_a_no_op_when_every_player_already_has_history():
    original = _gw_data()
    filled, still_missing = proxy_predict.ensure_players_present(original, [1, 2], _bootstrap())
    assert still_missing == []
    assert len(filled) == len(original)


def test_a_player_bootstrap_does_not_know_is_reported_not_invented():
    """The caller turns this into a clear error naming the player.

    Still a failed run, but "FPL has no player data for squad member 99" is a different
    thing to be told than "No feasible solution found".
    """
    filled, still_missing = proxy_predict.ensure_players_present(
        _gw_data(), [1, NEW_SIGNING], _bootstrap(include_new_signing=False)
    )
    assert still_missing == [NEW_SIGNING]
    assert NEW_SIGNING not in set(filled["element"])


def test_an_empty_gw_data_frame_is_handled():
    filled, still_missing = proxy_predict.ensure_players_present(
        pd.DataFrame(columns=["element"]), [1], _bootstrap()
    )
    assert still_missing == []
    assert set(filled["element"]) == {1}


# --- and the pool the solver actually builds ---------------------------------


def _predictions() -> pd.DataFrame:
    """normalized_data for load_player_data. The new signing is absent here too."""
    return pd.DataFrame([
        {"element": 1, "GW": 1, "player_team_id": 4, "predicted_points": 4.0},
        {"element": 2, "GW": 1, "player_team_id": 7, "predicted_points": 5.0},
    ])


def test_the_player_reaches_the_solver_pool_with_a_club():
    """The club matters: self.players['team'] IS the max-3-per-club map.

    He is absent from predictions, so the left join leaves his club NaN and he would
    quietly escape the club limit. load_player_data falls back to gw_data's own club.
    """
    filled, _ = proxy_predict.ensure_players_present(
        _gw_data(), [1, 2, NEW_SIGNING], _bootstrap()
    )

    solver = FPLSolver(planning_horizon=2, budget=1000, start_gw=1)
    pool = solver.load_player_data(
        filled, _predictions(), player_subset=[1, 2, NEW_SIGNING]
    )

    assert NEW_SIGNING in set(pool["element"])
    assert pool[pool["element"] == NEW_SIGNING].iloc[0]["team"] == 11
    # The players who were there all along keep the club predictions gave them.
    assert pool[pool["element"] == 1].iloc[0]["team"] == 4


def test_a_missing_owner_makes_the_squad_size_constraint_infeasible():
    """The failure this prevents, reduced to its arithmetic.

    x(p,1) = owned + in - out per player, transfers in == transfers out, and squad size
    == N every gameweek. With one owner absent from the pool the sum can only ever
    reach N-1, so no assignment satisfies it — which is why a brand-new signing used to
    kill the entire job rather than merely being left out of the plan.
    """
    squad = [1, 2, NEW_SIGNING]

    def _size_model(pool_ids):
        solver = FPLSolver(planning_horizon=1, budget=1000, start_gw=1)
        solver.players = pd.DataFrame({
            "element": pool_ids,
            "value": [50] * len(pool_ids),
            "position": ["MID"] * len(pool_ids),
            "team": [1] * len(pool_ids),
        })
        solver.initial_squad = squad
        solver.create_decision_variables()
        solver.prob = pulp.LpProblem("size", pulp.LpMaximize)
        solver.add_squad_flow_constraints()
        solver.prob += (
            pulp.lpSum(solver.variables["x"][(p, 1)] for p in pool_ids) == len(squad),
            "Squad_Size_1",
        )
        solver.prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return pulp.LpStatus[solver.prob.status]

    assert _size_model([1, 2]) == "Infeasible"
    assert _size_model(squad) == "Optimal"


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
