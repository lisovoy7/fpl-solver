"""Replay the real queued request that exposed the club-limit bug, and audit its plan.

Dev job 76711f57 (team 124578, four Arsenal players, Free Hit forced on GW2 over a
GW2-4 horizon) died with `No feasible solution found`. A synthetic test passing is not
evidence here: the last MILP change that looked proven by one still broke prod, because
the real failing path used proxy predictions and a forced Free Hit that the synthetic
model never exercised.

So this runs the actual `solver_jobs.params` payload through `_optimize_inner`, more
than once — near CBC's feasibility-pump cliff a single pass proves nothing — and then
walks the returned plan asserting the rule the MILP is supposed to encode:

    the squad may carry a grandfathered excess for as long as it is untouched,
    but any gameweek that transfers must end fully compliant.

Checking the plan, not just the status, is the point. A relaxation that was too loose
would still solve happily and quietly hand back an illegal squad.

The audit runs twice, against two club maps, and only one of them decides the verdict.
The **model** map is what `load_player_data` actually gave the MILP; the constraint can
only be held to that. The **live FPL** map is the truth the manager plays under, and
where the two disagree a plan can be compliant as modelled and still illegal in reality.
That is a real defect — but in the freshness of the club data, not in the constraint —
so it is reported as a STALE-CLUB WARNING rather than a failure. Folding it into the
verdict would let a genuine constraint regression hide behind a known data gap.

    python tests/verify_club_limit_replay.py [path/to/params.json] [--runs N] [--horizon H]

Needs network (FPL bootstrap + fixtures). Writes nothing.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server import OptimizeRequest, _optimize_inner  # noqa: E402
from fpl.api import fetch_bootstrap_data  # noqa: E402
from fpl.solver import MAX_PLAYERS_PER_CLUB  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARAMS = os.path.join(HERE, "job_76711f57_params.json")


def _request(params, horizon=None):
    """Rebuild the OptimizeRequest, dropping keys the API model doesn't declare.

    `params` is what trigger_solver stored, which carries a few fields the frontend
    tracks for itself (agent_overrides, forced_chips). Passing them through would just
    be a validation error.
    """
    fields = set(OptimizeRequest.model_fields)
    payload = {k: v for k, v in params.items() if k in fields}
    if horizon is not None:
        payload["planning_horizon"] = horizon
    return OptimizeRequest(**payload), sorted(set(params) - set(payload))


def _model_club_map():
    """element -> club exactly as the MILP sees it.

    `load_player_data` takes club membership from the *predictions* frame, not from
    bootstrap (fpl/solver.py — it joins `normalized_data['player_team_id']`). Pre-season
    that frame is proxy_predictions.csv. This is the map the constraint is enforced
    against, so it is the map the verdict has to be based on: the MILP can only be held
    to the data it was handed.
    """
    snapshot_path = os.path.join(os.path.dirname(HERE), "data", "proxy_predictions.csv")
    frame = pd.read_csv(snapshot_path, usecols=["player_id", "player_team_id"])
    return frame.drop_duplicates("player_id").set_index("player_id")[
        "player_team_id"
    ].to_dict()


def _live_club_map():
    """element -> club per live FPL, which is the truth the manager actually plays under.

    Deliberately audited separately rather than folded into the verdict. Where the two
    maps disagree the plan can be compliant as modelled and still illegal in reality —
    a real defect, but a defect in the freshness of the club data, not in the
    constraint. Conflating them would make this script fail for a reason it is not
    testing, and would hide the constraint regressing behind a known data gap.
    """
    return {e["id"]: e["team"] for e in fetch_bootstrap_data()["elements"]}


def audit(result, initial_squad, clubs):
    """Walk the plan's transfers and report every breach of the rule.

    Returns a list of human-readable problems, empty when the plan is clean.
    """
    problems = []
    squad = set(initial_squad)

    def counts(members):
        out = {}
        for p in members:
            club = clubs.get(p)
            if club is not None:
                out[club] = out.get(club, 0) + 1
        return out

    opening = {c: n for c, n in counts(squad).items() if n > MAX_PLAYERS_PER_CLUB}

    for gw in result["gameweeks"]:
        moved_in = [p["id"] for p in gw["transfers_in"]]
        moved_out = [p["id"] for p in gw["transfers_out"]]
        squad = (squad - set(moved_out)) | set(moved_in)

        if len(squad) != 15:
            problems.append(
                f"GW{gw['gw']}: squad is {len(squad)} players after "
                f"{len(moved_out)} out / {len(moved_in)} in"
            )

        over = {c: n for c, n in counts(squad).items() if n > MAX_PLAYERS_PER_CLUB}
        touched = bool(moved_in or moved_out)

        if touched and over:
            problems.append(
                f"GW{gw['gw']}: {len(moved_in)} transfer(s) made, but club(s) {over} "
                f"still over the limit — a transfer must restore compliance"
            )
        if not touched:
            crept = {c: n for c, n in over.items() if n > opening.get(c, MAX_PLAYERS_PER_CLUB)}
            if crept:
                problems.append(
                    f"GW{gw['gw']}: no transfers, yet club(s) {crept} grew beyond the "
                    f"grandfathered opening count"
                )

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params", nargs="?", default=DEFAULT_PARAMS)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=None,
                    help="override planning_horizon (the job itself used 3)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    with open(args.params) as fh:
        params = json.load(fh)

    req, dropped = _request(params, args.horizon)
    model_clubs = _model_club_map()
    live_clubs = _live_club_map()
    squad = req.squad or []

    print(f"team {req.team_id}  horizon {req.planning_horizon}  "
          f"force_free_hit_gw {req.force_free_hit_gw}  "
          f"time_limit {req.time_limit_per_scenario}s")
    if dropped:
        print(f"dropped non-API keys: {', '.join(dropped)}")

    for label, clubs in (("model", model_clubs), ("live FPL", live_clubs)):
        opening = {}
        for p in squad:
            club = clubs.get(p)
            if club is not None:
                opening[club] = opening.get(club, 0) + 1
        over = {c: n for c, n in opening.items() if n > MAX_PLAYERS_PER_CLUB}
        print(f"  opening squad over the limit per {label}: {over or 'none'}")

    disputed = [p for p in squad if model_clubs.get(p) != live_clubs.get(p)]
    print()

    failures = 0
    stale_warnings = 0
    for run in range(1, args.runs + 1):
        started = time.monotonic()
        try:
            result = asyncio.run(_optimize_inner(req))
        except Exception as exc:  # noqa: BLE001 - reporting any failure is the point
            failures += 1
            print(f"run {run}/{args.runs}  SOLVE FAIL  {type(exc).__name__}: {exc}  "
                  f"({time.monotonic() - started:.0f}s)")
            continue

        elapsed = time.monotonic() - started
        problems = audit(result, squad, model_clubs)
        stale = [p for p in audit(result, squad, live_clubs) if p not in problems]

        verdict = "OK   " if not problems else "AUDIT FAIL"
        print(f"run {run}/{args.runs}  {verdict}  {result['objective']} pts  "
              f"GW{result['start_gw']}-{result['end_gw']}  "
              f"[{result['scenario']}]  ({elapsed:.0f}s)")
        for problem in problems:
            failures += 1
            print(f"    {problem}")
        for problem in stale:
            stale_warnings += 1
            print(f"    STALE-CLUB WARNING  {problem}")

    print()
    print("FAILED — the constraint did not hold on the data the solver was given"
          if failures else
          "every run produced a plan that honours the rule on the model's club map")

    if stale_warnings:
        print()
        print(f"{stale_warnings} stale-club warning(s). These are NOT constraint "
              f"failures: the plan is compliant on the club map the MILP was given and "
              f"illegal only under live FPL, because the two disagree about "
              f"{disputed or 'some players'}.")
        print("The club limit is only ever as correct as the club data. Fixing it means "
              "refreshing player_team_id (or sourcing clubs from bootstrap rather than "
              "a predictions snapshot) — a data-pipeline change, not a model change.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
