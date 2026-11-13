"""
Suspended players (`status == 's'`) must have `predicted_points = 0.0` in
`player_predictions` for every gameweek up through their suspension return date,
while keeping their normal predicted points from their return gameweek onwards —
and injured/doubtful players (`status == 'i'` / `'d'`) must NOT be zeroed in the
table.

Standalone:
    python tests/test_suspended_zero_points.py
"""

from datetime import datetime, timezone
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "")

import pandas as pd  # noqa: E402

from api_server import _suspended_until_gw, _zero_suspended_predictions  # noqa: E402

NOW = datetime(2026, 11, 13, 12, 0, 0, tzinfo=timezone.utc)

THIAGO = 106   # Brentford (team 4), Suspended until 5 Dec -> misses GW11, 12, 13; back GW14 (5 Dec)
MATETA = 223   # Crystal Palace (team 7), Suspended until 2 Dec -> misses GW11, 12; back GW13 (2 Dec)
DOWMAN = 20    # Arsenal (team 1), Suspended until 29 Nov -> misses GW11; back GW12 (29 Nov)
INJURED = 208  # Injured with return date -> must NOT be zeroed in player_predictions
DOUBTFUL = 356 # Doubtful 75% -> must NOT be zeroed in player_predictions
FIT = 411      # Fit -> untouched


BOOTSTRAP = {
    "events": [
        {"id": 11, "deadline_time": "2026-11-21T11:00:00Z"},
        {"id": 12, "deadline_time": "2026-11-27T18:30:00Z"},
        {"id": 13, "deadline_time": "2026-12-01T18:30:00Z"},
        {"id": 14, "deadline_time": "2026-12-05T11:00:00Z"},
        {"id": 15, "deadline_time": "2026-12-12T13:30:00Z"},
    ],
    "elements": [
        {"id": THIAGO, "team": 4, "status": "s", "news": "Suspended until 5 Dec"},
        {"id": MATETA, "team": 7, "status": "s", "news": "Suspended until 2 Dec"},
        {"id": DOWMAN, "team": 1, "status": "s", "news": "Suspended until 29 Nov"},
        {"id": INJURED, "team": 7, "status": "i", "news": "Groin injury - Expected back 5 Dec"},
        {"id": DOUBTFUL, "team": 1, "status": "d", "news": "Knock - 75% chance of playing"},
        {"id": FIT, "team": 1, "status": "a", "news": ""},
    ],
}

FIXTURES = pd.DataFrame([
    # Brentford (team 4)
    {"id": 101, "event": 11, "team_h": 4, "team_a": 10, "kickoff_time": "2026-11-21T15:00:00Z"},
    {"id": 102, "event": 12, "team_h": 11, "team_a": 4, "kickoff_time": "2026-11-29T14:00:00Z"},
    {"id": 103, "event": 13, "team_h": 4, "team_a": 12, "kickoff_time": "2026-12-02T19:30:00Z"},
    {"id": 104, "event": 14, "team_h": 13, "team_a": 4, "kickoff_time": "2026-12-05T15:00:00Z"},
    {"id": 105, "event": 15, "team_h": 4, "team_a": 14, "kickoff_time": "2026-12-12T15:00:00Z"},
    # Crystal Palace (team 7)
    {"id": 201, "event": 11, "team_h": 7, "team_a": 10, "kickoff_time": "2026-11-21T15:00:00Z"},
    {"id": 202, "event": 12, "team_h": 11, "team_a": 7, "kickoff_time": "2026-11-28T15:00:00Z"},
    {"id": 203, "event": 13, "team_h": 7, "team_a": 12, "kickoff_time": "2026-12-02T20:00:00Z"},
    {"id": 204, "event": 14, "team_h": 13, "team_a": 7, "kickoff_time": "2026-12-06T14:00:00Z"},
    {"id": 205, "event": 15, "team_h": 7, "team_a": 14, "kickoff_time": "2026-12-12T15:00:00Z"},
    # Arsenal (team 1)
    {"id": 301, "event": 11, "team_h": 1, "team_a": 10, "kickoff_time": "2026-11-22T16:30:00Z"},
    {"id": 302, "event": 12, "team_h": 11, "team_a": 1, "kickoff_time": "2026-11-29T16:30:00Z"},
    {"id": 303, "event": 13, "team_h": 1, "team_a": 12, "kickoff_time": "2026-12-03T20:15:00Z"},
    {"id": 304, "event": 14, "team_h": 13, "team_a": 1, "kickoff_time": "2026-12-05T17:30:00Z"},
    {"id": 305, "event": 15, "team_h": 1, "team_a": 14, "kickoff_time": "2026-12-13T14:00:00Z"},
])


def _predictions():
    rows = []
    for gw in [11, 12, 13, 14, 15]:
        rows.extend([
            {"element": THIAGO, "event": gw, "predicted_points": 3.3},
            {"element": MATETA, "event": gw, "predicted_points": 4.5},
            {"element": DOWMAN, "event": gw, "predicted_points": 2.0},
            {"element": INJURED, "event": gw, "predicted_points": 3.8},
            {"element": DOUBTFUL, "event": gw, "predicted_points": 4.1},
            {"element": FIT, "event": gw, "predicted_points": 6.5},
        ])
    return pd.DataFrame(rows)


def test_suspended_until_gw_matches_return_date_and_only_suspensions():
    got = _suspended_until_gw(BOOTSTRAP, FIXTURES, now=NOW)
    assert got == {
        THIAGO: 13,
        MATETA: 12,
        DOWMAN: 11,
    }, f"unexpected suspended_until_gw: {got}"


def test_zero_suspended_predictions_zeros_missed_gws_and_keeps_return_gws():
    preds = _predictions()
    out, applied = _zero_suspended_predictions(preds, BOOTSTRAP, FIXTURES, now=NOW)

    assert applied == {THIAGO: 13, MATETA: 12, DOWMAN: 11}

    # Igor Thiago: 0 in GW11-13, 3.3 in GW14-15
    thiago_by_gw = dict(zip(
        out[out["element"] == THIAGO]["event"],
        out[out["element"] == THIAGO]["predicted_points"],
    ))
    assert thiago_by_gw == {11: 0.0, 12: 0.0, 13: 0.0, 14: 3.3, 15: 3.3}

    # Mateta: 0 in GW11-12, 4.5 in GW13-15
    mateta_by_gw = dict(zip(
        out[out["element"] == MATETA]["event"],
        out[out["element"] == MATETA]["predicted_points"],
    ))
    assert mateta_by_gw == {11: 0.0, 12: 0.0, 13: 4.5, 14: 4.5, 15: 4.5}

    # Dowman: 0 in GW11, 2.0 in GW12-15
    dowman_by_gw = dict(zip(
        out[out["element"] == DOWMAN]["event"],
        out[out["element"] == DOWMAN]["predicted_points"],
    ))
    assert dowman_by_gw == {11: 0.0, 12: 2.0, 13: 2.0, 14: 2.0, 15: 2.0}

    # Injured, doubtful, and fit players are completely untouched
    for pid, expected_xp in ((INJURED, 3.8), (DOUBTFUL, 4.1), (FIT, 6.5)):
        vals = out[out["element"] == pid]["predicted_points"].tolist()
        assert vals == [expected_xp] * 5, f"player {pid} altered: {vals}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
