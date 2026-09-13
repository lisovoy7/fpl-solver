"""Conceding is a club event, so every player at a club gets the club's xGC.

FPL's per-player `expected_goals_conceded` is the opposition xG generated while
that player was on the pitch. For a 90-minute player it is the whole match; for
anyone substituted it is a fraction. Averaging it per player therefore measured
who stayed on, not who defends: at 2026-27 GW4 every Arsenal player who finished
the Sunderland match read 1.64, Rice and Tzolis (80 minutes) read 1.20, and Ben
White (45) read 0.13 — and since MIN_MINUTES drops a sub-60 appearance from the
window altogether, White's worst match vanished and he was priced for a clean
sheet at 2.78 against Gabriel's 2.02. Same club, same fixture, 27% apart.

These tests pin the fix: the club's match xGC is the MAX across its players in
that fixture (exact, not an estimate — a 90-minute player carries the whole match
by construction), it is taken over the club's own last N matches regardless of who
featured, and every player at the club inherits it.

Runnable directly (``python tests/test_team_clean_sheet.py``) or under pytest.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.predict import (  # noqa: E402
    _team_match_xgc,
    _team_xgc_averages,
    generate_predictions,
)

SEASON = "2026-27"

# One club (team 1) across two played matches, plus an unplayed third fixture to
# predict for. Minutes are the whole point of the fixture set: `ever_present` plays
# every minute, `eighty` is withdrawn at 80, `half` at 45, `benched` never appears.
PLAYED = [
    # (element, name, position, fixture, minutes, personal xgc)
    (1, "ever_present", "DEF", 101, 90, 1.64),
    (2, "eighty", "DEF", 101, 80, 1.20),
    (3, "half", "DEF", 101, 45, 0.13),
    (4, "benched", "DEF", 101, 0, 0.00),
    (5, "keeper", "GK", 101, 90, 1.64),
    (1, "ever_present", "DEF", 102, 90, 0.40),
    (2, "eighty", "DEF", 102, 90, 0.40),
    (3, "half", "DEF", 102, 90, 0.40),
    (4, "benched", "DEF", 102, 0, 0.00),
    (5, "keeper", "GK", 102, 90, 0.40),
]

ZERO_COMPONENTS = [
    "assists",
    "bonus",
    "clean_sheets",
    "expected_assists",
    "expected_goals",
    "goals_conceded",
    "goals_scored",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "saves",
    "yellow_cards",
    "defensive_contribution",
]


def _gw_data():
    rows = []
    for element, name, position, fixture, minutes, xgc in PLAYED:
        row = {
            "element": element,
            "name": name,
            "position": position,
            "fixture": fixture,
            "minutes": minutes,
            "expected_goals_conceded": xgc,
            "was_home": True,
            "opponent_team": 2,
            "round": 1 if fixture == 101 else 2,
            "GW": 1 if fixture == 101 else 2,
            "kickoff_time": "2026-08-15T14:00:00Z" if fixture == 101 else "2026-08-22T14:00:00Z",
        }
        row.update({c: 0.0 for c in ZERO_COMPONENTS})
        rows.append(row)
    return pd.DataFrame(rows)


def _fixtures():
    return pd.DataFrame(
        [
            {"id": 101, "event": 1, "GW": 1, "team_h": 1, "team_a": 2,
             "kickoff_time": "2026-08-15T14:00:00Z"},
            {"id": 102, "event": 2, "GW": 2, "team_h": 1, "team_a": 2,
             "kickoff_time": "2026-08-22T14:00:00Z"},
            {"id": 103, "event": 3, "GW": 3, "team_h": 1, "team_a": 2,
             "kickoff_time": "2026-08-29T14:00:00Z"},
        ]
    )


def _team_tiers():
    return pd.DataFrame(
        [
            {"team_id": 1, "team_tier": 3.0, "team_name": "Home FC", "season": SEASON},
            {"team_id": 2, "team_tier": 3.0, "team_name": "Away FC", "season": SEASON},
        ]
    )


def _multipliers():
    """Flat 1.0 everywhere, so a normalised average is just the raw average."""
    rows = []
    for component in ZERO_COMPONENTS + ["expected_goals_conceded"]:
        for position in ("GK", "DEF", "MID", "FWD"):
            for is_home in (0, 1):
                rows.append(
                    {
                        "component_type": component,
                        "position": position,
                        "player_team_tier": 3.0,
                        "opponent_team_tier": 3.0,
                        "is_home": is_home,
                        "multiplier": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_club_match_xgc_is_the_full_match():
    """The club's figure is the 90-minute value, not any substitute's fraction."""
    matches = _team_match_xgc(_gw_data(), _fixtures())
    by_fixture = matches.set_index("fixture")["raw_xgc"].to_dict()
    assert by_fixture[101] == 1.64, by_fixture
    assert by_fixture[102] == 0.40, by_fixture
    assert len(matches) == 2, "one row per club-match, not per player"


def test_club_average_ignores_who_played():
    """A club has one xGC record; minutes never enter it."""
    matches = _team_match_xgc(_gw_data(), _fixtures())
    averages = _team_xgc_averages(matches, _multipliers(), _team_tiers())
    for position in ("GK", "DEF", "MID", "FWD"):
        row = averages[
            (averages["player_team_id"] == 1) & (averages["position"] == position)
        ]
        assert len(row) == 1, position
        assert row["team_xgc_games"].iloc[0] == 2, position
        assert abs(row["avg_norm_team_xgc"].iloc[0] - 1.02) < 1e-9, position


def test_teammates_share_a_clean_sheet_prediction():
    """The bug this exists for: same club, same fixture, same defensive outlook."""
    predictions = generate_predictions(
        _gw_data(), _fixtures(), _multipliers(), _team_tiers(), SEASON
    )
    defensive = predictions[
        predictions["component_type"].isin(["clean_sheet", "conceded_goals"])
    ]
    for component, group in defensive.groupby("component_type"):
        defenders = group[group["position"] == "DEF"]
        assert len(defenders) >= 3, component
        spread = defenders["predicted_points"].max() - defenders["predicted_points"].min()
        assert spread < 1e-9, f"{component} differs between teammates: {spread}"

    # And it lands on the club's own record, not on the best-attended player's.
    # avg club xGC 1.02, flat multipliers, so P(CS) = exp(-1.02) and DEF gets 4x.
    import math

    clean_sheet = defensive[
        (defensive["component_type"] == "clean_sheet") & (defensive["position"] == "DEF")
    ]["predicted_points"].iloc[0]
    assert abs(clean_sheet - 4.0 * math.exp(-1.02)) < 1e-9, clean_sheet


def test_a_player_who_never_featured_still_gets_his_clubs_record():
    """`benched` has no 60-minute game at all, so he has no xGC of his own."""
    predictions = generate_predictions(
        _gw_data(), _fixtures(), _multipliers(), _team_tiers(), SEASON
    )
    names = set(predictions["name"].unique())
    # He only reaches the prediction frame if he has some 60+ minute history;
    # `half` is the case that matters either way — 45 minutes in one match and 90
    # in the other, so his own window holds one game and the club's holds two.
    assert "half" in names
    half = predictions[
        (predictions["name"] == "half")
        & (predictions["component_type"] == "clean_sheet")
    ]["predicted_points"].iloc[0]
    ever = predictions[
        (predictions["name"] == "ever_present")
        & (predictions["component_type"] == "clean_sheet")
    ]["predicted_points"].iloc[0]
    assert abs(half - ever) < 1e-9, (half, ever)


if __name__ == "__main__":
    test_club_match_xgc_is_the_full_match()
    test_club_average_ignores_who_played()
    test_teammates_share_a_clean_sheet_prediction()
    test_a_player_who_never_featured_still_gets_his_clubs_record()
    print("all team clean-sheet tests passed")
