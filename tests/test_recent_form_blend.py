"""Tests for the 80/20 recent-form xP blend (capped [1, 10], padded with 2s)."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.predict import (  # noqa: E402
    _recent_form_average,
    generate_predictions,
)

SEASON = "2026-27"

ZERO_COMPONENTS = [
    "assists",
    "bonus",
    "clean_sheets",
    "expected_assists",
    "expected_goals",
    "expected_goals_conceded",
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


def test_recent_form_average_capping_and_padding() -> None:
    # 3+ games: uses last 3, caps to [1, 10]
    assert abs(_recent_form_average([5, -2, 13, 12]) - (1.0 + 10.0 + 10.0) / 3.0) < 1e-9
    # 2 games: pads 3rd game with 2
    assert abs(_recent_form_average([10, 8]) - (10.0 + 8.0 + 2.0) / 3.0) < 1e-9
    # 1 game: caps and pads 2nd & 3rd games with 2
    assert abs(_recent_form_average([16]) - (10.0 + 2.0 + 2.0) / 3.0) < 1e-9
    assert abs(_recent_form_average([0]) - (1.0 + 2.0 + 2.0) / 3.0) < 1e-9
    # 0 games: returns pad value 2.0
    assert abs(_recent_form_average([]) - 2.0) < 1e-9


def test_generate_predictions_blends_80_20_and_ignores_sub_55_cameos() -> None:
    # Player 1: 3 games of 90m with total_points [9, 12, 8] -> capped [9, 10, 8] -> avg 9.0
    # Plus a GW4 10-minute cameo of 1 point that must be ignored (minutes < 55).
    # Player 2: 1 game of 90m with total_points [16] -> capped 10 + pad 2, 2 -> avg 14/3
    played = [
        (1, "ThreeGames", "MID", 101, 1, "2026-08-15T14:00:00Z", 90, 9),
        (1, "ThreeGames", "MID", 102, 2, "2026-08-22T14:00:00Z", 90, 12),
        (1, "ThreeGames", "MID", 103, 3, "2026-08-29T14:00:00Z", 90, 8),
        (1, "ThreeGames", "MID", 104, 4, "2026-09-05T14:00:00Z", 10, 1),
        (2, "OneGame", "MID", 101, 1, "2026-08-15T14:00:00Z", 90, 16),
    ]
    rows = []
    for el, name, pos, fx, gw, ko, mins, pts in played:
        r = {
            "element": el,
            "name": name,
            "position": pos,
            "fixture": fx,
            "minutes": mins,
            "total_points": pts,
            "was_home": True,
            "opponent_team": 2,
            "round": gw,
            "GW": gw,
            "kickoff_time": ko,
        }
        r.update({c: 0.0 for c in ZERO_COMPONENTS})
        rows.append(r)
    gw_data = pd.DataFrame(rows)

    fixtures = pd.DataFrame(
        [
            {"id": 101, "event": 1, "GW": 1, "team_h": 1, "team_a": 2, "kickoff_time": "2026-08-15T14:00:00Z"},
            {"id": 102, "event": 2, "GW": 2, "team_h": 1, "team_a": 2, "kickoff_time": "2026-08-22T14:00:00Z"},
            {"id": 103, "event": 3, "GW": 3, "team_h": 1, "team_a": 2, "kickoff_time": "2026-08-29T14:00:00Z"},
            {"id": 104, "event": 4, "GW": 4, "team_h": 1, "team_a": 2, "kickoff_time": "2026-09-05T14:00:00Z"},
            {"id": 105, "event": 5, "GW": 5, "team_h": 1, "team_a": 2, "kickoff_time": "2026-09-12T14:00:00Z"},
        ]
    )
    team_tiers = pd.DataFrame(
        [
            {"team_id": 1, "team_tier": 3.0, "team_name": "Home FC", "season": SEASON},
            {"team_id": 2, "team_tier": 3.0, "team_name": "Away FC", "season": SEASON},
        ]
    )
    mult_rows = []
    for comp in ZERO_COMPONENTS:
        for pos in ("GK", "DEF", "MID", "FWD"):
            for is_home in (0, 1):
                mult_rows.append(
                    {
                        "component_type": comp,
                        "position": pos,
                        "player_team_tier": 3.0,
                        "opponent_team_tier": 3.0,
                        "is_home": is_home,
                        "multiplier": 1.0,
                    }
                )
    multipliers = pd.DataFrame(mult_rows)

    unblended = generate_predictions(
        gw_data, fixtures, multipliers, team_tiers, SEASON, recent_form_weight=0.0
    )
    blended = generate_predictions(
        gw_data, fixtures, multipliers, team_tiers, SEASON, recent_form_weight=0.20
    )

    unblended_tot = unblended.groupby("element")["predicted_points"].sum().to_dict()
    blended_tot = blended.groupby("element")["predicted_points"].sum().to_dict()

    # Player 1: 0.80 * unblended + 0.20 * 9.0
    want_1 = 0.80 * unblended_tot[1] + 0.20 * 9.0
    assert abs(blended_tot[1] - want_1) < 1e-9, (blended_tot[1], want_1)

    # Player 2: 0.80 * unblended + 0.20 * (14 / 3)
    want_2 = 0.80 * unblended_tot[2] + 0.20 * (14.0 / 3.0)
    assert abs(blended_tot[2] - want_2) < 1e-9, (blended_tot[2], want_2)


if __name__ == "__main__":
    test_recent_form_average_capping_and_padding()
    test_generate_predictions_blends_80_20_and_ignores_sub_55_cameos()
    print("all recent-form blend tests passed")
