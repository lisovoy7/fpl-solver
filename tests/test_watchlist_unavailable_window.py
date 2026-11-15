"""
Unavailable fixtures (`was_unavailable = True` or an early red card under `min_minutes`)
are skipped when building each player's 6-game eligibility window, while healthy 0-minute
benchings still count against him.

Covers the exact cases agreed with the user:
  - Bobby Thomas (173): 90, 77, 90, 90, 90, 90, 90, 59 in GW1-8, then 0, 0 in GW9-10
    while injured -> last 6 available are GW3-8 (6/6 starts, no absent streak) -> eligible.
  - Anton Stach (335): 90, 90, 90, 87, 90, 90, 90 in GW1-7, then 24, 25, 0 in GW8-10
    while injured -> last 6 available are GW2-7 (6/6 starts, no absent streak) -> eligible.
  - Igor Thiago (106): 82, 90, 90, 90, 90, 90, 90, 90, 90 in GW1-9, then 22' + red card
    in GW10 and 0, 0, 0 while suspended in GW11-13 -> last 6 available are GW4-9 -> eligible.
  - João Pedro (165): 90, 90, 90, 90 in GW1-4, 0 (injured) in GW5, 71, 90 in GW6-7,
    0, 0, 0 (injured) in GW8-10 -> last 6 available are GW1-4, 6, 7 (6/6 starts) -> eligible.
  - Rayan Cherki (399): 27, 81, 65, 45, 84, 90, 0 (injured), 45, 35, 90 -> even with GW7
    skipped as injured, his last 6 available (GW4-6, 8-10) only have 3 of 6 starts -> NOT eligible.
  - A player whose every row is `was_unavailable = True` (0 available fixtures) has 0
    starts against a minimum requirement of 1 -> NOT eligible.
"""

from datetime import datetime, timedelta, timezone
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from fpl.watchlist import create_watchlist, zero_ineligible_squad_predictions  # noqa: E402

LONG_AGO = datetime.now(timezone.utc) - timedelta(days=3)

THOMAS = 173
STACH = 335
THIAGO = 106
JOAO_PEDRO = 165
CHERKI = 399
BENCHED_FIT = 901
NEVER_FIT = 902


def _make_rows(element, entries):
    """entries: list of (gw, minutes, was_unavailable, red_cards)"""
    return [
        {
            "element": element,
            "GW": gw,
            "fixture": 1000 * element + gw,
            "minutes": mins,
            "was_unavailable": unavail,
            "red_cards": reds,
            "value": 50,
            "kickoff_time": (LONG_AGO - timedelta(days=7 * (20 - gw))).isoformat(),
        }
        for gw, mins, unavail, reds in entries
    ]


def _predictions(elements):
    return pd.DataFrame(
        [
            {"element": e, "event": 12, "predicted_points": 4.5, "hist_games": 6, "position": "MID"}
            for e in elements
        ]
    )


def test_thomas_and_stach_remain_eligible_and_keep_predicted_points():
    thomas_rows = _make_rows(
        THOMAS,
        [
            (1, 90, False, 0),
            (2, 77, False, 0),
            (3, 90, False, 0),
            (4, 90, False, 0),
            (5, 90, False, 0),
            (6, 90, False, 0),
            (7, 90, False, 0),
            (8, 59, False, 0),
            (9, 0, True, 0),
            (10, 0, True, 0),
        ],
    )
    stach_rows = _make_rows(
        STACH,
        [
            (1, 90, False, 0),
            (2, 90, False, 0),
            (3, 90, False, 0),
            (4, 87, False, 0),
            (5, 90, False, 0),
            (6, 90, False, 0),
            (7, 90, False, 0),
            (8, 24, True, 0),
            (9, 25, True, 0),
            (10, 0, True, 0),
        ],
    )
    benched_rows = _make_rows(
        BENCHED_FIT,
        [
            (5, 90, False, 0),
            (6, 90, False, 0),
            (7, 90, False, 0),
            (8, 90, False, 0),
            (9, 0, False, 0),
            (10, 0, False, 0),
        ],
    )

    gw_data = pd.DataFrame(thomas_rows + stach_rows + benched_rows)
    preds = _predictions([THOMAS, STACH, BENCHED_FIT])

    eligible = create_watchlist(
        preds,
        gw_data,
        min_hist_pct=0.6,
        max_hist_window=6,
        min_minutes=55,
        absent_streak_games=2,
        max_gw=10,
    )
    assert THOMAS in eligible, "Bobby Thomas should be eligible via GW3-8 fit window"
    assert STACH in eligible, "Anton Stach should be eligible via GW2-7 fit window"
    assert BENCHED_FIT not in eligible, "Healthy player benched in GW9-10 must still be dropped"

    adjusted, ineligible = zero_ineligible_squad_predictions(
        preds, [THOMAS, STACH, BENCHED_FIT], eligible
    )
    assert ineligible == [BENCHED_FIT]
    assert float(adjusted[adjusted["element"] == THOMAS]["predicted_points"].iloc[0]) == 4.5
    assert float(adjusted[adjusted["element"] == STACH]["predicted_points"].iloc[0]) == 4.5
    assert float(adjusted[adjusted["element"] == BENCHED_FIT]["predicted_points"].iloc[0]) == 0.0


def test_thiago_joao_pedro_cherki_and_never_fit():
    thiago_rows = _make_rows(
        THIAGO,
        [
            (1, 82, False, 0),
            (2, 90, False, 0),
            (3, 90, False, 0),
            (4, 90, False, 0),
            (5, 90, False, 0),
            (6, 90, False, 0),
            (7, 90, False, 0),
            (8, 90, False, 0),
            (9, 90, False, 0),
            (10, 22, False, 1),  # straight red at 22' — excluded even if was_unavailable=False
            (11, 0, True, 0),    # suspended
            (12, 0, True, 0),    # suspended
            (13, 0, True, 0),    # suspended
        ],
    )
    joao_pedro_rows = _make_rows(
        JOAO_PEDRO,
        [
            (1, 90, False, 0),
            (2, 90, False, 0),
            (3, 90, False, 0),
            (4, 90, False, 0),
            (5, 0, True, 0),
            (6, 71, False, 0),
            (7, 90, False, 0),
            (8, 0, True, 0),
            (9, 0, True, 0),
            (10, 0, True, 0),
        ],
    )
    cherki_rows = _make_rows(
        CHERKI,
        [
            (1, 27, False, 0),
            (2, 81, False, 0),
            (3, 65, False, 0),
            (4, 45, False, 0),
            (5, 84, False, 0),
            (6, 90, False, 0),
            (7, 0, True, 0),
            (8, 45, False, 0),
            (9, 35, False, 0),
            (10, 90, False, 0),
        ],
    )
    never_fit_rows = _make_rows(
        NEVER_FIT,
        [(gw, 0, True, 0) for gw in range(1, 11)],
    )

    gw_data = pd.DataFrame(thiago_rows + joao_pedro_rows + cherki_rows + never_fit_rows)
    preds = _predictions([THIAGO, JOAO_PEDRO, CHERKI, NEVER_FIT])

    eligible = create_watchlist(
        preds,
        gw_data,
        min_hist_pct=0.6,
        max_hist_window=6,
        min_minutes=55,
        absent_streak_games=2,
        max_gw=13,
    )
    assert THIAGO in eligible, "Igor Thiago should be eligible after his 3-match suspension"
    assert JOAO_PEDRO in eligible, "João Pedro should be eligible across his 6 fit appearances"
    assert CHERKI not in eligible, "Cherki has only 3 starts in his last 6 fit matches without extra_players"
    assert NEVER_FIT not in eligible, "A player with 0 available matches must not pass"

    # When Cherki (399) is included in standing extra_players (must_include), he passes
    # eligible_watchlist and keeps his full predicted points.
    eligible_with_cherki = create_watchlist(
        preds,
        gw_data,
        min_hist_pct=0.6,
        max_hist_window=6,
        min_minutes=55,
        absent_streak_games=2,
        must_include=[CHERKI],
        max_gw=13,
    )
    assert CHERKI in eligible_with_cherki, "Cherki in extra_players must be in eligible_watchlist"
    adjusted, ineligible = zero_ineligible_squad_predictions(
        preds, [CHERKI], eligible_with_cherki
    )
    assert ineligible == []
    assert float(adjusted[adjusted["element"] == CHERKI]["predicted_points"].iloc[0]) == 4.5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
