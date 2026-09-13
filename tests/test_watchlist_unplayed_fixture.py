"""
A club that has not played every gameweek in the window must not have its players
charged for the matches it did not play.

The bug this pins: `min_hist_games` was `ceil(window_size * min_hist_pct)` — one number
for the whole league. On 2026-09-13, with seven of ten GW4 fixtures played, the window was
4 for everyone, so a player whose own club had not kicked off yet needed 3 appearances out
of the 2 matches he had had the chance to make. 19 real players were dropped from the
candidate pool that way (Foden, Cherki, Dalot, Rashford, O'Reilly, Mainoo), hours before
their match began. A blank gameweek and a postponement did the same thing all season.
"""

import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from fpl.watchlist import create_watchlist

LONG_AGO = datetime.now(timezone.utc) - timedelta(days=3)
KICKED_OFF_SOON = datetime.now(timezone.utc) + timedelta(hours=4)


def _rows(element, gws, minutes, kickoff=LONG_AGO):
    return [
        {
            "element": element,
            "GW": gw,
            "fixture": 100 + gw,
            "minutes": m,
            "value": 50,
            "kickoff_time": kickoff.isoformat(),
        }
        for gw, m in zip(gws, minutes)
    ]


def _predictions(elements):
    return pd.DataFrame(
        [
            {"element": e, "predicted_points": 5.0, "hist_games": 3, "position": "MID"}
            for e in elements
        ]
    )


def test_club_yet_to_play_does_not_charge_its_players_a_missed_start():
    """Mainoo: started 2 of the 3 his club played, with GW4 still to kick off."""
    # A club that HAS played all four gameweeks, and one that has only played three.
    played_all = _rows(1, [1, 2, 3, 4], [90, 90, 0, 0])
    # The GW4 row exists (FPL publishes it on matchday) but the match is hours away.
    yet_to_play = _rows(2, [1, 2, 3], [23, 79, 83]) + _rows(
        2, [4], [0], kickoff=KICKED_OFF_SOON
    )
    gw_data = pd.DataFrame(played_all + yet_to_play)

    watchlist = create_watchlist(
        _predictions([1, 2]), gw_data, min_hist_pct=0.6, max_hist_window=6, min_minutes=60
    )

    # Player 2 made 2 appearances from 3 played fixtures: ceil(3 * 0.6) = 2. He qualifies.
    assert 2 in watchlist, "a player whose club has not played GW4 was charged for it"
    # Player 1 made 2 from 4: ceil(4 * 0.6) = 3. He does not.
    assert 1 not in watchlist


def test_blank_gameweek_is_not_a_missed_start():
    """A club with no fixture in a gameweek has no opportunity to be missed."""
    # Club plays GW1, 2 and 4 — GW3 is a blank. Player starts all three.
    gw_data = pd.DataFrame(_rows(7, [1, 2, 4], [90, 90, 90]))
    watchlist = create_watchlist(
        _predictions([7]), gw_data, min_hist_pct=0.6, max_hist_window=6, min_minutes=60
    )
    assert 7 in watchlist, "a blank gameweek was counted as a match the player missed"


def test_full_window_club_is_unchanged():
    """The ordinary case must keep behaving exactly as before."""
    regular = _rows(3, [1, 2, 3, 4], [90, 90, 90, 0])   # 3 of 4 -> needs 3 -> in
    fringe = _rows(4, [1, 2, 3, 4], [90, 0, 0, 15])     # 1 of 4 -> needs 3 -> out
    watchlist = create_watchlist(
        _predictions([3, 4]),
        pd.DataFrame(regular + fringe),
        min_hist_pct=0.6,
        max_hist_window=6,
        min_minutes=60,
    )
    assert 3 in watchlist
    assert 4 not in watchlist


def test_a_substitute_who_plays_the_threshold_counts():
    """The filter is minutes-based, not starts-based: coming off the bench counts."""
    sub = [
        {"element": 9, "GW": gw, "fixture": 200 + gw, "minutes": 50, "value": 50,
         "starts": 0, "kickoff_time": LONG_AGO.isoformat()}
        for gw in (1, 2, 3, 4)
    ]
    watchlist = create_watchlist(
        _predictions([9]), pd.DataFrame(sub),
        min_hist_pct=0.6, max_hist_window=6, min_minutes=45,
    )
    assert 9 in watchlist, "50 minutes off the bench should clear a 45-minute threshold"
