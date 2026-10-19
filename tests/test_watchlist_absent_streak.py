"""
A player who banked his starts early and has sat out his club's last two matches is not
a candidate, however the share-of-window test reads.

The case this pins (GW7, 2026-10-19): Awoniyi, Brobbey, Fofana and Rashford each had
exactly the 4 starts required from 6, all in GW2-5, then 0 minutes in both of the last
two — and each was in the top 5 of his price bucket.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from fpl.watchlist import create_watchlist

LONG_AGO = datetime.now(timezone.utc) - timedelta(days=3)


def _rows(element, gws, minutes):
    return [
        {"element": element, "GW": gw, "fixture": 100 * element + gw, "minutes": m,
         "value": 50, "kickoff_time": (LONG_AGO - timedelta(days=7 * (10 - gw))).isoformat()}
        for gw, m in zip(gws, minutes)
    ]


def _predictions(elements):
    return pd.DataFrame(
        [{"element": e, "predicted_points": 5.0, "hist_games": 4, "position": "MID"}
         for e in elements]
    )


def _watchlist(rows, elements, **kw):
    return create_watchlist(
        _predictions(elements), pd.DataFrame(rows),
        min_hist_pct=0.6, max_hist_window=6, min_minutes=45, **kw,
    )


GWS = [2, 3, 4, 5, 6, 7]


def test_benched_for_the_last_two_is_dropped():
    benched = _rows(1, GWS, [90, 90, 90, 70, 0, 0])   # 4 of 6 -> clears the share test
    assert 1 not in _watchlist(benched, [1])


def test_one_miss_is_not_a_streak():
    rested = _rows(2, GWS, [90, 90, 0, 90, 90, 0])
    cameo_then_start = _rows(3, GWS, [90, 90, 90, 0, 90, 20])
    wl = _watchlist(rested + cameo_then_start, [2, 3])
    assert 2 in wl and 3 in wl


def test_cameos_count_as_absences():
    # Gvardiol: 29 and 24 minutes — on the pitch, but under the appearance threshold.
    cameos = _rows(4, GWS, [90, 90, 90, 90, 29, 24])
    assert 4 not in _watchlist(cameos, [4])


def test_off_switch_restores_the_old_behaviour():
    benched = _rows(5, GWS, [90, 90, 90, 70, 0, 0])
    assert 5 in _watchlist(benched, [5], absent_streak_games=0)


def test_current_squad_is_exempt():
    benched = _rows(6, GWS, [90, 90, 90, 70, 0, 0])
    assert 6 in _watchlist(benched, [6], must_include=[6])


def test_a_blank_gameweek_is_not_part_of_the_streak():
    # No GW7 fixture for this club: his last two are GW5 and GW6, and he played GW6.
    blank = _rows(7, [2, 3, 4, 5, 6], [90, 90, 0, 0, 90])
    assert 7 in _watchlist(blank, [7])
