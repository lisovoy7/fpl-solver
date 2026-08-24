"""Excluding a player has to beat including him, including when he is in the squad.

`create_watchlist` takes a must_include list (the current squad, plus anything the caller
forced into consideration) and a must_exclude list ("never own this player"). Exclusion is
documented to win, and did — for one step. Step 5 dropped the excluded players out of the
merged frame, then step 6b added back every must_include player that was *missing* from
that frame, which step 5 had just guaranteed they were. Since the current squad is always
in must_include, "get rid of Salah" quietly did nothing for a player you already own.

It surfaced in prod on 2026-08-24: team 33695 sent `excluded_players: [448]` together with
`forced_lineup: [448 -> GW7]`. Exclusion should have won and dropped the forced start.
Instead Burn came back into the pool, the forced start was applied, and the run failed —
see tests/test_blank_gameweek_detection.py for the second half of that failure.

Runnable directly (``python tests/test_watchlist_exclusion.py``) or under pytest.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl.watchlist import create_watchlist  # noqa: E402

OWNED_WITH_FORECAST = 10
OWNED_WITHOUT_FORECAST = 11  # the Dan Burn shape: in the squad, never played 60 minutes
SPARE = 12

SQUAD = [OWNED_WITH_FORECAST, OWNED_WITHOUT_FORECAST]


def _predictions():
    """Everyone but OWNED_WITHOUT_FORECAST, who has no rows at all."""
    return pd.DataFrame([
        {'element': OWNED_WITH_FORECAST, 'predicted_points': 5.0, 'hist_games': 6, 'position': 'MID'},
        {'element': SPARE, 'predicted_points': 6.0, 'hist_games': 6, 'position': 'MID'},
    ])


def _gw_data():
    """Prices and appearances. Everyone has played, so nobody is filtered on minutes —
    the only thing under test here is include-vs-exclude."""
    return pd.DataFrame([
        {'element': e, 'GW': 1, 'minutes': 90, 'value': 50}
        for e in (OWNED_WITH_FORECAST, OWNED_WITHOUT_FORECAST, SPARE)
    ])


def _watchlist(must_exclude):
    return create_watchlist(
        _predictions(), _gw_data(),
        min_hist_pct=0.6, max_hist_window=6,
        must_include=list(SQUAD),
        must_exclude=must_exclude,
    )


def test_excluding_an_owned_player_with_a_forecast_removes_him():
    assert OWNED_WITH_FORECAST not in _watchlist([OWNED_WITH_FORECAST])


def test_excluding_an_owned_player_with_no_forecast_removes_him():
    """The one that regressed. He is absent from the predictions frame either way, so the
    add-back path could not tell "excluded" from "never had a forecast"."""
    assert OWNED_WITHOUT_FORECAST not in _watchlist([OWNED_WITHOUT_FORECAST])


def test_excluding_nobody_keeps_the_whole_squad():
    watchlist = _watchlist([])
    assert OWNED_WITH_FORECAST in watchlist
    assert OWNED_WITHOUT_FORECAST in watchlist


def test_excluding_one_player_leaves_the_rest_alone():
    watchlist = _watchlist([OWNED_WITHOUT_FORECAST])
    assert OWNED_WITH_FORECAST in watchlist
    assert SPARE in watchlist


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
