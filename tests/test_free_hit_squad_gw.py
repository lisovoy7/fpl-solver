"""A Free Hit squad is a one-week loan, not the squad the manager owns.

Picks on a Free Hit gameweek revert at the next deadline, so reading them as "the
current squad" is wrong for the whole week between the two. `_optimize_inner` always
stepped back a gameweek for this; `/api/squad` did not, so for that week it handed the
frontend 15 players the manager would not have. The verification card then asked them to
confirm that squad, and a confirmed card goes to the optimizer as the authoritative
`squad` override — the entire plan built on the wrong team. Selling prices came out wrong
too: `_build_purchase_prices` skips Free Hit week transfers, so those players fall back
to season-start prices.

Both callers now share one helper, so they cannot drift apart again.

Runnable directly (``python tests/test_free_hit_squad_gw.py``) or under pytest.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api_server as A  # noqa: E402


def test_an_ordinary_gameweek_is_left_alone():
    assert A._non_free_hit_squad_gw(10, free_hit_gws=[]) == 10
    assert A._non_free_hit_squad_gw(10, free_hit_gws=[3]) == 10


def test_a_free_hit_gameweek_steps_back_one():
    """GW10 was a Free Hit, so GW9's picks are the squad that carries into GW11."""
    assert A._non_free_hit_squad_gw(10, free_hit_gws=[10]) == 9


def test_both_halves_free_hits_are_handled():
    assert A._non_free_hit_squad_gw(27, free_hit_gws=[8, 27]) == 26
    assert A._non_free_hit_squad_gw(8, free_hit_gws=[8, 27]) == 7


def test_it_never_steps_back_past_the_managers_first_gameweek():
    """A mid-season joiner has no picks before they started.

    Rare enough to prefer the Free Hit squad, which at least exists, over no squad at
    all — the caller logs a warning when this happens.
    """
    assert A._non_free_hit_squad_gw(10, free_hit_gws=[10], first_gw=10) == 10
    assert A._non_free_hit_squad_gw(10, free_hit_gws=[10], first_gw=9) == 9


def test_it_never_steps_back_below_gw1():
    assert A._non_free_hit_squad_gw(1, free_hit_gws=[1]) == 1


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
