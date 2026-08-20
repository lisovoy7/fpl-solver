"""Where the optimizer's three money figures come from, and who wins.

`_resolve_money()` turns a request into (total_budget, bank, selling_discounts). The
caller's own numbers outrank the FPL API: they come from a squad the user confirmed,
and the API only ever reports a squad as of the last deadline — pre-season, nothing at
all. Falling back costs three requests, so it is skipped whenever the caller answered.

Runnable directly (``python tests/test_money_resolution.py``) or under pytest.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api_server as A  # noqa: E402
from fpl import api as fpl_api  # noqa: E402

# Two owned players, both appreciated: p1 cost 9.0 and is now 10.0, p2 cost 4.5 and is
# now 5.0. So p1 sells for 9.5 and p2 for 4.7 under FPL's keep-half-the-rise rule.
BOOTSTRAP = {"elements": [
    {"id": 1, "now_cost": 100},
    {"id": 2, "now_cost": 50},
    {"id": 3, "now_cost": 60},
]}
SQUAD = [1, 2]
SELLING = [{"player": 1, "selling_value": 9.5}, {"player": 2, "selling_value": 4.7}]
SQUAD_SALE_VALUE = 95 + 47
DISCOUNTS = {1: 5, 2: 3}


def _request(**kwargs):
    return A.OptimizeRequest(team_id=1, free_transfers=1, **kwargs)


def test_caller_figures_are_used_verbatim():
    req = _request(total_budget=(SQUAD_SALE_VALUE + 10) / 10, bank=1.0, selling_prices=SELLING)
    budget, bank, discounts = A._resolve_money(req, SQUAD, BOOTSTRAP, 1)
    assert budget == SQUAD_SALE_VALUE + 10
    assert bank == 10
    assert discounts == DISCOUNTS


def test_bank_is_inverted_out_of_the_budget_when_not_sent():
    """`total_budget` is squad sale value + bank by definition, so subtraction recovers it."""
    req = _request(total_budget=(SQUAD_SALE_VALUE + 19) / 10, selling_prices=SELLING)
    _, bank, _ = A._resolve_money(req, SQUAD, BOOTSTRAP, 1)
    assert bank == 19


def test_a_budget_too_small_for_its_own_squad_clamps_to_no_cash():
    """Only reachable from a squad and budget captured at different moments.

    Better to plan with no spare cash than to hand the MILP a negative opening balance,
    which makes even holding the squad infeasible and fails the whole job.
    """
    req = _request(total_budget=(SQUAD_SALE_VALUE - 30) / 10, selling_prices=SELLING)
    _, bank, _ = A._resolve_money(req, SQUAD, BOOTSTRAP, 1)
    assert bank == 0


def test_missing_selling_price_means_no_discount():
    """A player the caller sent no sale price for sells for his market price.

    That is the honest default: it is what the card itself shows for a player with no
    purchase history, and it never invents a rise the manager may not have.
    """
    req = _request(total_budget=15.0, bank=1.0, selling_prices=[SELLING[0]])
    _, _, discounts = A._resolve_money(req, SQUAD, BOOTSTRAP, 1)
    assert discounts == {1: 5}


def test_unreachable_fpl_api_degrades_instead_of_failing():
    """A confirmed budget is enough to plan with, so a failed lookup must not kill the run.

    This is the pre-season case: there are no picks to read purchase prices from, but the
    user has confirmed a squad on the card and expects a plan.
    """
    def boom(*_args, **_kwargs):
        raise RuntimeError("picks not public yet")

    original = fpl_api.get_squad_selling_prices
    A.api.get_squad_selling_prices = boom
    try:
        req = _request(total_budget=15.5)
        budget, bank, discounts = A._resolve_money(req, SQUAD, BOOTSTRAP, 1)
    finally:
        A.api.get_squad_selling_prices = original

    assert budget == 155
    assert discounts == {}, "no purchase history means no discount can be known"
    # Every owned player is assumed to sell for market price, so the bank is what's left
    # over the squad's market value — the behaviour that predates sale prices entirely.
    assert bank == 155 - (100 + 50)


def test_unreachable_fpl_api_still_fails_when_there_is_no_budget():
    """With no budget and no sale prices there is nothing to price a squad against."""
    def boom(*_args, **_kwargs):
        raise RuntimeError("picks not public yet")

    original = fpl_api.get_squad_selling_prices
    A.api.get_squad_selling_prices = boom
    try:
        try:
            A._resolve_money(_request(), SQUAD, BOOTSTRAP, 1)
        except RuntimeError:
            return
        raise AssertionError("expected the lookup failure to propagate")
    finally:
        A.api.get_squad_selling_prices = original


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
