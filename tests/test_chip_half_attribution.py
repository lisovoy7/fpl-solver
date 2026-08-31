"""A chip is spent in the half-season it was PLAYED in, not the first one.

Each chip is granted once per half (GW1-19, GW20-38), and callers used to derive the
split from the total count alone: `min(used, 1)` first half, `max(0, used - 1)`
second. That reads "one Bench Boost used" as "the first-half one", which is wrong for
every manager who let a first-half chip expire and played it after GW20 — the second
half then looks free and the winning plan schedules a chip they do not have. The
gameweek is in FPL's history payload all along, so it is kept and used.

An override is only a total and cannot say which half it means, so the two sources are
unioned: detection can add a spent half, a caller can add a spent half, and neither can
talk the other's spent half back into being available.

Runnable directly (``python tests/test_chip_half_attribution.py``) or under pytest.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api_server as A  # noqa: E402
import fpl.api as fpl_api  # noqa: E402
from fpl.free_hit import bench_boost_candidate_gws  # noqa: E402


def _detect(chips):
    """detect_chips_used against a stubbed history payload."""
    original = fpl_api._fetch_history_data
    fpl_api._fetch_history_data = lambda team_id: {"chips": chips}
    try:
        return fpl_api.detect_chips_used(1)
    finally:
        fpl_api._fetch_history_data = original


# --- detection knows which half a use belongs to -----------------------------


def test_a_second_half_bench_boost_is_attributed_to_the_second_half():
    detected = _detect([{"name": "bboost", "event": 22}])
    assert detected["bench_boost_used"] == 1
    assert detected["bench_boost_used_first_half"] == 0
    assert detected["bench_boost_used_second_half"] == 1


def test_a_first_half_bench_boost_is_attributed_to_the_first_half():
    detected = _detect([{"name": "bboost", "event": 5}])
    assert detected["bench_boost_used_first_half"] == 1
    assert detected["bench_boost_used_second_half"] == 0


def test_gw19_is_the_first_half_and_gw20_the_second():
    """The boundary itself, since every other case is decided by it."""
    assert _detect([{"name": "3xc", "event": 19}])["triple_captain_used_first_half"] == 1
    assert _detect([{"name": "3xc", "event": 20}])["triple_captain_used_second_half"] == 1


def test_both_halves_used():
    detected = _detect([
        {"name": "wildcard", "event": 8},
        {"name": "wildcard", "event": 30},
    ])
    assert detected["wildcards_used"] == 2
    assert detected["wildcards_used_first_half"] == 1
    assert detected["wildcards_used_second_half"] == 1


def test_free_hit_gameweeks_are_still_reported():
    """The squad-reversion logic reads this list — see _non_free_hit_squad_gw."""
    detected = _detect([{"name": "freehit", "event": 27}])
    assert detected["free_hit_gws"] == [27]
    assert detected["free_hits_used_second_half"] == 1


def test_an_entry_with_no_gameweek_falls_to_the_first_half():
    """Cannot be attributed, so it keeps the old assumption rather than guessing."""
    detected = _detect([{"name": "bboost", "event": None}])
    assert detected["bench_boost_used_first_half"] == 1
    assert detected["bench_boost_used_second_half"] == 0


def test_an_unknown_chip_name_is_ignored():
    detected = _detect([{"name": "some_new_2027_chip", "event": 4}])
    assert detected["wildcards_used"] == 0
    assert detected["bench_boost_used_first_half"] == 0


# --- overrides union with detection, never overwrite it ----------------------


def test_detection_wins_over_a_total_that_would_free_a_spent_half():
    """The bug, end to end at the resolution layer.

    Manager played Bench Boost in GW22. The frontend card reads the total of 1 as
    "first half gone, second half available" and sends 1 back. Splitting that total
    alone would hand back the chip they just used.
    """
    detected = _detect([{"name": "bboost", "event": 22}])
    assert A._chip_halves_from(detected, "bench_boost_used", 1) == (1, 1)


def test_a_total_of_one_still_means_available_now_when_nothing_was_detected():
    """The other direction, which must not regress.

    In the second half a caller encodes "still available" as 1, because an unplayed
    first-half chip expires rather than carrying over. With no detected use, that has
    to keep meaning available.
    """
    assert A._chip_halves_from(_detect([]), "bench_boost_used", 1) == (1, 0)


def test_a_first_half_use_plus_a_total_of_one_leaves_the_second_half_free():
    detected = _detect([{"name": "bboost", "event": 5}])
    assert A._chip_halves_from(detected, "bench_boost_used", 1) == (1, 0)


def test_no_override_passes_detection_through_untouched():
    detected = _detect([{"name": "bboost", "event": 22}])
    assert A._chip_halves_from(detected, "bench_boost_used", None) == (0, 1)


def test_a_caller_can_spend_a_chip_fpl_has_not_recorded():
    """"Plan as if my Free Hit were already gone" — the documented override case."""
    assert A._chip_halves_from(_detect([]), "free_hits_used", 2) == (1, 1)


def test_a_caller_cannot_unspend_a_chip_fpl_recorded():
    """Deliberate asymmetry: the reverse can only produce an unplayable plan."""
    detected = _detect([{"name": "bboost", "event": 22}])
    assert A._chip_halves_from(detected, "bench_boost_used", 0) == (0, 1)


# --- and the plan that came out of it ----------------------------------------


def test_no_second_bench_boost_is_offered_after_one_was_played_in_gw22():
    """What the user actually experienced: an illegal chip in the plan.

    Placement is post-hoc over `bench_boost_candidate_gws`, so this is the list that
    decided it. Under the old split (first=1, second=0) every remaining second-half
    gameweek was a candidate.
    """
    detected = _detect([{"name": "bboost", "event": 22}])
    first_half, second_half = A._chip_halves_from(detected, "bench_boost_used", 1)

    fixed = bench_boost_candidate_gws(
        start_gw=25, planning_horizon=10,
        used_first_half=first_half, used_second_half=second_half,
    )
    assert fixed == [], f"Bench Boost still offered in {fixed}"

    old_split = bench_boost_candidate_gws(
        start_gw=25, planning_horizon=10, used_first_half=1, used_second_half=0,
    )
    assert old_split, "old split should have offered candidates — test is not proving anything"


def test_an_unused_chip_is_still_offered_in_the_second_half():
    """The fix must not withhold a chip the manager really does still have."""
    first_half, second_half = A._chip_halves_from(_detect([]), "bench_boost_used", 1)
    offered = bench_boost_candidate_gws(
        start_gw=25, planning_horizon=10,
        used_first_half=first_half, used_second_half=second_half,
    )
    assert offered == list(range(25, 35))


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
