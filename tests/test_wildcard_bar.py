"""
The wildcard bar's arithmetic, without a solve.

Standalone, like every other test here (the repo has no pytest):
    python tests/test_wildcard_bar.py

The case that matters most is the last one: a first-half wildcard reaching GW19
unused must be played, whatever it scores. A chip that expires is worth strictly
less than a mediocre one, and the bar existing at all is what could waste it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "")

from api_server import WILDCARD_FORCE_WITHIN_GWS, wildcard_bar  # noqa: E402

RATE = 2.5
failures = []


def check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got}, want {want}")
    if not ok:
        failures.append(label)


print("\nFirst-half wildcard — the countdown runs to GW19, not to the horizon")
for gw, left in ((5, 15), (10, 10), (14, 6), (18, 2), (19, 1)):
    terms = wildcard_bar(current_gw=gw, wildcard_gw=gw, rate=RATE)
    check(f"GW{gw} has {left} gameweeks left", terms["gameweeks_left"], left)
    check(f"GW{gw} bar", terms["bar"], RATE * left)

print("\nThe bar stops applying at the deadline, and only there")
check("GW17 still judged", wildcard_bar(17, 17, RATE)["forced"], False)
check("GW18 still judged", wildcard_bar(18, 18, RATE)["forced"], False)
check("GW19 exempt", wildcard_bar(19, 19, RATE)["forced"], True)

print("\nA wildcard worth almost nothing is still played on its last gameweek")
worth = 0.4
terms = wildcard_bar(current_gw=19, wildcard_gw=19, rate=RATE)
check("GW19 exempt despite worth < bar", terms["forced"] or worth >= terms["bar"], True)
terms18 = wildcard_bar(current_gw=18, wildcard_gw=18, rate=RATE)
check("GW18 NOT exempt at the same worth", terms18["forced"] or worth >= terms18["bar"], False)

print("\nSecond-half wildcard counts down to GW38, not GW19")
terms = wildcard_bar(current_gw=25, wildcard_gw=25, rate=RATE)
check("GW25 deadline", terms["deadline_gw"], 38)
check("GW25 gameweeks left", terms["gameweeks_left"], 14)
check("GW25 is second half", terms["first_half"], False)
check("GW38 exempt", wildcard_bar(38, 38, RATE)["forced"], True)

print("\nThe boundary is read from the constant, not hardcoded here")
check(
    f"exempt exactly when gameweeks_left <= {WILDCARD_FORCE_WITHIN_GWS}",
    [wildcard_bar(gw, gw, RATE)["forced"] for gw in (17, 18, 19)],
    [19 - gw + 1 <= WILDCARD_FORCE_WITHIN_GWS for gw in (17, 18, 19)],
)

print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'All checks passed.'}")
sys.exit(1 if failures else 0)
