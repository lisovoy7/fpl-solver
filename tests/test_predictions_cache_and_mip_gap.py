"""
Tests for in-memory player_predictions caching and the 1.0% (0.01) default MIP gap.

Standalone:
    python tests/test_predictions_cache_and_mip_gap.py
"""

from datetime import datetime, timedelta, timezone
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "")

import pandas as pd  # noqa: E402

import api_server as A  # noqa: E402


class _FakeResp:
    def __init__(self, data):
        self.data = data


class _CountingSupabase:
    """Stub Supabase client that counts how many times player_predictions is queried."""

    def __init__(self, rows):
        self.rows = rows
        self.predictions_queries = 0
        self._table = None

    def table(self, name):
        self._table = name
        return self

    def select(self, _cols):
        return self

    def eq(self, _col, _val):
        return self

    def maybe_single(self):
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        if self._table == "player_predictions":
            self.predictions_queries += 1
            start, end = self._range
            return _FakeResp(self.rows[start : end + 1])
        return _FakeResp({"value": None})


BOOTSTRAP = {
    "elements": [
        {"id": 10, "first_name": "Bukayo", "second_name": "Saka", "element_type": 3, "team": 1},
        {"id": 20, "first_name": "Cole", "second_name": "Palmer", "element_type": 3, "team": 6},
    ]
}

GW_DATA = pd.DataFrame([
    {"element": 10, "GW": 1, "minutes": 90},
    {"element": 10, "GW": 2, "minutes": 90},
    {"element": 20, "GW": 1, "minutes": 30},
])


def _fresh_rows(age_hours: float = 1.0):
    ts = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
    return [
        {"player_id": 10, "event": 12, "predicted_points": 6.5, "generated_at": ts, "through_gw": 11},
        {"player_id": 20, "event": 12, "predicted_points": 5.8, "generated_at": ts, "through_gw": 11},
    ]


def test_default_mip_gap_is_one_percent():
    req = A.OptimizeRequest(team_id=12345, free_transfers=1)
    assert req.mip_gap == 0.01, f"expected default mip_gap 0.01, got {req.mip_gap}"


def test_predictions_cache_serves_repeat_calls_without_supabase_queries():
    stub = _CountingSupabase(_fresh_rows())
    A._supabase = stub
    A._invalidate_predictions_cache()

    df1, tgw1 = A._predictions_from_supabase(BOOTSTRAP, GW_DATA)
    assert df1 is not None and len(df1) == 2
    assert tgw1 == 11
    assert stub.predictions_queries == 1

    # Mutating the returned DataFrame must NOT corrupt the cached copy
    df1.loc[df1["element"] == 10, "predicted_points"] = 999.0

    df2, tgw2 = A._predictions_from_supabase(BOOTSTRAP, GW_DATA)
    assert df2 is not None and tgw2 == 11
    assert stub.predictions_queries == 1, "second call within TTL should hit in-memory cache"
    saka_pts = float(df2.loc[df2["element"] == 10, "predicted_points"].iloc[0])
    assert saka_pts == 6.5, f"cache was mutated by caller: got {saka_pts}"


def test_predictions_cache_refetches_after_ttl_or_invalidation():
    stub = _CountingSupabase(_fresh_rows())
    A._supabase = stub
    A._invalidate_predictions_cache()

    A._predictions_from_supabase(BOOTSTRAP, GW_DATA)
    assert stub.predictions_queries == 1

    # Simulate TTL expiry (older than PREDICTIONS_CACHE_MAX_AGE_SECONDS)
    A._predictions_cache["fetched_at"] -= A.PREDICTIONS_CACHE_MAX_AGE_SECONDS + 5
    A._predictions_from_supabase(BOOTSTRAP, GW_DATA)
    assert stub.predictions_queries == 2

    # Explicit invalidation forces a fresh query as well
    A._invalidate_predictions_cache()
    A._predictions_from_supabase(BOOTSTRAP, GW_DATA)
    assert stub.predictions_queries == 3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
