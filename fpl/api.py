"""
Single FPL API client for the fpl-solver project.

All Fantasy Premier League API interactions are consolidated here. This module
provides HTTP requests with retry logic, SSL fallback, bootstrap caching, and
all data-fetching functions needed for the solver.
"""

import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

_BOOTSTRAP_CACHE: Optional[Dict] = None

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{id}/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
ENTRY_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
ENTRY_TRANSFERS_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/transfers/"
ENTRY_HISTORY_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/history/"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def make_api_request(url: str, max_retries: int = 3) -> Optional[Dict]:
    """
    HTTP GET with retry logic. Try with SSL first, fall back to verify=False.

    Args:
        url: The API endpoint URL.
        max_retries: Maximum number of retry attempts.

    Returns:
        Parsed JSON dict or None if all attempts fail.
    """
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return json.loads(response.text)
        except requests.exceptions.SSLError:
            try:
                response = requests.get(url, headers=headers, verify=False, timeout=30)
                if response.status_code == 200:
                    return json.loads(response.text)
            except requests.exceptions.RequestException as e:
                logger.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
        except requests.exceptions.RequestException as e:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)

        if attempt < max_retries - 1:
            time.sleep(2)

    logger.error("Failed to fetch %s after %d attempts", url, max_retries)
    return None


def fetch_bootstrap_data() -> Dict:
    """
    Fetch bootstrap-static data. Cached in module-level variable `_bootstrap_cache`.

    Returns:
        Bootstrap data dict with elements, teams, events, etc.
    """
    global _BOOTSTRAP_CACHE
    if _BOOTSTRAP_CACHE is not None:
        return _BOOTSTRAP_CACHE
    data = make_api_request(BOOTSTRAP_URL)
    if data is None:
        raise RuntimeError("Failed to fetch bootstrap data")
    _BOOTSTRAP_CACHE = data
    logger.info("Fetched bootstrap data with %d players", len(data.get("elements", [])))
    return _BOOTSTRAP_CACHE


def detect_current_gw(bootstrap_data: Optional[Dict] = None) -> int:
    """
    Find the first event where deadline_time > now (the upcoming GW the user is planning for).

    Returns:
        The next gameweek number. If all deadlines have passed, returns last GW + 1 (end of season).
    """
    data = bootstrap_data or fetch_bootstrap_data()
    events = data.get("events", [])
    now = datetime.now(timezone.utc)
    for event in sorted(events, key=lambda e: e.get("id", 0)):
        deadline_str = event.get("deadline_time")
        if not deadline_str:
            continue
        try:
            deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
            if deadline > now:
                return int(event["id"])
        except (ValueError, TypeError):
            continue
    if events:
        last_id = max(int(e.get("id", 0)) for e in events)
        return last_id + 1
    return 1


def detect_current_season(bootstrap_data: Optional[Dict] = None) -> str:
    """
    Extract season from bootstrap data. Parses year from first event's kickoff_time.
    If first event is in Aug-Dec of year Y, season is "Y-(Y+1)". Format: "2025-26".

    Returns:
        Season string in format "YYYY-YY".
    """
    data = bootstrap_data or fetch_bootstrap_data()
    events = data.get("events", [])
    if not events:
        return "2024-25"
    first_event = min(events, key=lambda e: e.get("id", 999))
    kickoff_str = first_event.get("kickoff_time") or first_event.get("deadline_time")
    if not kickoff_str:
        return "2024-25"
    try:
        dt = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
        year = dt.year
        month = dt.month
        if month >= 8:
            return f"{year}-{str(year + 1)[-2:]}"
        return f"{year - 1}-{str(year)[-2:]}"
    except (ValueError, TypeError):
        return "2024-25"


def fetch_gameweek_data(bootstrap_data: Optional[Dict] = None) -> pd.DataFrame:
    """
    Fetch per-player per-GW data for ALL players in the current season.

    For each player in bootstrap elements, calls element-summary and extracts history.
    Builds a DataFrame with columns: name, position, element, assists, bonus, bps,
    clean_sheets, creativity, defensive_contribution, expected_assists,
    expected_goal_involvements, expected_goals, expected_goals_conceded, fixture,
    goals_conceded, goals_scored, ict_index, influence, kickoff_time, minutes,
    opponent_team, own_goals, penalties_missed, penalties_saved, red_cards, round,
    saves, selected, starts, team_a_score, team_h_score, threat, total_points,
    transfers_balance, transfers_in, transfers_out, value, was_home, yellow_cards, GW.

    Returns:
        DataFrame with one row per player per gameweek.
    """
    data = bootstrap_data or fetch_bootstrap_data()
    elements = data.get("elements", [])
    element_types = {str(e["id"]): int(e.get("element_type", 1)) for e in elements}
    total = len(elements)
    all_rows: List[Dict[str, Any]] = []

    for i, player in enumerate(elements, 1):
        if i % 50 == 0:
            logger.info("Processed %d/%d players", i, total)

        pid = player["id"]
        name = f"{player.get('first_name', '')} {player.get('second_name', '')}".strip()
        pos = POSITION_MAP.get(element_types.get(str(pid), 1), "UNKNOWN")

        url = ELEMENT_SUMMARY_URL.format(id=pid)
        resp = make_api_request(url)
        if resp is None:
            time.sleep(0.1)
            continue

        history = resp.get("history", [])
        for h in history:
            row = {
                "name": name,
                "position": pos,
                "element": int(pid),
                "assists": h.get("assists", 0),
                "bonus": h.get("bonus", 0),
                "bps": h.get("bps", 0),
                "clean_sheets": h.get("clean_sheets", 0),
                "creativity": h.get("creativity", 0),
                "defensive_contribution": h.get("defensive_contribution", 0),
                "expected_assists": h.get("expected_assists", 0),
                "expected_goal_involvements": h.get("expected_goal_involvements", 0),
                "expected_goals": h.get("expected_goals", 0),
                "expected_goals_conceded": h.get("expected_goals_conceded", 0),
                "fixture": h.get("fixture", 0),
                "goals_conceded": h.get("goals_conceded", 0),
                "goals_scored": h.get("goals_scored", 0),
                "ict_index": h.get("ict_index", 0),
                "influence": h.get("influence", 0),
                "kickoff_time": h.get("kickoff_time", ""),
                "minutes": h.get("minutes", 0),
                "opponent_team": h.get("opponent_team", 0),
                "own_goals": h.get("own_goals", 0),
                "penalties_missed": h.get("penalties_missed", 0),
                "penalties_saved": h.get("penalties_saved", 0),
                "red_cards": h.get("red_cards", 0),
                "round": h.get("round", 0),
                "saves": h.get("saves", 0),
                "selected": h.get("selected", 0),
                "starts": h.get("starts", 0),
                "team_a_score": h.get("team_a_score", 0),
                "team_h_score": h.get("team_h_score", 0),
                "threat": h.get("threat", 0),
                "total_points": h.get("total_points", 0),
                "transfers_balance": h.get("transfers_balance", 0),
                "transfers_in": h.get("transfers_in", 0),
                "transfers_out": h.get("transfers_out", 0),
                "value": h.get("value", 0),
                "was_home": h.get("was_home", False),
                "yellow_cards": h.get("yellow_cards", 0),
                "GW": h.get("round", 0),
            }
            all_rows.append(row)

        time.sleep(0.1)

    return pd.DataFrame(all_rows)


def fetch_current_fixtures(bootstrap_data: Optional[Dict] = None) -> pd.DataFrame:
    """
    Fetch fixtures from the FPL API.

    Returns:
        DataFrame with columns: id, event, kickoff_time, team_h, team_h_score,
        team_a, team_a_score, finished.
    """
    _ = bootstrap_data or fetch_bootstrap_data()
    data = make_api_request(FIXTURES_URL)
    if data is None:
        return pd.DataFrame()

    cols = ["id", "event", "kickoff_time", "team_h", "team_h_score", "team_a", "team_a_score", "finished"]
    rows = []
    for f in data:
        rows.append({
            "id": f.get("id"),
            "event": f.get("event"),
            "kickoff_time": f.get("kickoff_time"),
            "team_h": f.get("team_h"),
            "team_h_score": f.get("team_h_score"),
            "team_a": f.get("team_a"),
            "team_a_score": f.get("team_a_score"),
            "finished": f.get("finished"),
        })
    return pd.DataFrame(rows, columns=cols)


def fetch_team_data(team_id: int, gw: int) -> Dict:
    """
    Fetch team picks for a specific gameweek.

    Args:
        team_id: FPL team ID.
        gw: Gameweek number.

    Returns:
        Dict with keys: squad (list of 15 player IDs), bank (int, tenths), value (int, tenths).
    """
    url = ENTRY_PICKS_URL.format(team_id=team_id, gw=gw)
    data = make_api_request(url)
    if data is None:
        raise RuntimeError(f"Failed to fetch team data for team {team_id} GW {gw}")

    picks = data.get("picks", [])
    squad = [int(p["element"]) for p in picks]
    entry_history = data.get("entry_history", {})
    bank = int(entry_history.get("bank", 0))
    value = int(entry_history.get("value", 0))

    return {"squad": squad, "bank": bank, "value": value}


def calculate_selling_value(market_price: int, purchase_price: int) -> int:
    """
    Calculate FPL selling value using the official formula.

    If price rose: purchase_price + floor(rise / 2).
    If price fell: market_price.

    Args:
        market_price: Current market price in tenths.
        purchase_price: Original purchase price in tenths.

    Returns:
        Selling value in tenths.
    """
    if market_price >= purchase_price:
        rise = market_price - purchase_price
        return purchase_price + int(math.floor(rise / 2))
    return market_price


def _fetch_transfers_data(team_id: int) -> List[Dict]:
    """Fetch transfer history from /api/entry/{team_id}/transfers/."""
    url = ENTRY_TRANSFERS_URL.format(team_id=team_id)
    data = make_api_request(url)
    if data is None:
        return []
    return data if isinstance(data, list) else []


def _fetch_history_data(team_id: int) -> Dict:
    """Fetch history from /api/entry/{team_id}/history/."""
    url = ENTRY_HISTORY_URL.format(team_id=team_id)
    data = make_api_request(url)
    if data is None:
        return {}
    return data


def _build_purchase_prices(
    transfers: List[Dict],
    freehit_gws: List[int],
) -> Dict[int, Tuple[int, str]]:
    """
    Build player_id -> (purchase_price, source) from transfers, skipping Free Hit GWs.

    Returns:
        Dict mapping player_id to (purchase_price, source).
    """
    purchase_prices: Dict[int, Tuple[int, str]] = {}
    for t in sorted(transfers, key=lambda x: x.get("time", "")):
        gw = t.get("event")
        if gw in freehit_gws:
            continue
        player_in = t.get("element_in")
        cost_in = t.get("element_in_cost")
        if player_in is not None and cost_in is not None:
            purchase_prices[int(player_in)] = (int(cost_in), "transfer")
    return purchase_prices


def get_squad_selling_prices(
    team_id: int,
    gw: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Full selling price calculation for a squad.

    Args:
        team_id: FPL team ID.
        gw: Gameweek (None = current).

    Returns:
        Tuple of:
        - List of dicts with element, name, market_price, selling_value.
        - Summary dict with bank, correct_budget, selling_prices (player_id -> selling_value).
    """
    bootstrap = fetch_bootstrap_data()
    player_names = {p["id"]: f"{p.get('first_name', '')} {p.get('second_name', '')}".strip() for p in bootstrap["elements"]}
    player_costs = {str(p["id"]): int(p.get("now_cost", 0)) for p in bootstrap["elements"]}

    if gw is None:
        gw = detect_current_gw(bootstrap)

    picks_data = make_api_request(ENTRY_PICKS_URL.format(team_id=team_id, gw=gw))
    if picks_data is None:
        raise RuntimeError(f"Failed to fetch picks for team {team_id} GW {gw}")

    transfers = _fetch_transfers_data(team_id)
    history = _fetch_history_data(team_id)
    freehit_gws = [c["event"] for c in history.get("chips", []) if c.get("name") == "freehit"]
    purchase_prices = _build_purchase_prices(transfers, freehit_gws)

    bank = int(picks_data.get("entry_history", {}).get("bank", 0))
    result_list: List[Dict] = []
    total_selling = 0

    for pick in picks_data.get("picks", []):
        pid = int(pick["element"])
        name = player_names.get(str(pid), "Unknown")
        market_price = player_costs.get(str(pid), 0)
        purchase_price, _ = purchase_prices.get(pid, (market_price, "gw1_estimate"))
        selling_value = calculate_selling_value(market_price, purchase_price)
        total_selling += selling_value
        result_list.append({
            "element": pid,
            "name": name,
            "market_price": market_price,
            "selling_value": selling_value,
        })

    selling_prices = {r["element"]: r["selling_value"] for r in result_list}
    correct_budget = bank + total_selling

    summary = {
        "bank": bank,
        "correct_budget": correct_budget,
        "selling_prices": selling_prices,
    }
    return result_list, summary


def detect_chips_used(team_id: int) -> Dict:
    """
    Fetch history data and parse chips array.

    All four chip types follow the same rule: one allowed per half-season
    (GW 1-19 and GW 20-38), so each can be used 0, 1, or 2 times total.

    Returns:
        Dict with total counts for each chip type (int, 0-2) plus
        `free_hit_gws` (list of GWs where Free Hit was used).
    """
    history = _fetch_history_data(team_id)
    chips = history.get("chips", [])
    result: Dict[str, Any] = {
        "wildcards_used": 0,
        "free_hits_used": 0,
        "bench_boost_used": 0,
        "triple_captain_used": 0,
        "free_hit_gws": [],
    }
    for c in chips:
        name = c.get("name", "")
        event = c.get("event")
        if name == "wildcard":
            result["wildcards_used"] += 1
        elif name == "freehit":
            result["free_hits_used"] += 1
            if event is not None:
                result["free_hit_gws"].append(int(event))
        elif name == "bboost":
            result["bench_boost_used"] += 1
        elif name == "3xc":
            result["triple_captain_used"] += 1
    return result
