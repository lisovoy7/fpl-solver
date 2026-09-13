"""
FPL prediction engine: normalized stats and component-based point predictions.

Merges normalization and prediction logic. All data is passed as function parameters;
no hardcoded file paths, season strings, or file I/O in business logic.
"""

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Constants
MIN_MINUTES = 60
LAST_N_GAMES = 6

# Outlier cap + shrinkage, applied to the two spiky attacking components only.
#
# A 6-game window of xG takes a median 52% of its total from its single biggest
# game (xA: 49%); saves, xGC and defensive contribution sit near 27%, so they are
# left alone. Two separate failures follow from that spikiness and each needs its
# own correction:
#
#   1. Too few games. One 60-minute appearance was enough to make Hinshelwood the
#      highest-rated player in the league (1.31 normalised xG, 2026-27 GW3).
#      SHRINK_K pulls every average toward its positional prior, hard when the
#      window is thin and barely at all when it is full.
#   2. A full window with one freak game. Watkins had six games last season and
#      one of them normalised to 4.22 — more than the other five combined, and
#      no amount of "how many games" catches it. CAP_TO_SECOND_HIGHEST replaces
#      that single observation with the second-highest in the window.
#
# The cap needs CAP_MIN_GAMES, and 4 is not arbitrary: capping a 2-game window is
# the same as keeping only the worse game, and across the league it strips a
# median 90% of the average at 2 games and 62% at 3. Below the threshold
# shrinkage alone does the work.
SHRINK_K = 3.0
CAP_MIN_GAMES = 4
SHRUNK_COMPONENTS = ("norm_expected_goals", "norm_expected_assists")

# Bonus points get the same shrinkage and a much larger k, because they carry far
# less signal than xG: for the median player the ENTIRE 6-game bonus average comes
# from one game, and the first half of a window predicts the second at r=0.03.
#
# Measured on 2025-26 across six snapshot gameweeks, scored against what each
# player actually averaged over his following six games, the current
# straight-average estimator came LAST of nine (mean absolute error 0.326 vs 0.297
# at k=3, 0.280 at k=10, 0.277 at k=20) — and lost in all six snapshots. Using the
# positional average alone and ignoring the player's own history entirely scores
# 0.280, i.e. his bonus record is worth less than nothing.
#
# 10 takes essentially all of that gain while leaving ~1/3 of the weight on the
# player, which is what keeps a genuine bonus magnet distinguishable. Going
# further measures marginally better and reads worse: Bruno Fernandes on
# [0,0,3,0,3,3] went on to average 1.33, and k=20 would price him at 0.58.
#
# No cap here — capping a mostly-zero series is meaningless, and its one non-zero
# game is the only information in it.
BONUS_SHRINK_K = 10.0

# Conceding is a team event, not a personal one. FPL's per-player
# `expected_goals_conceded` is the opposition xG accumulated *while that player was
# on the pitch*, so for anyone who played the full 90 it IS the team's figure and
# for everyone else it is a fraction of it. Averaging it per player therefore
# measured availability, not defence: at 2026-27 GW4 Arsenal's whole XI carried
# 1.64 from the Sunderland match, while Ben White — subbed at half time — carried
# 0.13, and because MIN_MINUTES then dropped his 45-minute game from his window
# entirely he was priced for a clean sheet at 2.78 against Gabriel's 2.02. Two
# centre-backs, same club, same fixture, a 27% gap invented by a substitution.
#
# So the xGC that feeds `conceded_goals` and `clean_sheet` is rebuilt per CLUB over
# the club's own last TEAM_XGC_LAST_N matches, and every player at that club
# inherits it whether or not he featured in any of them. Both components are
# per-appearance estimates — what a player is worth IF he plays — so "was he on the
# pitch last month" has no business in them; whether he plays at all is the
# minutes/eligibility question, answered elsewhere.
#
# A match's team xGC is the MAX across that club's players in the fixture, which is
# exact rather than approximate: a 90-minute player's value is the whole match by
# construction, and nobody can exceed it.
TEAM_XGC_LAST_N = LAST_N_GAMES

GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}

NORMALIZABLE_COMPONENTS = [
    "assists",
    "bonus",
    "clean_sheets",
    "expected_assists",
    "expected_goals",
    "expected_goals_conceded",
    "goals_conceded",
    "goals_scored",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "saves",
    "yellow_cards",
]

PREDICTION_COMPONENTS = [
    "minutes_played",
    "goals_scored",
    "assists",
    "saves",
    "conceded_goals",
    "yellow_cards",
    "clean_sheet",
    "defensive_contribution",
    "bps",
]

# Population std for defensive contribution probability (when player history insufficient)
POPULATION_STD = {"DEF": 3.5, "MID": 3.9, "FWD": 2.8}

# Raw XGC multipliers for defensive fixture difficulty (tier -> multiplier)
RAW_XGC_MULTIPLIERS = {1.0: 1.126, 2.0: 1.040, 3.0: 1.000, 4.0: 0.735, 5.0: 0.641}


def _normalize_stats(
    gw_data: pd.DataFrame,
    fixtures: pd.DataFrame,
    multipliers: pd.DataFrame,
    team_tiers: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """
    Normalize player stats by removing fixture difficulty bias using multipliers.

    Filters to minutes >= MIN_MINUTES, merges with fixtures for team info,
    adds team tiers, and normalizes each component by dividing by fixture multiplier.
    """
    logger.debug("Normalizing stats for season %s", season)

    filtered = gw_data[gw_data["minutes"] >= MIN_MINUTES].copy()
    filtered["season"] = season
    logger.debug("Filtered to %d records with minutes >= %d", len(filtered), MIN_MINUTES)

    # Merge with fixtures
    gw_data_copy = filtered.copy()
    gw_data_copy["kickoff_time"] = pd.to_datetime(gw_data_copy["kickoff_time"])
    fixtures_copy = fixtures.copy()
    fixtures_copy["kickoff_time"] = pd.to_datetime(fixtures_copy["kickoff_time"])
    fixtures_copy["season"] = season

    fixture_cols = ["id", "season", "team_h", "team_a", "kickoff_time"]
    available_fixture_cols = [c for c in fixture_cols if c in fixtures_copy.columns]
    if "id" not in available_fixture_cols:
        raise ValueError("Fixtures must have 'id' column for merge")

    merged = gw_data_copy.merge(
        fixtures_copy[available_fixture_cols],
        left_on=["fixture", "season"],
        right_on=["id", "season"],
        how="left",
        suffixes=("", "_fixture"),
    )
    if "id" in merged.columns and "id_fixture" not in merged.columns:
        merged = merged.drop(columns=["id"], errors="ignore")

    # Derive player_team_id
    merged["player_team_id"] = merged.apply(
        lambda row: row["team_h"] if row["was_home"] else row["team_a"], axis=1
    )

    # Add team tiers
    player_tier_map = team_tiers[["team_id", "team_tier"]].rename(
        columns={"team_id": "player_team_id", "team_tier": "player_team_tier"}
    )
    merged = merged.merge(player_tier_map, on="player_team_id", how="left")

    opponent_tier_map = team_tiers[["team_id", "team_tier"]].rename(
        columns={"team_id": "opponent_team", "team_tier": "opponent_team_tier"}
    )
    merged = merged.merge(opponent_tier_map, on="opponent_team", how="left")

    merged["is_home"] = merged["was_home"].astype(int)

    # Cast normalizable components to numeric (API sometimes returns strings)
    for col in NORMALIZABLE_COMPONENTS:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    # Normalize each component
    merge_keys = ["position", "player_team_tier", "opponent_team_tier", "is_home"]
    for component in NORMALIZABLE_COMPONENTS:
        if component not in merged.columns:
            logger.warning("Component '%s' not in data, skipping", component)
            continue

        comp_mult = multipliers[multipliers["component_type"] == component]
        if len(comp_mult) == 0:
            logger.warning("No multipliers for '%s', skipping", component)
            continue

        missing = [k for k in merge_keys if k not in merged.columns]
        if missing:
            logger.warning("Missing keys for %s: %s", component, missing)
            continue

        merged = merged.merge(
            comp_mult[merge_keys + ["multiplier"]],
            on=merge_keys,
            how="left",
            suffixes=("", f"_{component}"),
        )
        mult_col = "multiplier"
        if f"multiplier_{component}" in merged.columns:
            mult_col = f"multiplier_{component}"

        norm_col = f"norm_{component}"
        merged[norm_col] = np.where(
            (merged[mult_col].isna()) | (merged[mult_col] == 0),
            merged[component],
            merged[component] / merged[mult_col],
        )
        merged = merged.drop(columns=[mult_col], errors="ignore")

    logger.debug("Normalization complete: %d records", len(merged))
    return merged


def _cap_top_observation(values: np.ndarray, min_games: int = CAP_MIN_GAMES) -> np.ndarray:
    """
    Replace the single largest observation with the second largest.

    Capping rather than dropping is deliberate. Deleting the best game taxes every
    player for having one: measured on 2025-26, deleting cost Haaland 19% of his
    average and Salah 12%, where capping cost them 4% and 3% — while still taking
    40% off Watkins and 65% off Diallo, whose windows are one game and five
    blanks. It also keeps the sample size, so nothing downstream has to cope with
    a 6-game window that turned into 5.

    A window shorter than `min_games` is returned untouched (see CAP_MIN_GAMES).
    Ties at the top are a no-op by construction: a value repeated twice is not an
    outlier.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) < max(2, min_games):
        return arr
    order = np.argsort(arr)
    capped = arr.copy()
    capped[order[-1]] = arr[order[-2]]
    return capped


def _positional_priors(
    normalized_stats: pd.DataFrame,
    columns: Sequence[str] = SHRUNK_COMPONENTS,
) -> Dict[str, Dict[str, float]]:
    """
    Per-game league average of each component, by position, from this run's own data.

    Pooled over every 60+ minute game in `normalized_stats`, not just the tail
    windows — same answer, more observations, and it does not wobble when a
    rotated player drops out of a window.

    Deliberately NOT persisted anywhere. It is eight numbers derived from exactly
    the data the player averages are derived from, so computing it here is what
    guarantees the two describe the same league. A stored prior would be a vintage
    that can go stale against the averages it corrects, which is the same class of
    bug as judging eligibility on today's data and points on yesterday's.

    The prior needs far less evidence than any individual: through GW3 of 2026-27
    the midfield xG prior sits on 264 player-games, and it moves very little all
    season (0.137 → 0.128 → 0.136 → 0.137 across GW3/10/20/36 of 2025-26).

    Returns {column: {position: mean, "": pooled mean as fallback}}.
    """
    priors: Dict[str, Dict[str, float]] = {}
    for col in columns:
        if col not in normalized_stats.columns:
            continue
        series = pd.to_numeric(normalized_stats[col], errors="coerce")
        by_pos: Dict[str, float] = {}
        if "position" in normalized_stats.columns:
            grouped = series.groupby(normalized_stats["position"]).mean()
            by_pos = {str(k): float(v) for k, v in grouped.items() if pd.notna(v)}
        overall = float(series.mean()) if len(series) else 0.0
        by_pos[""] = 0.0 if pd.isna(overall) else overall
        priors[col] = by_pos
        logger.info(
            "Prior (per game, by position) for %s: %s",
            col,
            {k: round(v, 4) for k, v in sorted(by_pos.items()) if k},
        )
    return priors


def _calculate_player_averages(
    normalized_stats: pd.DataFrame,
    gw_data: pd.DataFrame,
    last_n_games: int = LAST_N_GAMES,
    shrink_k: float = SHRINK_K,
    cap_min_games: int = CAP_MIN_GAMES,
    bonus_shrink_k: float = BONUS_SHRINK_K,
) -> pd.DataFrame:
    """
    Calculate player averages from last N games (normalized and raw components).

    `norm_expected_goals` and `norm_expected_assists` are capped and shrunk here
    (see SHRINK_K / CAP_MIN_GAMES) rather than downstream, because this is the
    last point at which the per-game observations still exist — by
    _create_component_predictions there is only an average left, and an average
    cannot be told apart from the same average made of one game and five blanks.

    Order matters and is fixed: normalise each game by its own fixture, take the
    window, cap the outlier, shrink toward the prior. Only then does
    _create_component_predictions multiply by the TARGET fixture's difficulty. The
    unshrunk means are kept alongside as `*_unshrunk` for diagnostics; nothing
    reads them, and they are why a trace can show what the correction did.

    `shrink_k=0` with `cap_min_games` above the window length reproduces the
    pre-2026-09-07 behaviour exactly, which is what the backtest sweeps against.
    """
    logger.debug("Calculating player averages from last %d games", last_n_games)

    gw_filtered = gw_data[gw_data["minutes"] >= MIN_MINUTES].copy()
    gw_filtered["kickoff_time"] = pd.to_datetime(gw_filtered["kickoff_time"])
    for col in ["defensive_contribution", "bonus", "yellow_cards"]:
        if col in gw_filtered.columns:
            gw_filtered[col] = pd.to_numeric(gw_filtered[col], errors="coerce").fillna(0)
    stats_sorted = normalized_stats.sort_values(["element", "kickoff_time"])
    priors = _positional_priors(stats_sorted, SHRUNK_COMPONENTS)
    bonus_priors = _positional_priors(gw_filtered, ("bonus",)).get("bonus", {})

    records = []
    for element, group in stats_sorted.groupby("element"):
        recent = group.tail(last_n_games)
        hist_games = len(recent)

        player_gw = gw_filtered[gw_filtered["element"] == element]
        if len(player_gw) > 0:
            gw_sorted = player_gw.sort_values("kickoff_time")
            recent_gw = gw_sorted.tail(last_n_games)
            avg_def = recent_gw["defensive_contribution"].mean()
            def_history = recent_gw["defensive_contribution"].tolist()
            avg_bonus = (
                recent_gw["bonus"].mean()
                if "bonus" in recent_gw.columns
                else 0.0
            )
            bonus_games = len(recent_gw) if "bonus" in recent_gw.columns else 0
        else:
            avg_def = 0.0
            def_history = []
            avg_bonus = 0.0
            bonus_games = 0

        position = (
            str(recent["position"].iloc[-1])
            if "position" in recent.columns and pd.notna(recent["position"].iloc[-1])
            else ""
        )
        corrected = {}
        for col in SHRUNK_COMPONENTS:
            if col not in recent.columns:
                corrected[col] = (0.0, 0.0)
                continue
            values = pd.to_numeric(recent[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            unshrunk = float(values.mean()) if len(values) else 0.0
            capped = float(_cap_top_observation(values, cap_min_games).mean()) if len(values) else 0.0
            prior = priors.get(col, {}).get(position)
            if prior is None:
                prior = priors.get(col, {}).get("", 0.0)
            shrunk = (
                (len(values) * capped + shrink_k * prior) / (len(values) + shrink_k)
                if len(values)
                else prior
            )
            corrected[col] = (float(shrunk), unshrunk)

        bonus_prior = bonus_priors.get(position)
        if bonus_prior is None:
            bonus_prior = bonus_priors.get("", 0.0)
        shrunk_bonus = (
            (bonus_games * avg_bonus + bonus_shrink_k * bonus_prior)
            / (bonus_games + bonus_shrink_k)
            if bonus_shrink_k > 0 or bonus_games
            else avg_bonus
        )

        records.append(
            {
                "element": element,
                "hist_games": hist_games,
                "avg_norm_expected_goals": corrected["norm_expected_goals"][0],
                "avg_norm_expected_goals_unshrunk": corrected["norm_expected_goals"][1],
                "avg_norm_expected_assists": corrected["norm_expected_assists"][0],
                "avg_norm_expected_assists_unshrunk": corrected["norm_expected_assists"][1],
                "avg_norm_saves": (
                    recent["norm_saves"].mean()
                    if "norm_saves" in recent.columns
                    else 0.0
                ),
                "avg_norm_expected_goals_conceded": (
                    recent["norm_expected_goals_conceded"].mean()
                    if "norm_expected_goals_conceded" in recent.columns
                    else 0.0
                ),
                "avg_yellow_cards": (
                    recent["yellow_cards"].mean()
                    if "yellow_cards" in recent.columns
                    else 0.0
                ),
                "avg_defensive_contribution": avg_def,
                "avg_bonus_points": float(shrunk_bonus),
                "avg_bonus_points_unshrunk": float(avg_bonus),
                "defensive_contribution_history": def_history,
            }
        )

    df = pd.DataFrame(records)
    logger.debug("Calculated averages for %d players", len(df))
    return df


def _team_match_xgc(
    gw_data: pd.DataFrame, fixtures: pd.DataFrame
) -> pd.DataFrame:
    """
    One row per (club, match): the whole match's expected goals conceded.

    Taken as the MAX of `expected_goals_conceded` across that club's players in the
    fixture, which is exact and not an estimate — the stat accumulates only while a
    player is on the pitch, so anyone who played 90 minutes carries the full match
    total and nobody can carry more. Verified on 2026-27 GW4 Sunderland-Arsenal:
    every Arsenal player with 90 minutes reads 1.64, the two half-time substitutions
    read 1.51 and 0.13.

    Deliberately built from UNFILTERED gw_data — MIN_MINUTES is a rule about whether
    a player's own performance is worth averaging, and it has nothing to say about
    how many chances his club gave up. Filtering here would be the original bug in a
    new place: a club whose defenders rotate would end up with fewer matches on
    record than a club whose defenders play every minute.
    """
    df = gw_data.copy()
    df["expected_goals_conceded"] = pd.to_numeric(
        df["expected_goals_conceded"], errors="coerce"
    ).fillna(0.0)

    fx = fixtures.copy()
    fx["kickoff_time"] = pd.to_datetime(fx["kickoff_time"])
    fixture_cols = [c for c in ["id", "team_h", "team_a", "kickoff_time"] if c in fx.columns]
    if "id" not in fixture_cols:
        raise ValueError("Fixtures must have 'id' column for merge")

    merged = df.drop(columns=["kickoff_time"], errors="ignore").merge(
        fx[fixture_cols], left_on="fixture", right_on="id", how="inner"
    )
    merged["player_team_id"] = np.where(
        merged["was_home"].astype(bool), merged["team_h"], merged["team_a"]
    )

    team_matches = merged.groupby(
        ["player_team_id", "fixture"], as_index=False
    ).agg(
        raw_xgc=("expected_goals_conceded", "max"),
        kickoff_time=("kickoff_time", "first"),
        opponent_team=("opponent_team", "first"),
        was_home=("was_home", "first"),
    )
    team_matches["is_home"] = team_matches["was_home"].astype(int)
    logger.debug(
        "Built team xGC for %d club-matches across %d clubs",
        len(team_matches),
        team_matches["player_team_id"].nunique(),
    )
    return team_matches


def _team_xgc_averages(
    team_matches: pd.DataFrame,
    multipliers: pd.DataFrame,
    team_tiers: pd.DataFrame,
    last_n_games: int = TEAM_XGC_LAST_N,
) -> pd.DataFrame:
    """
    Average normalised xGC per (club, position) over the club's last N matches.

    Keyed on position as well as club only because the xGC multiplier table is
    estimated per position. Those four columns describe the same team event and
    differ by sampling noise, but normalising with one position's multiplier and
    then de-normalising the target fixture with another's would bake that noise in
    as a bias, so each position is normalised with its own, exactly as before. Every
    player at the club gets his position's club number whether or not he featured.
    """
    xgc_mult = multipliers[multipliers["component_type"] == "expected_goals_conceded"]
    if len(xgc_mult) == 0:
        logger.warning("No expected_goals_conceded multipliers; team xGC unavailable")
        return pd.DataFrame(
            columns=["player_team_id", "position", "avg_norm_team_xgc", "team_xgc_games"]
        )

    tiers = team_tiers[["team_id", "team_tier"]].copy()
    tiers["team_tier"] = pd.to_numeric(tiers["team_tier"], errors="coerce")
    tm = team_matches.merge(
        tiers.rename(columns={"team_id": "player_team_id", "team_tier": "player_team_tier"}),
        on="player_team_id",
        how="left",
    ).merge(
        tiers.rename(columns={"team_id": "opponent_team", "team_tier": "opponent_team_tier"}),
        on="opponent_team",
        how="left",
    )
    tm = tm.sort_values(["player_team_id", "kickoff_time"])
    recent = tm.groupby("player_team_id", group_keys=False).tail(last_n_games)

    merge_keys = ["player_team_tier", "opponent_team_tier", "is_home"]
    frames = []
    for position in sorted(xgc_mult["position"].dropna().unique()):
        pos_mult = xgc_mult[xgc_mult["position"] == position][merge_keys + ["multiplier"]]
        m = recent.merge(pos_mult, on=merge_keys, how="left")
        mult = m["multiplier"]
        m["norm_xgc"] = np.where(
            mult.isna() | (mult == 0), m["raw_xgc"], m["raw_xgc"] / mult
        )
        agg = m.groupby("player_team_id", as_index=False).agg(
            avg_norm_team_xgc=("norm_xgc", "mean"),
            team_xgc_games=("norm_xgc", "size"),
        )
        agg["position"] = str(position)
        frames.append(agg)

    out = pd.concat(frames, ignore_index=True)
    logger.info(
        "Team xGC over last %d club matches: %d clubs, median %.3f",
        last_n_games,
        out["player_team_id"].nunique(),
        float(out["avg_norm_team_xgc"].median()) if len(out) else float("nan"),
    )
    return out


def _get_player_team_assignments(normalized_stats: pd.DataFrame) -> pd.DataFrame:
    """Get latest team assignment per player from normalized stats."""
    logger.debug("Determining player team assignments")
    sorted_stats = normalized_stats.sort_values(["element", "kickoff_time"])
    latest = sorted_stats.groupby("element").last().reset_index()
    assignments = latest[
        ["element", "name", "position", "player_team_id", "player_team_tier"]
    ].copy()
    logger.debug("Team assignments for %d players", len(assignments))
    return assignments


def _generate_player_fixture_combinations(
    player_assignments: pd.DataFrame,
    fixtures: pd.DataFrame,
    team_tiers: pd.DataFrame,
    last_played_gw: int,
) -> pd.DataFrame:
    """
    Generate player-fixture combinations for future fixtures only.
    """
    logger.debug("Generating player-fixture combinations (future fixtures only)")

    fixtures_copy = fixtures.copy()
    fixtures_copy["kickoff_time"] = pd.to_datetime(fixtures_copy["kickoff_time"])

    # Resolve event/gw column
    event_col = "event" if "event" in fixtures_copy.columns else "GW"
    if event_col not in fixtures_copy.columns:
        raise ValueError("Fixtures must have 'event' or 'GW' column")

    future_fixtures = fixtures_copy[
        fixtures_copy[event_col] > last_played_gw
    ].copy()
    logger.debug("Future fixtures: %d (event > %d)", len(future_fixtures), last_played_gw)

    combinations = []
    for _, prow in player_assignments.iterrows():
        pid = prow["player_team_id"]
        for _, frow in future_fixtures.iterrows():
            team_h = frow["team_h"]
            team_a = frow["team_a"]
            if pid == team_h:
                is_home = 1
                opponent = team_a
            elif pid == team_a:
                is_home = 0
                opponent = team_h
            else:
                continue

            opp_tier = team_tiers[team_tiers["team_id"] == opponent]["team_tier"]
            opp_tier_val = opp_tier.iloc[0] if len(opp_tier) > 0 else np.nan

            combinations.append(
                {
                    "element": prow["element"],
                    "name": prow["name"],
                    "position": prow["position"],
                    "player_team_id": pid,
                    "player_team_tier": prow["player_team_tier"],
                    "event": frow[event_col],
                    "kickoff_time": frow["kickoff_time"],
                    "opponent_team": opponent,
                    "opponent_team_tier": opp_tier_val,
                    "is_home": is_home,
                }
            )

    comb_df = pd.DataFrame(combinations)
    logger.debug("Created %d player-fixture combinations", len(comb_df))
    return comb_df


def _get_defensive_fixture_multiplier(
    opponent_team_tier: float, is_home: bool
) -> float:
    """Defensive fixture multiplier from opponent tier and location."""
    raw = RAW_XGC_MULTIPLIERS.get(
        float(opponent_team_tier) if not np.isnan(opponent_team_tier) else 3.0,
        1.0,
    )
    loc_mult = 0.846 if is_home else 1.000
    adjusted = raw * loc_mult
    return 0.269 * adjusted + 0.702


def _calculate_defensive_probability(
    predicted_value: float,
    position: str,
    player_history: Optional[List[float]] = None,
) -> float:
    """
    Probability of reaching defensive contribution threshold using normal distribution.
    Uses player-specific consistency when history available.
    """
    if position == "GK":
        return 0.0

    threshold = 10 if position == "DEF" else 12
    pop_std = POPULATION_STD.get(position, 3.5)

    if player_history and len(player_history) >= 1:
        hist = np.array(player_history)
        success_rate = (hist >= threshold).mean()
        player_std = hist.std() if len(hist) > 1 else 0.0
        mean_val = hist.mean()

        if len(hist) >= 4:
            if success_rate >= 0.8:
                floor = 0.70 if success_rate >= 0.9 else 0.60
                adj = success_rate * (
                    predicted_value / mean_val if mean_val > 0 else 1.0
                )
                return max(floor, min(0.95, adj))
            elif success_rate <= 0.2:
                adj = success_rate * (
                    predicted_value / mean_val if mean_val > 0 else 1.0
                )
                return min(0.30, max(0.0, adj))
            else:
                weight = min(0.7, len(hist) / 10.0)
                std_dev = weight * player_std + (1 - weight) * pop_std
        else:
            std_dev = pop_std

        if std_dev > 0:
            prob = 1 - stats.norm.cdf(threshold, predicted_value, std_dev)
            return max(0.0, min(1.0, prob))
        return 1.0 if predicted_value >= threshold else 0.0

    if pop_std > 0:
        prob = 1 - stats.norm.cdf(threshold, predicted_value, pop_std)
        return max(0.0, min(1.0, prob))
    return 1.0 if predicted_value >= threshold else 0.0


def _create_component_predictions(
    combinations: pd.DataFrame,
    player_averages: pd.DataFrame,
    multipliers: pd.DataFrame,
    team_xgc: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create predictions for all PREDICTION_COMPONENTS.

    `team_xgc` supplies the club-level expected goals conceded that `conceded_goals`
    and `clean_sheet` are built from (see TEAM_XGC_LAST_N). Omitted, both fall back
    to the player's own on-pitch average, which is the pre-2026-09-13 behaviour.
    """
    merge_keys = ["position", "player_team_tier", "opponent_team_tier", "is_home"]
    all_preds = []

    def _with_xgc_base(frame: pd.DataFrame) -> pd.Series:
        """Club xGC for this player's club and position, or his own as a fallback."""
        own = frame["avg_norm_expected_goals_conceded"]
        if team_xgc is None or len(team_xgc) == 0:
            return own.fillna(0.0)
        return frame["avg_norm_team_xgc"].fillna(own).fillna(0.0)

    # 1. minutes_played
    p = combinations.merge(
        player_averages[["element", "hist_games"]], on="element", how="left"
    )
    p["component_type"] = "minutes_played"
    p["predicted_points"] = 2.0
    p["hist_games"] = p["hist_games"].fillna(0)
    all_preds.append(p)

    # 2. goals_scored
    xg_mult = multipliers[multipliers["component_type"] == "expected_goals"]
    p = combinations.merge(
        player_averages[["element", "hist_games", "avg_norm_expected_goals"]],
        on="element",
        how="left",
    )
    p = p.merge(
        xg_mult[merge_keys + ["multiplier"]], on=merge_keys, how="left"
    )
    mult = p["multiplier"].fillna(1.0)
    pred_xg = p["avg_norm_expected_goals"].fillna(0) * mult
    p["predicted_points"] = pred_xg * p["position"].map(GOAL_POINTS).fillna(4)
    p["component_type"] = "goals_scored"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["multiplier"], errors="ignore")
    all_preds.append(p)

    # 3. assists
    xa_mult = multipliers[multipliers["component_type"] == "expected_assists"]
    p = combinations.merge(
        player_averages[["element", "hist_games", "avg_norm_expected_assists"]],
        on="element",
        how="left",
    )
    p = p.merge(
        xa_mult[merge_keys + ["multiplier"]], on=merge_keys, how="left"
    )
    mult = p["multiplier"].fillna(1.0)
    pred_xa = p["avg_norm_expected_assists"].fillna(0) * mult
    p["predicted_points"] = pred_xa * 3.0
    p["component_type"] = "assists"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["multiplier"], errors="ignore")
    all_preds.append(p)

    # 4. saves (GK only)
    saves_mult = multipliers[multipliers["component_type"] == "saves"]
    p = combinations.merge(
        player_averages[["element", "hist_games", "avg_norm_saves"]],
        on="element",
        how="left",
    )
    p = p.merge(
        saves_mult[merge_keys + ["multiplier"]], on=merge_keys, how="left"
    )
    mult = p["multiplier"].fillna(1.0)
    pred_saves = p["avg_norm_saves"].fillna(0) * mult
    p["predicted_points"] = np.where(
        p["position"] == "GK", pred_saves * (1.0 / 3.0), 0.0
    )
    p["component_type"] = "saves"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["multiplier"], errors="ignore")
    all_preds.append(p)

    # 5. conceded_goals (GK/DEF only)
    xgc_mult = multipliers[
        multipliers["component_type"] == "expected_goals_conceded"
    ]
    p = combinations.merge(
        player_averages[
            ["element", "hist_games", "avg_norm_expected_goals_conceded"]
        ],
        on="element",
        how="left",
    )
    if team_xgc is not None and len(team_xgc) > 0:
        p = p.merge(
            team_xgc[["player_team_id", "position", "avg_norm_team_xgc"]],
            on=["player_team_id", "position"],
            how="left",
        )
    p = p.merge(
        xgc_mult[merge_keys + ["multiplier"]], on=merge_keys, how="left"
    )
    mult = p["multiplier"].fillna(1.0)
    pred_xgc = _with_xgc_base(p) * mult
    p["predicted_points"] = np.where(
        p["position"].isin(["GK", "DEF"]), pred_xgc * (-0.5), 0.0
    )
    p["component_type"] = "conceded_goals"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(
        columns=["multiplier", "avg_norm_team_xgc", "avg_norm_expected_goals_conceded"],
        errors="ignore",
    )
    all_preds.append(p)

    # 6. yellow_cards (no multiplier)
    p = combinations.merge(
        player_averages[["element", "hist_games", "avg_yellow_cards"]],
        on="element",
        how="left",
    )
    p["predicted_points"] = p["avg_yellow_cards"].fillna(0) * (-1.0)
    p["component_type"] = "yellow_cards"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["avg_yellow_cards"], errors="ignore")
    all_preds.append(p)

    # 7. clean_sheet (Poisson P(CS) = exp(-xgc))
    p = combinations.merge(
        player_averages[
            ["element", "hist_games", "avg_norm_expected_goals_conceded"]
        ],
        on="element",
        how="left",
    )
    if team_xgc is not None and len(team_xgc) > 0:
        p = p.merge(
            team_xgc[["player_team_id", "position", "avg_norm_team_xgc"]],
            on=["player_team_id", "position"],
            how="left",
        )
    p = p.merge(
        xgc_mult[merge_keys + ["multiplier"]], on=merge_keys, how="left"
    )
    mult = p["multiplier"].fillna(1.0)
    pred_xgc = _with_xgc_base(p) * mult
    p["predicted_xgc"] = pred_xgc
    p["clean_sheet_prob"] = np.exp(-pred_xgc)
    cs_pts = np.where(
        p["position"].isin(["GK", "DEF"]),
        p["clean_sheet_prob"] * 4.0,
        np.where(p["position"] == "MID", p["clean_sheet_prob"] * 1.0, 0.0),
    )
    p["predicted_points"] = cs_pts
    p["component_type"] = "clean_sheet"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(
        columns=[
            "multiplier",
            "predicted_xgc",
            "clean_sheet_prob",
            "avg_norm_team_xgc",
            "avg_norm_expected_goals_conceded",
        ],
        errors="ignore",
    )
    all_preds.append(p)

    # 8. defensive_contribution
    p = combinations.merge(
        player_averages[
            [
                "element",
                "hist_games",
                "avg_defensive_contribution",
                "defensive_contribution_history",
            ]
        ],
        on="element",
        how="left",
    )
    pos_defaults = {"DEF": 7.15, "MID": 6.98, "FWD": 3.80, "GK": 0.0}
    pred_vals = []
    for _, row in p.iterrows():
        avg = row["avg_defensive_contribution"]
        if pd.isna(avg) or avg == 0:
            avg = pos_defaults.get(row["position"], 5.0)
        mult = _get_defensive_fixture_multiplier(
            row["opponent_team_tier"], bool(row["is_home"])
        )
        pred_val = avg * mult
        hist = row.get("defensive_contribution_history") or []
        prob = _calculate_defensive_probability(
            pred_val, row["position"], hist if isinstance(hist, list) else None
        )
        pred_vals.append(prob * 2.0)
    p["predicted_points"] = pred_vals
    p["component_type"] = "defensive_contribution"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(
        columns=[
            "avg_defensive_contribution",
            "defensive_contribution_history",
        ],
        errors="ignore",
    )
    all_preds.append(p)

    # 9. bps (direct average, no multiplier)
    p = combinations.merge(
        player_averages[["element", "hist_games", "avg_bonus_points"]],
        on="element",
        how="left",
    )
    p["predicted_points"] = p["avg_bonus_points"].fillna(0.0)
    p["component_type"] = "bps"
    p["hist_games"] = p["hist_games"].fillna(0)
    p = p.drop(columns=["avg_bonus_points"], errors="ignore")
    all_preds.append(p)

    # Standardize columns and concat
    base_cols = [
        "element",
        "name",
        "position",
        "player_team_id",
        "player_team_tier",
        "event",
        "kickoff_time",
        "opponent_team",
        "opponent_team_tier",
        "is_home",
        "hist_games",
        "component_type",
        "predicted_points",
    ]
    result = []
    for df in all_preds:
        cols = [c for c in base_cols if c in df.columns]
        result.append(df[cols].copy())

    out = pd.concat(result, ignore_index=True)
    logger.debug("Created %d total predictions across %d components", len(out), len(all_preds))
    return out


def _add_team_names(
    predictions: pd.DataFrame, team_tiers: pd.DataFrame
) -> pd.DataFrame:
    """Add player_team_name and opponent_team_name from team_tiers."""
    if "team_name" not in team_tiers.columns:
        logger.warning("team_tiers has no team_name column, skipping name mapping")
        return predictions

    id_to_name = dict(zip(team_tiers["team_id"], team_tiers["team_name"]))
    predictions = predictions.copy()
    predictions["player_team_name"] = predictions["player_team_id"].map(id_to_name)
    predictions["opponent_team_name"] = predictions["opponent_team"].map(id_to_name)
    return predictions


def generate_predictions(
    gw_data: pd.DataFrame,
    fixtures: pd.DataFrame,
    multipliers: pd.DataFrame,
    team_tiers: pd.DataFrame,
    season: str,
    shrink_k: float = SHRINK_K,
    cap_min_games: int = CAP_MIN_GAMES,
    bonus_shrink_k: float = BONUS_SHRINK_K,
) -> pd.DataFrame:
    """
    Generate component-based FPL point predictions for all players and future fixtures.

    Args:
        gw_data: Gameweek player data (must include element, name, position, minutes,
                 fixture, was_home, opponent_team, kickoff_time, defensive_contribution,
                 bonus, and NORMALIZABLE_COMPONENTS).
        fixtures: Fixture data with id, event (or GW), team_h, team_a, kickoff_time.
        multipliers: Component multipliers with component_type, position,
                     player_team_tier, opponent_team_tier, is_home, multiplier.
        team_tiers: Team metadata with team_id, team_tier, team_name (optional).
        season: Season string (e.g. '2025-26').
        shrink_k: Strength of the pull toward the positional prior for xG/xA,
                  in units of games. 0 disables it. See SHRINK_K.
        cap_min_games: Shortest window the outlier cap may touch. See CAP_MIN_GAMES.
        bonus_shrink_k: Same, for bonus points. 0 disables it. See BONUS_SHRINK_K.

    Returns:
        DataFrame with columns: element, name, position, player_team_id, player_team_tier,
        event, kickoff_time, opponent_team, opponent_team_tier, is_home, hist_games,
        component_type, predicted_points, player_team_name, opponent_team_name.
    """
    logger.info("Generating predictions for season %s", season)

    normalized = _normalize_stats(gw_data, fixtures, multipliers, team_tiers, season)
    player_averages = _calculate_player_averages(
        normalized,
        gw_data,
        LAST_N_GAMES,
        shrink_k=shrink_k,
        cap_min_games=cap_min_games,
        bonus_shrink_k=bonus_shrink_k,
    )
    player_assignments = _get_player_team_assignments(normalized)

    gw_col = "GW" if "GW" in gw_data.columns else "round"
    if gw_col not in gw_data.columns:
        raise ValueError("gw_data must have 'GW' or 'round' column")
    last_played_gw = int(gw_data[gw_col].max())

    combinations = _generate_player_fixture_combinations(
        player_assignments, fixtures, team_tiers, last_played_gw
    )

    team_matches = _team_match_xgc(gw_data, fixtures)
    team_xgc = _team_xgc_averages(team_matches, multipliers, team_tiers)

    predictions = _create_component_predictions(
        combinations, player_averages, multipliers, team_xgc=team_xgc
    )
    predictions = _add_team_names(predictions, team_tiers)

    logger.info("Prediction generation complete: %d rows", len(predictions))
    return predictions
