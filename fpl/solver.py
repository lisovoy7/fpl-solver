"""
MILP holistic solver for FPL season optimization.

This module solves the complete FPL problem including:
- Squad management and transfers over the planning horizon
- Lineup selection (11 starters, 4 bench - no ordering)
- Captain selection with bonus points
- Chip usage optimization (Wildcard, Bench Boost, Triple Captain)
- Budget and squad composition constraints

The solver maximizes total expected points over the planning horizon while
respecting all FPL rules and constraints.

Predictions and player data are passed as DataFrames - no hardcoded file paths.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pulp

logger = logging.getLogger(__name__)

# FPL Rules Constants
SQUAD_COMPOSITION = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
LINEUP_CONSTRAINTS = {'GK': (1, 1), 'DEF': (3, 5), 'MID': (2, 5), 'FWD': (1, 3)}
TOTAL_SQUAD_SIZE = 15
TOTAL_LINEUP_SIZE = 11
MAX_PLAYERS_PER_CLUB = 3
MAX_FREE_TRANSFERS = 5
TRANSFER_PENALTY_POINTS = -4
FREE_HIT_TRANSFER_PENALTY = -1000
FIRST_GW_TRANSFER_PENALTY = -2
CHIP_WINDOWS = {'first_half': (1, 19), 'second_half': (20, 38)}


class FPLSolver:
    """
    FPL optimization solver using Mixed Integer Linear Programming.

    Solves the complete FPL problem including transfers, lineups, captaincy,
    and chip usage over a specified planning horizon. All data is passed
    as DataFrames - no file paths or season strings.
    """

    def __init__(
        self,
        planning_horizon: int,
        budget: float,
        start_gw: int,
        solver_name: str = 'CBC',
        afcon_enabled: bool = True,
        afcon_trigger_gw: int = 15,
        afcon_transfer_count: int = 5,
        points_multiplier_override: Optional[List[tuple]] = None,
        forced_lineup_players: Optional[List[tuple]] = None,
        non_playing_players: Optional[List[tuple]] = None,
        free_hit_gws: Optional[List[int]] = None,
        first_gw_transfer_penalty: Optional[float] = None,
        sub_probability: float = 0.0,
        bench_boost_gw: int = -1,
        triple_captain_gw: int = -1,
    ):
        """
        Initialize the FPL solver.

        Args:
            planning_horizon: Number of gameweeks to optimize.
            budget: Total budget in units (100 = 10.0M).
            start_gw: Starting gameweek for optimization.
            solver_name: MILP solver to use ('CBC', 'GUROBI', etc.).
            afcon_enabled: Enable AFCON transfer top-up rule.
            afcon_trigger_gw: Gameweek after which AFCON transfers are topped up.
            afcon_transfer_count: Number of transfers to top up to for AFCON.
            points_multiplier_override: List of (player_id, multiplier) tuples.
            forced_lineup_players: List of (player_id, [gw_list]) for forced starters.
            non_playing_players: List of (player_id, [gw_list]) for 0-point overrides.
            free_hit_gws: Gameweeks where Free Hit is used.
            first_gw_transfer_penalty: Penalty for transfers in first GW.
            sub_probability: Probability lineup players won't play (bench valuation).
            bench_boost_gw: Gameweek for Bench Boost chip (-1 = disabled).
            triple_captain_gw: Gameweek for Triple Captain chip (-1 = disabled).
        """
        self.T = planning_horizon
        self.budget = budget
        self.start_gw = start_gw
        self.solver_name = solver_name
        self.afcon_enabled = afcon_enabled
        self.afcon_trigger_gw = afcon_trigger_gw
        self.afcon_transfer_count = afcon_transfer_count
        self.points_multiplier_override = points_multiplier_override or []
        self.forced_lineup_players = forced_lineup_players or []
        self.non_playing_players = non_playing_players or []
        self.free_hit_gws = free_hit_gws or []
        self.first_gw_transfer_penalty = (
            first_gw_transfer_penalty if first_gw_transfer_penalty is not None else FIRST_GW_TRANSFER_PENALTY
        )
        self.sub_probability = sub_probability
        self.bench_boost_gw = bench_boost_gw
        self.triple_captain_gw = triple_captain_gw

        self.players = None
        self.predictions = None
        self.initial_squad = None
        self.initial_transfers = 1
        self.prob = None
        self.variables = {}

        logger.info("Initialized FPL solver with %d GW horizon", planning_horizon)

    def load_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Load player predictions for the planning horizon.

        Args:
            predictions_df: DataFrame with columns element, event, predicted_points,
                and optionally name, position. Multiple rows per (element, event)
                are summed.

        Returns:
            DataFrame with aggregated predictions.
        """
        logger.info("Loading predictions from DataFrame")

        predictions = predictions_df.copy()
        end_gw = self.start_gw + self.T - 1
        predictions = predictions[
            (predictions['event'] >= self.start_gw) & (predictions['event'] <= end_gw)
        ].copy()

        agg_dict = {'predicted_points': 'sum'}
        if 'name' in predictions.columns:
            agg_dict['name'] = 'first'
        if 'position' in predictions.columns:
            agg_dict['position'] = 'first'

        self.predictions = predictions.groupby(['element', 'event']).agg(agg_dict).reset_index()

        self._apply_points_multiplier_override()
        self._apply_free_hit_points_override()

        logger.info("Loaded predictions for %d players", self.predictions['element'].nunique())
        logger.info("Gameweeks: %s", sorted(self.predictions['event'].unique()))

        return self.predictions

    def _apply_points_multiplier_override(self) -> None:
        """Apply points multiplier overrides to predicted points."""
        if not self.points_multiplier_override:
            return

        logger.info("Applying points multiplier overrides")

        for player_id, multiplier in self.points_multiplier_override:
            player_mask = self.predictions['element'] == player_id
            affected_rows = self.predictions[player_mask]

            if len(affected_rows) > 0:
                original_total = self.predictions.loc[player_mask, 'predicted_points'].sum()
                self.predictions.loc[player_mask, 'predicted_points'] *= multiplier
                new_total = self.predictions.loc[player_mask, 'predicted_points'].sum()
                player_name = affected_rows.iloc[0].get('name', player_id)
                logger.info(
                    "  Player %d (%s): %.1fx multiplier applied, total %.1f -> %.1f",
                    player_id, player_name, multiplier, original_total, new_total,
                )
            else:
                logger.warning("  Player %d: No predictions found", player_id)

    def _apply_free_hit_points_override(self) -> None:
        """Override all player points to 0 for Free Hit gameweeks."""
        if not self.free_hit_gws:
            return

        logger.info("Applying Free Hit points override (set to 0) for GWs: %s", self.free_hit_gws)

        for fh_gw in self.free_hit_gws:
            gw_mask = self.predictions['event'] == fh_gw
            affected_rows = self.predictions[gw_mask]

            if len(affected_rows) > 0:
                original_total = self.predictions.loc[gw_mask, 'predicted_points'].sum()
                self.predictions.loc[gw_mask, 'predicted_points'] = 0
                logger.info("  GW %d: %d players set to 0 points (was %.1f total)", fh_gw, len(affected_rows), original_total)
            else:
                logger.warning("  GW %d: No predictions found for Free Hit override", fh_gw)

    def load_player_data(
        self,
        gw_data: pd.DataFrame,
        normalized_data: pd.DataFrame,
        player_subset: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Build player metadata from gw_data and normalized_data.

        Args:
            gw_data: DataFrame with element, GW, name, position, value.
            normalized_data: DataFrame with element, GW, player_team_id.
            player_subset: Optional list of player IDs to include.

        Returns:
            DataFrame with element, name, position, value, team.
        """
        logger.info("Loading player metadata from DataFrames")

        gw_col = next((c for c in ['GW', 'event', 'round'] if c in gw_data.columns), None)
        if gw_col is None:
            raise ValueError("gw_data must have 'GW', 'event', or 'round' column for ordering")
        latest_values = gw_data.sort_values(['element', gw_col]).groupby('element').last().reset_index()

        norm_gw_col = next((c for c in ['GW', 'event', 'round'] if c in normalized_data.columns), None)
        if norm_gw_col is None:
            raise ValueError("normalized_data must have 'GW', 'event', or 'round' column for ordering")
        latest_teams = (
            normalized_data.sort_values(['element', norm_gw_col])
            .groupby('element')
            .last()
            .reset_index()
        )

        required_cols = ['element', 'position', 'value']
        name_col = 'name' if 'name' in latest_values.columns else None
        if name_col is None and 'first_name' in latest_values.columns and 'second_name' in latest_values.columns:
            latest_values = latest_values.copy()
            latest_values['name'] = latest_values['first_name'] + ' ' + latest_values['second_name']
            name_col = 'name'
        elif name_col is None:
            latest_values = latest_values.copy()
            latest_values['name'] = latest_values['element'].astype(str)
            name_col = 'name'

        player_data = latest_values[['element', name_col, 'position', 'value']].merge(
            latest_teams[['element', 'player_team_id']],
            on='element',
            how='left',
        )
        if name_col != 'name':
            player_data = player_data.rename(columns={name_col: 'name'})

        self.players = player_data.rename(columns={'player_team_id': 'team'})

        if player_subset is not None:
            self.players = self.players[self.players['element'].isin(player_subset)].copy()
            logger.info("Filtered to %d players from subset", len(self.players))

        logger.info("Loaded data for %d players", len(self.players))
        return self.players

    def set_initial_squad(self, squad_player_ids: List[int], available_transfers: int = 1) -> None:
        """
        Set the initial squad composition.

        Args:
            squad_player_ids: List of 15 player IDs in current squad.
            available_transfers: Number of free transfers available at start.
        """
        if len(squad_player_ids) != TOTAL_SQUAD_SIZE:
            raise ValueError(f"Initial squad must have exactly {TOTAL_SQUAD_SIZE} players")

        self.initial_squad = squad_player_ids
        self.initial_transfers = available_transfers

        logger.info("Set initial squad with %d players", len(squad_player_ids))
        logger.info("Available transfers: %d", available_transfers)

    def set_chip_state(
        self,
        wildcard_first_half: int = 0,
        wildcard_second_half: int = 0,
    ) -> None:
        """
        Set the current state of chip usage by half-season.

        Args:
            wildcard_first_half: Wildcards used in first half (GW 1-19).
            wildcard_second_half: Wildcards used in second half (GW 20-38).
        """
        self.chips_used = {
            'wildcard_first_half': wildcard_first_half,
            'wildcard_second_half': wildcard_second_half,
        }

        logger.info("Chip state set: first half wildcards %d/1, second half %d/1", wildcard_first_half, wildcard_second_half)

    def create_decision_variables(self) -> None:
        """Create all MILP decision variables."""
        logger.info("Creating MILP decision variables")

        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))

        variables = {}
        variables['x'] = pulp.LpVariable.dicts(
            "own",
            [(p, t) for p in players for t in gameweeks],
            cat='Binary',
        )
        variables['y'] = pulp.LpVariable.dicts(
            "start",
            [(p, t) for p in players for t in gameweeks],
            cat='Binary',
        )
        variables['c'] = pulp.LpVariable.dicts(
            "captain",
            [(p, t) for p in players for t in gameweeks],
            cat='Binary',
        )
        variables['penalty_transfers'] = pulp.LpVariable.dicts(
            "penalty_transfers",
            gameweeks,
            lowBound=0,
            cat='Integer',
        )
        variables['first_gw_penalty_transfers'] = pulp.LpVariable.dicts(
            "first_gw_penalty_transfers",
            gameweeks,
            lowBound=0,
            cat='Integer',
        )
        variables['s'] = pulp.LpVariable.dicts(
            "transfer_in",
            [(p, t) for p in players for t in gameweeks],
            cat='Binary',
        )
        variables['r'] = pulp.LpVariable.dicts(
            "transfer_out",
            [(p, t) for p in players for t in gameweeks],
            cat='Binary',
        )
        variables['u'] = pulp.LpVariable.dicts(
            "transfers_used",
            gameweeks,
            lowBound=0,
            cat='Integer',
        )
        variables['A'] = pulp.LpVariable.dicts(
            "transfers_available",
            gameweeks,
            lowBound=0,
            upBound=MAX_FREE_TRANSFERS,
            cat='Integer',
        )
        variables['wildcard'] = pulp.LpVariable.dicts("wildcard", gameweeks, cat='Binary')

        self.variables = variables
        logger.info("Created all decision variables")

    def create_objective(self) -> None:
        """Create the objective function to maximize total expected points."""
        logger.info("Creating objective function")

        if self.predictions is None:
            raise ValueError("Predictions must be loaded before creating objective")

        self.expected_points = {}
        for _, row in self.predictions.iterrows():
            key = (row['element'], row['event'])
            if key in self.expected_points:
                self.expected_points[key] += row['predicted_points']
            else:
                self.expected_points[key] = row['predicted_points']

        expected_points = self.expected_points
        objective_terms = []
        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))

        lineup_weight = 1.0 - self.sub_probability
        bench_weight = (11.0 * self.sub_probability) / 4.0
        lineup_complement = self.sub_probability
        bench_complement = 1.0 - bench_weight

        if self.sub_probability > 0:
            logger.info(
                "Bench valuation enabled: sub_probability=%.2f, lineup weight=%.2f, bench weight=%.2f",
                self.sub_probability, lineup_weight, bench_weight,
            )

        if self.bench_boost_gw > 0:
            logger.info("Bench Boost enabled for GW %d", self.bench_boost_gw)

        if self.triple_captain_gw > 0:
            logger.info("Triple Captain enabled for GW %d", self.triple_captain_gw)

        for t in gameweeks:
            actual_gw = self.start_gw + t - 1
            is_bench_boost_gw = actual_gw == self.bench_boost_gw
            is_triple_captain_gw = actual_gw == self.triple_captain_gw

            for p in players:
                if (p, actual_gw) not in expected_points:
                    continue

                E_pt = expected_points[(p, actual_gw)]
                is_non_playing = any(
                    p == player_id and actual_gw in gw_list
                    for player_id, gw_list in self.non_playing_players
                )
                if is_non_playing:
                    E_pt = 0

                objective_terms.append(lineup_weight * E_pt * self.variables['y'][(p, t)])

                if self.sub_probability > 0:
                    objective_terms.append(
                        bench_weight * E_pt * (self.variables['x'][(p, t)] - self.variables['y'][(p, t)])
                    )

                objective_terms.append(lineup_weight * E_pt * self.variables['c'][(p, t)])

                if is_bench_boost_gw:
                    objective_terms.append(lineup_complement * E_pt * self.variables['y'][(p, t)])
                    objective_terms.append(
                        bench_complement * E_pt * (self.variables['x'][(p, t)] - self.variables['y'][(p, t)])
                    )
                    objective_terms.append(lineup_complement * E_pt * self.variables['c'][(p, t)])

                if is_triple_captain_gw:
                    objective_terms.append(lineup_weight * E_pt * self.variables['c'][(p, t)])
                    if is_bench_boost_gw:
                        objective_terms.append(lineup_complement * E_pt * self.variables['c'][(p, t)])

        for t in gameweeks:
            actual_gw = self.start_gw + t - 1
            transfer_penalty = FREE_HIT_TRANSFER_PENALTY if actual_gw in self.free_hit_gws else TRANSFER_PENALTY_POINTS
            objective_terms.append(transfer_penalty * self.variables['penalty_transfers'][t])

            if t == 1:
                objective_terms.append(
                    self.first_gw_transfer_penalty * self.variables['first_gw_penalty_transfers'][t]
                )

        self.prob += pulp.lpSum(objective_terms), "Total_Expected_Points"
        logger.info("Objective function created")

    def add_squad_flow_constraints(self) -> None:
        """Add constraints for squad ownership flow and transfers."""
        logger.info("Adding squad flow constraints")

        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))

        for p in players:
            initial_owns = 1 if p in self.initial_squad else 0
            self.prob += (
                self.variables['x'][(p, 1)] == initial_owns + self.variables['s'][(p, 1)] - self.variables['r'][(p, 1)],
                f"Initial_Squad_{p}",
            )

        for t in range(2, self.T + 1):
            for p in players:
                self.prob += (
                    self.variables['x'][(p, t)]
                    == self.variables['x'][(p, t - 1)] + self.variables['s'][(p, t)] - self.variables['r'][(p, t)],
                    f"Squad_Flow_{p}_{t}",
                )

        for t in gameweeks:
            self.prob += (
                self.variables['u'][t] == pulp.lpSum([self.variables['s'][(p, t)] for p in players]),
                f"Transfer_Count_In_{t}",
            )
            self.prob += (
                self.variables['u'][t] == pulp.lpSum([self.variables['r'][(p, t)] for p in players]),
                f"Transfer_Count_Out_{t}",
            )

        logger.info("Squad flow constraints added")

    def add_transfer_banking_constraints(self) -> None:
        """Add constraints for transfer banking and usage."""
        logger.info("Adding transfer banking constraints")

        gameweeks = list(range(1, self.T + 1))

        if self.afcon_enabled and self.start_gw == self.afcon_trigger_gw + 1:
            effective_initial_transfers = self.afcon_transfer_count
            logger.info("AFCON rule applied: initial transfers set to %d", effective_initial_transfers)
        else:
            effective_initial_transfers = self.initial_transfers

        self.prob += (self.variables['A'][1] == effective_initial_transfers, "Initial_Transfers")

        M = TOTAL_SQUAD_SIZE

        for t in range(1, self.T):
            actual_gw = self.start_gw + t - 1
            free_hit_override = actual_gw in self.free_hit_gws
            afcon_override = self.afcon_enabled and actual_gw == self.afcon_trigger_gw

            if free_hit_override:
                self.prob += (self.variables['A'][t + 1] == self.variables['A'][t], f"FreeHit_Transfer_Preserve_{actual_gw}")
                self.prob += (self.variables['A'][t + 1] <= MAX_FREE_TRANSFERS, f"Transfer_Cap_{t}")
                continue
            elif afcon_override:
                self.prob += (self.variables['A'][t + 1] == self.afcon_transfer_count, f"AFCON_Transfer_Override_{t}")
                self.prob += (self.variables['A'][t + 1] <= MAX_FREE_TRANSFERS, f"Transfer_Cap_{t}")
                continue

            free_transfers_used = self.variables['u'][t] - self.variables['penalty_transfers'][t]
            self.prob += (
                self.variables['A'][t + 1]
                <= self.variables['A'][t] - free_transfers_used + 1 + M * self.variables['wildcard'][t],
                f"Transfer_Banking_Normal_{t}",
            )
            self.prob += (
                self.variables['A'][t + 1] <= self.variables['A'][t] + M * (1 - self.variables['wildcard'][t]),
                f"Transfer_Banking_Wildcard_Upper_{t}",
            )
            self.prob += (
                self.variables['A'][t + 1] >= self.variables['A'][t] - M * (1 - self.variables['wildcard'][t]),
                f"Transfer_Banking_Wildcard_Lower_{t}",
            )
            self.prob += (self.variables['A'][t + 1] <= MAX_FREE_TRANSFERS, f"Transfer_Cap_{t}")

        logger.info("Transfer banking constraints added")

    def add_squad_composition_constraints(self) -> None:
        """Add squad composition and budget constraints."""
        logger.info("Adding squad composition constraints")

        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))
        player_position = dict(zip(self.players['element'], self.players['position']))
        player_price = dict(zip(self.players['element'], self.players['value']))
        player_club = dict(zip(self.players['element'], self.players['team']))

        for t in gameweeks:
            self.prob += (
                pulp.lpSum([self.variables['x'][(p, t)] for p in players]) == TOTAL_SQUAD_SIZE,
                f"Squad_Size_{t}",
            )
            self.prob += (
                pulp.lpSum([player_price[p] * self.variables['x'][(p, t)] for p in players]) <= self.budget,
                f"Budget_{t}",
            )
            for position, required_count in SQUAD_COMPOSITION.items():
                position_players = [p for p in players if player_position[p] == position]
                self.prob += (
                    pulp.lpSum([self.variables['x'][(p, t)] for p in position_players]) == required_count,
                    f"Squad_{position}_{t}",
                )
            clubs = set(player_club.values())
            for club in clubs:
                club_players = [p for p in players if player_club[p] == club]
                if club_players:
                    self.prob += (
                        pulp.lpSum([self.variables['x'][(p, t)] for p in club_players]) <= MAX_PLAYERS_PER_CLUB,
                        f"Club_{club}_{t}",
                    )

        logger.info("Squad composition constraints added")

    def add_lineup_constraints(self) -> None:
        """Add lineup selection constraints."""
        logger.info("Adding lineup constraints")

        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))
        player_position = dict(zip(self.players['element'], self.players['position']))

        for t in gameweeks:
            for p in players:
                self.prob += (
                    self.variables['y'][(p, t)] <= self.variables['x'][(p, t)],
                    f"Start_Owned_{p}_{t}",
                )
            self.prob += (
                pulp.lpSum([self.variables['y'][(p, t)] for p in players]) == TOTAL_LINEUP_SIZE,
                f"Lineup_Size_{t}",
            )
            for position, (min_count, max_count) in LINEUP_CONSTRAINTS.items():
                position_players = [p for p in players if player_position[p] == position]
                self.prob += (
                    pulp.lpSum([self.variables['y'][(p, t)] for p in position_players]) >= min_count,
                    f"Min_{position}_{t}",
                )
                self.prob += (
                    pulp.lpSum([self.variables['y'][(p, t)] for p in position_players]) <= max_count,
                    f"Max_{position}_{t}",
                )
            self.prob += (
                pulp.lpSum([self.variables['c'][(p, t)] for p in players]) == 1,
                f"One_Captain_{t}",
            )
            for p in players:
                self.prob += (
                    self.variables['c'][(p, t)] <= self.variables['y'][(p, t)],
                    f"Captain_Starter_{p}_{t}",
                )

        logger.info("Lineup constraints added")

    def add_advanced_constraints(self) -> None:
        """Add advanced constraints (forced lineup, non-playing, BGW)."""
        logger.info("Adding advanced constraints")
        self._add_forced_lineup_constraints()
        self._add_non_playing_player_constraints()
        self._add_bgw_constraints()
        logger.info("Advanced constraints added")

    def _add_forced_lineup_constraints(self) -> None:
        """Add constraints to force specific players to start."""
        if not self.forced_lineup_players:
            return

        logger.info("Adding forced lineup constraints")

        for player_id, forced_gws in self.forced_lineup_players:
            player_name = "Unknown"
            if self.players is not None:
                player_data = self.players[self.players['element'] == player_id]
                if len(player_data) > 0:
                    player_name = player_data.iloc[0]['name']
                else:
                    logger.warning("Player %d not found in watchlist - skipping forced lineup", player_id)
                    continue

            for gw in forced_gws:
                internal_gw = gw - self.start_gw + 1
                if 1 <= internal_gw <= self.T:
                    if (player_id, internal_gw) not in self.variables['y']:
                        logger.warning("Player %d not in optimization model - cannot force lineup for GW %d", player_id, gw)
                        continue
                    self.prob += (
                        self.variables['y'][(player_id, internal_gw)] == 1,
                        f"Forced_Lineup_{player_id}_GW{gw}",
                    )
                    logger.info("  Player %d (%s) forced to start in GW %d", player_id, player_name, gw)
                else:
                    logger.warning(
                        "  Player %d forced lineup for GW %d outside planning horizon",
                        player_id, gw,
                    )

    def _add_non_playing_player_constraints(self) -> None:
        """Log non-playing overrides (handled in objective)."""
        if not self.non_playing_players:
            return

        logger.info("Adding non-playing player constraints (0 points override)")

        for player_id, non_playing_gws in self.non_playing_players:
            player_name = "Unknown"
            if self.players is not None:
                player_data = self.players[self.players['element'] == player_id]
                if len(player_data) > 0:
                    player_name = player_data.iloc[0]['name']

            for gw in non_playing_gws:
                internal_gw = gw - self.start_gw + 1
                if 1 <= internal_gw <= self.T:
                    logger.info("  Player %d (%s) will get 0 points in GW %d", player_id, player_name, gw)
                else:
                    logger.warning(
                        "  Player %d non-playing override for GW %d outside planning horizon",
                        player_id, gw,
                    )

    def _add_bgw_constraints(self) -> None:
        """Prevent starting/captaining players with no fixture (BGW)."""
        if not hasattr(self, 'expected_points') or self.expected_points is None:
            logger.warning("Expected points not available - skipping BGW constraints")
            return

        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))
        bgw_combinations = []
        bgw_by_gw = {}

        for t in gameweeks:
            actual_gw = self.start_gw + t - 1
            bgw_by_gw[actual_gw] = []
            for p in players:
                if (p, actual_gw) not in self.expected_points:
                    bgw_combinations.append((p, t, actual_gw))
                    bgw_by_gw[actual_gw].append(p)

        if not bgw_combinations:
            logger.info("No Blank Game Weeks detected")
            return

        logger.info("Detected %d BGW player-GW combinations", len(bgw_combinations))
        constraints_added = 0
        for p, t, actual_gw in bgw_combinations:
            if (p, t) in self.variables['y']:
                self.prob += (self.variables['y'][(p, t)] == 0, f"BGW_No_Start_{p}_GW{actual_gw}")
                constraints_added += 1
            if (p, t) in self.variables['c']:
                self.prob += (self.variables['c'][(p, t)] == 0, f"BGW_No_Captain_{p}_GW{actual_gw}")
                constraints_added += 1

        logger.info("Added %d BGW constraints", constraints_added)

    def add_chip_constraints(self) -> None:
        """Add chip usage constraints."""
        logger.info("Adding chip constraints")

        gameweeks = list(range(1, self.T + 1))
        chips_used = getattr(self, 'chips_used', {})

        first_half_gws = [
            t for t in gameweeks
            if CHIP_WINDOWS['first_half'][0] <= (self.start_gw + t - 1) <= CHIP_WINDOWS['first_half'][1]
        ]
        second_half_gws = [
            t for t in gameweeks
            if CHIP_WINDOWS['second_half'][0] <= (self.start_gw + t - 1) <= CHIP_WINDOWS['second_half'][1]
        ]

        wc_used_first = chips_used.get('wildcard_first_half', 0)
        wc_used_second = chips_used.get('wildcard_second_half', 0)

        if first_half_gws:
            remaining_wc_first = max(0, 1 - wc_used_first)
            self.prob += (
                pulp.lpSum([self.variables['wildcard'][t] for t in first_half_gws]) <= remaining_wc_first,
                "Max_Wildcard_First_Half",
            )

        if second_half_gws:
            remaining_wc_second = max(0, 1 - wc_used_second)
            self.prob += (
                pulp.lpSum([self.variables['wildcard'][t] for t in second_half_gws]) <= remaining_wc_second,
                "Max_Wildcard_Second_Half",
            )

        if self.bench_boost_gw > 0:
            bb_internal = self.bench_boost_gw - self.start_gw + 1
            if 1 <= bb_internal <= self.T:
                self.prob += (
                    self.variables['wildcard'][bb_internal] == 0,
                    f"No_Wildcard_BenchBoost_GW{self.bench_boost_gw}",
                )
                logger.info("  Wildcard blocked in GW %d (Bench Boost)", self.bench_boost_gw)

        if self.triple_captain_gw > 0:
            tc_internal = self.triple_captain_gw - self.start_gw + 1
            if 1 <= tc_internal <= self.T:
                self.prob += (
                    self.variables['wildcard'][tc_internal] == 0,
                    f"No_Wildcard_TripleCaptain_GW{self.triple_captain_gw}",
                )
                logger.info("  Wildcard blocked in GW %d (Triple Captain)", self.triple_captain_gw)

        for fh_gw in self.free_hit_gws:
            fh_internal = fh_gw - self.start_gw + 1
            if 1 <= fh_internal <= self.T:
                self.prob += (
                    self.variables['wildcard'][fh_internal] == 0,
                    f"No_Wildcard_FreeHit_GW{fh_gw}",
                )
                logger.info("  Wildcard blocked in GW %d (Free Hit)", fh_gw)

        M_transfers = 15
        for t in gameweeks:
            self.prob += (
                self.variables['penalty_transfers'][t]
                >= self.variables['u'][t] - self.variables['A'][t] - M_transfers * self.variables['wildcard'][t],
                f"Penalty_Lower_Bound_{t}",
            )
            self.prob += (
                self.variables['penalty_transfers'][t] <= self.variables['u'][t],
                f"Penalty_Upper_Bound_{t}",
            )
            self.prob += (
                self.variables['penalty_transfers'][t] <= M_transfers * (1 - self.variables['wildcard'][t]),
                f"Penalty_Wildcard_Zero_{t}",
            )

        for t in gameweeks:
            self.prob += (
                self.variables['first_gw_penalty_transfers'][t] <= self.variables['u'][t],
                f"First_GW_Penalty_Upper_{t}",
            )
            self.prob += (
                self.variables['first_gw_penalty_transfers'][t] <= M_transfers * (1 - self.variables['wildcard'][t]),
                f"First_GW_Penalty_Wildcard_Zero_{t}",
            )
            self.prob += (
                self.variables['first_gw_penalty_transfers'][t] >= self.variables['u'][t] - M_transfers * self.variables['wildcard'][t],
                f"First_GW_Penalty_Lower_{t}",
            )

        logger.info("Chip constraints added")

    def build_model(self) -> None:
        """Build the complete MILP model."""
        logger.info("Building complete MILP model")

        self.prob = pulp.LpProblem("FPL_Optimization", pulp.LpMaximize)
        self.create_decision_variables()
        self.create_objective()
        self.add_squad_flow_constraints()
        self.add_transfer_banking_constraints()
        self.add_squad_composition_constraints()
        self.add_lineup_constraints()
        self.add_chip_constraints()
        self.add_advanced_constraints()

        logger.info("MILP model built successfully")
        logger.info("Variables: %d, Constraints: %d", len(self.prob.variables()), len(self.prob.constraints))

    def solve(self, time_limit: Optional[int] = None) -> bool:
        """
        Solve the MILP model.

        Args:
            time_limit: Maximum solving time in seconds.

        Returns:
            True if optimal solution found, False otherwise.
        """
        logger.info("Solving MILP with %s", self.solver_name)

        if self.prob is None:
            raise ValueError("Model must be built before solving")

        if self.solver_name.upper() == 'CBC':
            solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit)
        elif self.solver_name.upper() == 'GUROBI':
            solver = pulp.GUROBI_CMD(msg=1, timeLimit=time_limit)
        else:
            solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit)

        self.prob.solve(solver)
        status = pulp.LpStatus[self.prob.status]
        logger.info("Solver status: %s", status)

        if self.prob.status == pulp.LpStatusOptimal:
            logger.info("Optimal objective value: %.2f", pulp.value(self.prob.objective))
            return True
        else:
            logger.warning("No optimal solution found")
            return False

    def extract_solution(self) -> Dict:
        """
        Extract the solution from the solved model.

        Returns:
            Dictionary with objective_value, start_gw, squads, lineups, captains,
            transfers, chips (including 'triple_captain' when applicable).
        """
        if self.prob.status != pulp.LpStatusOptimal:
            raise ValueError("Model must be solved optimally before extracting solution")

        logger.info("Extracting solution")

        solution = {
            'objective_value': float(pulp.value(self.prob.objective)),
            'start_gw': self.start_gw,
            'squads': {},
            'lineups': {},
            'captains': {},
            'transfers': {},
            'chips': {},
        }

        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))

        for t in gameweeks:
            squad = [p for p in players if self.variables['x'][(p, t)].varValue == 1]
            solution['squads'][t] = squad

        for t in gameweeks:
            starters = [p for p in players if self.variables['y'][(p, t)].varValue == 1]
            squad = [p for p in players if self.variables['x'][(p, t)].varValue == 1]
            bench = [p for p in squad if p not in starters]
            solution['lineups'][t] = {'starters': starters, 'bench': bench}

        for t in gameweeks:
            captain = [p for p in players if self.variables['c'][(p, t)].varValue == 1]
            if captain:
                solution['captains'][t] = captain[0]

        for t in gameweeks:
            transfers_in = [p for p in players if self.variables['s'][(p, t)].varValue == 1]
            transfers_out = [p for p in players if self.variables['r'][(p, t)].varValue == 1]
            transfers_used = int(self.variables['u'][t].varValue)
            free_transfers_available = int(self.variables['A'][t].varValue)
            wildcard_active = self.variables['wildcard'][t].varValue == 1
            penalty_transfers_count = int(self.variables['penalty_transfers'][t].varValue)

            if wildcard_active:
                free_transfers = transfers_used
                paid_transfers = 0
            else:
                paid_transfers = penalty_transfers_count
                free_transfers = transfers_used - paid_transfers

            solution['transfers'][t] = {
                'in': transfers_in,
                'out': transfers_out,
                'count': transfers_used,
                'free_transfers': free_transfers,
                'paid_transfers': paid_transfers,
                'available_transfers': free_transfers_available,
                'wildcard_active': wildcard_active,
            }

        for t in gameweeks:
            chips_used = []
            if self.variables['wildcard'][t].varValue == 1:
                chips_used.append('wildcard')
            actual_gw = self.start_gw + t - 1
            if actual_gw == self.bench_boost_gw:
                chips_used.append('bench_boost')
            if actual_gw == self.triple_captain_gw:
                chips_used.append('triple_captain')
            if actual_gw in self.free_hit_gws:
                chips_used.append('free_hit')
            solution['chips'][t] = chips_used

        logger.info("Solution extracted successfully")
        return solution
