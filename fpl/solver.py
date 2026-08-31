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
        points_multiplier_override: Optional[List[tuple]] = None,
        forced_lineup_players: Optional[List[tuple]] = None,
        non_playing_players: Optional[List[tuple]] = None,
        free_hit_gws: Optional[List[int]] = None,
        first_gw_transfer_penalty: Optional[float] = None,
        sub_probability: float = 0.0,
        bench_boost_gw: int = -1,
        triple_captain_gw: int = -1,
        force_wildcard_gw: Optional[int] = None,
        bank: Optional[float] = None,
        selling_discounts: Optional[Dict[int, int]] = None,
        player_clubs: Optional[Dict[int, int]] = None,
        club_gameweeks: Optional[Dict[int, set]] = None,
    ):
        """
        Initialize the FPL solver.

        Args:
            planning_horizon: Number of gameweeks to optimize.
            budget: Total budget in units (100 = 10.0M).
            start_gw: Starting gameweek for optimization.
            solver_name: MILP solver to use ('CBC', 'GUROBI', etc.).
            points_multiplier_override: List of (player_id, multiplier) tuples.
            forced_lineup_players: List of (player_id, [gw_list]) for forced starters.
            non_playing_players: List of (player_id, [gw_list]) for 0-point overrides.
            free_hit_gws: Gameweeks where Free Hit is used.
            first_gw_transfer_penalty: Penalty for transfers in first GW.
            sub_probability: Probability lineup players won't play (bench valuation).
            bench_boost_gw: Gameweek for Bench Boost chip (-1 = disabled).
            triple_captain_gw: Gameweek for Triple Captain chip (-1 = disabled).
            force_wildcard_gw: Force wildcard on this GW (None = let solver decide).
            bank: Cash in hand in units (5 = 0.5M). None = derive it from `budget` minus
                what the initial squad is worth at its own sale prices.
            selling_discounts: {player_id: units} the manager loses by selling a player
                they already own, i.e. market price minus FPL selling price. Only players
                in the initial squad can have one; everyone else sells for what they cost.
            player_clubs: {player_id: club_id} for every player in the game.
            club_gameweeks: {club_id: set of gameweeks that club has a fixture in}.
                Together these two are how a blank gameweek is detected — see
                _add_bgw_constraints. Omit both and it falls back to the old test.
        """
        self.T = planning_horizon
        self.budget = budget
        self.start_gw = start_gw
        self.solver_name = solver_name
        self.points_multiplier_override = points_multiplier_override or []
        self.forced_lineup_players = forced_lineup_players or []
        self.non_playing_players = non_playing_players or []
        self.free_hit_gws = free_hit_gws or []
        self.first_gw_transfer_penalty = first_gw_transfer_penalty if first_gw_transfer_penalty is not None else -1
        self.sub_probability = sub_probability
        self.bench_boost_gw = bench_boost_gw
        self.triple_captain_gw = triple_captain_gw
        self.force_wildcard_gw = force_wildcard_gw
        self.bank = bank
        self.selling_discounts = selling_discounts or {}
        self.player_clubs = player_clubs or {}
        self.club_gameweeks = club_gameweeks
        self._club_cache = None

        self.players = None
        self.predictions = None
        self.initial_squad = None
        self.initial_transfers = 1
        self.prob = None
        self.variables = {}
        self.proven_optimal = False

        logger.debug("Initialized FPL solver with %d GW horizon", planning_horizon)

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
        logger.debug("Loading predictions from DataFrame")

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

        logger.debug("Loaded predictions for %d players", self.predictions['element'].nunique())
        logger.debug("Gameweeks: %s", sorted(self.predictions['event'].unique()))

        return self.predictions

    def _apply_points_multiplier_override(self) -> None:
        """Apply points multiplier overrides to predicted points."""
        if not self.points_multiplier_override:
            return

        logger.debug("Applying points multiplier overrides")

        for player_id, multiplier in self.points_multiplier_override:
            player_mask = self.predictions['element'] == player_id
            affected_rows = self.predictions[player_mask]

            if len(affected_rows) > 0:
                original_total = self.predictions.loc[player_mask, 'predicted_points'].sum()
                self.predictions.loc[player_mask, 'predicted_points'] *= multiplier
                new_total = self.predictions.loc[player_mask, 'predicted_points'].sum()
                player_name = affected_rows.iloc[0].get('name', player_id)
                logger.debug(
                    "  Player %d (%s): %.1fx multiplier applied, total %.1f -> %.1f",
                    player_id, player_name, multiplier, original_total, new_total,
                )
            else:
                logger.warning("  Player %d: No predictions found", player_id)

    def _apply_free_hit_points_override(self) -> None:
        """Override all player points to 0 for Free Hit gameweeks."""
        if not self.free_hit_gws:
            return

        logger.debug("Applying Free Hit points override (set to 0) for GWs: %s", self.free_hit_gws)

        for fh_gw in self.free_hit_gws:
            gw_mask = self.predictions['event'] == fh_gw
            affected_rows = self.predictions[gw_mask]

            if len(affected_rows) > 0:
                original_total = self.predictions.loc[gw_mask, 'predicted_points'].sum()
                self.predictions.loc[gw_mask, 'predicted_points'] = 0
                logger.debug("  GW %d: %d players set to 0 points (was %.1f total)", fh_gw, len(affected_rows), original_total)
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
        logger.debug("Loading player metadata from DataFrames")

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

        # A player can be in gw_data and absent from predictions — a squad member with
        # no history, synthesized by proxy_predict.ensure_players_present. The left
        # merge above leaves his club NaN, and self.players['team'] IS the club map
        # behind the max-3-per-club constraint, so he would quietly escape it. gw_data
        # carries the club for exactly these rows; prefer it over nothing.
        if 'team' in latest_values.columns:
            fallback_clubs = dict(zip(latest_values['element'], latest_values['team']))
            player_data['player_team_id'] = player_data['player_team_id'].fillna(
                player_data['element'].map(fallback_clubs)
            )

        self.players = player_data.rename(columns={'player_team_id': 'team'})

        if player_subset is not None:
            self.players = self.players[self.players['element'].isin(player_subset)].copy()
            logger.debug("Filtered to %d players from subset", len(self.players))

        logger.debug("Loaded data for %d players", len(self.players))
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

        # Verify all squad members are in the player pool
        if self.players is not None:
            pool_ids = set(self.players['element'].tolist())
            missing = [p for p in squad_player_ids if p not in pool_ids]
            if missing:
                logger.warning(
                    "%d initial squad players missing from player pool: %s "
                    "(they will have 0 expected points in all GWs)",
                    len(missing), missing,
                )

        logger.debug("Set initial squad with %d players", len(squad_player_ids))
        logger.debug("Available transfers: %d", available_transfers)

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

        logger.debug("Chip state set: first half wildcards %d/1, second half %d/1", wildcard_first_half, wildcard_second_half)

    def create_decision_variables(self) -> None:
        """Create all MILP decision variables."""
        logger.debug("Creating MILP decision variables")

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
        # Cash in hand at the end of each gameweek. Continuous and floored at zero, which
        # is the whole budget rule: every purchase has to be funded out of the bank plus
        # what that gameweek's sales raised. Continuous rather than integer on purpose —
        # prices are already whole tenths, so the arithmetic lands on integers anyway and
        # declaring them integer would only give CBC more to branch on.
        variables['bank'] = pulp.LpVariable.dicts(
            "bank",
            gameweeks,
            lowBound=0,
            cat='Continuous',
        )

        self.variables = variables
        logger.debug("Created all decision variables")

    def create_objective(self) -> None:
        """Create the objective function to maximize total expected points."""
        logger.debug("Creating objective function")

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
            logger.debug(
                "Bench valuation enabled: sub_probability=%.2f, lineup weight=%.2f, bench weight=%.2f",
                self.sub_probability, lineup_weight, bench_weight,
            )

        if self.bench_boost_gw > 0:
            logger.debug("Bench Boost enabled for GW %d", self.bench_boost_gw)

        if self.triple_captain_gw > 0:
            logger.debug("Triple Captain enabled for GW %d", self.triple_captain_gw)

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
        logger.debug("Objective function created")

    def add_squad_flow_constraints(self) -> None:
        """Add constraints for squad ownership flow and transfers."""
        logger.debug("Adding squad flow constraints")

        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))

        fh_internal_gws = {
            gw - self.start_gw + 1
            for gw in self.free_hit_gws
            if 1 <= gw - self.start_gw + 1 <= self.T
        }

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

        # Free Hit GWs: freeze the real squad — no transfers in or out.
        # The FH sub-problem picks an independent optimal squad for that GW.
        if fh_internal_gws:
            logger.debug("Freezing squad flow for Free Hit GWs: %s", sorted(
                self.start_gw + t - 1 for t in fh_internal_gws))
            for t in fh_internal_gws:
                for p in players:
                    self.prob += (self.variables['s'][(p, t)] == 0, f"FH_No_In_{p}_{t}")
                    self.prob += (self.variables['r'][(p, t)] == 0, f"FH_No_Out_{p}_{t}")

        for t in gameweeks:
            self.prob += (
                self.variables['u'][t] == pulp.lpSum([self.variables['s'][(p, t)] for p in players]),
                f"Transfer_Count_In_{t}",
            )
            self.prob += (
                self.variables['u'][t] == pulp.lpSum([self.variables['r'][(p, t)] for p in players]),
                f"Transfer_Count_Out_{t}",
            )

        logger.debug("Squad flow constraints added")

    def add_transfer_banking_constraints(self) -> None:
        """Add constraints for transfer banking and usage."""
        logger.debug("Adding transfer banking constraints")

        gameweeks = list(range(1, self.T + 1))

        self.prob += (self.variables['A'][1] == self.initial_transfers, "Initial_Transfers")

        M = TOTAL_SQUAD_SIZE

        for t in range(1, self.T):
            actual_gw = self.start_gw + t - 1
            free_hit_override = actual_gw in self.free_hit_gws

            if free_hit_override:
                self.prob += (self.variables['A'][t + 1] == self.variables['A'][t], f"FreeHit_Transfer_Preserve_{actual_gw}")
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

        logger.debug("Transfer banking constraints added")

    def add_squad_composition_constraints(self) -> None:
        """Add squad size, positional and per-club constraints.

        Money is handled separately, by add_budget_constraints().
        """
        logger.debug("Adding squad composition constraints")

        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))
        player_position = dict(zip(self.players['element'], self.players['position']))
        player_club = dict(zip(self.players['element'], self.players['team']))

        # Computed from the same player_club map the constraints below use, so the two
        # can never disagree about which club a player belongs to.
        club_excess = self._grandfathered_club_excess(player_club)

        for t in gameweeks:
            self.prob += (
                pulp.lpSum([self.variables['x'][(p, t)] for p in players]) == TOTAL_SQUAD_SIZE,
                f"Squad_Size_{t}",
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
                if not club_players:
                    continue

                held = pulp.lpSum([self.variables['x'][(p, t)] for p in club_players])
                extra = club_excess.get(club, 0)

                if not extra:
                    self.prob += (held <= MAX_PLAYERS_PER_CLUB, f"Club_{club}_{t}")
                    continue

                # A grandfathered over-limit club (see _grandfathered_club_excess): the
                # squad may carry the excess for as long as it is left alone, but any
                # transfer in this GW has to land on a fully compliant squad.
                #
                # One row per player, rather than one aggregate row, is what keeps this
                # exact without a new binary. `s[p, t]` is already binary, so each row
                # independently says "if this transfer happens, the club is capped at
                # MAX_PLAYERS_PER_CLUB". Writing it once against u[t] instead would
                # over-restrict: `held + extra * u[t] <= 3 + extra` forces the club down
                # to two players on a double transfer. And introducing a per-GW "did I
                # transfer" binary is what broke prod in a1879db — the extra integers
                # push CBC past its feasibility-pump cliff at horizon 19 and scenarios
                # come back INFEASIBLE.
                #
                # No monotonicity chain is needed to stop the excess reappearing later:
                # climbing back to 4 requires a transfer in, and that transfer's own row
                # caps the club at 3.
                for p in players:
                    self.prob += (
                        held + extra * self.variables['s'][(p, t)]
                        <= MAX_PLAYERS_PER_CLUB + extra,
                        f"Club_{club}_{t}_Grandfathered_{p}",
                    )

        logger.debug("Squad composition constraints added")

    def _grandfathered_club_excess(self, player_club: Dict[int, int]) -> Dict[int, int]:
        """
        Clubs the initial squad already breaches the per-club limit for, and by how much.

        FPL applies the three-per-club limit at transfer time, not continuously: when a
        player you own moves to a club you already hold three of, you keep all four. So a
        squad handed to us can be legitimately over the limit, and rejecting it is wrong
        — a real dev job (76711f57, team 124578, four Arsenal players) died with
        `No feasible solution found` for exactly this reason. Letting the excess survive
        a transfer would be equally wrong, which is what the caller enforces.

        Returns {club: excess} for over-limit clubs only, so a legal squad leaves the
        model byte-identical to its long-tested shape.
        """
        if not self.initial_squad:
            return {}

        # `team` arrives as float64 (the join that builds it admits missing values), so
        # a club can be NaN as well as absent. The caller skips a NaN club anyway — it
        # matches no player, since NaN != NaN — but counting one here would invent an
        # excess for a club that does not exist.
        counts: Dict[int, int] = {}
        for p in self.initial_squad:
            club = player_club.get(p)
            if club is not None and not pd.isna(club):
                counts[club] = counts.get(club, 0) + 1

        excess = {
            club: count - MAX_PLAYERS_PER_CLUB
            for club, count in counts.items()
            if count > MAX_PLAYERS_PER_CLUB
        }

        if excess:
            logger.info(
                "Initial squad holds more than %d players from club(s) %s — "
                "grandfathering it; any transfer must restore compliance",
                MAX_PLAYERS_PER_CLUB,
                {club: MAX_PLAYERS_PER_CLUB + e for club, e in excess.items()},
            )

        return excess

    def sale_prices(self) -> Dict[int, float]:
        """
        What each player raises when sold, in units.

        Everyone sells for their market price except the players the manager already
        owns, who sell for the FPL selling price — purchase price plus half of any rise.
        `selling_discounts` carries that shortfall per player.

        Applied on every sale of an owned player, including a sell-then-buy-back-then-sell
        round trip within the horizon, where strictly it should only apply the first time.
        Tracking that needs a per-player indicator chain to distinguish the original
        holding from a re-bought one, and buying a player back at a higher price than he
        sold for is rarely part of a good plan anyway. Erring on the pessimistic side of
        a rare case is the cheap trade.
        """
        prices = dict(zip(self.players['element'], self.players['value']))
        return {
            p: max(0.0, float(price) - float(self.selling_discounts.get(p, 0)))
            for p, price in prices.items()
        }

    def opening_bank(self) -> float:
        """
        Cash in hand before the first gameweek of the plan, in units.

        Supplied by the caller when known. Otherwise derived from `budget`, which is
        "what the squad would raise if sold, plus the bank" — so subtracting the sale
        value of the opening squad leaves the bank.
        """
        if self.bank is not None:
            return max(0.0, float(self.bank))

        sale = self.sale_prices()
        held = sum(sale.get(p, 0.0) for p in (self.initial_squad or []))
        derived = float(self.budget) - held
        if derived < 0:
            # Only reachable when the caller gave a budget that can't fund its own squad
            # at the model's prices — a squad/budget pair from different points in time,
            # or price data that has moved since. Clamping keeps the plan solvable and
            # simply means no spare cash rather than negative cash.
            logger.warning(
                "Derived opening bank was negative (%.1f); clamping to 0", derived / 10
            )
            return 0.0
        return derived

    def add_budget_constraints(self) -> None:
        """
        Track cash gameweek by gameweek and never let it go below zero.

        This replaces the old "squad market value <= budget" check, which measured the
        wrong thing: the budget it compared against was the squad's *selling* value plus
        the bank, so a squad that had risen in price read as unaffordable to keep, and the
        model was pushed into selling players purely to balance an arithmetic error.

        Cash flow is what FPL actually enforces — a purchase has to be funded, holding a
        player costs nothing however much he has appreciated — and it needs no Free Hit
        exception: FH gameweeks force every transfer variable to zero, so the balance just
        carries across untouched.
        """
        logger.debug("Adding budget (cash flow) constraints")

        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))
        buy_price = dict(zip(self.players['element'], self.players['value']))
        sell_price = self.sale_prices()
        opening = self.opening_bank()

        for t in gameweeks:
            spent = pulp.lpSum(
                [float(buy_price[p]) * self.variables['s'][(p, t)] for p in players]
            )
            raised = pulp.lpSum(
                [float(sell_price[p]) * self.variables['r'][(p, t)] for p in players]
            )
            previous = opening if t == 1 else self.variables['bank'][t - 1]
            self.prob += (
                self.variables['bank'][t] == previous + raised - spent,
                f"Bank_Balance_{t}",
            )

        logger.debug(
            "Budget constraints added (opening bank %.1fM, %d discounted players)",
            opening / 10, len(self.selling_discounts),
        )

    def add_lineup_constraints(self) -> None:
        """Add lineup selection constraints.

        Free Hit GWs are skipped: the main solver ignores lineup/captain
        selection because the FH sub-problem handles it independently.
        All y/c variables are implicitly 0 on FH GWs (no points to gain).
        """
        logger.debug("Adding lineup constraints")

        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))
        player_position = dict(zip(self.players['element'], self.players['position']))

        fh_internal_gws = {
            gw - self.start_gw + 1
            for gw in self.free_hit_gws
            if 1 <= gw - self.start_gw + 1 <= self.T
        }

        for t in gameweeks:
            for p in players:
                self.prob += (
                    self.variables['y'][(p, t)] <= self.variables['x'][(p, t)],
                    f"Start_Owned_{p}_{t}",
                )

            if t in fh_internal_gws:
                continue

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

        logger.debug("Lineup constraints added")

    def add_advanced_constraints(self) -> None:
        """Add advanced constraints (forced lineup, non-playing, BGW)."""
        logger.debug("Adding advanced constraints")
        self._add_forced_lineup_constraints()
        self._add_non_playing_player_constraints()
        self._add_bgw_constraints()
        logger.debug("Advanced constraints added")

    def _add_forced_lineup_constraints(self) -> None:
        """Add constraints to force specific players to start.

        Free Hit GWs are skipped because lineup selection is handled
        by the FH sub-problem independently.
        """
        if not self.forced_lineup_players:
            return

        logger.debug("Adding forced lineup constraints")

        fh_gw_set = set(self.free_hit_gws)

        for player_id, forced_gws in self.forced_lineup_players:
            player_name = "Unknown"
            if self.players is not None:
                player_data = self.players[self.players['element'] == player_id]
                if len(player_data) > 0:
                    player_name = player_data.iloc[0]['name']
                else:
                    logger.warning("Player %d not found in watchlist - skipping forced lineup", player_id)
                    continue

            out_of_range_gws = []
            for gw in forced_gws:
                if gw in fh_gw_set:
                    logger.debug("  Player %d (%s) forced lineup skipped for FH GW %d", player_id, player_name, gw)
                    continue
                internal_gw = gw - self.start_gw + 1
                if 1 <= internal_gw <= self.T:
                    if (player_id, internal_gw) not in self.variables['y']:
                        logger.warning("Player %d not in optimization model - cannot force lineup for GW %d", player_id, gw)
                        continue
                    # A club with no fixture that week can't field anyone, so pinning a
                    # start here would contradict the blank-gameweek rule and make the
                    # whole scenario infeasible. Skip it loudly instead: one impossible
                    # instruction should cost the caller that instruction, not the plan.
                    # Callers that can report back should refuse before it gets this far.
                    if self.club_gameweeks is not None:
                        club = self._club_of(player_id)
                        if club is not None and gw not in self.club_gameweeks.get(club, set()):
                            logger.warning(
                                "Player %d (%s) forced to start in GW %d but his club has no "
                                "fixture then - ignoring that gameweek",
                                player_id, player_name, gw,
                            )
                            continue
                    self.prob += (
                        self.variables['y'][(player_id, internal_gw)] == 1,
                        f"Forced_Lineup_{player_id}_GW{gw}",
                    )
                    logger.debug("  Player %d (%s) forced to start in GW %d", player_id, player_name, gw)
                else:
                    out_of_range_gws.append(gw)

            if out_of_range_gws:
                # One line per player, not per GW — see the matching comment in
                # _add_non_playing_player_constraints for why that matters here.
                logger.debug(
                    "  Player %d (%s) forced lineup outside planning horizon for GWs %s",
                    player_id, player_name, out_of_range_gws,
                )

    def _add_non_playing_player_constraints(self) -> None:
        """Log non-playing overrides (handled in objective)."""
        if not self.non_playing_players:
            return

        logger.debug("Adding non-playing player constraints (0 points override)")

        for player_id, non_playing_gws in self.non_playing_players:
            player_name = "Unknown"
            if self.players is not None:
                player_data = self.players[self.players['element'] == player_id]
                if len(player_data) > 0:
                    player_name = player_data.iloc[0]['name']

            in_range = [gw for gw in non_playing_gws if 1 <= gw - self.start_gw + 1 <= self.T]
            out_of_range_count = len(non_playing_gws) - len(in_range)

            if in_range:
                logger.debug("  Player %d (%s) will get 0 points in GWs %s", player_id, player_name, in_range)
            if out_of_range_count:
                # Routine, not actionable — an "unknown return date" absence commonly spans
                # the rest of the season while the horizon only covers a few GWs. One line
                # per player, not per GW: this loop runs once per (player, gw) in
                # non_playing_players, which for a long absence and a short horizon can mean
                # dozens of out-of-range GWs per player — logging each at WARNING is enough
                # log volume, under Cloud Run's request-scoped CPU throttling, to stretch an
                # otherwise sub-second call into minutes.
                logger.debug(
                    "  Player %d (%s): %d non-playing GWs outside planning horizon",
                    player_id, player_name, out_of_range_count,
                )

    def _club_of(self, player_id: int):
        """The club a player belongs to, or None if we genuinely don't know.

        Prefers the caller-supplied map (built from the FPL bootstrap, so it covers
        every player in the game) and falls back to the team column on the player
        frame, which is only populated for players who appear in the predictions.

        Built once and cached: the blank-gameweek rule asks this for every player in
        every gameweek, and scanning the frame each time would be thousands of lookups.
        """
        if self._club_cache is None:
            cache = {}
            if self.players is not None and 'team' in self.players.columns:
                for element, club in zip(self.players['element'], self.players['team']):
                    if club is not None and not pd.isna(club):
                        cache[int(element)] = int(club)
            # The bootstrap map wins — it is complete, where the frame's column is only
            # filled in for players who made it into the predictions.
            for element, club in self.player_clubs.items():
                cache[int(element)] = int(club)
            self._club_cache = cache
        return self._club_cache.get(player_id)

    def _add_bgw_constraints(self) -> None:
        """Prevent starting/captaining players whose club has no fixture (BGW).

        The test is the CLUB's fixture list, not whether this particular player has a
        points forecast. Those are different questions and conflating them was a real
        bug: forecasts are only built for players with a 60+ minute appearance this
        season, so two gameweeks in, roughly two thirds of the game has none. Every one
        of those players was being read as "his club isn't playing" and banned from every
        lineup — which quietly benched squad members who were perfectly fit, and turned
        any `forced_lineup` naming one of them into a flat contradiction ("must start" and
        "cannot start") that failed the entire solve. A player we can't forecast is worth
        0 points, which the objective already handles; he is not unavailable.

        Free Hit GWs are skipped because lineup selection is handled by the
        FH sub-problem; the main solver's y/c variables are unconstrained
        (and have 0 expected points) on those GWs.
        """
        if not hasattr(self, 'expected_points') or self.expected_points is None:
            logger.warning("Expected points not available - skipping BGW constraints")
            return

        players = self.players['element'].tolist()
        gameweeks = list(range(1, self.T + 1))

        fh_internal_gws = {
            gw - self.start_gw + 1
            for gw in self.free_hit_gws
            if 1 <= gw - self.start_gw + 1 <= self.T
        }

        if self.club_gameweeks is None:
            logger.warning(
                "No club fixture map supplied - falling back to forecast presence for BGW "
                "detection, which mis-reads an unforecast player as having no fixture"
            )

        bgw_combinations = []
        bgw_by_gw = {}
        unknown_clubs = set()

        for t in gameweeks:
            if t in fh_internal_gws:
                continue
            actual_gw = self.start_gw + t - 1
            bgw_by_gw[actual_gw] = []
            for p in players:
                if (p, actual_gw) in self.expected_points:
                    continue
                if self.club_gameweeks is not None:
                    club = self._club_of(p)
                    if club is None:
                        # No club, so no way to tell a blank from a missing forecast.
                        # He is worth 0 either way, so leave him startable rather than
                        # risk banning a player the caller may have forced in.
                        unknown_clubs.add(p)
                        continue
                    if actual_gw in self.club_gameweeks.get(club, set()):
                        # His club plays. He simply has no forecast — 0 points, but
                        # available.
                        continue
                bgw_combinations.append((p, t, actual_gw))
                bgw_by_gw[actual_gw].append(p)

        if unknown_clubs:
            logger.warning(
                "No club known for %d players - left startable: %s",
                len(unknown_clubs), sorted(unknown_clubs)[:20],
            )

        if not bgw_combinations:
            logger.debug("No Blank Game Weeks detected")
            return

        logger.debug("Detected %d BGW player-GW combinations", len(bgw_combinations))
        constraints_added = 0
        for p, t, actual_gw in bgw_combinations:
            if (p, t) in self.variables['y']:
                self.prob += (self.variables['y'][(p, t)] == 0, f"BGW_No_Start_{p}_GW{actual_gw}")
                constraints_added += 1
            if (p, t) in self.variables['c']:
                self.prob += (self.variables['c'][(p, t)] == 0, f"BGW_No_Captain_{p}_GW{actual_gw}")
                constraints_added += 1

        logger.debug("Added %d BGW constraints", constraints_added)

    def add_chip_constraints(self) -> None:
        """Add chip usage constraints."""
        logger.debug("Adding chip constraints")

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

        # Force wildcard on a specific GW if requested
        if self.force_wildcard_gw is not None:
            wc_internal = self.force_wildcard_gw - self.start_gw + 1
            if 1 <= wc_internal <= self.T:
                self.prob += (
                    self.variables['wildcard'][wc_internal] == 1,
                    f"Force_Wildcard_GW{self.force_wildcard_gw}",
                )
                for t in gameweeks:
                    if t != wc_internal:
                        self.prob += (
                            self.variables['wildcard'][t] == 0,
                            f"No_Wildcard_Other_GW{self.start_gw + t - 1}",
                        )
                logger.debug("  Wildcard forced on GW %d", self.force_wildcard_gw)

        if self.bench_boost_gw > 0:
            bb_internal = self.bench_boost_gw - self.start_gw + 1
            if 1 <= bb_internal <= self.T:
                self.prob += (
                    self.variables['wildcard'][bb_internal] == 0,
                    f"No_Wildcard_BenchBoost_GW{self.bench_boost_gw}",
                )
                logger.debug("  Wildcard blocked in GW %d (Bench Boost)", self.bench_boost_gw)

        if self.triple_captain_gw > 0:
            tc_internal = self.triple_captain_gw - self.start_gw + 1
            if 1 <= tc_internal <= self.T:
                self.prob += (
                    self.variables['wildcard'][tc_internal] == 0,
                    f"No_Wildcard_TripleCaptain_GW{self.triple_captain_gw}",
                )
                logger.debug("  Wildcard blocked in GW %d (Triple Captain)", self.triple_captain_gw)

        for fh_gw in self.free_hit_gws:
            fh_internal = fh_gw - self.start_gw + 1
            if 1 <= fh_internal <= self.T:
                self.prob += (
                    self.variables['wildcard'][fh_internal] == 0,
                    f"No_Wildcard_FreeHit_GW{fh_gw}",
                )
                logger.debug("  Wildcard blocked in GW %d (Free Hit)", fh_gw)

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

        logger.debug("Chip constraints added")

    def build_model(self) -> None:
        """Build the complete MILP model."""
        logger.debug("Building complete MILP model")

        self.prob = pulp.LpProblem("FPL_Optimization", pulp.LpMaximize)
        self.create_decision_variables()
        self.create_objective()
        self.add_squad_flow_constraints()
        self.add_transfer_banking_constraints()
        self.add_squad_composition_constraints()
        self.add_budget_constraints()
        self.add_lineup_constraints()
        self.add_chip_constraints()
        self.add_advanced_constraints()

        logger.debug("MILP model built: %d vars, %d constraints",
                     len(self.prob.variables()), len(self.prob.constraints))

    def solve(
        self,
        time_limit: Optional[int] = None,
        threads: Optional[int] = None,
        mip_gap: Optional[float] = None,
    ) -> bool:
        """
        Solve the MILP model.

        Args:
            time_limit: Maximum solving time in seconds.
            threads: CBC worker threads. None leaves CBC's default (1). Only set
                this when scenarios are solved one at a time — when scenarios run
                in parallel processes, threads must stay 1 or the processes
                oversubscribe the CPU and every solve gets slower.
            mip_gap: Relative optimality gap to stop at (e.g. 0.005 = 0.5%).
                None solves to proven optimality or the time limit.

        Returns:
            True if a usable solution was found, False otherwise.

        Note:
            True does not mean "proven optimal". CBC reports LpStatusOptimal both
            for a proven optimum and for a feasible incumbent it was still
            improving when the time limit fired; the two are only distinguishable
            via `prob.sol_status`. `self.proven_optimal` records which one it was.
        """
        logger.debug("Solving MILP with %s", self.solver_name)

        if self.prob is None:
            raise ValueError("Model must be built before solving")

        options = [f"ratio {mip_gap}"] if mip_gap is not None else []
        kwargs = {'msg': 0, 'timeLimit': time_limit}
        if threads is not None:
            kwargs['threads'] = threads

        if self.solver_name.upper() == 'GUROBI':
            solver = pulp.GUROBI_CMD(msg=0, timeLimit=time_limit)
        else:
            solver = pulp.PULP_CBC_CMD(options=options, **kwargs)

        self.prob.solve(solver)
        status = pulp.LpStatus[self.prob.status]
        self.proven_optimal = self.prob.sol_status == pulp.LpSolutionOptimal
        logger.debug("Solver status: %s (proven optimal: %s)", status, self.proven_optimal)

        if self.prob.status == pulp.LpStatusOptimal:
            if not self.proven_optimal:
                logger.debug(
                    "Time limit reached — returning best incumbent (%.2f), search was not complete",
                    pulp.value(self.prob.objective),
                )
            else:
                logger.debug("Optimal objective value: %.2f", pulp.value(self.prob.objective))
            return True
        else:
            logger.warning("No optimal solution found")
            return False

    def extract_solution(self) -> Dict:
        """
        Extract the solution from the solved model.

        Returns:
            Dictionary with objective_value, start_gw, squads, lineups, captains,
            transfers, bank, chips (including 'triple_captain' when applicable).
        """
        if self.prob.status != pulp.LpStatusOptimal:
            raise ValueError("Model must be solved optimally before extracting solution")

        logger.debug("Extracting solution")

        solution = {
            'objective_value': float(pulp.value(self.prob.objective)),
            'start_gw': self.start_gw,
            'squads': {},
            'lineups': {},
            'captains': {},
            'transfers': {},
            'bank': {},
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

        # Replay the free-transfer ledger, spending free transfers before taking hits.
        #
        # The MILP bounds penalty_transfers below by (u - A) but never pins it there, and
        # add_transfer_banking_constraints() banks A[t+1] off `u - penalty_transfers` — so
        # a solve can over-declare a hit to un-spend a free transfer and roll it forward,
        # reporting -4 in a gameweek that had a transfer spare. FPL applies free transfers
        # first and gives you no way to decline one, so such a plan is not executable as
        # printed. Pinning penalty_transfers in the model needs an indicator per gameweek,
        # and at horizon 19 those extra binaries push CBC past the feasibility-pump cliff
        # described in TECHNICAL.md (scenarios come back INFEASIBLE and the job dies), so
        # the ledger is rebuilt here instead.
        #
        # This loses nothing. Over-declaring is exactly break-even — a -4 buys a banked
        # transfer worth at most one future -4 — so the schedule the solver picked is
        # still optimal, and replaying it honestly yields the same total hits. Only the
        # gameweeks they are attributed to change.
        honest_available = {}
        honest_paid = {}
        available = self.initial_transfers
        for t in gameweeks:
            honest_available[t] = available
            used = int(self.variables['u'][t].varValue)
            if self.variables['wildcard'][t].varValue == 1 or (self.start_gw + t - 1) in self.free_hit_gws:
                # Both chips make the gameweek's transfers free and leave the balance
                # untouched, matching Transfer_Banking_Wildcard_* / FreeHit_Transfer_Preserve.
                honest_paid[t] = 0
                continue
            honest_paid[t] = max(0, used - available)
            available = min(MAX_FREE_TRANSFERS, available - min(used, available) + 1)

        for t in gameweeks:
            transfers_in = [p for p in players if self.variables['s'][(p, t)].varValue == 1]
            transfers_out = [p for p in players if self.variables['r'][(p, t)].varValue == 1]
            transfers_used = int(self.variables['u'][t].varValue)
            free_transfers_available = honest_available[t]
            wildcard_active = self.variables['wildcard'][t].varValue == 1
            paid_transfers = honest_paid[t]
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

        # Cash left after that gameweek's moves, in units. Read straight off the model so
        # the number reported can never disagree with the constraint that produced it.
        # Rounded to the nearest tenth: prices are whole tenths, so anything else is
        # floating-point noise from the LP relaxation.
        for t in gameweeks:
            value = self.variables['bank'][t].varValue
            solution['bank'][t] = round(float(value)) if value is not None else None

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

        logger.debug("Solution extracted successfully")
        return solution
