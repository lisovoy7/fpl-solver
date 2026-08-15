# Technical Documentation

Detailed internals of the fpl-solver prediction engine and MILP decision engine.

## Table of Contents

- [Prediction Engine](#prediction-engine)
  - [Data Flow](#data-flow)
  - [Step 1: Stat Normalization](#step-1-stat-normalization)
  - [Step 2: Player Averages](#step-2-player-averages)
  - [Step 3: Player-Fixture Combinations](#step-3-player-fixture-combinations)
  - [Step 4: Component Predictions](#step-4-component-predictions)
  - [Prediction Components Detail](#prediction-components-detail)
- [MILP Decision Engine](#milp-decision-engine)
  - [Problem Formulation Overview](#problem-formulation-overview)
  - [Sets and Indices](#sets-and-indices)
  - [Decision Variables](#decision-variables)
  - [Objective Function](#objective-function)
  - [Constraints](#constraints)
  - [Chip Handling](#chip-handling)
  - [Free Hit Sub-Problem](#free-hit-sub-problem)
  - [Chip Scenario Enumeration](#chip-scenario-enumeration)
- [End-to-End Pipeline](#end-to-end-pipeline)

---

## Prediction Engine

**Module:** `fpl/predict.py`

The prediction engine generates per-player, per-fixture expected FPL points for every future gameweek. It uses a **component-based** approach — each source of FPL points (goals, assists, clean sheets, etc.) is predicted independently, then summed.

### Data Flow

```
Raw GW data (FPL API)        Fixtures (FPL API)        Multipliers (bundled CSV)
         |                          |                           |
         v                          v                           v
   ┌──────────────────────────────────────────────────────┐
   │  Step 1: Stat Normalization (_normalize_stats)       │
   │  Remove fixture difficulty bias from historical stats│
   └───────────────────────┬──────────────────────────────┘
                           v
   ┌──────────────────────────────────────────────────────┐
   │  Step 2: Player Averages (_calculate_player_averages)│
   │  Rolling averages from last N normalized games       │
   └───────────────────────┬──────────────────────────────┘
                           v
   ┌──────────────────────────────────────────────────────┐
   │  Step 3: Player-Fixture Combinations                 │
   │  Cross-join players × future fixtures                │
   └───────────────────────┬──────────────────────────────┘
                           v
   ┌──────────────────────────────────────────────────────┐
   │  Step 4: Component Predictions                       │
   │  Apply multipliers to re-scale for each fixture      │
   └───────────────────────┬──────────────────────────────┘
                           v
                   Predictions DataFrame
           (element, event, component_type, predicted_points)
```

### Step 1: Stat Normalization

**Purpose:** Remove the effect of fixture difficulty from historical stats so that player averages represent "true ability" independent of opponents faced.

**Process:**

1. Filter to appearances with **minutes >= 60** (ensures meaningful samples).
2. Merge each game with its fixture metadata to determine: player's team, opponent team, home/away status.
3. Look up team tiers (1-5) for both the player's team and the opponent.
4. For each stat component (xG, xA, xGC, saves, bonus, etc.), find the corresponding fixture multiplier from the bundled `multipliers.csv`.
5. **Normalize:** divide the raw stat by the multiplier.

**Example:** A midfielder scores xG = 0.6 in a game where the multiplier for "expected_goals" (MID, tier 2, vs tier 5, home) = 1.4. The normalized value is 0.6 / 1.4 = 0.43. This represents what the player would have scored against a "neutral" opponent.

**Normalized components:** `assists`, `bonus`, `clean_sheets`, `expected_assists`, `expected_goals`, `expected_goals_conceded`, `goals_conceded`, `goals_scored`, `own_goals`, `penalties_missed`, `penalties_saved`, `red_cards`, `saves`, `yellow_cards`.

Key detail: we normalize using **expected stats (xG, xA, xGC)** rather than actual goals/assists. Expected stats are derived from shot quality and are more stable predictors of future output.

### Step 2: Player Averages

**Purpose:** Compute a single "baseline ability" number per player per component.

**Process:**

1. Sort each player's normalized appearances by kickoff time.
2. Take the **last 10 games** (configurable via `LAST_N_GAMES`).
3. Calculate the mean of each normalized component across those games.
4. Additionally extract:
   - `avg_defensive_contribution` — mean of raw `defensive_contribution` stat (not normalized, as it has a separate prediction path).
   - `defensive_contribution_history` — full list of recent values (used for probability estimation).
   - `avg_bonus_points` — mean bonus points.

**Output:** One row per player with fields like `avg_norm_expected_goals`, `avg_norm_expected_assists`, `avg_norm_saves`, etc.

### Step 3: Player-Fixture Combinations

**Purpose:** Create every (player, future fixture) pair where the player's team is involved.

For each player, scan all fixtures with `event > last_played_gw`. If the player's team is the home team or away team, create a row with:
- `element` (player ID)
- `event` (gameweek number)
- `opponent_team` and `opponent_team_tier`
- `is_home` (1 or 0)

This naturally handles **double gameweeks** — if a team has two fixtures in the same GW, the player gets two rows for that GW, and their predictions are summed later.

### Step 4: Component Predictions

For each (player, fixture) combination, compute predicted points for 9 components:

### Prediction Components Detail

#### 1. Minutes Played
- **Formula:** `predicted_points = 2.0` (flat)
- **Logic:** Every player who qualifies (meets `min_hist_pct`) is assumed to play 60+ minutes and earn the 2-point appearance award. No multiplier applied.

#### 2. Goals Scored (xG-based)
- **Formula:** `predicted_points = avg_norm_xG × fixture_multiplier × goal_points[position]`
- **Multiplier source:** `expected_goals` from `multipliers.csv`, keyed by (position, player_team_tier, opponent_team_tier, is_home).
- **Goal points by position:** GK = 10, DEF = 6, MID = 5, FWD = 4.
- **Intuition:** A player's baseline xG (normalized) is scaled up/down by how favorable the upcoming fixture is. A FWD with avg_norm_xG = 0.4 facing a tier 5 team at home (multiplier ~1.3) gets 0.4 × 1.3 × 4 = 2.08 predicted goal points.

#### 3. Assists (xA-based)
- **Formula:** `predicted_points = avg_norm_xA × fixture_multiplier × 3.0`
- **Multiplier source:** `expected_assists` from `multipliers.csv`.
- **All positions earn 3 points per assist** in FPL.

#### 4. Saves (GK only)
- **Formula:** `predicted_points = avg_norm_saves × fixture_multiplier × (1/3)`
- **Multiplier source:** `saves` from `multipliers.csv`.
- **FPL rule:** 1 point per 3 saves. Non-GK positions get 0.

#### 5. Goals Conceded (GK/DEF only)
- **Formula:** `predicted_points = avg_norm_xGC × fixture_multiplier × (-0.5)`
- **Multiplier source:** `expected_goals_conceded` from `multipliers.csv`.
- **FPL rule:** -1 point per 2 goals conceded for GK/DEF. Non-GK/DEF positions get 0.

#### 6. Yellow Cards
- **Formula:** `predicted_points = avg_yellow_cards × (-1.0)`
- **No fixture multiplier** — card rate is treated as player-specific, not fixture-dependent.

#### 7. Clean Sheet (Poisson model)
- **Formula:** `P(CS) = exp(-predicted_xGC)`, then:
  - GK/DEF: `predicted_points = P(CS) × 4.0`
  - MID: `predicted_points = P(CS) × 1.0`
  - FWD: `predicted_points = 0`
- **Derivation:** Under a Poisson model, the probability of zero goals conceded is $e^{-\lambda}$ where $\lambda$ = expected goals conceded for the fixture.
- **predicted_xGC** = `avg_norm_xGC × fixture_multiplier` (same as component 5).

#### 8. Defensive Contribution (BPS-based)
- **Formula:** Uses a **normal distribution probability model**.
- **FPL BPS defensive threshold:** DEF needs ≥ 10 BPS, MID needs ≥ 12 BPS from defensive actions to earn 2 bonus points from this component.
- **Process:**
  1. Compute `predicted_value = avg_def_contribution × defensive_fixture_multiplier`.
  2. The defensive fixture multiplier combines opponent tier (from `RAW_XGC_MULTIPLIERS`) and home/away factor (home = 0.846, away = 1.0), then linearly transformed: `0.269 × adjusted + 0.702`.
  3. Estimate probability of exceeding the threshold using `1 - CDF(threshold, predicted_value, std_dev)`.
  4. Standard deviation: uses player-specific historical consistency when ≥ 4 games available (weighted blend of player std and population std). Falls back to population std (`DEF: 3.5`, `MID: 3.9`, `FWD: 2.8`) otherwise.
  5. Special cases: players with ≥ 80% historical success rate get a floor of 0.60-0.70 probability; players with ≤ 20% success rate get capped at 0.30.
  6. `predicted_points = probability × 2.0`.
- **GK always gets 0** for this component.

#### 9. Bonus Points (BPS)
- **Formula:** `predicted_points = avg_bonus_points` (direct average, no multiplier).
- **Logic:** Bonus points are awarded to the top 3 BPS scorers per match. Predicting this from fixtures is noisy, so we use the simple rolling average.

---

## MILP Decision Engine

**Module:** `fpl/solver.py`

The decision engine formulates the entire remaining FPL season as a **single Mixed Integer Linear Program** and solves it with the CBC solver (via PuLP).

### Problem Formulation Overview

**Goal:** Maximize total expected points across all gameweeks in the planning horizon, accounting for:
- Transfer costs (-4 per paid transfer)
- Captain bonus (captain scores double; triple with TC chip)
- Bench value (weighted by substitution probability)
- Chip timing (Wildcard, Free Hit, Bench Boost, Triple Captain)

### Sets and Indices

| Symbol | Description |
|---|---|
| $P$ | Set of all candidate players (the watchlist) |
| $T = \{1, 2, ..., H\}$ | Set of gameweeks in the planning horizon (internal indexing) |
| $E_{p,t}$ | Expected points for player $p$ in gameweek $t$ |

### Decision Variables

| Variable | Type | Description |
|---|---|---|
| $x_{p,t}$ | Binary | 1 if player $p$ is in the squad at GW $t$ |
| $y_{p,t}$ | Binary | 1 if player $p$ is in the starting XI at GW $t$ |
| $c_{p,t}$ | Binary | 1 if player $p$ is captain at GW $t$ |
| $s_{p,t}$ | Binary | 1 if player $p$ is transferred IN at GW $t$ |
| $r_{p,t}$ | Binary | 1 if player $p$ is transferred OUT at GW $t$ |
| $u_t$ | Integer ≥ 0 | Total transfers used at GW $t$ |
| $A_t$ | Integer [0, 5] | Free transfers available at start of GW $t$ |
| $h_t$ | Integer ≥ 0 | Paid (penalty) transfers at GW $t$ |
| $w_t$ | Binary | 1 if Wildcard is played at GW $t$ |

### Objective Function

$$
\max \sum_{t \in T} \left[ \sum_{p \in P} \left( \alpha \cdot E_{p,t} \cdot y_{p,t} + \beta \cdot E_{p,t} \cdot (x_{p,t} - y_{p,t}) + \alpha \cdot E_{p,t} \cdot c_{p,t} \right) - 4 \cdot h_t \right]
$$

Where:
- $\alpha = 1 - p_{sub}$ — lineup weight (probability the starter actually plays)
- $\beta = \frac{11 \cdot p_{sub}}{4}$ — bench weight (expected value of each bench player as a substitute)
- $p_{sub}$ — substitution probability (`sub_probability` config param, default 0.10)
- The captain term `α · E · c` adds an extra copy of the captain's points (making it 2× total)
- $h_t$ — paid transfers, penalized at **-4 points each** (FPL rule)
- `first_gw_penalty` — small artificial penalty (-1 or -2) on GW 1 transfers to avoid front-loading

**Bench Boost GW:** When BB is active, bench players score full points. The complement weights are added:

$$
(1-\alpha) \cdot E_{p,t} \cdot y_{p,t} + (1-\beta) \cdot E_{p,t} \cdot (x_{p,t} - y_{p,t}) + (1-\alpha) \cdot E_{p,t} \cdot c_{p,t}
$$

This makes $\alpha + (1-\alpha) = 1$ for starters and bench alike — everyone scores full points.

**Triple Captain GW:** When TC is active, the captain scores **3×** instead of 2×. An additional captain term is added:

$$
\alpha \cdot E_{p,t} \cdot c_{p,t}
$$

(If both BB and TC are on the same GW — which is blocked by chip conflict rules — a fourth term would also apply.)

**Free Hit GW:** All player points for that GW are set to 0 in the main solver (the FH squad is solved separately). Transfers in a FH GW are penalized at -1000 to force the solver to use the FH mechanism rather than real transfers.

### Constraints

#### Squad Flow

Tracks squad evolution through transfers:

$$
x_{p,1} = \text{initial}_{p} + s_{p,1} - r_{p,1}
$$

$$
x_{p,t} = x_{p,t-1} + s_{p,t} - r_{p,t} \quad \forall t \geq 2
$$

Transfer counts are symmetric (transfers in = transfers out):

$$
u_t = \sum_{p} s_{p,t} = \sum_{p} r_{p,t}
$$

#### Squad Composition

Every GW must maintain:
- Exactly 15 players total
- Exactly 2 GK, 5 DEF, 5 MID, 3 FWD
- At most 3 players from any single club
- Total squad value ≤ budget

#### Lineup Selection

Every GW the starting XI must have:
- Exactly 11 starters
- Exactly 1 GK
- 3-5 DEF, 2-5 MID, 1-3 FWD (valid FPL formations)
- Starters must be in the squad: $y_{p,t} \leq x_{p,t}$
- Captain must be a starter: $c_{p,t} \leq y_{p,t}$
- Exactly 1 captain per GW

#### Transfer Banking

Free transfers accumulate: unused transfers roll over, capped at 5.

$$
A_{t+1} \leq A_t - (u_t - h_t) + 1 + M \cdot w_t
$$

Where $M = 15$ (Big-M) deactivates the constraint when Wildcard is played (making all transfers free).

The upper bound also resets when Wildcard is used:

$$
A_{t+1} \leq A_t + M \cdot (1 - w_t)
$$

$$
A_{t+1} \geq A_t - M \cdot (1 - w_t)
$$

#### Paid Transfer Penalties

Paid transfers = max(0, transfers_used - free_transfers_available):

$$
h_t \geq u_t - A_t - M \cdot w_t
$$

$$
h_t \leq u_t
$$

$$
h_t \leq M \cdot (1 - w_t)
$$

The third constraint ensures $h_t = 0$ when Wildcard is active (all transfers are free during WC).

#### Blank Gameweek (BGW) Handling

Players with no fixture in a GW (no entry in `expected_points`) are prevented from starting or being captain:

$$
y_{p,t} = 0 \quad \text{and} \quad c_{p,t} = 0 \quad \text{if } (p, t) \notin E
$$

They can still be owned in the squad (useful if the BGW is temporary and they play the following week).

#### Non-Playing Player Overrides

For manually flagged non-playing players (injury, suspension), their expected points $E_{p,t}$ are set to 0 in the objective function for the specified GWs. No hard constraint is added — the solver naturally avoids starting them since they contribute 0 points, but may keep them in the squad if they're expected to return.

#### Forced Lineup Overrides

For forced starters, an equality constraint is added:

$$
y_{p,t} = 1 \quad \text{for specified } (p, t)
$$

### Chip Handling

#### Wildcard

- One per half-season (GW 1-19, GW 20-38).
- When active: all transfers are free, transfer banking resets.
- Constraint: sum of wildcard variables over the half ≤ 1 minus already-used count (and similarly for second half).
- Cannot overlap with BB, TC, or FH in the same GW.

#### Bench Boost and Triple Captain

Both are handled **outside** the main MILP, as a fixed parameter passed to the solver that adds objective terms for that GW — but *which* GW gets chosen normally comes from post-hoc scoring against an already-solved plan, not scenario enumeration (see "Chip Scenario Enumeration" below). Enumeration only happens when a chip is forced to a specific GW via config, or already fully used (both collapse to a single fixed option anyway).

#### Free Hit

Free Hit GWs are also determined by scenario enumeration. For a FH GW:
1. All player points in the main solver are set to 0 for that GW.
2. **Squad is frozen**: transfer-in and transfer-out variables are forced to 0 for all players, ensuring the real squad flows through the FH GW unchanged (matching FPL's revert-after-FH rule).
3. Transfer banking is preserved across the FH GW (available transfers carry through).
4. **Lineup, captain, BGW, budget, and forced-lineup constraints are relaxed** for the FH GW in the main solver. Because the FH sub-problem handles lineup selection independently, the main solver's lineup/captain variables are irrelevant. Budget is relaxed because the frozen squad may have a market value exceeding the selling-price-based budget (due to player appreciation) — this is expected and not a real constraint violation.
5. A **separate MILP** (`fpl/free_hit.py`) solves the optimal squad from scratch for that single GW, subject to budget and composition constraints. The FH points are added to the main solver's total afterwards.

### Free Hit Sub-Problem

**Module:** `fpl/free_hit.py` — `_solve_free_hit_milp()`

A standalone single-GW MILP that picks the best possible 15-man squad (unconstrained by the current squad):

**Variables:** $x_p$ (squad), $y_p$ (lineup), $c_p$ (captain) — all binary.

**Objective:** $\max \sum_p E_p \cdot y_p + E_p \cdot c_p$

**Constraints:**
- 15 players, 2/5/5/3 composition, ≤ 3 per club
- 11 starters, valid formation, captain in lineup
- Total cost ≤ budget

This is solved once per FH GW in the planning horizon, and the resulting points are added to the scenario's total.

### Chip Scenario Enumeration

**Module:** `fpl/free_hit.py` — `generate_chip_scenarios()`

Free Hit is a discrete choice that interacts multiplicatively with the continuous optimization (it effectively picks a different squad for one GW), so it is decomposed into **scenarios** the same way the whole chip decision used to be. Each scenario specifies:
- Which GWs to use Free Hit (0, 1, or 2 — one per half-season)
- Which GW to use Bench Boost (-1 = none, or a specific GW)
- Which GW to use Triple Captain (-1 = none, or a specific GW)

**Generation logic:**
1. FH scenarios: enumerate all valid single/double FH placements, respecting the 1-per-half rule.
2. BB options: normally just `[-1]` (deferred — see below); enumerated per-half-available GWs only when `force_bench_boost_gw` is set or BB is already fully used, in which case there's only one legal option anyway.
3. TC options: same rule, mirrored for `force_triple_captain_gw`.
4. Cartesian product of FH × BB × TC, filtered by the **1-chip-per-GW conflict rule** (no two chips on the same GW).

In the common case (both chips available, neither forced) this cartesian product collapses to just the FH options — BB and TC are placed afterwards, not enumerated here.

Each scenario is solved independently by the main MILP solver. The scenario with the highest total expected points (solver objective + FH benefits) wins.

**Forced chip GWs:** When `force_free_hit_gw`, `force_bench_boost_gw`, `force_triple_captain_gw`, or `force_wildcard_gw` is set in config, that chip is pinned to the specified GW and no other options are enumerated for it. For Wildcard, the solver adds a hard constraint forcing the wildcard decision variable to 1 on the specified GW and 0 elsewhere.

**Complexity control:** The `max_scenarios` config param (default: 100) caps the number of scenarios tested — mostly a backstop now that BB/TC deferral keeps the FH-only scenario count small (roughly linear in horizon, not quadratic) in the common case.

### Post-Hoc Bench Boost / Triple Captain Placement

**Module:** `fpl/free_hit.py` — `find_best_bench_boost_gw()`, `find_best_triple_captain_gw()`

Once the FH-only scenarios above are solved, each chip is scored against every solved plan **without re-solving**, then only the top-scored `(FH plan, BB GW, TC GW)` combinations — deduplicated, `chip_reselect_candidates` of them (default 5) — get a full MILP re-solve with both chips fixed, to confirm the true objective. Whichever of those re-solves scores best wins.

The two chips are not equally exact:

- **Triple Captain** only ever adds `lineup_weight × E_pt[captain]` on top of an otherwise-unaffected plan (`create_objective`'s TC term is additive and uniform across players, so TC can't change who's captained). The best legal GW for an already-solved plan is exactly its own highest-scoring legal captain GW — no re-solve needed to find it; the re-solve stage exists to confirm the number, not to search for a better placement.
- **Bench Boost** raises *every* squad member (starters via `lineup_complement`, bench via `bench_complement`) to full weight on its GW — it changes the objective's valuation of the squad the model already picked, rather than adding a fixed bonus. A solve that knew BB was coming could legitimately pick a different bench or different transfers to exploit that GW harder. `find_best_bench_boost_gw()` scores what the *already-chosen* lineup would gain, which is a real approximation of the jointly-optimal choice, not an exact shortcut to it — the re-solve stage is what catches cases where this estimate misleads, not just confirms it.

BB is picked first (excluding FH/wildcard GWs) and excluded from TC's candidate GWs afterwards, rather than jointly optimizing the pair — cheap either way, but scoring every `(BB GW, TC GW)` combination together buys little given both get a full re-solve regardless.

---

## End-to-End Pipeline

**Module:** `run.py`

The full pipeline executed by `python run.py`:

1. **Load config** — parse `config.yaml`, apply defaults.
2. **Auto-detect** — fetch bootstrap data from FPL API, detect current GW, season, and chips used. Merge with config (config overrides API).
3. **Fetch team data** — squad composition, bank balance, selling prices for budget calculation.
4. **Fetch fixtures** — all 380 PL fixtures. Apply `fixture_overrides` from config (DGW/BGW corrections).
5. **Fetch GW data** — per-player per-GW stats for all ~820 players (this is the slow step, ~2-3 minutes).
6. **Generate predictions** — run the prediction engine (Steps 1-4 above).
7. **Build watchlist** — filter players by `min_hist_pct` within the last `max_hist_window` GWs (window shrinks early-season), apply `must_include`/`must_exclude`.
8. **Enumerate chip scenarios** — generate all valid chip combinations.
9. **Pre-calculate Free Hit benefits** — solve the FH sub-problem for every GW in the horizon.
10. **Solve each scenario** — scenarios are independent MILPs and are solved in parallel worker processes (`fpl/scenario_runner.py`). The best solution wins.
11. **Output** — display the optimal strategy (transfers, lineup, captain, chips per GW) on the console and save to `output/strategy_gw{N}.json`.

### Runtime Characteristics

| Step | Typical time |
|---|---|
| API data fetch (GW data) | 2-3 minutes |
| Prediction generation | 5-10 seconds |
| Free Hit pre-calculation | 5-10 seconds |
| Each MILP scenario solve | up to `time_limit_per_scenario` |
| Total (10-GW horizon, all chips available, 3 workers) | ~1.5 minutes — 11 FH-only scenarios (BB/TC deferred, not enumerated) plus up to `chip_reselect_candidates` re-solves. Measured trade-off vs. full FH×BB enumeration: ~0.5% fewer points (637.4 → 634.0) for a ~5.8x speedup (517s → 89s) |

The solver uses the **CBC** (Coin-or Branch and Cut) solver, which is open-source and bundled with PuLP. No external solver installation required. For faster solves, GUROBI can be used by changing `solver_name` (requires a separate license).

### Why the time limit can't just be lowered

`time_limit_per_scenario` behaves like a quality knob at short horizons and like a **coverage** knob at long ones, and the second is far less forgiving.

CBC reports `LpStatusOptimal` both for a proven optimum and for a time-limited incumbent, so a solve that runs out of time normally returns its best find so far. At horizon 19 (~55k binaries) that stops being true: CBC's feasibility pump often finds *no* integer-feasible point at all, so there is no incumbent to return, the scenario is dropped, and the search silently narrows. If every scenario is dropped the job fails outright with `No feasible solution found`.

Measured end-to-end on Cloud Run (horizon 19, all four chips available, 6 workers, 8 vCPU):

| stage-1 limit | scenarios dropped | objective | wall clock |
|---|---|---|---|
| 90s | 0 / 20 | 1187.0 | 386s |
| 60s | 17 / 20 | 1187.0 | 267s |
| 15s | 20 / 20 | job failed | — |

The 60s row is the trap: it returned the correct plan having actually evaluated only three Free Hit placements, because the winner happened to survive. The same config dropped only 4 of 25 on a different day, so the failure rate is not even stable. Treat a non-zero INFEASIBLE count in the logs as lost coverage, not noise.

**A warm start does not fix this** — implemented, measured, rejected. Seeding CBC with a trivially feasible plan (hold the squad, no transfers, no chips) works mechanically (`MIPStart values read for 54530 variables`) but anchors the search to the do-nothing neighbourhood: the full pipeline scored **1130.5 seeded vs 1187.0 unseeded** at the same 90s limit, and the seeded plan dropped the wildcard entirely. Cut generation also has to be disabled for a seeded solve to branch at all, costing further quality. And it does not rescue short limits anyway: CBC applies a MIPStart only after the MPS parse, presolve and root LP — past 15s on a shared vCPU — so a 15s limit still dropped every scenario with the seed present and unread.

The way to make this faster is a smaller model (shorter horizon, tighter watchlist), not a shorter limit or a cleverer start.

### Scenario Parallelism

**Module:** `fpl/scenario_runner.py`

Building a scenario's model costs ~0.2s; CBC's branch-and-bound is effectively all
of the remaining runtime. Since scenarios share no state, they are dispatched to a
`ProcessPoolExecutor` — one CBC thread per worker, because spending cores on
scenarios beats spending them on a single scenario's search tree and the two
compete for the same CPUs.

Measured on a 10-core machine, 20 scenarios at a 6-GW horizon:

| Configuration | Wall clock | Solution quality |
|---|---|---|
| Sequential, 15s limit | 189s | baseline |
| 8 workers, 15s limit | 34s (5.5x) | marginally worse on 6/20 scenarios |
| 8 workers, 60s limit | 41s (4.6x) | better on 2/20, worse on none |

The middle row is the trap: `time_limit` is wall-clock inside CBC, so under load
each scenario gets slightly less search. The fix is to spend part of the speedup
on a longer limit — the bottom row is faster *and* strictly better than the
sequential baseline.

Worker count comes from `--workers`, else `FPL_SOLVER_WORKERS`, else CPU count - 1.
`--workers 1` runs in-process, which is what you want when debugging a solve.

**Results are order-preserved, not completion-ordered**, so the chosen best
scenario is reproducible across runs.

### Time Limits and "Optimal"

CBC reports `LpStatus == Optimal` both for a proven optimum and for a feasible
incumbent it was still improving when the time limit fired; only `prob.sol_status`
distinguishes them. `FPLSolver.solve()` records this as `self.proven_optimal`, and
the scenario runner tags time-limited results in its progress output.

This matters for interpreting results: at a 6-GW horizon a single scenario needed
~400s to prove optimality, and the default 15s limit landed ~4% below that
objective. Scenarios are therefore ranked partly on how much search each happened
to get. Raising `time_limit_per_scenario` (cheap now that scenarios run in
parallel) narrows that spread; `solver.mip_gap` sets a relative gap to stop at
instead.
