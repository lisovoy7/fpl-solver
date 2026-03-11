# fpl-solver

Holistic Fantasy Premier League optimization using Mixed Integer Linear Programming (MILP).

**fpl-solver** plans your entire remaining FPL season in one go: transfers, lineup, captaincy, and chip timing (Wildcard, Free Hit, Bench Boost, Triple Captain) — all solved simultaneously under FPL's real constraints.

## How It Works

The tool has two engines that run in sequence:

### 1. Prediction Engine

Generates per-player expected points for every future fixture using:

- **Component-based prediction** — goals, assists, clean sheets, saves, bonus, defensive contribution, etc. are each predicted independently using **expected stats (xG, xA, xGC)** rather than actual goals/assists, which reduces variance and gives more stable projections
- **Fixture multipliers** — historical data (5 seasons) reveals how each stat component varies by position, opponent tier, and home/away. Player averages are normalized to remove fixture bias, then re-scaled for each upcoming fixture
- **Poisson model** for clean sheet probability (from expected goals conceded)
- **Player-specific consistency** for defensive contribution (DEF/MID threshold probability)

### 2. Decision Engine (MILP Solver)

Formulates the entire remaining season as a single Mixed Integer Linear Program using [PuLP](https://coin-or.github.io/pulp/):

- **Decision variables**: squad ownership, starting XI, captain, transfers in/out — per player per gameweek
- **Objective**: maximize total expected points across all gameweeks (with captain bonus, bench value, and transfer penalties)
- **Constraints**: budget, squad composition (2 GK / 5 DEF / 5 MID / 3 FWD), formation rules (3-5-2, 4-4-2, etc.), max 3 per club, transfer banking (up to 5 free transfers), chip rules
- **Chip enumeration**: Wildcard, Free Hit, Bench Boost, and Triple Captain are decomposed into scenarios and solved independently — the best combination wins. All four chips follow the same rule: one allowed per half-season (GW 1-19 and GW 20-38), giving two uses per season

## Quick Start

```bash
# 1. Clone
git clone https://github.com/lisovoy7/fpl-solver.git
cd fpl-solver

# 2. Install dependencies
pip install -r requirements.txt

# 3. Edit config.yaml with your team ID and free transfers
#    (see Configuration Reference below)

# 4. Run
python run.py
```

First run takes a few minutes because it fetches per-player stats from the FPL API (~900 players). Subsequent runs with `--skip-predictions` reuse cached predictions.

### CLI Flags

| Flag | Description |
|---|---|
| `--skip-predictions` | Reuse `output/predictions.csv` from a previous run |
| `--horizon N` | Override planning horizon (default: rest of season) |
| `--no-chips` | Disable chip optimization (faster, no FH/BB/TC) |
| `--config PATH` | Use a different config file |

## Configuration Reference

All parameters live in `config.yaml`. Below is a complete reference for every parameter.

### Required Parameters

| Parameter | Type | Description |
|---|---|---|
| `team_id` | int | Your FPL team ID. Find it in your team URL: `fantasy.premierleague.com/entry/YOUR_ID/` |
| `free_transfers` | int | Free transfers you currently have available. Must be set manually — the FPL API does not reliably expose this value |

### Auto-detected Parameters

These are fetched from the FPL API at runtime. You only need to set them if auto-detection gives the wrong result.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `current_gw` | int | auto | The upcoming gameweek (first GW whose deadline hasn't passed). Uncomment to override |

### Solver Parameters

Control how the MILP optimization behaves. All have sensible defaults.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `solver.planning_horizon` | int or `"rest_of_season"` | `rest_of_season` | How many GWs ahead to plan. `rest_of_season` plans through GW 38. An integer like `5` plans a short horizon (faster, useful for testing) |
| `solver.min_hist_games` | int | `7` | Minimum number of 60+ minute appearances a player must have this season to enter the candidate pool. Lower values include more players but with less reliable stats. Higher values are more conservative |
| `solver.sub_probability` | float | `0.10` | Probability that each starting-XI player won't play on any given GW (rotation risk). This determines how much bench value matters. `0.10` means ~1.1 expected substitutions per GW. Set to `0.0` to ignore bench value entirely |
| `solver.first_gw_transfer_penalty` | float | `-1` | Artificial points penalty per transfer made in the first GW of the horizon. Prevents the solver from making transfers that only look good because GW 1 is the most "certain" in the plan. Negative value = mild penalty |
| `solver.time_limit_per_scenario` | int | `15` | Maximum seconds the MILP solver spends on each chip scenario. Increase if solutions are suboptimal (solver reports gap > 0) |
| `solver.max_scenarios` | int | `100` | Cap on total chip scenarios to evaluate. Prevents combinatorial explosion when many chips are unused and the horizon is long |

### Chip Usage State

How many times each chip has been used this season. Auto-detected from the FPL API if omitted. All four chip types follow the same rule: **one per half-season** (GW 1-19 and GW 20-38), so each can be 0, 1, or 2. The solver automatically enumerates all unused chips — there are no enable/disable toggles.

If you set a value here, it overrides the API detection. If you omit a value (or set it to `null`), the API-detected count is used.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `chips.wildcards_used` | int (0-2) | auto | How many Wildcards have been used. 1 = first-half used, 2 = both halves used |
| `chips.free_hits_used` | int (0-2) | auto | How many Free Hits have been used |
| `chips.bench_boost_used` | int (0-2) | auto | How many Bench Boosts have been used |
| `chips.triple_captain_used` | int (0-2) | auto | How many Triple Captains have been used |

### Transfer Top-up

Models a mid-season transfer window (e.g. AFCON, injury crisis) where the solver is allowed extra free transfers at a specific GW.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `transfer_topup.enabled` | bool | `true` | Whether the top-up rule is active |
| `transfer_topup.trigger_gw` | int | `15` | The GW at which extra transfers become available |
| `transfer_topup.transfer_count` | int | `5` | Number of extra free transfers granted at `trigger_gw` |

### Fixture Overrides

Override the gameweek assignment of specific fixtures. Use this when the Premier League reschedules matches (postponements, double gameweeks, blank gameweeks) and the FPL API hasn't been updated yet. Each entry moves a fixture (identified by home and away team IDs) to a new GW. Team IDs can be found in `data/team_tiers.csv`.

| Parameter | Type | Description |
|---|---|---|
| `fixture_overrides[].home_team` | int | FPL team ID of the home side |
| `fixture_overrides[].away_team` | int | FPL team ID of the away side |
| `fixture_overrides[].gameweek` | int | New GW number for this fixture |

```yaml
fixture_overrides:
  # Arsenal (1) vs Newcastle (15) rescheduled to GW 33 (double gameweek)
  - home_team: 1
    away_team: 15
    gameweek: 33
  # Brighton (6) vs Chelsea (7) moved to GW 33
  - home_team: 6
    away_team: 7
    gameweek: 33
```

### Player Overrides

These let you inject domain knowledge the model can't detect on its own. The FPL API does NOT reflect transfers you've already made for the upcoming gameweek, and it has no injury/suspension data. Use these overrides to correct for that.

#### `non_playing`

Players who won't play in specific GWs (injury, suspension, rotation). The solver sets their expected points to 0 for those GWs but may keep them in the squad if it expects them back later.

```yaml
non_playing:
  - player: 661     # Ekitike — knee injury
    gameweeks: [28, 29, 30]
  - player: 441     # Dorgu — suspended
    gameweeks: [30]
```

#### `forced_lineup`

Force specific players into the starting XI for specific GWs. Use when you have strong conviction or want to lock a differential.

```yaml
forced_lineup:
  - player: 235     # Palmer
    gameweeks: [30, 31]
```

#### `points_multiplier`

Scale a player's predicted points up or down across all GWs. Useful when you believe the model under/over-estimates a player (e.g. a new signing with little history, or a player whose form has clearly shifted).

```yaml
points_multiplier:
  - player: 235     # Palmer — model underestimates
    multiplier: 1.5
  - player: 267     # Sarr — form has dropped
    multiplier: 0.7
```

#### `excluded_players`

Player IDs to permanently remove from the candidate pool. The solver will never consider buying or owning these players.

```yaml
excluded_players:
  - 5      # Gabriel — never want to own
  - 237    # Enzo
```

#### `extra_players`

Player IDs to force into the candidate pool even if they don't meet the `min_hist_games` threshold. By default, a player needs at least `min_hist_games` appearances of 60+ minutes to be considered. Use `extra_players` to bypass this filter for players you trust despite limited game time — e.g. new signings, returning-from-injury players, or promising players who've only recently broken into the starting XI.

```yaml
extra_players:
  - 256    # Munoz — new signing, only 3 starts but looks strong
  - 267    # Sarr — returning from injury, want solver to consider him
```

## Output

- **Console**: full GW-by-GW strategy (transfers, lineup, captain, chips)
- **File**: `output/strategy_gw{N}.json` with structured results

## Disclaimers

### Pending Transfers Not Visible

The FPL API shows your squad as of the **last completed gameweek**. If you have already made transfers for the upcoming gameweek in the FPL app, the solver **will not see them** — your squad and free transfer count will be stale. Always run the solver **before** making transfers in the app.

### Free Transfers

Free transfers cannot be reliably detected from the API. Always set `free_transfers` manually in `config.yaml`.

### Injuries and Suspensions

The solver has **no injury/suspension detection**. You must manually add injured or suspended players to the `non_playing` section in `config.yaml` (see Player Overrides above).

### Bundled Data

#### `data/multipliers.csv` — Fixture Multipliers

Pre-computed from 5 seasons of FPL data (2020-2025). Each row describes how a stat component (e.g. expected_goals) varies based on:
- **Position** (GK, DEF, MID, FWD)
- **Player's team tier** (1-5, where 1 = strongest)
- **Opponent's team tier** (1-5)
- **Home/Away**

A multiplier of 1.2 means "players in this situation historically produce 20% more than baseline."

#### `data/team_tiers.csv` — Team Tiers

Maps each Premier League team to a tier (1-5). Tiers are based on historical performance and are used by the multiplier system. **Update this file at the start of each new season** to reflect promoted/relegated teams and any tier changes.

Format: `season,team_id,team_name,team_tier`

## Project Structure

```
fpl-solver/
  run.py              # CLI entry point
  config.yaml         # All user configuration
  requirements.txt
  data/
    multipliers.csv   # Pre-computed fixture multipliers (5 seasons)
    team_tiers.csv    # Team tier assignments (current season)
  fpl/
    __init__.py
    api.py            # All FPL API calls (single source of truth)
    config.py         # YAML config loader
    predict.py        # Prediction engine (normalize + predict)
    solver.py         # MILP holistic solver
    watchlist.py      # Player filtering for solver
    free_hit.py       # Free Hit calculator + chip scenario generation
  output/             # Generated at runtime (.gitignored)
```

## License

MIT
