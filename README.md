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

All parameters live in `config.yaml`. The only **required** fields are:

| Field | Type | Description |
|---|---|---|
| `team_id` | int | Your FPL team ID (from your team URL) |
| `free_transfers` | int | Free transfers available right now |

Everything else has sensible defaults or is auto-detected from the FPL API. See `config.yaml` for the full annotated reference.

### Auto-detected Values

| Value | Source | Override in config |
|---|---|---|
| Current gameweek | First event with deadline > now | `current_gw` |
| Chips used | Team history endpoint | `chips` section |
| Current season | Bootstrap data | (not overridable) |

### Player Overrides

These let you inject domain knowledge the model can't detect on its own.

**`non_playing`** — Players who won't play in specific GWs (injury, suspension, rotation). The solver sets their points to 0 but may keep them in the squad if it expects them back later.

```yaml
non_playing:
  - player: 661     # Ekitike — knee injury
    gameweeks: [28, 29, 30]
  - player: 441     # Dorgu — suspended
    gameweeks: [30]
```

**`forced_lineup`** — Force specific players into the starting XI for specific GWs. Use when you have strong conviction or want to lock a differential.

```yaml
forced_lineup:
  - player: 235     # Palmer
    gameweeks: [30, 31]
```

**`points_multiplier`** — Scale a player's predicted points up or down across all GWs. Useful when you believe the model under/over-estimates a player (e.g. a new signing with little history).

```yaml
points_multiplier:
  - player: 235     # Palmer — model underestimates
    multiplier: 1.5
  - player: 267     # Sarr — form has dropped
    multiplier: 0.7
```

**`excluded_players`** — Player IDs to permanently remove from the optimization pool.

```yaml
excluded_players:
  - 5      # Gabriel — never want to own
  - 237    # Enzo
```

**`extra_players`** — Player IDs to always include in the pool (beyond your current squad). Use for transfer targets.

```yaml
extra_players:
  - 256    # Munoz
  - 267    # Sarr
```

### Fixture Overrides (DGW / BGW)

When the Premier League reschedules fixtures (postponements, double gameweeks, blank gameweeks), the FPL API may not be updated immediately. Use `fixture_overrides` to manually correct the schedule before running the solver.

Each entry moves a specific fixture (identified by home and away team IDs) to a new gameweek. Team IDs can be found in `data/team_tiers.csv`.

```yaml
fixture_overrides:
  # Arsenal (1) vs Chelsea (7) postponed, rescheduled to GW 33
  - home_team: 1
    away_team: 7
    gameweek: 33
  # Liverpool (12) vs Man City (13) moved to GW 29 (double gameweek)
  - home_team: 12
    away_team: 13
    gameweek: 29
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
