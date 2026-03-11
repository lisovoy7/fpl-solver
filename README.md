# fpl-solver

Holistic Fantasy Premier League optimization using Mixed Integer Linear Programming (MILP).

**fpl-solver** plans your entire remaining FPL season in one go: transfers, lineup, captaincy, and chip timing (Wildcard, Free Hit, Bench Boost, Triple Captain) — all solved simultaneously under FPL's real constraints.

## How It Works

The tool has two engines that run in sequence:

### 1. Prediction Engine

Generates per-player expected points for every future fixture using:

- **Component-based prediction** — goals, assists, clean sheets, saves, bonus, defensive contribution, etc. are each predicted independently
- **Fixture multipliers** — historical data (5 seasons) reveals how each stat component varies by position, opponent tier, and home/away. Player averages are normalized to remove fixture bias, then re-scaled for each upcoming fixture
- **Poisson model** for clean sheet probability (from expected goals conceded)
- **Player-specific consistency** for defensive contribution (DEF/MID threshold probability)

### 2. Decision Engine (MILP Solver)

Formulates the entire remaining season as a single Mixed Integer Linear Program using [PuLP](https://coin-or.github.io/pulp/):

- **Decision variables**: squad ownership, starting XI, captain, transfers in/out — per player per gameweek
- **Objective**: maximize total expected points across all gameweeks (with captain bonus, bench value, and transfer penalties)
- **Constraints**: budget, squad composition (2 GK / 5 DEF / 5 MID / 3 FWD), formation rules (3-5-2, 4-4-2, etc.), max 3 per club, transfer banking (up to 5 free transfers), chip rules
- **Chip enumeration**: Wildcard, Free Hit, Bench Boost, and Triple Captain are decomposed into scenarios and solved independently — the best combination wins

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

## Output

- **Console**: full GW-by-GW strategy (transfers, lineup, captain, chips)
- **File**: `output/strategy_gw{N}.json` with structured results

## Disclaimers

### API Limitations

- The FPL API reflects your squad as of the **last completed gameweek**. Pending transfers are NOT visible. If you've already made transfers in the FPL app, the solver doesn't know about them.
- Free transfers cannot be reliably detected from the API. Always set `free_transfers` manually in `config.yaml`.

### Injuries and Suspensions

The solver has **no injury/suspension detection**. You must manually set players as non-playing in `config.yaml`:

```yaml
non_playing:
  - player: 661     # Ekitike
    gameweeks: [21, 22]
```

### Bundled Data

#### `data/multipliers.csv` — Fixture Multipliers

Pre-computed from 5 seasons of FPL data (2020-2025). Each row describes how a stat component (e.g. expected_goals) varies based on:
- **Position** (GK, DEF, MID, FWD)
- **Player's team tier** (1-5, where 1 = strongest)
- **Opponent's team tier** (1-5)
- **Home/Away**

A multiplier of 1.2 means "players in this situation historically produce 20% more than baseline."

#### `data/team_tiers.csv` — Team Tiers

Maps each team to a tier (1-5) per season. Tiers are based on historical performance and are used by the multiplier system. **You should update this file at the start of each new season** to reflect promoted/relegated teams and any tier changes.

Format: `season,team_id,team_name,team_tier`

## Project Structure

```
fpl-solver/
  run.py              # CLI entry point
  config.yaml         # All user configuration
  requirements.txt
  data/
    multipliers.csv   # Pre-computed fixture multipliers (5 seasons)
    team_tiers.csv    # Team tier assignments per season
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
