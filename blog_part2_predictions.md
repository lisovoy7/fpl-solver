# I Built a MILP Solver to Manage My Fantasy Premier League Team — Part 2: The Prediction Engine

*Part 2 of 2. Part 1 covered the optimization engine that decides transfers, lineups, captaincy, and chip timing. This post covers the prediction engine — the system that estimates expected FPL points for every player in every future gameweek.*

---

## The Input The Optimizer Needs

In Part 1, we treated expected points $E_{p,t}$ as a given: "Salah is expected to score 7.2 points in GW30." But where does that number come from?

Most FPL analytics sites give you a single "expected points" number with no explanation. Some use proprietary models. Others just average recent scores. None of them tell you *how* they handle the fact that scoring 0.6 xG against Burnley is very different from scoring 0.6 xG against Arsenal.

I built a prediction engine from scratch that:

1. **Decomposes** FPL points into 9 independent components (goals, assists, clean sheets, saves, etc.)
2. **Normalizes** historical stats to remove fixture difficulty bias
3. **Re-scales** for each upcoming fixture using empirically derived multipliers
4. Uses **probability models** (Poisson, Normal distribution) where point-scoring is non-linear

The result: a per-player, per-fixture expected points value that accounts for opponent strength, home advantage, and position-specific scoring rules. Let's walk through each step.

---

## Step 1: Why Raw Averages Are Misleading

The naive approach: "Salah averaged 7 points over his last 5 games, so predict 7." This fails spectacularly because it ignores **who he played against**.

If those 5 games were against bottom-half teams at home, his "true ability" is lower than 7. If they were against top-6 teams away, it's higher. Raw averages conflate player quality with fixture difficulty.

### The Normalization Trick

We fix this by **dividing out the fixture effect** from every historical stat.

For each game a player has played (filtering to 60+ minutes for meaningful samples), we:

1. Look up the **team tiers** (1–5) for both the player's team and the opponent. Tier 1 = elite (Manchester City, Arsenal), Tier 5 = weakest defenses/attacks.
2. Find the corresponding **fixture multiplier** from a pre-computed lookup table. These multipliers are derived from multiple seasons of data — they capture how much a stat inflates or deflates based on opponent and venue.
3. **Normalize** by dividing the raw stat by the multiplier.

**Example:** A midfielder records xG = 0.6 in a home game against a Tier 5 opponent. The multiplier for (MID, Tier 2, vs Tier 5, home) is 1.4, meaning midfielders in this context tend to over-perform their baseline by 40%. The normalized xG is:

$$
\text{xG}_{\text{norm}} = \frac{0.6}{1.4} = 0.43
$$

This 0.43 represents the player's "true" underlying rate — what they'd produce against a neutral opponent. We normalize **expected stats (xG, xA, xGC)** rather than actual goals and assists, because expected stats are derived from shot quality and are more stable and predictive.

### The Fixture Multiplier Table

The multipliers are the backbone of the entire prediction system. They're stored in a CSV with ~2,800 rows covering every combination of:

- **Position:** GK, DEF, MID, FWD
- **Player's team tier:** 1–5
- **Opponent's team tier:** 1–5
- **Home/Away**
- **Stat component:** expected_goals, expected_assists, expected_goals_conceded, saves, etc.

Each multiplier answers: "How much does this stat inflate or deflate in this specific context?" A value of 1.0 means neutral. 1.3 means the stat tends to be 30% higher than baseline. 0.7 means 30% lower.

These multipliers were computed from aggregated historical data across multiple Premier League seasons. They're shipped as a static file with the solver — you don't need to regenerate them.

---

## Step 2: From Normalized Stats to Player Baselines

After normalization, each player has a sequence of "fixture-adjusted" stats for every game they've played this season. We compute a **rolling average** over their last 10 games to get a single baseline number per component:

- `avg_norm_xG` — normalized expected goals per game
- `avg_norm_xA` — normalized expected assists per game
- `avg_norm_xGC` — normalized expected goals conceded per game
- `avg_norm_saves` — normalized saves per game
- `avg_yellow_cards` — average yellow cards (not normalized — card rates are player-specific)
- `avg_bonus_points` — average bonus points
- `avg_defensive_contribution` — average BPS from defensive actions

**Why 10 games?** It's a bias-variance tradeoff. Too few games and you're chasing noise (a defender who scored a freak goal looks like a goal threat). Too many and you miss genuine form changes (a player who's been dropped or returned from injury).

These baselines represent each player's **underlying per-game ability**, stripped of fixture effects. The next step is to project them into the future.

---

## Step 3: Player × Fixture Combinations

For every player, we scan all remaining fixtures where their team is involved. If a team plays twice in a gameweek (a double gameweek), the player gets **two rows** — their predictions are summed later. This naturally handles DGWs without any special logic.

Each row contains:
- Player ID and position
- Gameweek number
- Opponent team tier
- Home or away

This cross-product typically generates ~4,000 player-fixture combinations for a 9-gameweek horizon with ~440 active players.

---

## Step 4: Component Predictions — The 9 Building Blocks

Here's where it all comes together. For each (player, fixture) combination, we compute predicted FPL points across **9 independent components**, then sum them. Each component has its own formula, reflecting the specific FPL scoring rules.

### Component 1: Minutes Points

$$
\text{predicted} = 2.0
$$

Simple: every player in the candidate pool is assumed to play 60+ minutes and earn the 2-point appearance award. Players who haven't been playing regularly are filtered out earlier (via `min_hist_games`).

### Component 2: Goals (xG-based)

$$
\text{predicted} = \underbrace{\text{avg\_norm\_xG}}_{\text{player baseline}} \times \underbrace{\text{fixture\_multiplier}}_{\text{opponent context}} \times \underbrace{\text{goal\_points}}_{\text{position}}
$$

Goal points vary by position: **GK = 10, DEF = 6, MID = 5, FWD = 4**.

**Example:** A forward with `avg_norm_xG = 0.40` faces a Tier 5 team at home. The fixture multiplier is 1.3. Predicted goal points: $0.40 \times 1.3 \times 4 = 2.08$.

The same forward away to a Tier 1 team might see a multiplier of 0.7: $0.40 \times 0.7 \times 4 = 1.12$. The fixture context makes nearly a 2× difference.

### Component 3: Assists (xA-based)

$$
\text{predicted} = \text{avg\_norm\_xA} \times \text{fixture\_multiplier} \times 3.0
$$

All positions earn **3 points per assist** in FPL. The multiplier comes from the `expected_assists` column in the multiplier table.

### Component 4: Saves (GK only)

$$
\text{predicted} = \text{avg\_norm\_saves} \times \text{fixture\_multiplier} \times \frac{1}{3}
$$

FPL awards **1 point per 3 saves**. Goalkeepers facing stronger attacks tend to make more saves — the multiplier captures this.

### Component 5: Goals Conceded (GK/DEF only)

$$
\text{predicted} = \text{avg\_norm\_xGC} \times \text{fixture\_multiplier} \times (-0.5)
$$

FPL deducts **1 point per 2 goals conceded** for goalkeepers and defenders. The negative sign makes this a penalty. Facing a stronger attack means higher expected goals conceded and more penalty points.

### Component 6: Yellow Cards

$$
\text{predicted} = \text{avg\_yellow\_cards} \times (-1.0)
$$

**No fixture multiplier** — we treat card rates as player-specific rather than fixture-dependent. Some players just get booked more.

### Component 7: Clean Sheets (Poisson Model) 🎯

This is where the math gets interesting. Clean sheets are binary (you either keep one or you don't), but we need a *probability*.

Under a **Poisson distribution**, if the expected number of goals conceded in a fixture is $\lambda$, the probability of conceding exactly zero goals is:

$$
P(\text{clean sheet}) = e^{-\lambda}
$$

where $\lambda = \text{avg\_norm\_xGC} \times \text{fixture\_multiplier}$.

Then:
- **GK/DEF:** predicted points = $P(\text{CS}) \times 4.0$
- **MID:** predicted points = $P(\text{CS}) \times 1.0$
- **FWD:** 0 (forwards don't earn clean sheet points)

**Example:** A Tier 1 defense (avg_norm_xGC = 0.8) faces a Tier 4 attack at home (multiplier = 0.75). Expected goals conceded: $\lambda = 0.8 \times 0.75 = 0.6$. Clean sheet probability: $e^{-0.6} = 0.549$. A defender's predicted CS points: $0.549 \times 4 = 2.20$.

The same defender away to a Tier 1 attack (multiplier = 1.4): $\lambda = 0.8 \times 1.4 = 1.12$, $P(\text{CS}) = e^{-1.12} = 0.326$, predicted points = $0.326 \times 4 = 1.30$.

The Poisson model captures the non-linearity: the difference between $\lambda = 0.5$ and $\lambda = 1.0$ is much bigger in probability terms ($0.61 \to 0.37$) than the difference between $\lambda = 2.0$ and $\lambda = 2.5$ ($0.14 \to 0.08$). Low-xGC fixtures are disproportionately valuable for clean sheets.

### Component 8: Defensive Contribution (Normal Distribution Model)

FPL's **Bonus Points System (BPS)** awards extra points based on in-game actions. Defenders and midfielders who make tackles, interceptions, and clearances accumulate "defensive BPS." If they exceed a threshold (10 for DEF, 12 for MID), they earn bonus points.

We model this with a **normal distribution**:

1. Compute the expected defensive contribution for the fixture: $\mu = \text{avg\_def\_contribution} \times \text{defensive\_multiplier}$
2. Estimate the standard deviation from the player's historical consistency (with a fallback to population-level estimates)
3. Calculate $P(\text{exceed threshold}) = 1 - \Phi\left(\frac{\text{threshold} - \mu}{\sigma}\right)$
4. predicted points $= P \times 2.0$

This component is small in absolute terms (0–2 points) but matters for defenders — it's the difference between a 4-point and a 6-point gameweek.

### Component 9: Bonus Points

$$
\text{predicted} = \text{avg\_bonus\_points}
$$

The simplest component. Bonus points (1, 2, or 3 awarded to the top BPS scorers in each match) are noisy and hard to predict from fixtures alone. We use the straight rolling average.

---

## Putting It All Together

The final expected points for a player in a gameweek is the sum of all 9 components:

$$
E_{p,t} = \sum_{i=1}^{9} \text{component}_i
$$

For double gameweeks, the player has two fixture rows. Each fixture produces its own 9-component prediction, and they're summed — giving DGW players roughly (but not exactly) double their single-GW expectation.

**Example total for a top defender (single GW, favorable fixture):**

| Component | Points |
|-----------|--------|
| Minutes | 2.00 |
| Goals (xG) | 0.35 |
| Assists (xA) | 0.25 |
| Saves | 0.00 |
| Goals conceded | -0.40 |
| Yellow cards | -0.12 |
| Clean sheet | 2.20 |
| Defensive BPS | 0.90 |
| Bonus | 0.80 |
| **Total** | **5.98** |

This number feeds directly into the MILP optimizer from Part 1 as $E_{p,t}$.

---

## The Normalize → Average → Re-Scale Pipeline

The beauty of this approach is the **three-stage pipeline**:

```
Historical stat ──÷ multiplier──→ Normalized stat ──avg──→ Baseline ──× multiplier──→ Prediction
```

1. **Normalize** past games by dividing out fixture effects → reveals true player ability
2. **Average** the normalized values → stable baseline per player
3. **Re-scale** by multiplying with the future fixture's multiplier → context-aware prediction

The multiplier appears twice — once to remove context from history, once to add context for the future. This means the prediction system is internally consistent: a player's predicted output against the same opponent tier at the same venue would equal their historical average in that context.

---

## What This Engine Does NOT Do

Honest about the limitations:

- **No injury prediction.** If a player is fit, we assume they play. Injuries must be manually flagged via config overrides.
- **No tactical prediction.** We don't model manager decisions (rotation, formation changes). If a player has been playing regularly, we assume they continue.
- **No transfer market prediction.** New signings or January transfers aren't automatically incorporated — though the watchlist mechanism handles this via `extra_players`.
- **No "form" beyond the window.** We use the last 10 games. A player who was world-class for 8 games and poor for 2 still looks good. Conversely, a player on a hot streak of 3 games doesn't instantly become elite.
- **Expected stats, not actual stats.** We deliberately use xG/xA/xGC rather than actual goals/assists/goals conceded. This is a feature, not a bug — expected stats are more predictive of future output. A striker who scored 5 goals from 1.5 xG is due for regression; one who scored 1 goal from 3.0 xG is due for a breakout.

### Manual Prediction Adjustments

To address some of these limitations, the solver includes a `points_multiplier` override:

```yaml
points_multiplier:
  - player: 8       # Timber
    multiplier: 0.75
```

This is especially relevant early in the season, when there's not enough data and the model's predictions can be heavily skewed by a single outlier performance. For example, when Timber scored two goals and provided an assist in the second gameweek — the model might overestimate his future points. Here, I can use a multiplier to dampen his projected returns, manually correcting for the noise until the stats stabilize. The optimizer then re-solves the whole plan with this custom adjustment, so every transfer, captaincy, and chip choice reflects a more realistic expectation.

---

## Why Not Use Machine Learning?

A fair question. Why this hand-crafted component model instead of XGBoost / neural networks / LLMs?

**Interpretability.** I can tell you exactly why the model rates Player A over Player B: "A has higher xG, a better fixture multiplier this week, and plays for a team with a 55% clean sheet probability." With a black-box ML model, you get a number and a shrug.

**Data scarcity.** Each player has at most ~30 games per season. Training a complex model on 30 data points per entity is a recipe for overfitting. The component approach uses domain knowledge (FPL scoring rules) to structure the problem, requiring far fewer parameters to estimate.

**Stability.** Expected stats (xG, xA) are already the output of sophisticated ML models run by Opta/StatsBomb. We're standing on the shoulders of giants — then applying FPL-specific domain knowledge to translate those stats into points.

**Debuggability.** When a prediction looks wrong, I can trace it to a specific component. "Oh, the clean sheet probability is too high because the opponent tier is wrong." Try doing that with a 1,000-tree gradient boosting model.

That said, the multiplier table itself could be improved with more sophisticated estimation techniques. And there's room for ML in specific sub-problems (like bonus point prediction). But for the core engine, the component approach hits the right balance of accuracy and transparency.

---

## The Full Pipeline

Here's how prediction and optimization fit together:

```
FPL API
  ├─ Bootstrap data (players, teams, positions)
  ├─ Fixtures (380 matches, with DGW/BGW overrides)
  └─ GW data (per-player per-GW stats for ~820 players)
         │
    ┌────┴────────────────────┐
    │   PREDICTION ENGINE     │
    │                         │
    │  1. Normalize stats     │
    │  2. Compute baselines   │
    │  3. Cross with fixtures │
    │  4. 9-component predict │
    └────┬────────────────────┘
         │
    E_{p,t} for all players × GWs
         │
    ┌────┴────────────────────┐
    │   MILP OPTIMIZER        │
    │   (from Part 1)         │
    │                         │
    │  Transfers, lineups,    │
    │  captains, chip timing  │
    └────┬────────────────────┘
         │
    Optimal strategy for the
    rest of the season
```

The prediction engine runs in ~5 seconds (after the initial data fetch). The optimizer takes 15–45 minutes depending on the number of chip scenarios. Together, they produce a complete gameweek-by-gameweek plan: which players to transfer in/out, who to start, who to captain, and when to play each chip.

---

## Does It Work?

Refer back to the rank trajectory from Part 1. Since implementing the holistic solver around GW7, my overall rank has climbed from roughly 2,000,000 to **123,709** — top 1.3% out of ~10 million managers. The trend accelerated precisely when the prediction engine's outputs started feeding into the multi-week optimizer rather than a simple greedy algorithm.

The prediction engine isn't perfect — no model is. Players get injured, managers rotate unexpectedly, and sometimes a Tier 5 team parks the bus and keeps a clean sheet against Manchester City. But over a 30+ gameweek horizon, the noise averages out and the systematic advantage compounds.

---

## Try It Yourself

The full solver — prediction engine and optimizer — is open-source:

**[github.com/lisovoy7/fpl-solver](https://github.com/lisovoy7/fpl-solver)**

```bash
git clone https://github.com/lisovoy7/fpl-solver.git
cd fpl-solver
pip install -r requirements.txt
# Edit config.yaml with your team_id and free_transfers
python run.py
```

Everything is automated. The only manual inputs are your FPL team ID and your current free transfer count. Fixture overrides for upcoming DGW/BGW can be added in the config when they're announced.

---

*Thanks for reading both parts. If you use the solver, I'd love to hear how it goes — especially if it finds a chip sequence that surprises you as much as mine surprised me.*
