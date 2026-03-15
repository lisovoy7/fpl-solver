# I Built a MILP Solver to Manage My Fantasy Premier League Team — Part 1: The Optimization Engine

FPL looks deceptively simple on the surface: pick 15 players, choose 11 to start each week, captain someone. But the moment you try to think even a few gameweeks ahead, the complexity explodes.

Consider what a single decision actually involves. You're choosing from **~300 players** to fill 15 squad slots under strict constraints: exactly 2 GK, 5 DEF, 5 MID, 3 FWD, no more than 3 from any one club, and a £100m budget. From those 15, you pick 11 starters each week in a valid formation, plus a captain whose points are doubled. You have **one free transfer per gameweek** (unused ones roll over, up to a maximum of 5), and any extra transfers beyond your free allowance cost you 4 points each — permanently. And spread across the season, you have **4 chips** (Wildcard, Free Hit, Bench Boost, Triple Captain), each playable once per half-season, never two in the same gameweek.

Now try to decide: should you use your wildcard this week or next? Is it worth taking a -4 hit for that urgent upgrade? Should you save the bench boost for the double gameweek in GW33, or is there a better window? How do you balance building for the long run against the player who's on fire right now?

Every one of these decisions interacts with every other. Selling a player today changes your budget for the next five weeks. Using a chip now means it's gone when a better moment arrives. The number of possible squad sequences, lineup combinations, transfer paths, and chip schedules across a full season is astronomically large — far beyond what any human can reason through.

This is why most FPL managers — even experienced ones — rely on gut feel, Reddit consensus, and greedy one-week thinking. And this is exactly the kind of problem that **mathematical optimization** was designed to solve.

---

Over the past months, I've built an open-source framework that tackles both sides of the problem: a **prediction engine** that estimates how many points each player is expected to score in every upcoming gameweek, and an **optimization engine** that turns those predictions into a full season plan — a concrete, week-by-week schedule that tells you exactly which players to buy and sell, who to captain, whether to roll your free transfer or spend it, and when to play each of your four chips. The whole thing runs in under an hour, outputs a readable strategy, and the **full code is freely available on [GitHub](https://github.com/lisovoy7/fpl-solver)**.

To keep this digestible, I've split it into two posts. Part 1 (this post) covers the **optimization engine**. Part 2 covers the **prediction engine**.

Why this order? Prediction is the more intuitive problem — concepts like xG, xA, and fixture difficulty are widely discussed in the FPL community. Optimization is where the real leverage is, and where most tools fall short. You can have perfect predictions and still make poor decisions if you're thinking one week at a time. So we start here, assuming predictions already exist. Think of it as: *"given that we know Salah is expected to score 7.2 points in GW30 — what should we actually do?"*

---

## The Claim Every FPL Manager "Knows" Is Wrong

Let's take an example. It's the middle of March. Double and blank gameweeks are approaching. Reddit, Twitter, and every FPL podcast converge on the same endgame chip strategy:

> **Wildcard GW32 → Bench Boost DGW33 → Free Hit BGW34**

Stack your squad with double-gameweek players, boost your bench when they all play twice, dodge the blank with a Free Hit. Sounds intuitive. Everyone's doing it.

> **Note:** At the time of writing, the exact teams that will play in each double or blank gameweek aren't even finalized. We're all operating with uncertainty, and realistically all we can do is test a few different possible scenarios (for which teams blank or double when), and see how strategies hold up under each. I did exactly that: generated a range of plausible fixture slates and ran the solver on them all.

Here's the really striking result: **No matter which scenario I tried, the solver *never* chose "Wildcard GW32 → Bench Boost DGW33 → Free Hit BGW34" as the optimal chip sequence.** In every case, it found a chip schedule worth 30+ more expected points than the conventional wisdom — often one a human would rarely consider.

This result comes from my specific squad, my remaining chips, my budget, and the modeled fixtures for each scenario — your mileage will vary depending on your situation. But the principle stands: **the "obvious" chip strategy is rarely the mathematically optimal one**, because humans underestimate opportunity costs and overvalue double gameweeks.

So, how does the solver figure this out? By evaluating *every* legal combination of transfers, lineups, captains, and chip timings simultaneously — and picking the one that maximizes total expected points over the entire horizon, for every plausible fixture schedule.

This post explains how.

---

## Enter Mixed Integer Linear Programming

**Mixed Integer Linear Programming (MILP)** is the workhorse of operations research. Airlines use it to schedule crews. Logistics companies use it to route trucks. And we're going to use it to pick a fantasy football team.

The idea is simple:

1. **Define decision variables** — what the solver can control
2. **Write an objective function** — what we want to maximize
3. **Add constraints** — what the rules of FPL require
4. **Hand it to a solver** — let it find the mathematically optimal solution

If every constraint and the objective can be expressed as linear expressions (sums of variables times constants), and some variables are restricted to integers (0 or 1), the solver *guarantees* it will find the best possible solution — or tell you how close it got within the time limit.

### A Toy Example

Before diving into the full FPL model, here's the intuition. Suppose you have 3 players and need to pick 1 captain:

$$
\max \quad 5x_A + 3x_B + 7x_C
$$

$$
\text{subject to:} \quad x_A + x_B + x_C = 1, \quad x_i \in \{0, 1\}
$$

The solver instantly returns x_C = 1. Trivial here, but the same framework scales to thousands of variables and constraints — and that's exactly what we need.

---

## The Full FPL Model

### Decision Variables

For every player $p$ and every gameweek $t$ in the planning horizon, the solver controls:

| Variable | Type | Meaning |
|----------|------|---------|
| $x_{p,t}$ | Binary | Is player $p$ in the squad at GW $t$? |
| $y_{p,t}$ | Binary | Is player $p$ in the starting XI at GW $t$? |
| $c_{p,t}$ | Binary | Is player $p$ the captain at GW $t$? |
| $s_{p,t}$ | Binary | Is player $p$ transferred IN at GW $t$? |
| $r_{p,t}$ | Binary | Is player $p$ transferred OUT at GW $t$? |
| $w_t$     | Binary | Is Wildcard played at GW $t$? |
| $A_t$     | Integer | Free transfers available at start of GW $t$ |
| $h_t$     | Integer | Paid transfers at GW $t$ (costing -4 pts each) |

With ~300 candidate players and a full-season horizon, that's potentially tens of thousands of binary decision variables. Even for a short gameweek horizon (late season), we get roughly **7,200 variables**. This is where the "mixed integer" part earns its keep — a continuous relaxation could say "half-own Salah," but binary constraints force real yes/no decisions.

### The Objective Function

We want to maximize total expected points, accounting for captaincy and transfer costs:

$$
\max \sum_{t=1}^{H} \left[ \sum_{p \in P} \left( E_{p,t} \cdot y_{p,t} + E_{p,t} \cdot c_{p,t} \right) - 4h_t \right]
$$

Let's unpack each term:

$$
E_{p,t} \cdot y_{p,t}
$$

Expected points from each starter. If player *p* is in the starting XI at gameweek *t*, their full expected points count toward the total.

$$
E_{p,t} \cdot c_{p,t}
$$

Captain bonus. The captain's points are doubled in FPL, so this extra copy adds the second instance. Combined with the starter term above, the captain effectively contributes 2 × E.

$$
-4h_t
$$

Transfer penalty. Each paid transfer costs 4 points, exactly as in FPL.

> **For the non-technical reader:** This formula captures the big picture for the entire timeframe you’re planning for—whether that’s just the next 10 gameweeks or the whole rest of the season. It adds up all the expected points from your starting players across every week in your chosen horizon, doubles the captain’s points each gameweek, and subtracts 4 points for every paid transfer you make. In other words, it’s optimizing your total score over your whole planning period, not just for a single gameweek.

### The Constraints

#### Squad Flow — Tracking Transfers Over Time

The squad at GW $t$ equals the previous GW's squad plus transfers in, minus transfers out:

$$
x_{p,t} = x_{p,t-1} + s_{p,t} - r_{p,t}
$$

This single equation does a remarkable amount of work. It means the solver can't magically swap half the squad — every change must go through a transfer, and every transfer is tracked and potentially penalized. It also creates the "butterfly effect": selling a £4.5m defender in GW5 frees up budget that ripples through every subsequent gameweek.

#### Squad Composition — FPL Rules

Every gameweek, the squad must satisfy:

- Exactly **15 players** total
- Exactly **2 GK, 5 DEF, 5 MID, 3 FWD**
- At most **3 players from any single club**
- Total squad value **≤ budget**

#### Lineup Selection — Valid Formations

The starting XI must have:

- Exactly **11 players** (all from the squad)
- Exactly **1 GK** starting
- **3–5 DEF, 2–5 MID, 1–3 FWD** (all valid FPL formations)
- Exactly **1 captain**, who must be a starter

#### Transfer Banking — The Rollover Mechanic

Unused free transfers roll over (capped at 5). This creates an intertemporal tradeoff: save a transfer now to have 2 free transfers next week, or use it now for an immediate upgrade? The MILP handles this natively:

$$
A_{t+1} \leq A_t - u_t + h_t + 1 + M \cdot w_t
$$

where $M = 15$ is a "Big-M" constant that deactivates this constraint when Wildcard is played (making all transfers free that week). The Big-M method is a classic MILP technique for encoding conditional logic: when $w_t = 1$, the constraint becomes trivially satisfied, effectively "turning off" the transfer banking rules for that week.

---

## Chip Handling — The Combinatorial Challenge

Chips are the hardest part. Each chip fundamentally changes the optimization landscape:

- **Wildcard**: All transfers are free. The solver can rebuild the entire squad in one move.
- **Bench Boost**: All 15 players score, not just 11. Changes lineup selection for that GW.
- **Triple Captain**: Captain scores 3× instead of 2×. Changes captain choice for that GW.
- **Free Hit**: Temporary squad for one GW — pick any 15 players, then revert next week.

The problem: chip timing interacts with everything else. Using Wildcard in GW10 changes which players are available for the rest of the season, which affects whether Bench Boost in GW15 or GW25 is better, and so on.

### Why Chips Are Decomposed Into Scenarios

You might wonder: why not just add Bench Boost, Triple Captain, and Free Hit as decision variables inside the MILP, like we do with Wildcard?

The answer is **computational complexity**. Each chip introduces non-linear interactions with the objective function. Bench Boost, for example, changes how bench players are scored — but only for the GW where it's active. Modeling this inside the MILP requires auxiliary binary variables for every player-gameweek combination, plus Big-M constraints to link them. Multiply by three chips and the model bloats dramatically — solver runtime can explode from minutes to hours.

Instead, we **decompose** the problem. Wildcard stays inside the MILP (it's elegantly handled by the Big-M method on transfer constraints). But BB, TC, and Free Hit are enumerated as **scenarios**:

1. Generate every valid combination of (BB GW, TC GW, FH GW), respecting the 1-per-half-season rule and the no-two-chips-same-GW rule.
2. For each combination, solve the full MILP with those chips hard-coded.
3. The scenario with the highest total points wins.

For example, for a late-season planning horizon with just one Bench Boost and one Free Hit chip remaining, this approach typically yields around 91 possible scenarios. If it's earlier in the season and you have all chips available, the number of scenarios can be even higher. Each scenario generally takes 15–30 seconds to solve, and the entire set can be evaluated in parallel — so if you want, you can run them all at the same time.

### The Free Hit Sub-Problem

Free Hit is special: it temporarily replaces your entire squad for one gameweek, then reverts. The main MILP handles this by **freezing** the squad flow during the FH gameweek — transfer-in and transfer-out variables are forced to zero, so the "real" squad passes through unchanged. Meanwhile, a separate, smaller MILP solves the optimal one-week squad:

$$
\max \sum_{p} E_p \cdot y_p + E_p \cdot c_p
$$

subject to standard squad constraints (15 players, 2/5/5/3, ≤ 3 per club, budget). No transfer flow, no multi-week planning — just "pick the best possible squad for this one week." The resulting points are added back to the main solver's total.

---

## Example: How the Solver Prepares for Double and Blank Gameweeks

One of the most dramatic ways mathematical optimization outperforms human intuition is in navigating the chaos of double and blank gameweeks (DGWs/BGWs), but this is just one example of its broader forward-planning power.

Whenever the Premier League shuffles fixtures—postponing games to create blanks, or combining fixtures into doubles—human managers often panic. It's tempting to focus only on the next DGW ("I need to stack 15 DGW players!") and lose sight of the bigger picture. The solver is different: it calmly sees the entire horizon and adjusts for the long term.

For example, weeks before any blanks or doubles were even confirmed, the solver was already **shaping my squad** to minimize future disruption. It strategically managed my Arsenal and Manchester City exposure, avoided stacking players from teams projected to blank, and as the blank in GW31 approached, I was able to field a strong XI with hardly any transfers. No panic hits, no last-minute fire sales—the solver had quietly positioned my squad to handle chaos well before it actually hit.

This proactive planning is tough for humans, who are naturally wired to focus on *this week*. The solver, however, optimizes for *all remaining weeks together*—meaning sometimes the key to a successful GW33 is a seemingly odd transfer in GW28.

The solver also makes it easy to run **what-if scenarios** for fixture changes. Since the FPL schedule often isn't finalized until just before the deadline, I can input hypothetical fixture overrides (e.g., "what if Arsenal vs Newcastle moves to GW33?") and immediately see how optimal strategy changes.

---

## The Human Touch

A solver that ignores human judgment is doomed to miss crucial details. While the FPL API provides some data—like basic injury flags or suspension status—it rarely has reliable updates on tactical rotations, nuanced fitness news, or the true duration of an absence. This is where human insight, judgment, and active intervention become essential: maybe you know a player is at risk of being benched for tactical reasons, hear a trusted report about a manager’s preferences, or believe a return-from-injury timeline is shorter (or longer) than stated. This type of knowledge is exactly the contextual input you can feed into the model—letting the optimizer reflect realities and subtleties that raw data alone will never capture.

I built several override mechanisms that let me interact with the optimization without breaking it:

### Player Availability Overrides

```yaml
non_playing:
  - player: 661     # Ekitike
    gameweeks: [21, 22]
```

When I know a player will miss specific gameweeks (e.g., rotation / injury news), I can flag them as non-playing. The solver sets their expected points to 0 for those weeks but may keep them in the squad if they're expected to return — it optimizes *around* the absence.

### Forced Lineup

```yaml
forced_lineup:
  - player: 235     # Palmer
    gameweeks: [27]
```

Sometimes I want to guarantee a player starts, regardless of what the model thinks. Maybe I have a gut feeling, a hunch, or simply want to take a chance on a differential. The solver respects the constraint and re-optimizes everything else around it.

### Player Exclusions

```yaml
excluded_players:
  - 5    # Gabriel
```

Players I never want to own — maybe they're template picks I want to avoid as differentials, or players I simply don't trust regardless of what the numbers say.

The key insight: **these overrides don't just change one decision — they re-optimize everything.** If I force Palmer into the lineup, the solver doesn't just add him; it re-evaluates every other transfer, every captain pick, every chip timing in light of that constraint. It turns the tool from a dictator into a co-pilot.

---

## Why "Conventional Wisdom" Gets Chip Timing Wrong

Back to the opening claim. The "WC32 → BB33 → FH34" strategy sounds logical for managers with those chips remaining:

1. Wildcard to load up on DGW33 players
2. Bench Boost when they all play twice
3. Free Hit to dodge the blank GW34

But the solver reveals several flaws in this reasoning:

**The opportunity cost of Bench Boost on a DGW is often overestimated.** Yes, DGW players score more that week. But your bench during a DGW is typically filled with low-quality players bought only for the double — their "extra" game might only be worth 2–3 points each. Bench Boost on a regular gameweek with a well-constructed bench (strong players who happen to sit due to formation choices) can provide *more marginal value*.

— According to the solver's optimal decision-making, **using Free Hit in the DGW isn't necessarily about outscoring Bench Boost in that specific gameweek — it's about optimizing your total expected points across all remaining weeks.** By playing Free Hit during the DGW, the solver can assemble a perfect squad just for that week, without having to compromise future gameweeks by loading up on short-term options via Wildcard or Bench Boost. This global approach often leads to a higher cumulative score than simply maximizing points in a single DGW.

**The compounding effect of better transfers matters more than one big GW.** A well-timed Wildcard cascades into better squads for *every subsequent gameweek*. The 30+ point gap between the "obvious" strategy and the solver's choice isn't from one brilliant GW — it's from the accumulation of slightly better decisions across every remaining week.

Of course, this varies by manager. Your remaining chips, current squad, budget, and free transfers all change the calculus. The point isn't that one specific chip sequence is universally optimal — it's that **the optimal sequence is almost never the one your gut tells you**, and the only way to find it is to check every possibility mathematically.

---

## Personal Notes

A confession: this is actually my first season playing FPL. Even though I like football, I didn't really follow the Premier League that closely before this project. I knew the top clubs, recognized the big-name players, but that was about it. With so little domain knowledge, I did something naive at the start: I took the FPL player prices as a proxy for expected performance, assuming the organizers had done the hard work of encoding quality into price. That was my "prediction engine" for the first few weeks. Simple, arguably stupid, but it was a starting point.

Unfortunately, it wasn't until **Gameweek 7** that I finished developing and integrated the MILP solver into my workflow. Up to that point, I was relying solely on my point prediction model, using a very basic and greedy long-term optimizer to guide my decisions. Now, with the MILP solver (plus the component-based prediction engine covered in Part 2), I've upgraded from that simple approach—but even so, I treat the solver as a **co-pilot**, not an autopilot.

My typical workflow:
1. Run the solver with the base config
2. Look at the suggested strategy
3. Ask "what if?" — adjust player overrides, force/exclude players, tweak fixture assumptions
4. Re-run and compare
5. If the adjusted strategy is within 1–2 points of the original, I go with my gut. If the adjustment costs 5+ points, I trust the math.

Here's how my overall rank percentile evolved over the season (out of ~10 million managers):

![Overall Rank Percentile](/covers/rank_percentile_chart.png)

The early weeks were noisy — bouncing between the 69th and 90th percentile with no real trend. After the solver kicked in, there's a gradual upward drift, and by GW 29 I'd reached the **98.8th percentile**. There's plenty of variance in FPL, and some of this is surely luck. But the trend is there, and it coincides with when the solver started making decisions.

I want to be careful not to overclaim. A single season is a small sample, and FPL has enough randomness that any approach can look brilliant or foolish over a stretch of weeks. Still, consistently making the mathematically better-informed decision — even if it's the boring one — seems to compound over time.

The season isn't over. There are 9 gameweeks left, double and blank gameweeks ahead, and chips still to play.

But honestly? The math and the ranks aren't even the main thing I got out of this. This project made me genuinely curious about the Premier League. I went from casually knowing a handful of top players to watching most games every week, learning every squad, tracking every fixture. And maybe the best part — my five-year-old son started watching the games with me. Last week I opened an FPL account for him. Whatever my final rank ends up being, that's the real achievement of this season.

---

## Try It Yourself

The full solver is open-source: **[github.com/lisovoy7/fpl-solver](https://github.com/lisovoy7/fpl-solver)**

```bash
git clone https://github.com/lisovoy7/fpl-solver.git
cd fpl-solver
pip install -r requirements.txt
# Edit config.yaml with your team_id and free_transfers
python run.py
```

It fetches your squad from the FPL API, generates predictions, and outputs the optimal transfer and chip strategy for the rest of the season. No manual data entry beyond two numbers in the config file.

---

*Next up — **Part 2: The Prediction Engine.** How do we transform raw Premier League data into accurate expected FPL points? It all starts with the rich "expected goals" (xG), "expected assists" (xA), and other advanced stats that the FPL API makes freely available for every player — a goldmine that lets us go far beyond basic past points.*

*Our prediction engine learns from past seasons and breaks down every possible way a player can earn FPL points: scoring, assisting, clean sheets, defensive contributions, bonus points, etc. For each of these components, we build separate predictive models that account for not just player quality, but also the future difficulty of every fixture.*

*The result: granular, component-wise point forecasts for every player, every GW, grounded both in stats and in real-world fixture context. Stay tuned for all the details.*
