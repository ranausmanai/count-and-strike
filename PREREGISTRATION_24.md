# Project Underdog: Round 24 pre-registration ("The Slingshot", an invented tactic tested in simulation)

*Written 2026-10-17, before any run.*

This round tests an invented tactic. It was not found in data. It is built from three facts that survived our real-data tests:
- **Baiting works:** working the ball on one side shifts the block and raises the far-lane overload (agent bait, H1, confirmed in all three populations).
- **+1 at the landing spot:** retention is 48% / 72% / 89% when outnumbered / level / +1 (agent chain; the round-22 stress test passed).
- **Count:** counters are worth about twice as much with 3 or fewer opponents behind the ball.

We test it in the agent-based simulator, which earlier invented "95% win" exploits. So the bar is **robustness across simulator physics and opponents**, not the raw win rate.

## The tactic (new knobs in `src/sim/engine2.py`; knobs 0–9 are unchanged)

**Knob 10 `asym` (0–1). PULL, the lopsided shape.**
- In possession, the team shifts (0.25 + 0.5·asym) × (ball_y − centre) toward the ball side; the old shift was 0.25.
- One wide midfielder on the far side stays high and wide (y = 4 m from the far touchline) as an isolated outlet.

**Knob 11 `release` (0–1.5). RELEASE.**
- Applies once the ball has been on one side of the pitch for ≥ 3 s in the current possession.
- Passes to a free receiver (> 5 m from the nearest opponent) more than 25 m across the pitch get a value bonus of release × 0.012.

**Knob 12 `escape` (0/1). ESCAPE.**
- In our own third, the two forwards move toward the long-ball landing zone.
- A long ball is aimed at the candidate landing point (3 candidates across the width) with the best local balance: ours minus theirs within 10 m. Otherwise the aim is random, as before.

**Physics change applied to EVERY team and arm (grounding in data):**
- Long-ball success = 0.45 + 0.17 × clip(ours10 − theirs10, −2, +1).
- This replaces the fixed 0.45, so the +1 landing effect from real data is built into the world for both sides.

## Arms

| Arm | Underdog tactic |
|---|---|
| A0 | Baseline: default tactic, new knobs at 0 |
| A1 | Pull only: asym 0.8 |
| A2 | Pull + Release: asym 0.8, release 1.0 |
| A3 | Escape only |
| A4 | Full Slingshot: asym 0.8, release 1.0, escape 1 |
| A5 | Free search over all 13 knobs |

**A5 search:**
- 600 random tactics × 120 matches against the default strong team, in the calibrated physics.
- The best 20 are re-run on fresh seeds; the best fresh one is evaluated like the other arms.

## Evaluation

**Matchups:** the underdog has weak skills (as in `sim/search.py`) against the strong team.
- 5 opponent styles: default, high_press, deep_block, direct, possession.
- 8 physics variants: calibration ×U(0.7, 1.3) on lane_k, risk_k, tackle_k and engage_r; shot_b ± 0.3; seeded.
- Each matchup is played 600 times per arm, on common random seeds across arms.

**Outcome:** points per match, as each arm minus A0.

**Success rule for the Slingshot (A4):**
- gain > 0 in ≥ 7 of 8 physics variants AND in ≥ 4 of 5 opponent styles;
- AND the pooled gain is ≥ 0.05 points per match with a 95% CI above 0;
- AND realism holds: pass completion and goals per match stay within ±25% of A0.

## Exploit flags (reported)
- Any arm with a pooled win rate > 60% for the underdog, or a gain > 0.5 points per match, is flagged as a likely simulator exploit.

## Interpretation (stated in advance)
- A pass would mean the Slingshot is a coherent, physics-robust hypothesis worth trying on a pitch. It would not prove the tactic works.
- A fail would be reported as such.
