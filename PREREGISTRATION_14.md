# Project Underdog — Round 14 pre-registration: real-time moment triggers and team habits that create moments

*Written 2026-10-14 before running.*

## 14A: real-time trigger, "defenders back" at the moment of a regain (360 data)

At every open-play regain, `opp_goalside` is the number of visible outfield opponents between the ball and their goal. The outcome is xG in the next 20 s.

**H14A.** Holding regain location fixed (cubic in x, centrality), plus game state and minute, counter value falls as `opp_goalside` rises. The coefficient must be negative with one-sided p < 0.05 in the discovery set **and** in each confirmation set (round-13 split).

If confirmed, we publish a trigger table: the value of a regain by zone × defenders back, i.e. *"if ≤ k of them are back, go"*.

## 14B: team habits that create valuable moments (event data)

* **Cross-fitting.** Habits are measured on a team's odd-numbered matches and outcomes on its even-numbered matches, then the halves are swapped. Controls: the team's odd-match xG for and against per match (strength).
* **Data.** Discovery: StatsBomb 2015/16 non-EPL. Confirmation: StatsBomb EPL 2015/16 + StatsBomb unseen WSL/NWSL/ISL, pooled with competition FE.
* **Primary outcome.** Net xG per match (for − against).

**Habits (pre-specified causal pairs; secondary outcome = the moment each habit should create):**

| Habit | Moment it should create |
|---|---|
| H1 counter-press rate: pressures within 5 s of losing the ball, per possession lost | high regains (x ≥ 80) per match |
| H2 press height: mean x of pressures | high regains per match |
| H3 directness after regain: share of regains whose first action gains ≥ 10 yds | final-third entries ≤ 10 s after a regain, per match |
| H4 crossing rate: crosses per final-third possession | box entries per match |
| H5 tempo: passes per minute of own possession | shots per match |

**Tests:** 5 habits × {net xG, moment} = 10. **Shortlist:** BH 10% in discovery. **Confirmed:** same sign and pooled one-sided p < 0.05/k (Holm). Team-level n is small, so low power is expected, and we will say so.
