# Project Underdog — Round 12 pre-registration: do mined decision rules work as team *habits*?

*Written 2026-10-14 after round 11 (whose action-level rules replicate 90–97% in sign but are confounded by in-the-moment selection), before running anything below.*

**Design (intention-to-treat on habits, cross-fitted).**
* For each team and situation cell (the 2,916 cells of round 11), compute the team's **propensity** to choose action A: its share of decisions in that cell, measured on the team's odd-numbered matches.
* Compute the **outcome** on its even-numbered matches: mean net xG (own − opponent, next 20 s) over **all** decisions in that cell, whatever action was taken. Then swap odd and even and average the two estimates.
* Regression: outcome ~ propensity_z (z-scored within cell) + team FE + cell FE, weighted by decisions, with SEs clustered by team.
* Because the outcome covers every decision in the cell, and propensity and outcome come from different matches, in-the-moment selection ("he shot because the shot was on") cannot create the effect. Team FE removes overall quality; cell FE removes the situation.

**Tests.** Actions A ∈ {carry_fwd, short_fwd, long_fwd, cross, shot, dribble, recycle} × zone groups {all cells, own half (x<60), middle third (60–96), final third (x≥96)} = 28 tests.

**Discovery:** StatsBomb 2015/16 non-EPL. **Shortlist:** BH-FDR 10%. **Confirmation:** StatsBomb EPL 2015/16 and the unseen WSL/NWSL/ISL. Confirmed if the pooled estimate has the same sign with one-sided p < 0.05/k (Holm) **and** the same sign in both sets.

**Interpretation rule.** A confirmed positive β_A means teams that habitually choose A more in those situations get better outcomes there, holding team and situation fixed. That is a tactic a coach can install. A null means the round-11 rules describe *which moments are good*, not *which choices are good*.

**Value.** For confirmed habits: the net xG per season from moving a team's propensity by +1 SD (between-team, within cell). We report it in xG and in points via the season simulator.
