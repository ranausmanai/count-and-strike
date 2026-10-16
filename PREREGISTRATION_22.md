# Project Underdog: Round 22 pre-registration (mistakes, underdogs, and stress-testing Strike)

*Written 2026-10-16, before running. Data: StatsBomb 360 (418 matches).*

**Populations:**
- Discovery: men's international.
- Confirmation: men's club AND women's. Both are required.

**Motivation:** an outside review asked two things.
1. Is Strike through causal, or do players just shoot when the shot is good? Things like keeper position and a clean strike are unobserved.
2. Are there decisions where the usual choice is worse *specifically for weaker teams*?

## 22A: Strike-through stress tests

**Sample:**
- Open-play decisions with x ≥ 96 and |y−40| ≤ 20.
- Each decision is a shot or a recycle (lateral, back, short forward), as in the shooter agent.

**Outcome:** net xG over 20 s. The 60 s version is reported as a robustness check.

**Rich features**, rebuilt from 360 frames and events:
- Distance and visible angle to goal.
- Opponents in the shooting lane.
- Opponents within 2 yds and within 5 yds; nearest opponent.
- Teammates ahead; the event's under_pressure flag.
- Keeper: whether visible, distance from the goal centre, lateral offset from the ball–goal-centre line, distance to the ball.
- Previous event: whether the ball arrived in the air.
- Possession duration so far, minute, score difference.

**Tests:**
- **S1 Keeper control.** The shooter-agent cells plus keeper bins: keeper > 4 yds off the line (yes/no) and lateral offset > 2 yds (yes/no). Keeper-visible decisions only. Shoot − recycle > 0.
- **S2 Doubly robust (AIPW), cross-fitted by match** with gradient-boosting nuisance models on all rich features. Effect on the decisions that *recycled* (ATU; the extra shots a coach would add) > 0.
- **S3 Low-propensity stratum.** Among decisions whose cross-fitted P(shoot) is in the bottom 40% of their population: shoot − recycle (AIPW within the stratum) > 0.
- **S4 Placebo (pre-trend).**
  - Net xG in the 20 s *before* the decision should not differ between shoot and recycle after adjustment.
  - Fails if pooled confirmation p < 0.05 *in favour of shots*.
- **S5 (report only).**
  - Coefficient stability: raw → cell → AIPW.
  - Oster δ for a zero effect, with R²max = 1.3 × R².
- **S6 (report only; likely underpowered).** Team-level ITT:
  - Exposure: a team's cross-fitted "shoots more than the rich model expects" rate in these situations, from its *other* matches.
  - Outcome: net xG per situation in this match, controlling for team rating.
  - Expected slope > 0.

**Pass rules:**
- S1–S3 each pass only with a positive sign and one-sided p < .05 in discovery, AND positive in both confirmations with pooled one-sided p < .05.
- **Strike "survives stress"** if S1, S2 and S3 pass and S4 does not fail.

## 22B: Underdog-specific rules

**Team rating:** leave-one-match-out mean of (non-penalty xG for − xG against) per team × competition-season. A rating requires ≥ 2 other matches.

**Strength gap and tiers:** gap = own rating − opponent rating.
- *Underdog* decisions: bottom tercile of gap, within each population.
- *Favourite* decisions: top tercile.

**Tests:**
- **B1 Underdog-only mining.**
  - Rerun the R15 procedure (attack and pressing families, cell vs the modal action) on underdog decisions only.
  - BH 5% in discovery, Holm + same sign in both confirmations.
  - Report confirmed rules that are not already in R15's confirmed list.
- **B2 Underdog × favourite interaction mining.** For every cell × action with ≥ 30 decisions per side in discovery for both tiers:
  - interaction = (action − modal) for underdogs − the same for favourites;
  - BH 5% in discovery;
  - confirmation: Holm + same sign in both confirmation populations.
  - These would be advantages that exist for underdogs only (or more strongly for them).
- **B3 (report only).** Each of the 8 rules' effect for underdogs vs favourites.

## 22C: Mistake ledger (descriptive)

For every confirmed rule (R15, R18, agents, B1):
- the share of times players choose the worse option;
- the cost per match, and how that cost differs for underdogs vs favourites.
