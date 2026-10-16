# Pre-registration — agent "subs": substitution timing and type

Written before any outcome was computed. Code: `src/agent_subs_*.py`. Output: `results/agent_subs.json`, `results/agents/subs_summary.md`.

## Background
Myers (2012, JQAS) proposed that a trailing team should make its 1st sub before 58', 2nd before 73', 3rd before 79'. Our earlier event study (`src/event_study.py`, 2015/16) found no effect of attacking coaching changes on xG rates. Here we test timing (H1, H2) and type (H3) on new populations.

## Populations (event-based protocol)
- **Discovery**: StatsBomb big-five 2015/16 (`data/raw/sb`, 1,551 matches).
- **Confirmation**: (a) StatsBomb women's club, WSL/NWSL/ISL (`data/raw/sb2`, 577 matches); (b) StatsBomb 360-competition event files (`data/raw/sb360`, 477 matches, men's and women's international plus men's club; event data only); (c) Wyscout 2017/18 (7 competitions). Wyscout events have no substitution events, but `matches_*.json` lists `teamsData.formation.substitutions` with minutes. So Wyscout is used for H1/H2 (timing only). It has no player positions, so it is **skipped for H3**. Wyscout xG comes from `data/proc/xg_model.pkl` (penalties 0.76).
- Regulation time only (periods 1–2). Clock = minute in match (2nd half starts at 45). Goals include own goals credited to the beneficiary.

## Strength and momentum covariates
- Strength: 2015/16 uses pre-match market odds (`market.parquet` via `team_map.json`, as in `frontier.load_segments`): `str = log(pwin/ploss)`. Elsewhere: leave-one-out xG strength = (team's mean xG for − xG against in the other matches of the same competition-season) − (the same for the opponent).
- Momentum: xG difference (own − opp) over 0–T0, plus xG difference over 45–T0 (recent).
- Home indicator.

## H1 — early sub when trailing by 1 (Myers)
- Population: team trails by exactly 1 at 55:00 **and** at 62:00 (no goal in 55–62), made no sub before 55:00, and no red card for either side before 62:00.
- Treatment: EARLY = first sub in [55, 62). LATE = first sub at ≥ 70 or none. Teams with first sub in [62, 70) are excluded.
- Outcomes (all measured from 62:00, after treatment is fixed): (i) net xG (own − opp) from 62 to the end of regulation; (ii) equalise = team is level or ahead at any point after 62.
- Direction: EARLY > LATE (positive).
- Estimator: OLS (net xG) and logit (equalise; reported as average marginal effect), adjusting for strength, xGD 0–55, xGD 45–55 and home, with SE clustered by match. Robustness (pre-registered): (R1) "landmark" contrast EARLY vs no sub by 62 (whatever happens later), which avoids conditioning on post-62 behaviour; (R2) exact-stratified difference within strength-tercile × momentum-tercile cells.

## H2 — early sub when level at 60
- Population: level at 60:00 and 67:00, no sub before 60, no red before 67. Both teams of a match can enter (SE clustered by match).
- EARLY = first sub in [60, 67); LATE = first sub ≥ 75 or none. Outcomes from 67: net xG 67–end; win = final GD > 0. Same covariates (momentum over 0–60 and 45–60) and robustness.
- Direction: EARLY > LATE.

## H3 — attacking vs like-for-like subs when trailing (event study)
- Position classes from StatsBomb names: GK; DEF = name contains "Back"; FWD = contains "Forward", "Striker" or "Wing" (but not "Back"); MID = other. Off player class = position on the Substitution event; on player class = most common position on the replacement's later events (fallback to the off class).
- Decision = a team's subs within 2 minutes of each other (cluster), made in the 2nd half while trailing, at clock ≤ 80, no red card within ±3 min. ATTACKING = at least one sub is FWD-on for DEF/MID-off and none brings on a lower class than it takes off. LIKE = every sub is same class. Others excluded.
- Windows as in `event_study.py`: pre = up to 15 min before (stopping at previous goal or 45:00); post = up to 15 min after (stopping at next goal, red card, end of regulation); both ≥ 3 min.
- Outcome: own xG in the post window. Poisson GLM, offset log(post minutes), covariates: shrunk log pre-window own-xG rate (momentum), log pre-window opp-xG rate, strength, clock, trailing-by-2+, home, opponent sub in post window; cluster-robust SE by match. Effect = coefficient on ATTACKING (log rate ratio). Secondary (not part of success rule): opponent xG and net xG per 15 min.
- Direction: ATTACKING > LIKE (positive). Populations: discovery sb; confirmation sb2 and sb360.

## Success rule (per hypothesis; primary outcome = net xG for H1/H2, own-xG log rate ratio for H3)
- Discovery: one-sided p < 0.05 in the stated direction in 2015/16 (three hypotheses → also report Holm across H1–H3).
- Confirmed if: same sign in ≥ 2 confirmation populations AND inverse-variance pooled confirmation one-sided p < 0.05 (Holm across the hypotheses that passed discovery). H3 has only two confirmation populations, so both must agree.
- A confirmed hypothesis must survive R1 (landmark) with the same sign.
- Equalise/win probabilities are secondary and reported alongside.

## Confounding (stated up front)
Coaches sub when things go badly (or when the team is weaker, tired, or has an injured player). An early sub therefore marks teams that were doing worse before the sub. Adjusting for strength and pre-sub xG difference removes part but not all of this (unobserved fatigue, injuries, the coach's private information). The LATE group (no sub until 70) is also defined by what happens during 62–70 (immortal-time style selection): teams that are doing fine keep their XI. R1 removes that by fixing groups at 62. No result here is causal. A null is a null about the association after adjustment, not proof of zero effect.
