# Project Underdog — Round 2 pre-registration: a discovery engine with an out-of-time lock

*Written 2026-10-12 before running any of the analyses below. Round 1 (PREREGISTRATION.md) is complete; its data have been seen, but none of the features below were tested against outcomes in round 1. The one exception is "total shots" (P4), which is included here again as a known positive control.*

## Goal

Find any **habitual style trait of an underdog**, or any **style matchup** between an underdog and a favourite, that predicts the underdog beating the closing betting market. The trait must replicate on seasons that were never touched during the search. A trait that beats the market is, by construction, something bookmakers and (presumably) clubs do not already price in. That is our operational definition of "discovered".

## Data and split

football-data.co.uk, big-five leagues, matches with closing (or best-available) odds and match stats.
* **Discovery set:** seasons 2005/06 to 2016/17. This is the only data used for searching.
* **Locked holdout:** seasons 2017/18 to 2025/26. It is not read by any analysis until the shortlist is frozen and committed. `src/discovery.py` enforces this with a flag.

Underdog = side with the lower market win probability.

## Outcomes

* **Primary:** goal-difference residual = (underdog goals − favourite goals) − (λ_u − λ_f), where λ are the market-implied goal rates. It has more power than points.
* **Secondary:** points residual = points − market expected points.

## Feature space (fixed now)

Habitual traits are the rolling mean over each team's previous 10 league matches (min 10), demeaned within league-season. They are computed for both underdog (U) and favourite (F):

`shots_for, shots_against, sot_rate_for (SOT/shots), sot_rate_against, fouls_committed, fouls_won, corners_for, corners_against, yellows, gd_resid_form` (that last one is the team's own market GD residual over its last 10, i.e. recent over-performance).

Tests:
1. U main effects (10).
2. F main effects (10).
3. U × F interactions, all 100 pairs. Each model controls for both main effects, the market gap and home.

Total: 120 tests, each OLS with SEs clustered by underdog team-season.

## Selection rule (discovery set only)

Benjamini–Hochberg at FDR 10% across the 120 tests. If nothing passes BH, the shortlist is the 5 smallest p-values, and they are labelled "weak candidates".

## Confirmation rule (holdout)

A shortlisted effect is **confirmed** if it has the same sign in the holdout, with a one-sided p below 0.05/k (Holm–Bonferroni over the k shortlisted effects). We report all shortlisted effects' holdout estimates regardless of outcome.

## ML arm

LightGBM predicts the GD residual from all 20 U/F traits plus the market gap. Fit on discovery seasons, early-stopped on 2015/16–2016/17, evaluated once on the holdout. **Success:** holdout MSE is lower than that of predicting zero (the market) with a match-bootstrap 95% CI excluding 0.

## Positive control

U shots_for + shots_against (≈ P4 tempo) should reappear in the discovery set with a negative sign. If it does not, the pipeline is suspect.
