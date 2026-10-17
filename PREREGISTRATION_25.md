# Round 25 pre-registration: grid search in the realism-calibrated simulator

*Written 2026-10-17, before the search.*

## Simulator
`src/sim/engine4.py` with `calibration4.json`, in which both teams mark the far side. The calibration was checked on 480 fresh matches:

| Measure | Simulator | Real target |
|---|---|---|
| Goals per team | 1.54 | 1.35 |
| Shots | 13.5 | 12.5 |
| Pass completion | 0.76 | 0.83 |
| Counter ratio (≤3 back vs 8+ back) | 1.74 | 2.0 |

## Underdog
The underdog's skills are scaled down until its baseline is about 0.9 points per match against the strong default team. That matches a typical bottom-half team against a top side.

## Grid
- 13 underdog knobs at 2–4 levels each. From the full product, 3,000 points are sampled uniformly without replacement.
- **Stage 1:** each point plays 100 matches against the default strong team.
- **Stage 2:** the top 50 replay 1,000 fresh matches each.
- **Stage 3:** the top 10 are tested for robustness against 5 opponent styles × 6 physics variants (calibration ×U(0.8, 1.2) on 4 parameters, shot_b ± 0.2), 300 matches each.

## Guards (stated in advance)
- **Exploit:** a tactic is flagged if its gain is > 0.5 points per match, its win rate is > 55%, or goals per match (both teams) fall outside ±40% of the baseline.
- **Supported in simulation:** a tactic must:
  - beat the baseline in ≥ 5 of 6 physics variants and ≥ 4 of 5 opponent styles;
  - have a pooled gain with a 95% CI above 0;
  - not be flagged as an exploit.

## Knob importance
Partial dependence of stage-1 points on each knob level is reported.

## Interpretation
- A "supported in simulation" result is a hypothesis to check against real data where possible. It is never proof.
