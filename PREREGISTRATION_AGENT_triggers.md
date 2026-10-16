# Pre-registration — agent "triggers" (classic pressing triggers)

Written before any outcome is computed.

## Question
Do classic pressing triggers make pressing more valuable? Within matched situations, is
(press − hold | trigger present) > (press − hold | trigger absent)?

## Unit, choice, outcome
- Unit: opponent open-play Ball Receipt* (play pattern Regular Play / From Counter) in StatsBomb 360 matches, periods 1–2,
  with a 360 freeze frame, whose preceding pass (via `related_events`, else the previous Pass event of the receiving team)
  is a non-restart pass. Both complete and incomplete receipts are included (needed for trigger c).
- Choice: we (non-receiving team) record a Pressure event within [t0, t0+2 s] (press) vs not (hold).
- Outcome: OUR net xG (own − opponent, non-penalty StatsBomb xG) in the 20 s after the receipt (`r15_360.windows`).

## Triggers (all defined from pre-receipt information except c)
- a `backward`: pass end_x < start_x − 3.
- b `touchline`: receipt location y ≤ 8 or y ≥ 72.
- c `poor_touch`: receipt outcome Incomplete, or the receiver's next own event within 3 s is a Miscontrol.
  (Caveat declared in advance: a miscontrol can be caused by our pressure — partial outcome leakage.)
- d `long_air`: pass length > 30 yd and height = High Pass.
- e `cm_crowded`: receiver's position is a central/defensive midfielder (name contains "Center Midfield" or
  "Defensive Midfield") and ≥ 2 of our outfield players within 8 yd of the receipt location (freeze frame).

## Cells
zone (r15 `cellify`: 6 x-bands × 3 y-channels) × receiver support distance (nearest receiver teammate <8 / 8–15 / >15 yd)
× number of our players within 10 yd (0 / 1 / 2+).

## Estimators
Per cell (requiring ≥5 rows in each press/hold × trigger/no-trigger group used):
- conditional effect C_k = mean(press|T) − mean(hold|T), pooled across cells with weights 1/(1/n_p + 1/n_h);
- interaction I_k = C_k(T) − C_k(no T), pooled with weights 1/Σ(1/n).
SE by match-cluster bootstrap (300 reps). Tests are z-tests.

## Hypotheses and direction
For each trigger: H1 (primary) interaction I > 0; H2 (secondary) conditional effect C > 0 (pressing beats holding when the trigger is present).

## Success rule
Discovery = men's international (DISC). BH 5% across the 10 discovery tests (two-sided p), but a discovery hit only counts
if its sign is the pre-registered (positive) one. Confirmation: same sign in BOTH men's club and women's, pooled
(inverse-variance) one-sided p < 0.05, Holm-corrected over the hits carried forward.

## Robustness for anything confirmed
(1) 45 s horizon; (2) finer cells adding our goal-side count (≤5/6–8/9+) and "already pressed" (<3 yd);
(3) excluding receipts where any of our players already within 3 yd.
