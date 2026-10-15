# Project Underdog — Round 17 pre-registration: opponent weaknesses × own strengths (the "game-plan" test)

*Written 2026-10-15 before running. Motivation: a reader argued that the real opportunity is analysing specific opponents' weaknesses and matching them to your own strengths. Round 3-B tested this at a coarse level (6 channels) and found defender profiles unreliable. This round tests it at route level, across three full-league populations.*

## Routes (14)

Every non-penalty shot is assigned to the route that created it:
* set pieces: `corner`, `free_kick`, `throw_in`;
* transitions: `counter`, `high_regain`;
* build-up, by assist type: `cross_A`, `cross_B` (by flank), `cutback`, `through_ball`, `pass_central`, `pass_wide`, `aerial_pass`;
* unassisted: `solo_box` (shot in the box) or `long_shot`.

**Populations.** StatsBomb big-five 2015/16 (1,551 matches); Wyscout big-five 2017/18 (1,826; own xG model; cutback approximated from geometry); StatsBomb women's/ISL (577).

**Profiles (built only from prior matches).**
* *Attacking strength* of a team on route k = its xG via k / expected.
* *Defensive weakness* of a team on route k = xG conceded via k / expected.
* Expected = league route rate × the team's overall attack (or defence) level. Profiles are shrunk toward 1 with k = 8 pseudo-matches.

## Tests

**H17-1 (do weaknesses persist?).** Split-half reliability (odd vs even matches, Spearman–Brown) of each team's route-specific concession share. Success: ≥ 3 routes with reliability ≥ 0.3 in at least 2 of 3 populations. Attacking route profiles are reported as a reference.

**H17-2 (do weaknesses predict?).** For team-match-route xG, a Poisson model with an offset from overall strengths. Terms: log attack_k (attacker's prior route strength), log weakness_k (defender's prior route weakness), and their product (the matchup). Evaluation is leave-one-league-out, with deviance gain over the model without the defender terms. Success: the defender-weakness term (or the matchup term) has a positive coefficient **and** a positive out-of-sample deviance gain, with a match-bootstrap 95% CI > 0, in at least 2 of 3 populations.

**H17-3 (does exploiting them work? natural experiment).** In each match, a team's exploitation index = Σ_k (its route share in this match − its usual share) × the opponent's prior weakness on k. Outcome: total xG for − expected (from strength), and xG difference. Controls: own and opponent overall levels. Success: coefficient > 0, one-sided p < 0.05, in at least 2 of 3 populations.

**H17-4 (value, simulated; only if H17-2 passes).** A game-plan optimiser reallocates up to 20% of a team's attacking volume toward routes where (own strength × opponent weakness) is highest, using the H17-2 coefficients. We report the xG per match and points per season, evaluated on held-out leagues.
