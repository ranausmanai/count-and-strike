# Project Underdog — Round 18 pre-registration: thinking like a coach (overloads and formation matchups)

*Written 2026-10-15 before running.*

## 18A: play into the overload (360 data; same discovery/confirmation split as rounds 13–15)

At every open-play pass from the middle third onward (x ≥ 40), we count the visible players **ahead of the ball** (x > ball x − 5) in three lanes: left (y < 26.7), centre, right (y > 53.3). For each lane: **overload** = our players − their players. A pass's *target lane* is the lane of its end location.

* **H18-1.** Within the same situation cell (zone × opponents goal-side × nearest-opponent distance), passes into a lane with overload ≥ +1 yield higher net xG (20 s) than passes into a lane with overload ≤ −1. One-sided p < 0.05 in discovery, **and** the same sign with pooled one-sided p < 0.05 in the two confirmation populations (men's club, women's).
* **H18-2.** Switches (passes that change lane by two, e.g. left → right) into a lane with overload ≥ +1 beat short passes that stay in a lane with overload ≤ −1, within the same cell. Same criterion.
* **H18-3 (dose–response).** Net xG rises with the target lane's overload (linear term, cell fixed effects). Same criterion.

## 18B: formation matchups (StatsBomb starting formations)

* **Families.** Back line: 3/5 vs 4 (first digit). Strikers: 1 / 2 / 3+ (last digit).
* **Outcome.** For StatsBomb 2015/16 (market odds available): goal-difference residual vs the closing market. All populations: xG difference residual vs leave-one-out xG strength.
* **H18-4.** A matchup model (team back-line family × opponent striker family, and vice versa) improves held-out prediction (leave-one-competition-out) over strength + own-formation main effects. Success: OOS gain with bootstrap CI > 0, in at least 2 of the 3 populations that have formations (SB 2015/16, SB women's/ISL, SB 360 competitions).
* **H18-5 (coaching lore, directional).** A back three facing two strikers does better (xG difference) than a back four facing two strikers, relative to strength. One-sided p < 0.05 pooled, same sign in ≥ 2 populations.
