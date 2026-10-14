# Project Underdog — Round 15 pre-registration: the full advantage search (attack + pressing)

*Written 2026-10-14 before running. Same 360 split as round 13: discovery = men's international tournaments; confirmation = men's club and women's (both required).*

## A. Attacking choices (14 options)

Options: `shot`, `dribble`, `carry_fwd`, `carry_other`, `through_ball`, `cutback`, `switch`, `cross_low`, `cross_high`, `long_fwd`, `short_fwd_ground`, `short_fwd_air`, `back_pass`, `lateral`. These come from StatsBomb pass attributes (through_ball, cut_back, switch, height) plus geometry.

Cells: zone (6×3) × space (opp_goalside 3 bins × nearest_opp 3 bins × mates_ahead 3 bins), as in round 13. Outcome: net xG over the next 20 s.

## B. Pressing choices (new)

* **Unit:** every opponent open-play ball receipt that has a 360 frame.
* **Choice:** our team presses (a Pressure event by us within 2 s) vs holds.
* **Cells:** zone of the receipt (in the opponent's attacking frame, 6×3) × the receiver's nearest teammate distance (support: <8 / 8–15 / >15 yds) × our players goal-side of the ball (≤5 / 6–8 / ≥9) × receiver under existing pressure (nearest of our players <3 yds vs not).
* **Outcome:** OUR net xG over the next 20 s.

## Rules

For both families, compare each option against the cell's most common option (≥30 uses each). Shortlist = BH 5% in discovery. **Confirmed** = same sign in both confirmation sets and pooled one-sided p < 0.05/k (Holm, applied per family). Attacking rules that involve shots also get the lane check from round 13.

## Synthesis (fixed in advance)

Every confirmed rule is valued as |advantage| × how often the worse option is currently chosen per team-match × 50% adoption × 38. The ranked list is the playbook.
