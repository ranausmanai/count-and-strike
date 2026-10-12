# Project Underdog — Round 8 pre-registration: the dead-ball decision audit

*Written 2026-10-12 before running. Rationale: set pieces are the one channel where quality matters least (confirmed in 4 populations). If teams systematically choose worse dead-ball options, fixing that is a free, coachable edge for underdogs.*

**Unit.** Every attacking dead ball:
* corners;
* free kicks taken 15–40 yards from the goal centre;
* throw-ins within 35 yards of the opponent's goal line.

**Outcome.** Non-penalty xG of the attacking team in the 20 s after the restart, including second balls. Secondary: xG conceded in the next 30 s (counter risk).

**Options (fixed now).**
* Corners: `short` (end location outside the penalty area) vs `delivered`. Delivered corners are split by technique (`inswing` / `outswing` / `other`, StatsBomb only) and by target zone: `near` (y within 6 yds of the near post), `central` (6-yard box between the posts), `far` (beyond the far post), `penalty_area` (x 102–108, central), `edge` (x < 102).
* Free kicks: `shot` vs `cross` (end in the box, high) vs `pass` (other). Location cells: distance 15–20, 20–25, 25–30, 30–40 yds × central/wide (|angle| < 25°).
* Throw-ins: `long_box` (end in the penalty area) vs `short`.

**Discovery (StatsBomb 2015/16, non-EPL leagues).** For each decision type and each location cell, estimate mean outcome by option, with corners adjusted for side and league via GLM. Define a **misallocation** where an option has higher mean outcome than the most-used option in that cell by ≥ 20% and ≥ 0.005 xG, with p < 0.05. The shortlist is every misallocation found, plus the single best corner option overall.

**Confirmation.** StatsBomb EPL 2015/16, Wyscout 2017/18 (corner zones, FK shot/cross/pass and throw-ins only; no technique), and the unseen StatsBomb WSL/NWSL/ISL. A shortlisted effect is **confirmed** if it has the same sign with one-sided p < 0.05 pooled across the confirmation sets (meta-analytic, inverse-variance), and the same sign in at least 2 of 3 sets.

**Value.** For confirmed items, the xG per season gained by reallocating the current mix to the better option, converted to points via the season simulator.

**Caveat.** Option choice depends on the defensive setup the kicker sees, so selection can bias any option comparison. That is why we confirm across populations and report the risk outcome too.
