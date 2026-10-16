# Project Underdog — Round 21 pre-registration: exploiting disorganised moments

*Written 2026-10-16 before running. Four event-data populations: StatsBomb 2015/16 (discovery), StatsBomb unseen women/ISL, StatsBomb 360 competitions (event layer), Wyscout 2017/18. Confirmation requires the discovery sign in at least 3 of the 3 others, with pooled one-sided p < 0.05.*

## 21A: quick restarts

For free kicks and throw-ins in the attacking half (x ≥ 60), *restart delay* = seconds between the stoppage (the last event before the restart) and the restart.
* Quick: < 6 s. Slow: > 15 s.
* Outcome: the restarting team's net xG over the next 20 s.
* Comparison is within restart type × zone (x band 60–80 / 80–96 / ≥ 96 × centre/wide).

**H21-1:** quick > slow. Wyscout uses event timestamps in the same way.

## 21B: vulnerability windows

Goal hazard per minute by window:
* minutes 0–5 after the team's own goal (just scored);
* minutes 0–5 after conceding;
* minutes 0–5 after kick-off of each half;
* the last 5 minutes of the first half.

Each is compared with the team's own baseline concession rate in the same match-state cell (score difference −1/0/+1 and strength tier). We estimate the conceding hazard and the scoring hazard.

* **H21-2:** teams concede more in the 5 minutes after scoring than their baseline (the "most vulnerable after scoring" lore).
* **H21-3:** a team that just conceded scores more in the next 5 minutes than its baseline (the "strike back while they celebrate" lore).

Confirmation: the same sign in ≥ 3 of the 4 populations and pooled p < 0.05. The direction is registered as the lore states it.

## 21C: goal kicks against pressing

Goal kicks are *short* (pass length < 32 yds) or *long*. Outcome: the kicking team's net xG over 30 s.
* Stratified by the opponent's pressing intensity: the opponent's pressures per opponent pass in the previous 15 minutes (StatsBomb only), in tertiles.

**H21-4:** short minus long is lower (more negative) against the high-press tertile than against the low-press tertile. Requires the same sign in all three StatsBomb populations.
