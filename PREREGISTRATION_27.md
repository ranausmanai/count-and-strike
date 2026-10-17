# Round 27 pre-registration: fast, flat long balls (a refinement of Escape)

*Written 2026-10-17, before outcomes.*

## Lead
Tracking data (agent track, E3) found that defenders close in on the landing area while the ball is in the air. Flights of more than 2.5 s were kept 57% of the time, against 65% for shorter ones.

## Sample
StatsBomb 360 long balls (≥ 35 yds, open play): `data/proc/agent_chain_L.parquet`.
- Flight speed = pass length / StatsBomb pass `duration` (yds/s), joined from raw events.
- Passes with duration ≤ 0.2 s or > 8 s are excluded.

## Hypothesis
**H1: faster flights are retained more often.** Fast = the top tercile of speed within each population; slow = the bottom tercile.
- **Outcome:** keep5, meaning we still have the ball 5 s later.
- **Matching cells:**
  - length band (35–45, 45–60, 60+);
  - height (Ground / Low / High);
  - local balance at the kick (ours10 − theirs10, clipped to −1…+2, landing area on camera);
  - start zone (x < 40, 40–60, > 60).
- **Test rule:** discovery is men's international (one-sided p < .05). Confirmation requires the same sign in men's club AND women's, with pooled one-sided p < .05.

**H2 (stress test):** a cross-fitted AIPW estimate of fast vs slow, using the same features as continuous variables, with the same pass rule.

## Report only
- **R1:** keep5 by speed tercile × balance.
- **R2:** net xG over 20 s and 45 s, fast vs slow, cell-matched.
