# Progress log (resume from here)

## Plan
1. [x] Pre-registration (commit d94684e)
2. [x] Data: StatsBomb 2015/16 events (src/fetch_statsbomb.py → data/raw/sb), football-data odds (data/raw/fd)
3. [x] Event processing → possessions/segments with style features + xG (src/build_segments.py)
4. [x] Frontier model fit (non-EPL) + P1 held-out EPL test
5. [x] Match DP (minute × goal diff) + policy optimisation → P2 with bootstrap
6. [x] P3 (habitual style vs market, EPL) and P4 (tempo vs market, 20 seasons)
7. [x] Season sim (Brentford vs 2025/26 PL) → P5
8. [x] Leicester case study, exploratory
9. [x] Paper + figures + artifact page

## Log
- 2026-10-11 01:00 started; downloads running.
- 02:40 Results so far (all in results/*.json):
  - P1 FAIL (lagged style adds nothing out of sample; contemporaneous 'wins' only through mechanical confounding)
  - P2 FAIL on the honest cross-fit (0.015 pts/match, CI −0.04..0.08). The literal criterion 'passes' only via the optimiser's curse
  - P3 null (EPL +0.72, CI −0.44..1.88)
  - P4 SUPPORTED, modest: −0.0069 pts per habitual shot (p=.008); same sign in all 5 leagues; fades after 2015
  - Event study (post-hoc): attacking/defensive coaching changes barely move xG (±5%)
  - Channels (post-hoc): set pieces flat (elasticity .41 vs open play 1.09); counters steepest (1.19), so NOT an equaliser.
    Cheap skills: corner xG/corner and throw xG/throw (persistent, unrelated to wealth)
- NEXT: theory-vs-P4 check; season sim (Brentford 2025/26) incl. P5 risk dial + set-piece playbook value; Leicester case; paper
- 03:30 DONE: season sim, P5 (not supported), Brentford-vs-City policy, Leicester case, Wyscout 2017/18 replication
  (set-piece flatness replicates; counter-elasticity claim withdrawn), set-piece market test (null), figures, paper PDF.
- Possible next steps: licensed tracking data to re-test P1/P2; Transfermarkt squad values for a cleaner wealth axis;
  more seasons of event data for cheap-skill reliability.

## Round 2 (2026-10-12) — discovery engine, out-of-time lock (PREREGISTRATION_2.md, commit 56a3532)
- Discovery 2005-16 (19,605 underdog matches): positive control reproduced; BH shortlist = F:shots_for, F:shots_against, U:yellows×F:fouls_won
- Holdout 2017-25 (15,818): NONE confirmed (all shrank ~85-95% toward 0). ML arm: holdout gain −0.0011 (CI −0.0024..+0.0002) → null
- Conclusion: post-2017 market prices all box-score style info. Edges must come from finer-grained (event) information.
## Round 3 — counter-attack vulnerability windows + opponent weak-spot matchups (PREREGISTRATION_3.md)
