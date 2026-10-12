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
- R3-A (regains, 110k SB + 431k WY): H3a-1 FAIL (opp-commitment adds nothing beyond game state OOS); H3a-2 FAIL as registered, REVERSE replicates in 3 samples (regain after opponent pushed deep ≈1.7× counter xG); H3a-3 pass (top20% carry 42-66%)
- R3-B: H3b-1 FAIL (WY profiles unreliable); H3b-2 passes literally (+4.9 [1.5,8.5]; +2.5 [0.2,4.6]) but gain = attacker specialisation; DEFENDER weak-spot term null in both → no exploitable weak spots at channel level
## Round 4 — does defensive structure beat xG? (PREREGISTRATION_4.md)
- R4 (freeze frames): H4-1 null (structure already inside SB xG); H4-2 PASS (cone density per shot rel .75, wealth r .07); H4-3 PASS (GA−xGA rel .45/.34) but mostly goalkeeper, partly wealth-linked
- R5 (confirm cone density, prereg): H5-1 FAIL (p=.28), H5-2 FAIL (wrong sign, unseen WSL/NWSL/ISL), H5-3 FAIL (p=.10) → round-4 candidate dies
- R6 (confirm on unseen WSL/NWSL/ISL, 577 matches): H6-1 CONFIRMED set pieces flat (0.72 vs 1.02, diff CI [.11,.50]); H6-2 not confirmed (right sign);
  H6-3 CONFIRMED vulnerability window (p=.004). Same spec in 4 populations: ×1.69, ×1.57, ×1.37, ×1.52 counter xG when opponent had pushed deep (all CIs > 1)
- Bait check: deep blocks put more regains in the window but NOT more counter xG/match → in-moment decision rule, not a reason to sit deeper
- NEXT: paper v2 (all rounds), figs 8-9
## Round 7 — chase timing (PREREGISTRATION_7.md): FAIL
- H7-1: trailing-team xG escalation 45-60 → 75+ is only +10-12% (CIs include 0 in SB15 & WY; negative in SB2)
- H7-2 invalid: late "all-in" multipliers have trailing team conceding LESS (×0.67-0.87) → reflects leader's passivity (pre-registered confound). Gains 0.008-0.025, below threshold in 2/3
## Round 8 — dead-ball decision audit (PREREGISTRATION_8.md)
## Round 8 — dead-ball audit: confirmed on the letter, KILLED by registered selection checks
- Shortlist: corner→penalty-spot area (×1.41 disc, ×1.18 conf 3/3) and long throw into box (×1.67, ×2.25 conf 3/3)
- Same-location throws: effect ≈ 0 in all 4 populations (location confound). Corners with team FE: discovery sign flips (−0.33), confirmation mixed → not robust
## Round 9 — referee natural experiment (PREREGISTRATION_9.md)
- R9 referees: NULL in discovery and confirmation (H9-1 +0.011/+0.024; H9-2 wrong sign; H9-3 null); assignment unrelated to gap
## Round 10 — power + surrogate endpoints (the constructive thesis)
- Power: a +20% corner routine needs 15 club-seasons (0.7 league-season); a +0.1 pts/match tactic needs 57 club-seasons → football tactics live in a low-power regime (explains all observational nulls)
- Surrogate 'final-third entry ≤10 s' after regains: 20% rate, 66% of regain xG, team-level predictive validity r=.27 (xG itself .31); needs 0.7 club-seasons for +20% (10× fewer than xG)
- Corners: 'first contact' surrogate INVALID (r=−.19); corners need consortium
- Window effect visible in surrogate: OR 1.47 [1.36,1.59] SB15; 1.26 [1.08,1.47] unseen → testable by one club within ~1-1.5 seasons
- NEXT: paper v3 (thesis: tactics problem = power problem; trial protocol)
