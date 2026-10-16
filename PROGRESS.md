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
## Round 11 — Tactic Miner (2.9M decisions, 2,916 situations, 6,847 rules)
- 749 nominal hits vs ~342 by chance; 0 pass BH 5% individually
- Aggregate replication (post-hoc, DEVIATIONS #4): strong hits replicate in sign 90% (EPL) / 97% (unseen), r=.85 → REAL, stable patterns
- But the combined policy implies +21 goals/season (impossible) → patterns describe good MOMENTS (selection), mostly "go forward > recycle"
## Round 12 — habit (ITT, cross-fitted, team+cell FE) test of mined rules: 28/28 NULL (BH 0). Choices-as-habits don't move outcomes
- Next frontier: StatsBomb 360 freeze frames (every event) to condition on visible space → separate choice from situation
## Round 14B — habits → moments: tempo (passes/min) passed registered confirmation (+0.12 disc, +0.17 conf, p=.004), then DIED with market-rating control (−0.03, p=.17; corr with rating .57) → quality proxy. Directness → more fast attacks but no net xG. Press height: strong in confirmation only (not shortlisted)
## Round 13 — 360 Tactic Miner (418 matches, ~730k decisions; disc = men's intl, confirm = men's club + women's)
- Space conditioning shrinks the forward premium only 20-30%
- 771 rules → 24 BH → 14 testable → 13 CONFIRMED in both confirmation populations; shot rules survive a lane-blocker control (shots taken with MORE blockers)
- Contrarian: central 25-40 yds vs packed+pressed → shoot > recycle (+0.013-0.018, p<1e-6 ×3); teams recycle 7× more than they shoot
- Rule-set value ≈ 4.1 xG/season at 50% adoption
## Round 14A — counter trigger CONFIRMED ×3: each extra opponent behind the ball at the regain −11-19% counter xG (middle third: ≤3 back 0.0144 vs 8+ 0.0072)
- Match-plan generator now includes the real-time rule card
## Round 15 — full advantage search: attack 997 rules → 11 confirmed (new pass types: none); pressing 102 → 1 confirmed (press exposed receiver near our box)
- Patience test: shot rules hold at 45/60 s (7/8 fully; long-range weakens)
- SYSTEM: "Count & Strike" (paper/count_and_strike.pdf) — Count at regain, Strike through set blocks, Run into space, Press exposed receiver, Dead balls
## Round 16 — possession simulator ("chess engine"), constrained policy iteration on 360 MDP
- H16-1 technically passes (cross-fitted gain +0.044/+0.061 per possession, CIs > 0) but the implied +90-130 xG/season is IMPOSSIBLE → structural selection bias (same in every population, so cross-fitting can't remove it); simulator overstates ~3-4×
- 68 plays (35 with adjustment carries removed) "confirmed" in both populations: all point the same way (forward/cross/shoot instead of back/lateral) — consistent with Strike Through, but R12 showed forward HABITS don't help → treat as moment-descriptions
- New candidate plays (unproven): high cross from left edge vs set defence instead of back pass (+0.03); dribble in the box when pressed instead of lateral (+0.02)
- Lesson: simulation/EPV "tactics engines" on event data overclaim; direction yes, magnitude no
## Round 17 — opponent weaknesses × own strengths (14 routes, 3 populations, ~4,000 matches): ALL FAIL
- H17-1 defensive route profiles reliable ≥0.3: SB15 2/14, women 5/14 (few teams), WY 0/14 → FAIL (attack profiles reliable up to .5-.7)
- H17-2 defender-weakness terms OOS: −1.4 [−3.0,.2], +2.4 [−1.5,6.2], −0.7 [−1.5,.03] → FAIL
- H17-3 exploiting opponent weak routes: −0.046 (SB15, sig. NEGATIVE), −0.026, −0.012 xG/SD → FAIL
- Exploratory: leaning into own strong routes ≈ 0 (+.003, −.009, +.018) → no consistent gain
- Conclusion: data-driven "find the opponent's weak spot" game plans don't work at route level; weakness profiles are mostly noise. Player-level/tracking weaknesses untested (needs tracking data → ASI)
## Round 18 — think like a coach
- 18A OVERLOAD: CONFIRMED. Passing into a lane where we outnumber them (players ahead of ball, 360) vs the crowded lane, same cell: +0.0014/+0.0024/+0.0010 net xG; dose-response +0.0006-0.0007 per player (p<.001 ×3); with lane-change matched +0.0010-0.0013 (p<.001 ×3); gain from own chances (not risk); completion 94% vs 84%. Players pass into crowded lane 66% vs overload 23% when both exist.
- H18-2 switch-to-overload: right sign, underpowered (not confirmed)
- 18B formation matchups: NULL (OOS gain negative in all 3; back3 vs 2 strikers null)
- Locked prediction addendum O1/O2 (PREDICTIONS.md)
## Round 19 — sweet-spot search (~290 testable cells + count/overload sweeps): nothing new confirmed
- Tempo (first-time vs controlled): 201 cells, BH 0. Pressing traps: 86 cells, BH 0. Count sweep: 'go fast' positive in own half/middle third up to ~6 back (descriptive), no single threshold confirmed ×3. Overload sweep: aggregate confirmed (R18), slices underpowered
- Rule value (simulated, 50% adoption, 2025/26 PL): Strike ≈1.4-2.0 pts, Overload ≈1.4-2.1, Dead balls ≈1.5-1.8, Match mgmt (model) ≈1.4-2.1, Run ≈0.3, Count (assumed) ≈0.2, Press ≈0.1; ALL ≈ 6.9-8.3 pts/season
## Agent-based simulator (src/sim/): 22-player numba engine, calibrated (≈1.2 goals, 13 shots, 49 tackles per team; completion 67% too low)
- Underdog (5-15% worse skills) baseline 0.51 pts/match. 2,000 tactics × 150 matches (300k) + top-40 × 2,000 fresh + top-10 × 5 opponent styles
- Best tactics ~2.8 pts/match (≈95% wins) → simulator exploits. Even the default underdog tactic gets 2.9 pts vs a 'possession' strong side → sim's tactic space dominated by artefacts
- Partial dependence (R² .89): defensive compactness ~25 m, attacking width ~40 m, no extra pressers, shoot more
## Round 20 — real-data bridge (full-width 360 frames): sim claims FAIL except 'shoot more' (already confirmed)
- Attacking width 32-48 vs >48: +.0027/+.0024/−.0008 → not confirmed
- Narrow defending ≤32 m: −.008/−.009/−.001 → OPPOSITE of sim
- Pressing: inconsistent sign
- Lesson: agent sims find exploits of their own physics; only "shoot more" transferred (and was already known from data)
## Round 21 — disorganised moments (4 populations, ~4,400 matches, 160k restarts, 11.7k goals)
- Quick restarts (<6 s vs >15 s, same type/zone): positive in 4/4 (+.0018 sig, +.0014, +.0004, +.0003) but pooled others n.s. → NOT confirmed (consistent direction, small)
- 'Most vulnerable right after scoring': O/E conceding in 5 min after scoring = 0.77/0.79/0.59/0.79 (all CIs < 1). ≈ the ~1 min kickoff/celebration dead time → NO extra vulnerability: myth busted. Same for 'strike back after conceding'
- Goal kicks short vs long by opponent press: no consistent interaction
## Agent wave 1 (2026-10-16, 6 parallel agents; summaries in results/agents/)
- shooter: CONFIRMED shoot>recycle near box for EVERY role (FW +.113, AM/W +.079, CM/DM +.074, DEF +.078 pooled); CM/DM 20-30 yds +.023; finishing skill doesn't modulate
- combos: third-man release CONFIRMED (+.01 xG avg, +.10-.14 wide in box) — caveat: label includes completion/progress; one-two/lay-off/switch not robust
- cross: MIXED — cross>carry from wide (+.011 pooled) but no dose-response with box bodies (myth: 'only cross with bodies in box')
- restarts: quick restarts face 0.5-0.8 fewer defenders goal-side but NO chance gain; final-third free kicks: set up properly
- triggers: NONE of 5 pressing triggers confirmed
- subs: Myers timing & attacking subs NOT confirmed
## Agent wave 2
- secondball: CONFIRMED — aim long balls where we have local superiority (retention +5 pts per +1 player; net xG +.003/player; high balls superior−outnumbered +.007); ~25% of high balls land where outnumbered
- runners: NOT CONFIRMED (more runners at regain don't add xG once Count fixed; lead: fewer conceded)
- corners360: NOT CONFIRMED (box numbers / 6-yd overload no effect; low power)
- keeper: NOT CONFIRMED (quick release +.002 positive in all 3, fails Holm)
## Confirmed rule set now: Count, Strike through (all roles), Run into space, Press the exposed, Dead balls, Play into the overload, Second-ball overload (+ third-man release, asterisk)
## Round 22: outside review ("most rules are obvious; is Strike causal?"), so we tried to kill our own rules
- **22A Strike stress (pre-registered): FAILS.**
  - Keeper-bin cells still +.07–.09.
  - Rich-feature AIPW: ATE +.028; for the shots a coach would ADD (ATU) +.009 at 20 s and +.003 (n.s.) at 60 s.
  - Low-propensity situations ≈ 0 (+.000/+.000/+.005).
  - PLACEBO FAILS: the 20 s *before* a shot is already +.025 net xG better. Shots come in scrambles and good spells (selection).
  - Oster δ 0.7–0.9; team-habit ITT null.
  - The big shoot-vs-recycle numbers are mostly "players shoot when it's good".
- **22B underdog-specific rules: NONE.**
  - B2 interaction search: 345 attack + 24 press tests, BH 0.
  - B1 found only shoot-variants.
  - B3: rules work about equally for underdogs and favourites (overload slightly larger for underdogs, +.0025 vs +.0008, n.s.).
- **22C stress battery on every rule (post-hoc, DEVIATIONS #7):**
  - Overload: AIPW +.0025 disc / +.0058 pooled. The placebo is significant but tiny (+.0003, about 5% of the effect). Survives on magnitude; flagged.
  - Second ball: retention +20–26 pts SURVIVES, but net xG ≈ 0.
  - Run: disc +.019, pooled +.004 n.s. → FAILS.
  - Count-go (≤3 back): underpowered → FAILS.
  - Press the exposed: ≈ 0 → FAILS.
  - The Count *situation* gradient (counters worth less with more defenders back) remains a descriptive fact, not a proven choice.
- **22D ledger:**
  - When both exist, ALL tiers pass into the crowded lane 66% vs the overload 23% (about 90 times per team-match).
  - Underdogs aim long balls where outnumbered 29% vs favourites 18%.
- **Season value of the survivors** (Overload + Dead balls + match management): ≈ +5.5 pts (was +8). See results/clubs_playbook_v3.json.
- **New headline:** "The overload gap". Players know overloads but don't use them; that is the one decision rule that survives every test. The selection lesson (Strike) is the big methodological finding.
