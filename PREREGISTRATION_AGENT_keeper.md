# Pre-registration — agent "keeper": does a quick keeper release help?

Written before any outcome (xG) was computed on the keeper-release sample. Only structural checks were made (goalkeeper event types/outcomes, which events follow them, pass body parts, 360 frame availability at GK events).

## Coaching claim
After the keeper wins the ball in open play, release quickly (< ~5 s) by throw/roll/kick to start a counter while the opponent is still committed forward.

## Populations
- Discovery: StatsBomb big-five 2015/16 (`data/raw/sb`, 1,551 matches) = `SB_1516`.
- Confirmation (event-based): `SB_women_ISL` (`data/raw/sb2`: WSL/NWSL/ISL, 577 matches) and `SB_360comps` (`data/raw/sb360/*_events.json.gz`, 477 matches). With two other populations, "same sign in ≥ 2 of the other populations" means both.
- H2 (360 frames) uses the 360 split: discovery = men's international (`r15_360.DISC`), confirmation = women's (`WOMEN`) and men's club (rest).
- Periods 1–2 only.

## Sample definition
- Gain event: `Goal Keeper` event of type Collected, Keeper Sweeper, Shot Saved, Shot Saved Off Target, Shot Saved to Post, Smother or Save (any outcome; possession retention is defined by the sequence below). Gain time `t0` = its timestamp.
- Scan the following events (≤ 25 events). Allowed before release: the same keeper's Carry / Ball Receipt* / Goal Keeper events and opponent Pressure events. Release = the first `Pass` by the same keeper with no pass type (open-play distribution; `Recovery` type also allowed). The sequence is dropped if any other on-ball event happens first (any team-mate or opponent on-ball event, keeper Clearance/Miscontrol/Dispossessed/Shot etc.), if the keeper's pass is a Goal Kick/Free Kick/other set piece, or if any stoppage-type event occurs in between (Injury Stoppage, Substitution, Referee Ball-Drop, Foul Committed, Foul Won, Bad Behaviour, Player Off/On, Tactical Shift, Half End).
- Delay = t_release − t0. Drop delay < 0 or > 60 s. Quick = delay < 5 s; slow = delay > 10 s; 5–10 s only described.
- Release type: `hand` = body part Keeper Arm; `dropkick` = Drop Kick; `foot` = Left/Right Foot (ball on the ground); others dropped.
- Gain type: `save` (Shot Saved* / Save / Smother) vs `claim` (Collected) vs `sweeper` (Keeper Sweeper).
- Game state = keeper's team goal difference at t0 (−, 0, +).
- Length: pass `length` (yards). Short < 35, long ≥ 35.
- Outcome: own xG and opponent xG (non-penalty StatsBomb xG) in [t_release, t_release + 30 s]; net = own − opp. Robustness horizon 60 s.

## Cells and estimator
- H1 cell = release type × game state × gain type. H3 cell = game state × gain type (release type omitted because it is nearly collinear with length; a within-release-type version is reported as secondary).
- Estimate = within-cell difference of means (A − B), weighted by min(n_A, n_B); cells need ≥ 5 of each. SE from match-cluster bootstrap (500 reps). One-sided z-tests.

## Hypotheses
- **H1a**: quick − slow own xG(30 s) > 0.
- **H1b**: quick − slow net xG(30 s) > 0.
- **H3**: among quick releases, short − long net xG(30 s). Direction not fixed by lore: the discovery sign sets the direction; confirmation is one-sided in that direction (discovery must reach two-sided p < 0.05 for this to count).
- **H2 (360)**: n_opp_half = visible non-keeper opponents with x < 60 in the freeze frame at the gain event (frame of the first later keeper event if the gain event has none). High = above the discovery-set median of n_opp_half (computed before outcomes), low = at or below. Estimate = (quick − slow net | high) − (quick − slow net | low) > 0, both using H1 cells. Success: one-sided p < 0.05 in discovery (single test; BH with one test), same sign in both confirmation sets, pooled one-sided p < 0.05.

## Success rule (event-based H1a, H1b, H3)
Discovery `SB_1516` estimate in the predicted direction; same sign in both `SB_women_ISL` and `SB_360comps`; pooled inverse-variance one-sided p < 0.05 over the two confirmation populations, Holm-corrected over the three event tests. Discovery one-sided p is also reported.

## Robustness (for anything confirmed)
(i) 60 s horizon; (ii) finer cells adding `pressed` (any opponent Pressure between gain and release) and keeper x band (< 6, 6–18, > 18).

## Descriptive
Share of quick / 5–10 s / slow releases by population and release type; n_vis at 360 gains.

## Caveats acknowledged in advance
Delay is a choice confounded with what the keeper sees (team-mates available, opponents nearby); slow releases include time-wasting by teams ahead (game-state cells partly address this). 360 frames show only the visible area, so n_opp_half undercounts distant opponents. Observational; no causal claims beyond within-cell association.
