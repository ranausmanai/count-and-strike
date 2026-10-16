# Pre-registration — agent "chain": the long-ball overload chain and its cost at the back

*Written 2026-10-16 BEFORE any outcome for the new table is computed. Data: StatsBomb 360 (477 matches).
Discovery = men's international (`r15_360.DISC`); confirmation = men's club AND women's (`WOMEN`), both required.
Code: `src/agent_chain_build.py` (table), `src/agent_chain_analyse.py` (tests), `src/agent_chain_stress.py` (battery).*

Background: Overload (pass into the lane where we outnumber, r18a) and second-ball superiority (agent_secondball, ours10 vs theirs10)
both survive. Question: does the overload rule hold in the second-ball phase, is the chain worth more than its parts, what does
committing bodies to the landing zone cost at the back, and is "go long into superiority" a better goal-kick choice than short?

## Units
- **Long ball L**: `Pass`, periods 1–2, `pass.length` ≥ 35, type open play or `Goal Kick` (same as agent_secondball), with a 360 frame.
  From the frame at L: `ours10`/`theirs10` = visible outfield teammates (not actor, not keeper) / opponents (not keeper) within 10 yds of
  the end location; `sup = ours10 − theirs10`; `vis` = end location inside `visible_area`. Primary samples use `vis == 1`.
- **Follow-up pass F** ("second ball won, next pass"): the first non-restart `Pass` by L's team after L, within 15 s of L, in the SAME
  StatsBomb possession as L (= we kept / won the second ball without the opponent establishing possession), with its own 360 frame.
  Lane overload at F computed exactly as r18a: lanes y<26.7 / 26.7–53.3 / >53.3; count visible outfield teammates (not actor) minus
  opponents (not keeper) with x > x_F − 5 in the target lane → `target_ov`; also `start_ov`, `max_ov`, `min_ov`, `opp_goalside`,
  `nearest_opp`, r18a `cell` (x band from 39 × lane × goal-side band × nearest band). `O = 1` if target_ov ≥ 1, `O = 0` if ≤ −1 (0 dropped).
  `retained` = 1 if such an F exists (frame or not).
- **Goal kick G**: every `Goal Kick` pass with a frame (any length).

## Outcomes (non-penalty StatsBomb xG; own counts from the event time, opponent from strictly after)
- From L time: `own45L`, `opp45L`, `net45L` (= own − opp, 45 s), `net20L`.
- From F time: `net20F`, `net45F`.
- From G time: `net45G`, `opp45G`.
- `pre_net`: net xG in the 20 s BEFORE the decision (placebo, r22c.add_pre).

## Game state (built from events)
Score difference for the acting team at the event (goals from `Shot` outcome Goal plus `Own Goal For`), banded ≤−1 / 0 / ≥+1;
minute band 0–29 / 30–59 / 60–74 / 75+. Press height at G = visible opponent outfielders with x < 40 in the frame, banded 0–1 / 2–3 / 4+.

## Hypotheses (confirmatory family of 7; direction stated)
- **H1a (+)**: overload holds after a won second ball. Sample: F with x_F ≥ 40, target_ov ≠ 0, L vis-free (any L).
  Within r18a cells, mean(net20F | O=1) − mean(net20F | O=0); cells need ≥ 10 per arm; weights min(n1,n0); r18a SE formula.
- **H1b (+)**: same as H1a with net45F.
- **H1c (+, chain super-additivity)**: sample: retained L with vis = 1, sup ≠ 0, F with x_F ≥ 40 and target_ov ≠ 0. S = 1[sup > 0].
  OLS `net45L ~ S*O + C(ez) + C(height) + gk + pressure + C(fz)` (ez = L end zone as agent_secondball; fz = F x band × lane);
  coefficient on S:O > 0. Cluster-robust SE by match. The four cell means (S,O) are reported descriptively.
- **H2a (+ = cost)**: rest-defence cost. Sample: all L with vis = 1. OLS `opp45L ~ ours10c + C(theirs10c) + C(ez) + C(height) + gk + pressure`,
  ours10c = min(ours10, 4), theirs10c = min(theirs10, 4); coefficient on ours10c > 0 means each extra committed player raises xG conceded.
- **H2b (two-sided, optimum)**: same sample/controls, `net45L ~ C(supc)` with supc = clip(sup, −2, +2) (reference 0);
  contrast (sup=+1) − (sup=+2 or more). Positive = +1 is the optimum; negative = more commitment pays.
- **H2c (+)**: `net45L ~ ours10c + C(theirs10c) + controls`, coefficient on ours10c > 0 (commitment net of its cost).
- **H3 (+)**: goal kicks. T = long (length ≥ 35, vis = 1, ours10 > theirs10); C = short (length < 30). Within cells score band ×
  minute band × press band, mean(net45G | T) − mean(net45G | C); cells ≥ 5 per arm; weights min(n1,n0).
  Secondary (descriptive): long-NOT-superior vs short; opp45G.

## H4 (descriptive, not in the family)
Tiers from `r22b_underdog.tiers()` logic (r22_ratings.ratings() gap terciles within population; under / mid / fav). Report H1a, H1c,
H2a, H2c, H3 estimates in the under and fav tiers per population and the pooled (club+women) under − fav difference with SE.
Usage: share of L with sup > 0 / sup < 0 and retained rate by tier.

## Success rule
- Discovery: two-sided p (z = est/SE) for the 7 tests, Benjamini–Hochberg 5%, with the stated sign (H2b: either sign, sign then fixed).
- Confirmation: same sign in men's club AND women's, inverse-variance pooled one-sided p < 0.05, Holm across carried tests.
- **Stress battery** on every confirmed test (r22c crossfit + r22a aipw/verdict, 5-fold GroupKFold by match, ATE):
  binary contrast underlying the test — H1a/b: T = O on net20F/net45F, features F x, y, opp_goalside, nearest_opp, start_ov, max_ov,
  min_ov + L ex, ey, length, height code, gk; H1c: T = O within S = 1 (net45L); H2a/H2c: T = 1[ours10 ≥ 2] vs ours10 ≤ 1 with
  theirs10, ex, ey, x, y, length, height, gk, pressure as features; H3: T as above with score_diff, minute, press, x, y.
  Survives if AIPW `verdict` passes AND the 20-s pre-decision placebo does NOT (pooled > 0 with one-sided p < .05 = fail).
- Robustness for confirmed: finer matching (H1: add L height and gk to cell; H3: add L-player press count 4+ split by score) and
  the alternative horizon (20 s ↔ 45 s).
- A "CONFIRMED" verdict for the chain tactic requires H1a or H1b confirmed AND surviving the battery; H1c is the test of "worth more
  than either alone".

## Caveats anticipated
Frames are at the pass, not landing; the visibility filter selects on-camera landings; retention is post-treatment for L (H1c conditions on
it, so H1c is a within-retained comparison, not a policy effect); goal kicks rarely have landing on camera, so H3 T may be small.
Not causal. ≤ 2 worker processes.
