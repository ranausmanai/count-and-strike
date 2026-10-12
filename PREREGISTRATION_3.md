# Project Underdog — Round 3 pre-registration: when is a stronger opponent vulnerable?

*Written 2026-10-12 before any of these analyses were run. Round 1 computed possession channels but never modelled regain-level counter value or opponent concession profiles.*

## R3-A: Vulnerability windows (the brief's question: "when does an opponent become vulnerable to a counter-attack?")

Unit: every open-play **regain**, i.e. a possession won from the opponent in open play.
Outcome: xG of the regaining team's possession, measured in the first 20 seconds.

Situation features at the moment of the regain (none are chosen by the regaining team at that instant):
`regain_x, regain_centrality, regain_type (interception/tackle/recovery/other), opp_prev_passes (passes in the opponent's possession that just ended), opp_prev_max_x (deepest point the opponent reached, in the regainer's frame), opp_prev_secs, game_state, minute, log strength ratio`.

* **Train:** StatsBomb 2015/16, La Liga + Serie A + Ligue 1 + Bundesliga.
* **Test 1:** StatsBomb EPL 2015/16 (same provider).
* **Test 2:** Wyscout 2017/18, all 5 leagues (different provider and season; xG from our location model).
* **Model:** Poisson GLM (interpretable) and LightGBM.
* **Baseline:** regain location only (x, centrality).

**H3a-1:** the opponent-commitment features (`opp_prev_passes`, `opp_prev_max_x`) improve held-out Poisson deviance over the location-only baseline on BOTH tests, with match-bootstrap 95% CIs excluding 0.

**H3a-2 (direction):** holding regain location fixed, counter value *decreases* with how deep into our half the opponent had advanced (`opp_prev_max_x`, in our frame: low x = they were near our goal). Rationale: the regain is then further from their goal and their players are already back. Alternative: it *increases*, because more of their players are committed forward. We register the first direction; if the data show the second, H3a-2 fails.

**H3a-3 (concentration):** in the test sets, the top 20% of regains by predicted value carry ≥ 40% of regain-possession xG.

## R3-B: Opponent weak-spot matchups

For each team-season, compute a **concession profile**: xG conceded per match in each channel (build-up, counter, high regain, corner, free kick, throw-in), adjusted for opponent strength.

**H3b-1:** concession profiles are persistent beyond overall defensive quality. Split-half reliability of channel *shares* of xG conceded is > 0.3 for at least 2 channels, in both datasets.

**H3b-2:** a matchup model predicts held-out team-match channel xG better than an additive main-effects model (deviance, pooled across channels, using only *prior* matches to build profiles). The matchup model uses the attacker's channel rate × the defender's channel concession rate. Test seasons: SB 2015/16 EPL and Wyscout 2017/18.

If H3b-2 holds, a weaker team's game plan should target each opponent's channel weakness, and we quantify the points value of doing so.
