# Project Underdog — Pre-registration

*Written 2026-10-11, before any outcome data were inspected (the event download was still running; no team-level results, xG or odds had been looked at).
Committed to git before analysis; the commit timestamp is the registration time.*

## Research question

> Is there a systematic way for a weaker team to change the *structure* of a football match so that player quality matters less, and how much is it worth in league points and title probability?

We are not trying to predict matches. We want a **tactical policy**: a mapping from match state (minute, score, opponent strength) to a controllable style choice. We want to measure how much it is worth to a weak team, and we want to know where the claim breaks.

## Framing: the underdog's frontier

Model each team's chance production as two Poisson-like rates: λ_for (own xG per minute) and λ_against (opponent xG per minute). Any tactic moves the match along two axes:

* **Intensity** T = λ_for + λ_against: how *open* the game is
* **Share** S = λ_for / T: who dominates it

For fixed S < 0.5, lowering T raises the weaker side's expected points, because fewer goals means more draws and more noise relative to quality. A tactic helps an underdog when the intensity it removes is worth more than the share it gives up. The core empirical object is the **frontier**: the set of (λ_for, λ_against) pairs a team of given quality can actually reach by changing style, estimated from event data.

Policy optimisation is exact dynamic programming over (minute, goal difference). It is done **only over style regions observed in the data** (a support constraint, the offline-RL analogue of "don't trust the simulator where it has never seen football").

## Data

* StatsBomb Open Data: full 2015/16 seasons of the EPL, La Liga, Serie A and Ligue 1 (+34 Bundesliga matches). Events with StatsBomb xG.
* football-data.co.uk: results, shots and closing odds for the big-five leagues, 2000/01 to 2025/26.

**Hold-out:** the EPL 2015/16 events are *not* used to fit the frontier model. They are used only for the confirmatory tests P1 and P3.

## Primary hypotheses (direction stated in advance)

**P1 (frontier is real and transfers).** A model of segment-level xG rates that includes controllable style variables (defensive block height, pressing intensity/PPDA, directness, possession share) predicts held-out EPL xG-for/xG-against better than a model with only strength, home advantage, game state and minute. Metric: out-of-sample Poisson deviance. Success = improvement whose match-bootstrap 95% CI excludes 0.

**P2 (there is a tactical gain for underdogs, in the simulator).** For a heavy underdog (pre-match win probability ≈ 0.15), the DP-optimal state-dependent policy restricted to observed style support raises expected points per match versus the team's *typical* (empirically average-for-its-strength) style by **≥ 0.05 points/match**. The gain must survive refitting the frontier on bootstrap resamples (≥ 80% of resamples show gain > 0). We predict the optimal policy is *deeper / lower-intensity* at level scores, and *switches to higher intensity when trailing late*.

**P3 (habitual style vs. the market, events season).** In held-out EPL 2015/16, an underdog's habitual style (rolling mean over its *previous* matches) that is closer to the P2 prescription is associated with a positive points residual versus closing-market expected points. Direction: positive. We expect low power (one season) and report the estimate either way.

**P4 (tempo vs. the market, 20+ seasons).** Across big-five leagues 2005/06–2025/26, an underdog's habitual match tempo (rolling mean of *total* shots in its previous 10 league matches) is **negatively** associated with its points residual versus the closing market. A null here means the market already prices "shrinking the game"; that would be a meaningful result in its own right.

**P5 (points ≠ titles).** For a Brentford-strength team in a simulated 38-match season against a 2025/26-strength Premier League, the policy that maximises *title probability* differs from the one that maximises expected points and is more variance-seeking. Prediction: higher-intensity choices in matches against peers. We report both, plus the absolute title probability, which we expect to remain small.

## Secondary / exploratory (labelled as such in any write-up)

* Leicester 2015/16 (a real underdog champion in the data): where does their style sit on the frontier?
* Counter-attack vulnerability: xG per regain as a function of regain location and the opponent's possession share.
* Late-game decisions: drawing at 70', should the underdog open up or protect the point? (Read off the DP value function.)

## Analysis rules

* All thresholds, splits and seeds are fixed in code before the held-out evaluation is run. Any deviation is logged in `DEVIATIONS.md`.
* Nulls and failures are reported with the same prominence as successes.
* Causal language is avoided for observational associations. The frontier is a *conditional* model; style is chosen partly in response to the game. We control for game state and minute and say plainly that residual confounding remains.
