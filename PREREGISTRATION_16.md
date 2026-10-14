# Project Underdog — Round 16 pre-registration: the possession simulator ("football chess engine")

*Written 2026-10-14 before running.*

**Model.** A two-team Markov decision process over the 486 space-aware situation cells of round 15 (360 data).
* From each on-ball action we take: the immediate reward (the xG of a shot), and the next on-ball state within 30 s. That next state is either our own continuation or the opponent's state if the ball was lost (zero-sum: it counts as minus their value). It is terminal if there is no next action within 30 s.
* Transition probabilities and rewards are estimated per population (discovery: men's international; confirmation: men's club, women's).
* Values are solved jointly for our team (policy π) and the opponent (behaviour policy).

**Search.** Constrained policy iteration. In every cell we may switch to any action used ≥ 30 times there in the discovery data (support constraint: no unobserved moves). π* = the greedy policy on the discovery model.

**Primary test (H16-1).** π* is learned on discovery only. Its value gain over the behaviour policy, per possession-start, is evaluated in each confirmation model. Confirmed if it is positive in both, with a match-bootstrap 95% CI excluding 0 (200 resamples). We report naive (discovery) vs cross-fitted gains, which shows how much the simulator flatters itself.

**Plays (H16-2).** For each cell where π* changes the action, the advantage Q(s, a*) − Q(s, a_usual) is computed in each confirmation model. A **confirmed play** is one where this is positive in both confirmation models, with bootstrap CIs excluding 0. We separate the advantage into its immediate part (shot xG) and its downstream part (better next situations). Plays whose value is mostly downstream are new relative to rounds 13–15.

**Value.** Cross-fitted gain → xG per team-match → season.
