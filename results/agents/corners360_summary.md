# corners360: corner setup from 360 frames

**Question.** At the kick, do more of our players in the box (H1), a six-yard overload (H2), or fewer players held back (H3, counter cost) change corner value? Model: team x competition-season fixed effects, controls for opponents in the box and corner side, SE clustered by match.

**Data.** 4,373 corners. 2,415 have a frame, and 1,345 pass coverage (at least 85% of the box and the whole six-yard box visible): men's international 483, men's club 512, women's 350. Typical setup: 5.5 of ours in the box, 1.4 in the six-yard box, about 9 defenders.

| Test (est ± SE) | Men's intl (discovery) | Men's club | Women's | BH q |
|---|---|---|---|---|
| H1 own xG 20 s per extra box player | −0.0036±0.0044 | +0.0036±0.0047 | +0.0066±0.0065 | 0.79 |
| H2 ≥3 vs ≤1 in six-yard box | +0.0012±0.0058 | +0.0019±0.0159 | +0.0016±0.0182 | 0.62 |
| H3 counter xG 30 s per extra box player | +0.0054±0.0029 | −0.0130±0.0128 | −0.0001±0.0001 | 0.089 |

Robustness checks (95% coverage, 45/60 s horizon, no short corners) are also null. H3 in discovery has one-sided p≈0.03 but fails BH correction and does not reproduce in either confirmation population. The sweet-spot analysis by box count is noise: the best count differs by population and all differences are within about 1 SE.

**Verdict: NOT CONFIRMED.**

**Key caveat.** Low power: only effects of about 0.01 xG per player or larger can be detected. The coverage filter selects certain camera angles. Counter-attacks after corners are very rare, so H3 is poorly estimated. Box counts cluster at 5–6.

**Coaching takeaway.** None. There is no evidence that bodies in the box or a six-yard overload changes corner value.
