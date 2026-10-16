# Agent "runners": how many of our players should be ahead of the ball at a regain?

**Question.** The Count rule says each extra opponent goal-side cuts counter value. If we hold that count fixed, does having more of our own players ahead of the ball at the regain (`mates_ahead`) add attacking xG, cost us xG at the other end, or point to a best number of runners?

**Data.** StatsBomb 360 open-play regains with x < 80: discovery (men's international) 6,664 regains in 166 matches; men's club 5,623 in 126; women's 5,871 in 125. Quasi-Poisson / OLS models with match-clustered SEs. Controls: cubic x, centrality, opponents goal-side, visible opponents, game state and minute. Pre-registered in `PREREGISTRATION_AGENT_runners.md`.

| Test (per extra runner) | Discovery | Men's club | Women's | Pooled confirmation, one-sided p |
|---|---|---|---|---|
| H1 own xG 20 s (log β ± SE) | +0.083 ± 0.079 | −0.004 ± 0.079 | +0.059 ± 0.080 | 0.32 |
| H2 slope (≤4 back) − slope (≥7 back) | −0.23 ± 0.13 | −0.17 ± 0.17 | −0.10 ± 0.25 | 0.86 (wrong sign) |
| H3 conceded xG 30 s | −0.079 ± 0.100 | −0.104 ± 0.084 | −0.215 ± 0.110 | 0.99 (wrong sign) |
| Net xG 20 s (OLS) | +0.0005 ± 0.0004 | +0.0003 ± 0.0005 | +0.0003 ± 0.0003 | 0.12 |

**H4 sweet spot.** Bands were opponents back (≤4 / 5–6 / ≥7) × runner bins (0–2 / 3–4 / 5–6 / 7+). No contrast was both the same sign in all three populations and significant after Holm correction. The best candidates were "5–6 runners beat 3–4 when ≤4 back" (pooled p = 0.19) and "7+ beat 5–6 when 5–6 back" (pooled p = 0.08). Neither held at the 45 s horizon.

**Verdict: NOT CONFIRMED.** H1, H2 and H4 are null. H3 runs the opposite way: in all three populations, more runners goes with less xG conceded (confirmation pooled about −9 % per runner, two-sided p ≈ 0.03; finer controls give the same result). Discovery was not significant, so this is only a lead.

**Key caveat.** Runner count is not a choice made after the ball is won. It reflects how open the game already was, and it is tied closely to the opponents-back count and to frame visibility.
