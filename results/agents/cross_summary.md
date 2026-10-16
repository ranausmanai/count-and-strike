# Agent "cross": when should a team cross?

**Question.** Is "only cross with enough bodies in the box" true? Wide final-third open-play decisions (x≥96, |y−40|>18): CROSS (pass into box, not cutback) vs alternatives, within 24 cells: our box players (0–1/2/3/4+) × theirs (≤4/5–6/7+) × passer pressed. Outcome: net xG 20 s; cell fixed effects, match-clustered SEs. Pre-registered in `PREREGISTRATION_AGENT_cross.md`.

**Data.** StatsBomb 360 (frames showing the penalty spot). Discovery (men's intl) 4,693 decisions/166 matches; club 4,519/126; women 3,323/125. Usual alternative (fixed in discovery) = carry/dribble.

| Hyp | Discovery | Club | Women | Verdict |
|---|---|---|---|---|
| H1 gap slope per box level | +.0055±.0031 (q=.16) | −.0018±.0037 | −.0007±.0036 | Not confirmed |
| H2 cross−carry, 3+ ours | **+.0187±.0047** (q<.001) | +.0154±.0053 | +.0071±.0049 | **Confirmed** (pooled +.0109±.0036, p₁=.001) |
| H3 cross−carry, 0–1 ours (pred. <0) | +.0017±.0059 | +.0182±.0076 | +.0114±.0072 | Not confirmed (reversed) |
| H4 low−high cross, 7+ theirs | −.0080±.0082 | −.0167±.0129 | −.0257±.0181 | Not confirmed |

H2 arms (cross/carry): 573/482, 437/389, 437/333.

**Robustness (H2).** 45 s horizon pooled +.0077±.0045 (p₁=.043); finer cells +.0089±.0041; all frames +.0104±.0035.

**Verdict: MIXED.** H2 confirmed, but not the lore's mechanism: no dose-response, and crossing also beats carrying with 0–1 in the box in both confirmation populations. Crossing beats carrying from wide areas generally.

**Key caveat.** Observational: players cross when they see a target (unmeasured runs, delivery quality); counts cover visible players only; effect ≈0.01 xG per decision.

**Coaching sentence.** From wide in the final third, crossing is worth ~0.01 xG more than carrying, with 3+ teammates in the box and, on this evidence, with fewer too.
