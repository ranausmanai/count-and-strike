# Agent avail: is the overload pass actually ON?

**Question.** Does "play into the overload" hold when the overload pass is genuinely available? Available means a teammate in an ov>0 lane who is visible, 5–35 yds from the ball, has no opponent within 3 yds, and has no opponent within 2 yds of the passing line.

**Data.** StatsBomb 360, open-play passes with x≥40 where both lane types exist: 113,957 (this rebuild matches r18a exactly). The overload pass is on in 31–34% of them. With target ov≠0, the on samples are: discovery 13,852, club 12,653, women's 7,232. Of these, 6,277 / 6,420 / 1,564 are full-width frames. A team gets 46 / 53 / 33 overload-on situations per match (median 40 / 47 / 27), or 21 / 27 / 7 in full-width frames.

| Test (overload − crowded) | Disc | Club | Women | Pooled conf (p) |
|---|---|---|---|---|
| H2 cell net xG20, on | +.0020 (.0009) | +.0000 (.0013) | +.0011 (.0013) | +.0006 (.27) |
| H3 same, full-width | +.0021 (.0012) | −.0005 (.0017) | −.0005 (.0023) | −.0005 (.65) |
| H4 AIPW net xG20, on | +.0014 (.0010) | −.0001 (.0012) | +.0011 (.0017) | +.0003 (.37) |
| H4 placebo, pre-20 s | +.0002 | −.0006 | −.0006 | clean |
| Completion, on (secondary) | +.028 (.004) | +.033 (.004) | +.042 (.008) | +.035 (<.001) |
| Completion, on + full-width | +.017 (.006) | +.018 (.006) | +.035 (.020) | +.019 (<.001) |

**Results.** No primary test passes BH in discovery (q = .07 / .12 / .15). The 45 s horizon and finer cells are also null in confirmation.

**H1 is false.** When the overload pass is on, only 44–46% of passes go into the crowded lane, so players mostly take it. When it is not on, 87–89% go into the crowded lane. So the 66% crowded-lane usage mostly reflects situations where the overload pass is not available.

**Verdict: NOT CONFIRMED.** Among genuinely available options, the overload pass does not raise net xG. It does raise completion by 2–4 points.

**Key caveat.** Availability is geometric only: it ignores body shape, pass difficulty and player movement. Observational, not causal.
