# Agent secondball: win the second ball

**Question.** For long balls (≥35 yds, open play or goal kick), does the local balance (our minus their visible outfielders within 15 yds of the landing point, from the 360 frame at the pass) raise retention and net xG? Should high balls be aimed at an overload?

**Data.** StatsBomb 360. 33,460 long balls have frames. 9,951 land inside the visible area: discovery (men's international) 4,492, men's club 3,548, women's 1,911. Clearances are excluded because they have no end location.

| Test | Disc | Club | Women | Pooled conf |
|---|---|---|---|---|
| H1 keep@5s per +1 | +.058 (.007) | +.070 (.009) | +.023 (.011) | +.051 (.007) |
| H2 netxG20 per +1 | +.0029 (.0011) | +.0025 (.0014) | +.0039 (.0011) | +.0033 (.0009) |
| H3 High, sup−outn | +.016 (.005) | +.006 (.005) | +.009 (.005) | +.007 (.004) |

All three pass BH in discovery (q ≤ .009), keep the same sign in both confirmation populations, and pass Holm (H3 one-sided p = .02). **Verdict: CONFIRMED.**

**Robustness.**
- **Holds:** a 45 s horizon and a crowding control.
- **Weaker:**
  - H3 within zone × crowding cells shrinks in confirmation (+.004 club, +.008 women's, n.s.).
  - H2 at a 10-yd radius is about 0.

**Usage.** Long balls land where we are outnumbered 13% (discovery), 12% (club) and 19% (women's) of the time. For high balls the rate is 24%, 20% and 30%.

**Key caveat.** Positions are frozen at the pass, not at landing. Only about 30% of balls land on camera. Superiority partly means the pass was "on", so this is selection, not causation. The xG gains are small.

**Coaching.** Aim long balls, especially high ones, where you already outnumber them. About a quarter of high balls go where you are outnumbered.
