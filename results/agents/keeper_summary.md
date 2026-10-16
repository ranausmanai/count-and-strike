# Agent keeper: quick goalkeeper release after open-play possession

**Question.** After an open-play keeper gain (claim/save/sweep), does a quick release (<5 s) beat a slow one (>10 s) on own/net xG over 30 s (H1)? Is it bigger with more opponents in our half (H2)? Quick short (<35 yd) vs long (H3)?

**Data.** Keeper open-play distributions (no set pieces/stoppages): SB 2015/16 n=9,737 (discovery); SB women/ISL n=3,579; SB 360 comps n=2,695 (2,314 with frames for H2). Within-cell (release type × game state × gain type) differences, match-cluster bootstrap SE.

**How often quick?** <5 s: 36% (SB15/16), 25% (women/ISL), 29% (360). Medians 8–10 s. Throws 34–55% quick; kicks 10–18%.

| Test (xG per release) | SB 15/16 | Women/ISL | 360 comps | Pooled conf. p (Holm) |
|---|---|---|---|---|
| H1a own30, quick−slow | +0.0013 ± 0.0010 | +0.0025 ± 0.0020 | +0.0032 ± 0.0026 | 0.042 (0.13) |
| H1b net30, quick−slow | +0.0014 ± 0.0012 | +0.0031 ± 0.0033 | +0.0048 ± 0.0038 | 0.063 (0.13) |
| H3 net30, short−long (quick) | +0.0006 ± 0.0017 | −0.0007 ± 0.0043 | +0.0017 ± 0.0075 | 0.51 |
| H2 (360) high−low opp in half | DISC +0.003 ± 0.010 | WOMEN −0.019 ± 0.025 | CLUB +0.040 ± 0.033 | 0.45 |

Robustness: same sign at 60 s and with finer cells.

**Verdict: NOT CONFIRMED.** H1 is positive everywhere but discovery p≈0.11 and pooled p fails Holm. H2, H3 null.

**Key caveat.** Speed is chosen after reading the play (free team-mate, no pressure); quick releases show fewer visible players (11.7 vs 13.8), so even the lean may be selection. Size ≈ +0.002 xG/release, ≤0.02 xG/match.

**Coaching sentence.** None; at most a weak lean toward "release quickly when it's on".
