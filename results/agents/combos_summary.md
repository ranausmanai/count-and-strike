# Combination plays (agent "combos")

**Question:** From a pass received at x ≥ 40, do one-twos, third-man passes, lay-off-forward, switch-attack or solo carries beat the cell's most common continuation? Outcome: net xG over 20 s, compared within 360 cells.

**Data:** StatsBomb 360, 417 matches. Decision points: discovery 90,718; club 82,570; women 50,996. Uses per team-match (disc/club/women): third-man 55/66/42, solo 25/28/20, switch 12/13/8, one-two 9/12/6, lay-off 4/4/3. The reference is "other" (59%).

**Results:** 259 cell rules, 8 pass BH, 5 confirmed (Holm), all third-man > other.

| Test | Disc | Club | Women |
|---|---|---|---|
| Third-man, box wide (2 cells) | +0.10/+0.12 (±0.03) | +0.10/+0.12 (±0.04) | +0.14/+0.14 (±0.03) |
| Third-man, 3 cells x 80–108 | +0.02 to +0.04 | +0.02 to +0.07 | +0.02 to +0.08 |
| Third-man type-level, IV (pre-reg) | +0.0016±0.0002 | +0.0030±0.0003 | +0.0032±0.0003 |
| Third-man, n-weighted, cluster bootstrap | +0.010±0.001 | +0.012±0.001 | +0.014±0.001 |
| One-two/lay-off/switch, IV | −0.0007/−0.0016/−0.0007 | same sign | same sign |
| Same, n-weighted | +0.009/+0.005/+0.003 | + | + |

Robustness for third-man: the sign holds at 45 s, with shots within 3 s excluded, and when restricted to completed first passes.

**Verdict: MIXED.** Third-man is CONFIRMED and robust. The pre-registered "negatives" for one-two, lay-off and switch reverse under n-weighting, so they come from the weighting and are not robust. Solo carry is null.

**Caveat:** As defined, "third-man" is close to "B quickly completes a pass to someone ≥10 yds beyond the original passer". Completion and forward progress are part of the label, and the choice is not randomised.

**Coaching:** In the final third, a quick release to a teammate already beyond the passer beats holding or recycling, most of all around the box.
