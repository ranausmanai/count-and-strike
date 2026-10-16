# Agent "triggers": do classic pressing triggers make pressing pay?

**Question.** On opponent open-play receipts, does pressing within 2 s beat holding (our net xG, next 20 s) more when a classic trigger is present than when it is absent? Compared within zone × support distance × our players within 10 yd. Pre-registered in `PREREGISTRATION_AGENT_triggers.md`.

**Data.** StatsBomb 360 opponent receipts. Discovery (men's international): 60,063. Men's club: 56,632. Women's: 42,635. Press rate is 16–20%. Match-cluster bootstrap SEs.

**Results.** I = interaction, (press − hold | trigger) − (press − hold | no trigger), in net xG ×1000, ± SE.

| Trigger | Discovery I (BH q) | Club I | Women I | Pooled one-sided p |
|---|---|---|---|---|
| Backward pass | +3.2±1.4 (0.13) | +0.2±2.0 | +3.1±2.1 | 0.14 |
| Touchline (≤8 yd) | +0.6±1.1 (0.58) | −1.1±2.1 | −0.0±1.9 | 0.65 |
| Poor touch / incomplete | −9.5±5.4 (0.19) | −4.6±12.6 | −16.3±4.4 | 1.00 |
| Long ball in the air (>30 yd) | −6.0±4.1 (0.20) | +10.3±9.7 | −6.0±5.8 | 0.63 |
| Crowded central midfielder | +2.0±1.6 (0.26) | +1.2±2.3 | −1.8±2.0 | 0.62 |

None of the 10 discovery tests (interactions and press-vs-hold under each trigger) passed BH at 5%. Closest: backward pass, with a press effect of +2.4±1.3 (discovery) and +1.0 / +2.7 (club / women's). It points the right way everywhere, including at 45 s and in finer cells, but is not significant.

**Verdict: NOT CONFIRMED.** No trigger makes pressing measurably more valuable.

**Key caveat.** Pressing after a poor touch looks *worse* everywhere. This is likely selection: after a loose ball we only "press" if the opponent recovered it, and miscontrol leaks the outcome. Pressing is self-selected, so these are associations. Effects are at most about 0.003 xG per receipt.
