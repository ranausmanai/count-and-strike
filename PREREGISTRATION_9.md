# Project Underdog — Round 9 pre-registration: referees as a natural experiment

*Written 2026-10-12 before running. Teams do not choose the referee. If referee strictness changes the value of disruptive, foul-heavy underdog football, and the betting market does not price it, that is both a tactical insight (adapt your aggression to the official) and a market inefficiency.*

**Data.** EPL matches with closing odds and a named referee, 2005/06–2025/26 (football-data.co.uk).

**Strictness.** For each match, the referee's mean cards per match (yellow + 2 × red), and separately fouls per match, over **all of his previous EPL matches** (min 20), z-scored within season. Primary = cards.

**Outcomes.** GD residual of the underdog vs the market (as in round 2). Secondary: total-goals residual = actual total − (λ_h + λ_a).

**Hypotheses.**
* **H9-1:** strictness → underdog GD residual is **negative** (strict officials remove the underdog's disruptive tools).
* **H9-2:** strictness × underdog habitual fouls committed (rolling 10, demeaned) → **negative**.
* **H9-3 (two-sided):** strictness → total-goals residual ≠ 0.

Controls: market gap, home. SEs clustered by referee-season.

**Split.** Discovery 2005/06–2016/17. Confirmation 2017/18–2025/26, run only after the discovery estimates are committed. **Confirmed** = same sign and one-sided p < 0.05 in confirmation (two-sided for H9-3). Both periods are reported regardless.
