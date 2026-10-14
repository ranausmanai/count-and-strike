# Project Underdog — Round 11 pre-registration: the Tactic Miner (automated, large-scale decision search)

*Written 2026-10-14 before running. The user asked for a far larger automated search. The design keeps it honest: mine everything on discovery data, then confirm only on data the miner never saw.*

**Units.** Every open-play on-ball action (pass, carry, dribble, shot) by the team in possession, periods 1–2.

**Actions (8).**
* `shot`
* `cross` (pass from wide, |y−40| > 22 and x ≥ 96, ending in the box)
* `long_fwd` (pass ≥ 30 yds, ≥ 10 yds forward)
* `short_fwd` (pass < 30 yds, ≥ 5 yds forward)
* `recycle` (any other pass)
* `carry_fwd` (carry ≥ 5 yds forward)
* `carry_other`
* `dribble`

**Situation cells (2,916).**
* zone: x in 6 bands × y in 3 bands;
* under pressure: yes / no;
* game state: trailing / level / leading;
* seconds since our possession started: < 5 / 5–15 / > 15;
* period: 1st half / 2nd half < 75' / 75'+;
* team strength: underdog / even / favourite.

**Outcome.** Net xG = own non-penalty xG in the next 20 s − opponent xG in the next 20 s.

**Candidates.** In every cell with ≥ 30 uses of the action and ≥ 30 of the cell's most common action, compare action vs most-common action (difference in mean net xG, Welch SE). We expect ~10–20k candidate rules.

**Discovery:** StatsBomb 2015/16 La Liga, Serie A, Ligue 1, Bundesliga. **Shortlist:** BH-FDR 5%.

**Confirmation:** never seen by the miner. StatsBomb EPL 2015/16 and the unseen StatsBomb WSL/NWSL/ISL. A rule is **confirmed** if the pooled confirmation estimate (inverse-variance) has the discovery sign with one-sided p < 0.05 / k (Holm over the k shortlisted rules), and the same sign in both confirmation sets.

**Reporting.** Number searched, shortlisted and confirmed; effect sizes; how often teams currently choose the better option (under-use is the actionable part); and the season value of confirmed rules.

**Known limit (registered).** Players choose actions partly because of what they see (open space). Comparisons within a situation cell remove much, but not all, of this. Confirmed rules get a within-player robustness check before being called tactics.
