# Agent "combos" pre-registration: combination plays from a decision point

*Written 2026-10-16 before any outcome is computed. Data: StatsBomb 360 (477 matches). Discovery = men's international (`DISC`); confirmation = men's club and women's (`WOMEN`), both required.*

## Decision point P
A completed `Ball Receipt*` (no outcome) by player B of the possession team, at x ≥ 40 (middle or attacking third), periods 1–2, where the pass received is open play (not a corner/free kick/throw-in/goal kick/kick-off) and a 360 freeze frame exists for the receipt. A = the passer of that pass.

Situation cell at P (from the receipt freeze frame; same bins as Round 15):
zone (x bands 40–60 / 60–80 / 80–96 / 96–108 / 108+ × left/central/right) × opponents goal-side of B (≤3 / 4–6 / 7+) × nearest-opponent distance of B (<3 / 3–7 / >7 yds) × B's outfield teammates ahead (0–1 / 2–3 / 4+).

## Continuation types (defined only from events after P and BEFORE any shot; first matching rule wins)
Let B's "on-ball phase" = B's consecutive Carry/Dribble events after the receipt, ending in B's release (Pass or Shot) or a loss (Dispossessed, Miscontrol, opponent event, etc.). Action 1 = B's release; action 2 = the next on-ball action (Pass/Carry/Dribble/Shot) of the receiver of action 1. "Forward" uses x (attacking toward x = 120).
1. **shot**: B's first release is a shot. Reported for usage only; never tested (its outcome *is* the shot).
2. **solo_carry**: B's on-ball phase contains a Dribble, or B's carries gain ≥ 10 yds in x.
3. **one_two**: action 1 is a completed pass from B to A, released within 4 s of P, and A receives it ≥ 5 yds further forward (x) than where A played the pass to B.
4. **third_man**: action 1 is a completed pass, released within 4 s of P, to C ≠ A, and C receives it ≥ 10 yds further forward than A's pass origin.
5. **layoff_fwd**: action 1 is a completed backward pass (end x < start x − 2, length ≤ 15 yds) to D, and D's next on-ball action, starting within 4 s of receiving, is a pass with end x − start x ≥ 15 (attempted; completion not required).
6. **switch_attack**: action 1 is a completed pass that changes lateral lane (lanes y < 26.7 / 26.7–53.3 / > 53.3), with |Δy| ≥ 15 and |Δx| < 10 (or flagged `switch`), and the receiver's next on-ball action, starting within 4 s of receipt, is forward: a pass with Δx ≥ 10, a carry with Δx ≥ 10, or a dribble.
7. **other**: everything else (simple passes, failed passes, losses, fouls, etc.).

Leakage control: categories use only events strictly before the first shot after P. Action 2's success is never used. Categories 3–6 require action 1 to be completed (a necessary condition for a combination to exist); because this conditions on a successful first link, a **completion-matched** analysis is also run (see robustness).

## Outcome
Net xG (own − opponent, non-penalty StatsBomb xG) over 20 s from P (`r15_360.windows`).

## Hypotheses (two-sided in discovery; coaching direction = combination > reference)
- **Cell level (primary, as requested):** within each cell and population, reference = the cell's most common continuation type (excluding `shot` from candidate references and from the tested set). Each other type with n ≥ 30 and reference n ≥ 30 in discovery is one rule: diff = mean(type) − mean(reference), SE = sqrt(v1/n1 + v0/n0), two-sided normal p, BH 5% across all rules. Survivors are tested in men's club and women's (n ≥ 10 each side, same reference type): confirmed if the same sign in both and the pooled (inverse-variance) one-sided p < 0.05 after Holm across the survivors.
- **Type level (secondary):** for each of the five types (solo_carry, one_two, third_man, layoff_fwd, switch_attack), the inverse-variance-weighted mean of within-cell differences vs. the cell reference (cells where the type ≠ reference, n ≥ 5 per side). BH 5% over the five types in discovery; confirmation as above (same sign in both, pooled one-sided p < 0.05, Holm).

## Robustness for anything confirmed
(a) 45 s horizon; (b) drop sequences where the team shoots within 3 s of P; (c) completion-matched: restrict to P where B's first release is a completed pass (all non-solo categories then share the success condition; reference recomputed). The sign must hold in each.

## Also reported
Usage rates of each type per population, and per team-match.

## Caveats registered in advance
Choice is not randomised; the cell controls for observed context only. SEs treat sequences as independent (as in Rounds 15–21).

## Clarification (added while coding, before any outcome was computed)
StatsBomb logs almost every receipt as receipt → (short) Carry → Pass. "The receiver's next on-ball action" in types 5–6 is therefore read as the receiver's on-ball phase, defined exactly as for B (consecutive Carry/Dribble, then release): layoff_fwd requires D's release to be a pass with Δx ≥ 15 released within 4 s of D's receipt; switch_attack requires, in the receiver's phase, a dribble, or carries gaining ≥ 10 yds, or a release pass with Δx ≥ 10 released within 4 s of receipt. Event types ignored while walking a phase: Pressure, Ball Receipt*, Dribbled Past, Duel events that do not change possession_team. Any other event (opponent action, foul, miscontrol, dispossession, change of possession) ends the phase without a release.
