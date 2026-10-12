# Project Underdog — Round 7 pre-registration: does football chase too late? ("pull the goalie")

*Written 2026-10-12 before running. Motivation: hockey analytics showed coaches pull the goalie far too late (Asness & Brown 2018). Football's analogue is the moment a trailing team escalates to all-out attack. Round 1 estimated a late × trailing effect inside a GLM, but never measured the escalation profile by minute and never optimised its timing.*

## Measurement (identical in StatsBomb 2015/16, Wyscout 2017/18, and the unseen StatsBomb WSL/NWSL/ISL)

For team-segments in which the team trails by exactly 1 with no red cards, compute multipliers by 15-minute bucket (0–30, 30–45, 45–60, 60–75, 75–end): actual xG rate ÷ baseline rate.
* The baseline is the same team's rate at gd = 0 at the same minute, adjusted for strength. For StatsBomb 2015/16 the strength comes from market λ. For the other populations it is leave-one-match-out xG for and against.
* We compute m_for (own xG) and m_against (xG conceded).

**H7-1 (late escalation).** m_for in 75–end exceeds m_for in 45–60, with a match-bootstrap 95% CI on the log-ratio above 0, in each of the three populations.

## Decision model

The DP over (minute, goal difference) for a trailing underdog uses two modes:
* *normal* = the observed trailing multipliers at that minute;
* *all-in* = the observed 75–end multipliers (both m_for and m_against), available from any minute.

All-in is irreversible once chosen (one-way switch, like pulling the goalie). We compute:
* the optimal switch minute when trailing by 1;
* the expected points gained relative to switching at the *observed* escalation minute.

The observed escalation minute is the first bucket where m_for exceeds the midpoint between its 45–60 and 75–end values.

**H7-2 (teams chase too late).** For underdogs (pre-match win probability 0.15–0.35), the optimal switch is ≥ 10 minutes earlier than the observed escalation, and the gain is ≥ 0.02 points per trailing-by-1 situation. Both must hold in each population, *and* when multipliers estimated in one population are evaluated in another (cross-population fit).

## Caveat registered in advance

The 75–end multipliers include the leading team's behaviour (it may sit back late). An early all-in may not reproduce them. We will bound this by recomputing with m_for shrunk toward 1 by 25% and 50%, and report the switch minute under each.
