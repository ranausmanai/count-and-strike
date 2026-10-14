# Project Underdog — Round 13 pre-registration: Tactic Miner with 360 "what the player saw"

*Written 2026-10-14 while the 360 data were downloading; no 360 outcome analysis has been run.*

**Data.** StatsBomb open data with 360 frames (visible player positions at every event).
* **Discovery:** men's international tournaments (World Cup 2022, Euro 2020, Euro 2024, AFCON 2023).
* **Confirmation set 1:** men's club (Bundesliga 2023/24, La Liga 2020/21, Ligue 1 2021/22 and 2022/23, MLS 2023).
* **Confirmation set 2:** women's (Women's Euro 2022 and 2025, Women's World Cup 2023).

**Space features at each open-play action** (from the 360 frame, actor's attacking direction, GK excluded):
* `opp_goalside`: visible opponents ahead of the ball;
* `nearest_opp`: distance to the nearest opponent;
* `mates_ahead`: visible teammates ahead of the ball.

**Cells.** Zone (6 x-bands × 3 y-bands) × opp_goalside {≤3, 4–6, ≥7} × nearest_opp {<3, 3–7, >7 yds} × mates_ahead {0–1, 2–3, ≥4} = 486 cells. Actions and outcome are as in round 11 (net xG, 20 s).

* **H13-1 (descriptive).** The cell-weighted advantage of forward actions over recycling shrinks when space is conditioned on (zone-only cells vs zone×space cells). We report the ratio.
* **H13-2 (rules).** Mine action vs most-common action in each zone×space cell (≥30 each). Shortlist = BH 5% in discovery. Confirmed = same sign in both confirmation sets and pooled one-sided p < 0.05/k (Holm).
* **H13-3 (aggregate).** Sign-agreement rate of nominal discovery hits in each confirmation set, with bootstrap CI.
