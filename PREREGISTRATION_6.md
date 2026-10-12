# Project Underdog — Round 6 pre-registration: confirming the two replicated candidates on unseen leagues

*Written 2026-10-12 before running. Data: StatsBomb WSL 2018/19, 2019/20, 2020/21, NWSL 2023, ISL 2021/22 (577 matches, never used for these questions). No betting odds exist for these leagues, so team strength = leave-this-match-out xG for / against per match over the team's other matches that season.*

**H6-1 (set pieces are flat).** Pooled with competition FE, the elasticity of team-match set-piece xG to own attacking strength (log LOO xG-for per match) is lower than that of open-play xG. Success: the bootstrap 95% CI (resampling teams within competition) of the difference (open − set piece) excludes 0. Channel definitions are identical to src/possessions.py.

**H6-2 (underdogs keep more of the set-piece pie).** In mismatches (the weaker side's LOO xG difference per match is ≥ 0.5 lower than the opponent's), the underdog's share of set-piece xG exceeds its share of open-play xG. Success: match-bootstrap CI of the difference excludes 0.

**H6-3 (break after they commit).** For open-play regains, the xG of the regaining possession within 20 s *increases* the deeper the opponent had advanced before losing the ball. Operationally, the marginal effect of `opp_prev_max_x` (our frame; low = they reached near our goal) is *negative*, evaluated at regain x = 36 (the sample median from round 3). Controls: regain location (cubic x, centrality, interaction), game state, minute. Success: one-sided p < 0.05, SEs clustered by match. The direction is fixed from round 3, where it appeared in 3 of 3 samples.
