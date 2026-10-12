# Project Underdog — Round 5 pre-registration: confirming the "cone density" candidate

*Written 2026-10-12. Round 4 follow-ups (exploratory, src/r4_value.py) found that teams putting more defenders between the shooter and goal (per shot conceded) conceded fewer goals than the market expected: −1.06 goals/match per +1 defender, p = 0.026, StatsBomb 2015/16. The trait was persistent (0.75) and unrelated to rating (0.07). That result was post-hoc. This round tests it in ways that cannot have been tuned. None of the computations below have been run.*

## Definitions (fixed now)

* `cone_def`: defenders (not the GK) inside the triangle from the shooter to the posts (src/r4_structure.py).
* `cone_excess`: shot-level residual of cone_def after a Poisson GLM on shot geometry (distance, |angle|, distance², header flag), fitted within each competition-season. This removes the mechanical fact that long shots have bigger cones.
* Team trait = mean `cone_excess` over shots **conceded**.

## H5-1: prospective test (StatsBomb 2015/16 men, 4 full leagues + EPL)

The trait in each team's first 19 league matches predicts goals conceded minus market-expected goals conceded (λ_opp) per match in matches 20+. OLS with league FE and team rating (mean log λ_own/λ_opp over the first half). **Success:** coefficient < 0, one-sided p < 0.05.

## H5-2: unseen populations (WSL 2018/19, 2019/20, 2020/21; NWSL 2023; ISL 2021/22)

The first-half-of-season trait predicts (goals conceded − StatsBomb xG conceded) per match in the second half. Competition FE, controlling for the team's first-half xG difference per match (quality proxy). **Success:** coefficient < 0, one-sided p < 0.05.

## H5-3: shot level, unseen populations

goal ~ logit(StatsBomb xG) + cone_excess, pooled over the H5-2 competitions, SEs clustered by match. **Success:** cone_excess coefficient < 0, one-sided p < 0.05.

## Interpretation rule

If H5-1 and H5-2 hold, "cone density" is confirmed as a repeatable, wealth-independent defensive trait that beats xG (and, in 2015/16, the market). We then estimate its points value in the season simulator. If only H5-3 holds, the effect is real per shot but too small to matter at team level. If none hold, the round-4 association was noise or confounding (e.g. goalkeeper quality).
