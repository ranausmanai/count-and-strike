# Project Underdog — Round 4 pre-registration: can defensive structure beat xG?

*Written 2026-10-12 before running the analyses. Motivation: Leicester 2015/16 gained 18 points above their xG. If part of that is repeatable structure (bodies between ball and goal, pressure on the shooter), it is a lever a cheap squad can coach.*

Freeze-frame features for every StatsBomb 2015/16 non-penalty shot:
* `cone_def`: defenders (excluding the GK) inside the triangle from the shooter to the two posts
* `press_3y`: defenders within 3 yards of the shooter
* `gk_off`: the keeper's distance from the shooter–goal-centre line
* `gk_depth`: the keeper's distance from the goal line

**H4-1 (structure beyond xG).** A logistic model of goal ~ logit(statsbomb_xg) + structure features, fit on non-EPL, improves held-out EPL log loss over goal ~ logit(statsbomb_xg) alone. Success: a match-bootstrap 95% CI on the log-loss gain that excludes 0. *Note: StatsBomb xG may already use freeze frames, so a null is expected and informative.*

**H4-2 (structure is a coachable team trait).** For defending teams, the mean `cone_def` and `press_3y` per shot conceded each have split-half reliability > 0.4. Their correlation with team market rating has |r| < 0.4 (i.e. not simply bought).

**H4-3 (defensive over-performance repeats).** Team goals conceded minus xG conceded per match has split-half reliability > 0.2 in BOTH StatsBomb 2015/16 (StatsBomb xG) and Wyscout 2017/18 (our location xG). If it fails, Leicester-style over-performance is treated as non-repeatable.

If H4-2 holds but H4-1 fails, structure works *through* xG (it lowers the quality of shots conceded). We would then quantify its points value via the season simulator.
