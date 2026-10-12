"""Round 10: what would it take to *discover* a tactic? Power analysis from observed outcome variances.
Units and outcomes:
  corner          xG within 20 s (SB 2015/16)
  attacking FK    xG within 20 s
  attacking throw xG within 20 s
  regain          xG within 20 s (open-play regain)
  match           goal-difference residual vs market (EPL 2005-2025)   -> match-level tactic
Effect sizes: relative uplift r in the unit's mean xG (or absolute GD shift for match-level).
N per arm for two-arm randomised comparison, alpha = 0.05 two-sided, power 0.8.
Within-match randomisation (alternate routines within the same match) removes match-level noise; we estimate the
variance reduction from the within-match share of variance."""
import os, json, numpy as np, pandas as pd
from scipy.stats import norm
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
z = norm.ppf(0.975) + norm.ppf(0.8)
A = pd.read_parquet(os.path.join(PROC, "r8_deadballs.parquet")); A = A[A["pop"].isin(["discovery", "confirm_SB_EPL"])]
R = pd.read_parquet(os.path.join(PROC, "r3_regains_sb.parquet"))
units = {
    "corner": A[A.kind == "corner"][["match_id", "xg20"]],
    "attacking free kick": A[(A.kind == "fk") & A.fk_cell.notna()][["match_id", "xg20"]],
    "attacking throw-in": A[A.throw_opt.notna()][["match_id", "xg20"]],
    "open-play regain": R[["match_id", "xg20"]],
}
per_match_team = {"corner": 5.0, "attacking free kick": 3.0, "attacking throw-in": 9.0, "open-play regain": 72.0}
out = {}
for name, d in units.items():
    mu, sd = d.xg20.mean(), d.xg20.std()
    within = d.groupby("match_id").xg20.transform(lambda s: s - s.mean()).var() / d.xg20.var()
    row = dict(mean=float(mu), sd=float(sd), within_match_var_share=float(within), per_team_match=per_match_team[name])
    for r in (1.1, 1.2, 1.5):
        n = 2 * (z * sd / (mu * (r - 1))) ** 2   # total units, both arms
        row[f"units_needed_r{r}"] = float(n)
        row[f"club_seasons_r{r}"] = float(n / (per_match_team[name] * 38))
        row[f"league_seasons_r{r}"] = float(n / (per_match_team[name] * 38 * 20))
    out[name] = row
# match-level tactic: residual sd
U = pd.read_parquet(os.path.join(PROC, "team_matches_fd.parquet"))
U = U[U.season >= 2005]
gd_sd = float(((U.pts - U.xpts)).std())
row = dict(sd_points_resid=gd_sd)
for eff in (0.05, 0.1, 0.2):
    n = 2 * (z * gd_sd / eff) ** 2
    row[f"matches_needed_{eff}pts"] = float(n); row[f"club_seasons_{eff}pts"] = float(n / 38)
out["match-level tactic (points vs market)"] = row
for k, v in out.items():
    print(k, {a: round(b, 3) for a, b in v.items()})
json.dump(out, open(os.path.join(ROOT, "results", "r10_power.json"), "w"), indent=1)
