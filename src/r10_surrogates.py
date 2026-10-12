"""Round 10b: surrogate endpoints for tactical experiments. For regains and corners (SB 2015/16), candidate surrogates:
  regain:  final-third entry ≤10 s; box entry ≤20 s; shot ≤20 s
  corner:  attacking team has the next touch (first contact) ; shot ≤20 s
Validity: xG ratio (success vs failure) and how much of the between-TEAM variation in xG per unit the surrogate explains
(team-level R², split-half: surrogate rate in odd matches vs xG per unit in even matches).
Power: units for a 20% uplift in the surrogate rate."""
import os, json, numpy as np, pandas as pd
from scipy.stats import norm
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
z = norm.ppf(0.975) + norm.ppf(0.8)
ev = pd.read_parquet(os.path.join(PROC, "events.parquet"), columns=["match_id", "period", "t", "type", "team", "poss", "x", "y"])
ev = ev[ev.period <= 2]
p = pd.read_parquet(os.path.join(PROC, "possessions.parquet"))
R = pd.read_parquet(os.path.join(PROC, "r3_regains_sb.parquet"))
# rebuild regain keys (match, team, t0) from possessions to compute surrogates
q = p.sort_values(["match_id", "poss"]).copy(); prev = q.groupby("match_id").shift(1)
reg = q[(q.pattern == "Regular Play") & prev.poss_team.notna() & (prev.poss_team != q.poss_team) & (prev.period == q.period)][["match_id", "poss", "poss_team", "period", "t0", "start_x"]]
own = ev.merge(reg[["match_id", "poss", "poss_team", "t0"]], left_on=["match_id", "poss", "team"], right_on=["match_id", "poss", "poss_team"])
own["dt"] = (own.t - own.t0) * 60
g = own.groupby(["match_id", "poss"])
s1 = g.apply(lambda d: bool(((d.x >= 80) & (d.dt <= 10)).any())).rename("ft10")
s2 = g.apply(lambda d: bool(((d.x >= 102) & (d.y.between(18, 62)) & (d.dt <= 20)).any())).rename("box20")
s3 = g.apply(lambda d: bool(((d.type == "Shot") & (d.dt <= 20)).any())).rename("shot20")
reg = reg.set_index(["match_id", "poss"]).join(s1).join(s2).join(s3).reset_index()
reg = reg[reg.start_x < 80]  # regains not already in the final third
# xg20 per regain from possessions-level shots within 20s
sh = ev[ev.type == "Shot"].merge(reg[["match_id", "poss", "t0"]], on=["match_id", "poss"])
xg = pd.read_parquet(os.path.join(PROC, "events.parquet"), columns=["match_id", "poss", "t", "type", "xg", "sub"])
xs = xg[(xg.type == "Shot") & (xg["sub"] != "Penalty")].merge(reg[["match_id", "poss", "t0"]], on=["match_id", "poss"])
xs = xs[(xs.t - xs.t0) * 60 <= 20].groupby(["match_id", "poss"]).xg.sum().rename("xg20")
reg = reg.set_index(["match_id", "poss"]).join(xs).reset_index(); reg["xg20"] = reg.xg20.fillna(0)
out = {}
def assess(d, cols, unit, team_col, per_tm):
    res = {}
    mu, sd = d.xg20.mean(), d.xg20.std()
    res["xg_units_r1.2"] = float(2 * (z * sd / (mu * 0.2)) ** 2)
    for c in cols:
        pr = d[c].mean()
        n = 2 * z ** 2 * (1 - pr) / (pr * 0.2 ** 2)
        ratio = d[d[c]].xg20.mean() / max(d[~d[c]].xg20.mean(), 1e-9)
        share_xg = d[d[c]].xg20.sum() / d.xg20.sum()
        # team-level predictive validity (split-half: surrogate rate odd matches -> xg/unit even matches)
        rows = []
        for tm, gg in d.groupby(team_col):
            ms = sorted(gg.match_id.unique())
            if len(ms) < 20: continue
            a, b = gg[gg.match_id.isin(ms[0::2])], gg[gg.match_id.isin(ms[1::2])]
            rows.append((a[c].mean(), b.xg20.mean(), a.xg20.mean()))
        T = np.array(rows)
        res[c] = dict(rate=float(pr), units_needed_r1_2=float(n), club_seasons_r1_2=float(n / (per_tm * 38)),
                      xg_ratio_success_vs_fail=float(ratio), share_of_xg_in_success=float(share_xg),
                      r_surrogate_odd_vs_xg_even=float(np.corrcoef(T[:, 0], T[:, 1])[0, 1]),
                      r_xg_odd_vs_xg_even=float(np.corrcoef(T[:, 2], T[:, 1])[0, 1]))
    return res
out["regain"] = assess(reg, ["ft10", "box20", "shot20"], "regain", "poss_team", 72 * 0.8)
# corners: first contact & shot
A = pd.read_parquet(os.path.join(PROC, "r8_deadballs.parquet")); C = A[(A.kind == "corner") & A.comp.str.startswith("SB_")].copy()
evs = ev.sort_values(["match_id", "period", "t"])
nxt = []
E = {k: g for k, g in evs[evs.type.isin(["Pass", "Ball Receipt*", "Shot", "Clearance", "Duel", "Goal Keeper", "Interception", "Block", "Ball Recovery", "Miscontrol", "Carry"])].groupby(["match_id", "period"])}
for r in C.itertuples():
    g = E.get((r.match_id, r.period))
    if g is None: nxt.append(np.nan); continue
    w = g[(g.t * 60 > r.t + 0.3) & (g.t * 60 <= r.t + 6)]
    w = w[w.type != "Ball Receipt*"] if len(w[w.type != "Ball Receipt*"]) else w
    nxt.append(float(w.team.iloc[0] == r.team) if len(w) else np.nan)
C["first_contact"] = nxt; C = C.dropna(subset=["first_contact"]); C["first_contact"] = C.first_contact.astype(bool)
C["shot20"] = C.xg20 > 0
C["poss_team"] = C.team
out["corner"] = assess(C, ["first_contact", "shot20"], "corner", "poss_team", 5)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(ROOT, "results", "r10_surrogates.json"), "w"), indent=1)
