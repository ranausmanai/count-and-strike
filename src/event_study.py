"""Event study: effect of attacking / defensive coaching decisions on xG rates (post-hoc frontier, see DEVIATIONS #2).

Decision = a team's substitutions/formation shifts within 2 minutes (no red card within ±3 min).
Exposure delta = change in mean(formation load, personnel load).
Windows: pre = up to 15 min before (stopping at the previous goal/half start); post = up to 15 min after
(stopping at the next goal, red card, or the end of the half). The game state is constant within each window.
Outcome: post-window xG for / against, Poisson GLM with offset = expected xG under a strength+state baseline.
Controls: pre-window actual/expected ratio (momentum), game state, minute, strength, opponent's own decision delta.
"""
import difflib, json, os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(__file__))
from frontier import load_segments, design, fit_glm, StyleScaler, STYLE, ROOT, PROC

W = 15.0


def build():
    ch = pd.read_parquet(os.path.join(PROC, "changes_raw.parquet"))
    ev = pd.read_parquet(os.path.join(PROC, "events.parquet"),
                         columns=["match_id", "period", "t", "type", "team", "xg", "outcome"])
    ev = ev[ev.period <= 2]
    ev["clock"] = ev.t + np.where(ev.period == 2, 45.0, 0.0)
    seg = load_segments(min_minutes=0.0)
    # canonical names in events table already; map change-team names to canonical per match
    names = seg.groupby("match_id").team.unique().to_dict()
    ch["team"] = [n if n in names.get(m, []) else difflib.get_close_matches(n, list(names.get(m, [n])), 1, 0)[0]
                  for m, n in zip(ch.match_id, ch.team)]
    ch = ch.sort_values(["match_id", "team", "clock"]).reset_index(drop=True)
    reds = ch[ch.kind == "red"][["match_id", "clock"]]
    ch = ch[ch.kind != "red"]
    # cluster decisions
    ch["new"] = (ch.groupby(["match_id", "team"]).clock.diff().fillna(99) > 2.0).astype(int)
    ch["dec"] = ch.groupby(["match_id", "team"]).new.cumsum()
    dec = ch.groupby(["match_id", "team", "dec"]).agg(clock=("clock", "min"), clock_end=("clock", "max"),
                                                      period=("period", "first"),
                                                      d_form=("d_form", "sum"), d_pers=("d_pers", "sum"),
                                                      n_sub=("kind", lambda k: (k == "sub").sum()),
                                                      n_shift=("kind", lambda k: (k == "shift").sum())).reset_index()
    dec["delta"] = (dec.d_form + dec.d_pers) / 2
    # drop decisions near red cards
    rd = reds.groupby("match_id").clock.apply(list).to_dict()
    dec = dec[[not any(abs(c - r) <= 3 for r in rd.get(m, [])) for m, c in zip(dec.match_id, dec.clock)]]
    # match info: per-team strength from segments
    info = seg.drop_duplicates(["match_id", "team"])[["match_id", "team", "opp", "home", "lam_own", "lam_opp", "pwin", "ploss", "league"]]
    dec = dec.merge(info, on=["match_id", "team"], how="inner")

    shots = ev[ev.type == "Shot"]; goals = ev[((ev.type == "Shot") & (ev.outcome == "Goal")) | (ev.type == "Own Goal For")]
    goals_by = goals.groupby("match_id")
    shots_by = shots.groupby("match_id")
    gmap = {m: g for m, g in goals_by}; smap = {m: g for m, g in shots_by}
    allred = rd
    alldec = dec.groupby("match_id")
    rows = []
    for r in dec.itertuples():
        g = gmap.get(r.match_id, goals.iloc[:0]); s = smap.get(r.match_id, shots.iloc[:0])
        half0 = 0.0 if r.period == 1 else 45.0
        half1 = s[s.period == r.period].clock.max() if len(s[s.period == r.period]) else half0 + 47
        half1 = max(half1, ev[(ev.match_id == r.match_id) & (ev.period == r.period)].clock.max()) if False else max(half1, half0 + 45)
        prev_goal = g[g.clock < r.clock].clock.max() if (g.clock < r.clock).any() else -1
        next_goal = g[g.clock > r.clock_end].clock.min() if (g.clock > r.clock_end).any() else 1e9
        reds_after = [x for x in allred.get(r.match_id, []) if x > r.clock_end]
        pre0 = max(r.clock - W, half0, prev_goal); pre1 = r.clock
        post0 = r.clock_end; post1 = min(r.clock_end + W, next_goal, min(reds_after) if reds_after else 1e9, half1)
        if post1 - post0 < 3 or pre1 - pre0 < 3:
            continue
        # game state at decision
        gf = ((g.clock < r.clock) & (g.team == r.team)).sum(); ga = ((g.clock < r.clock) & (g.team != r.team)).sum()
        def x(t0, t1, own):
            m = (s.clock > t0) & (s.clock <= t1) & ((s.team == r.team) == own)
            return s.xg[m].sum()
        # opponent decisions in post window
        od = alldec.get_group(r.match_id)
        od = od[(od.team != r.team) & (od.clock >= post0 - 0.5) & (od.clock <= post1)]
        rows.append(dict(match_id=r.match_id, team=r.team, league=r.league, clock=r.clock, delta=r.delta,
                         n_sub=r.n_sub, n_shift=r.n_shift, gd=int(gf - ga), lam_own=r.lam_own, lam_opp=r.lam_opp,
                         pwin=r.pwin, ploss=r.ploss, home=r.home,
                         pre_min=pre1 - pre0, post_min=post1 - post0,
                         xf_pre=x(pre0, pre1, True), xa_pre=x(pre0, pre1, False),
                         xf_post=x(post0, post1, True), xa_post=x(post0, post1, False),
                         pre_mid=(pre0 + pre1) / 2, post_mid=(post0 + post1) / 2,
                         opp_delta=od.delta.sum()))
    d = pd.DataFrame(rows)
    d.to_parquet(os.path.join(PROC, "decisions.parquet"))
    return d


def baseline_rates(d, side):
    """Expected xG per window from the strength+state baseline GLM (fit on all segments)."""
    seg = load_segments()
    seg = seg.dropna(subset=["lam_own"])
    sc = StyleScaler().fit(seg, STYLE)
    X, off = design(seg, sc, "base")
    res = fit_glm(seg, X, off)
    def pred(clock_mid, minutes, gd, lam_own, lam_opp):
        q = pd.DataFrame(dict(gd_c=np.clip(gd, -2, 2), men_c=0, tmin=clock_mid / 90, log_ratio=np.log(lam_own / lam_opp),
                              minutes=minutes, lam_own=lam_own))
        Xq, oq = design(q, sc, "base")
        return np.exp(Xq.values @ res.params + oq.values)
    return pred, res


def analyse(d):
    pred, res = baseline_rates(d, "for")
    out = {}
    for side in ("f", "a"):
        if side == "f":
            ef_post = pred(d.post_mid, d.post_min, d.gd, d.lam_own, d.lam_opp)
            ef_pre = pred(d.pre_mid, d.pre_min, d.gd, d.lam_own, d.lam_opp)
            y, ypre = d.xf_post, d.xf_pre
        else:  # opponent's view
            ef_post = pred(d.post_mid, d.post_min, -d.gd, d.lam_opp, d.lam_own)
            ef_pre = pred(d.pre_mid, d.pre_min, -d.gd, d.lam_opp, d.lam_own)
            y, ypre = d.xa_post, d.xa_pre
        mom = np.log((ypre + 0.1 * ef_pre) / (1.1 * ef_pre))   # shrunk pre-window over/under-performance
        X = pd.DataFrame(dict(const=1.0, att=(d.delta >= 0.5).astype(float), dfn=(d.delta <= -0.5).astype(float),
                              mom=mom, opp_att=(d.opp_delta >= 0.5).astype(float), opp_dfn=(d.opp_delta <= -0.5).astype(float),
                              trail=(d.gd < 0).astype(float), lead=(d.gd > 0).astype(float)))
        m = sm.GLM(y.values, X.values, family=sm.families.Poisson(), offset=np.log(ef_post)).fit(
            scale="X2", cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
        out[side] = {n: (float(p), float(se)) for n, p, se in zip(X.columns, m.params, m.bse)}
        # continuous delta version
        Xc = X.drop(columns=["att", "dfn"]).assign(delta=d.delta.clip(-3, 3))
        mc = sm.GLM(y.values, Xc.values, family=sm.families.Poisson(), offset=np.log(ef_post)).fit(
            scale="X2", cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
        out[side + "_cont"] = {n: (float(p), float(se)) for n, p, se in zip(Xc.columns, mc.params, mc.bse)}
    return out


if __name__ == "__main__":
    d = build() if "--rebuild" in sys.argv or not os.path.exists(os.path.join(PROC, "decisions.parquet")) else pd.read_parquet(os.path.join(PROC, "decisions.parquet"))
    print(len(d), "decisions;", (d.delta >= 0.5).sum(), "attacking;", (d.delta <= -0.5).sum(), "defensive")
    print(d.groupby(np.sign(d.gd)).delta.describe())
    out = analyse(d)
    for k, v in out.items():
        print(k); [print(f"   {n:8s} {p:+.3f} ({se:.3f})  x{np.exp(p):.2f}") for n, (p, se) in v.items()]
    json.dump(out, open(os.path.join(ROOT, "results", "event_study.json"), "w"), indent=1)
