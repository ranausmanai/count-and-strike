"""Round 11: Tactic Miner (PREREGISTRATION_11.md).
  python3 src/r11_miner.py build   -> data/proc/r11_actions.parquet
  python3 src/r11_miner.py mine    -> discovery search + confirmation
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
RESTART = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
W = 20 / 60.0


def strength_sb15():
    from frontier import load_segments
    s = load_segments(min_minutes=0.0).drop_duplicates(["match_id", "team"])
    return dict(zip(zip(s.match_id, s.team), s.lam_own / (s.lam_own + s.lam_opp)))


def strength_sb2(ev):
    sh = ev[(ev.type == "Shot") & (ev["sub"] != "Penalty")]
    teams = ev.groupby("match_id").team.unique()
    rows = []
    for m, ts in teams.items():
        for t in ts:
            o = [x for x in ts if x != t]
            if not o:
                continue
            rows.append(dict(match_id=m, team=t, opp=o[0], comp=ev[ev.match_id == m].league.iloc[0] if False else None))
    T = pd.DataFrame(rows)
    xf = sh.groupby(["match_id", "team"]).xg.sum()
    T["xf"] = [xf.get((m, t), 0.0) for m, t in zip(T.match_id, T.team)]
    T["xa"] = [xf.get((m, o), 0.0) for m, o in zip(T.match_id, T.opp)]
    lg = ev.drop_duplicates("match_id").set_index("match_id").league
    T["comp"] = T.match_id.map(lg)
    g = T.groupby(["comp", "team"]); n = g.match_id.transform("size")
    T["att"] = (g.xf.transform("sum") - T.xf) / (n - 1); T["dfn"] = (g.xa.transform("sum") - T.xa) / (n - 1)
    o = T[["match_id", "team", "att", "dfn"]].rename(columns={"team": "opp", "att": "o_att", "dfn": "o_dfn"})
    T = T.merge(o, on=["match_id", "opp"])
    ef, ea = T.att * T.o_dfn, T.o_att * T.dfn
    return dict(zip(zip(T.match_id, T.team), ef / (ef + ea)))


def actions(ev, share):
    ev = ev[ev.period <= 2].sort_values(["match_id", "idx"]).reset_index(drop=True)
    ev["clock"] = ev.t + np.where(ev.period == 2, 45.0, 0.0)
    # goals timeline and game state
    goal = ((ev.type == "Shot") & (ev.outcome == "Goal")) | (ev.type == "Own Goal For")
    G = ev[goal][["match_id", "clock", "team"]]
    # shots for outcome windows
    S = ev[(ev.type == "Shot") & (ev["sub"] != "Penalty") & (ev["sub"] != "Free Kick")][["match_id", "period", "t", "team", "xg"]]
    # candidate actions
    a = ev[ev.type.isin(["Pass", "Carry", "Dribble", "Shot"]) & (ev.team == ev.poss_team)].copy()
    a = a[~((a.type == "Pass") & a["sub"].isin(RESTART))]
    a = a[~((a.type == "Shot") & a["sub"].isin(["Penalty", "Free Kick", "Corner", "Kick Off"]))]
    fwd = a.end_x - a.x
    wide = (a.y - 40).abs() > 22
    inbox = (a.end_x >= 102) & a.end_y.between(18, 62)
    act = np.select([a.type == "Shot", (a.type == "Dribble"),
                     (a.type == "Pass") & wide & (a.x >= 96) & inbox,
                     (a.type == "Pass") & (a.length >= 30) & (fwd >= 10),
                     (a.type == "Pass") & (a.length < 30) & (fwd >= 5),
                     (a.type == "Pass"),
                     (a.type == "Carry") & (fwd >= 5)],
                    ["shot", "dribble", "cross", "long_fwd", "short_fwd", "recycle", "carry_fwd"], "carry_other")
    a["action"] = act
    a = a[~((a.type == "Carry") & ((a.end_x - a.x).abs() + (a.end_y - a.y).abs() < 1))]  # drop zero-length carries
    # possession start time
    ps = ev.groupby(["match_id", "poss"]).t.min().rename("poss_t0")
    a = a.join(ps, on=["match_id", "poss"])
    a["since"] = (a.t - a.poss_t0) * 60
    # game state at action
    a = a.reset_index(drop=True)
    gd = np.zeros(len(a), dtype=int)
    gmap = {m: g for m, g in G.groupby("match_id")}
    for (m, tm), idx in a.groupby(["match_id", "team"]).indices.items():
        g = gmap.get(m)
        if g is None:
            continue
        mine = np.sort(g.clock.values[g.team.values == tm]); theirs = np.sort(g.clock.values[g.team.values != tm])
        c = a.clock.values[idx]
        gd[idx] = np.searchsorted(mine, c, "left") - np.searchsorted(theirs, c, "left")
    a["gd"] = gd
    # outcome: own xG in [t, t+W], opp xG in (t, t+W]
    own = np.zeros(len(a)); opp = np.zeros(len(a))
    smap = {k: g for k, g in S.groupby(["match_id", "period"])}
    for key, idx in a.groupby(["match_id", "period"]).indices.items():
        g = smap.get(key)
        if g is None:
            continue
        tt = a.t.values[idx]; tm = a.team.values[idx]
        st = g.t.values; sx = g.xg.values; steam = g.team.values
        order = np.argsort(st); st, sx, steam = st[order], sx[order], steam[order]
        for team in np.unique(tm):
            mine = steam == team
            cs_m = np.concatenate([[0], np.cumsum(np.where(mine, sx, 0))]); cs_o = np.concatenate([[0], np.cumsum(np.where(~mine, sx, 0))])
            sel = tm == team
            lo = np.searchsorted(st, tt[sel] - 1e-9, "left"); hi = np.searchsorted(st, tt[sel] + W, "right")
            lo2 = np.searchsorted(st, tt[sel] + 1e-9, "right")
            own[idx[sel]] = cs_m[hi] - cs_m[lo]; opp[idx[sel]] = cs_o[hi] - cs_o[lo2]
    a["net"] = own - opp; a["own20"] = own; a["opp20"] = opp
    a["share"] = [share.get((m, t), np.nan) for m, t in zip(a.match_id, a.team)]
    # situation cell
    a["zx"] = pd.cut(a.x, [-1, 40, 60, 80, 96, 108, 121], labels=False)
    a["zy"] = np.select([(a.y - 40).abs() <= 13, a.y < 40], [1, 0], 2)  # central / left / right
    a["pr"] = a.under_pressure.astype(int)
    a["gs"] = np.sign(a.gd).astype(int)
    a["sc"] = np.select([a.since < 5, a.since < 15], [0, 1], 2)
    a["per"] = np.select([a.period == 1, a.clock < 75], [0, 1], 2)
    a["st"] = np.select([a.share < 0.42, a.share <= 0.58], [0, 1], 2)
    a = a.dropna(subset=["share", "zx"])
    a["cell"] = (a.zx.astype(int).astype(str) + a.zy.astype(str) + a.pr.astype(str) + (a.gs + 1).astype(str) + a.sc.astype(str) + a.per.astype(str) + a.st.astype(str))
    return a[["match_id", "league", "team", "action", "cell", "zx", "zy", "pr", "gs", "sc", "per", "st", "x", "y", "net", "own20", "opp20"]]


def build():
    cols = ["match_id", "idx", "period", "t", "type", "team", "poss", "poss_team", "x", "y", "end_x", "end_y", "length", "xg", "outcome", "sub", "under_pressure", "league"]
    e1 = pd.read_parquet(os.path.join(PROC, "events.parquet"), columns=cols)
    a1 = actions(e1, strength_sb15()); a1["pop"] = np.where(a1.league == "EPL", "confirm_EPL", "discovery")
    e2 = pd.read_parquet(os.path.join(PROC, "sb2_events.parquet"), columns=cols)
    a2 = actions(e2, strength_sb2(e2)); a2["pop"] = "confirm_unseen"
    A = pd.concat([a1, a2], ignore_index=True)
    A.to_parquet(os.path.join(PROC, "r11_actions.parquet"))
    print(A.groupby(["pop", "action"]).size().unstack(0))


def cell_stats(d):
    g = d.groupby(["cell", "action"]).net.agg(["mean", "var", "size"]).reset_index()
    return g


def mine():
    A = pd.read_parquet(os.path.join(PROC, "r11_actions.parquet"))
    D = A[A["pop"] == "discovery"]
    st = cell_stats(D)
    ref = st.sort_values("size", ascending=False).drop_duplicates("cell").set_index("cell")
    c = st.merge(ref[["action", "mean", "var", "size"]].rename(columns=lambda x: "ref_" + x), left_on="cell", right_index=True)
    c = c[(c.action != c.ref_action) & (c["size"] >= 30) & (c.ref_size >= 30)].copy()
    c["diff"] = c["mean"] - c.ref_mean
    c["se"] = np.sqrt(c["var"] / c["size"] + c.ref_var / c.ref_size)
    c["p"] = 2 * norm.sf(np.abs(c["diff"] / c.se))
    c["q"] = multipletests(c.p, method="fdr_bh")[1]
    c["use_share"] = c["size"] / (c["size"] + c.ref_size)
    print("cells", st.cell.nunique(), "| candidate rules searched:", len(c), "| BH 5% shortlist:", int((c.q < 0.05).sum()), "| nominal p<.05:", int((c.p < 0.05).sum()))
    short = c[c.q < 0.05].copy()
    # confirmation
    out = []
    for popn in ("confirm_EPL", "confirm_unseen"):
        s2 = cell_stats(A[A["pop"] == popn]).set_index(["cell", "action"])
        m = []
        for r in short.itertuples():
            try:
                a = s2.loc[(r.cell, r.action)]; b = s2.loc[(r.cell, r.ref_action)]
                m.append((a["mean"] - b["mean"], np.sqrt(a["var"] / a["size"] + b["var"] / b["size"]), a["size"], b["size"]))
            except KeyError:
                m.append((np.nan, np.nan, 0, 0))
        m = np.array(m, dtype=float)
        short[f"{popn}_diff"] = m[:, 0]; short[f"{popn}_se"] = m[:, 1]; short[f"{popn}_n"] = m[:, 2]
    ok = short[(short.confirm_EPL_n >= 10) & (short.confirm_unseen_n >= 10)].copy()
    w1, w2 = 1 / ok.confirm_EPL_se ** 2, 1 / ok.confirm_unseen_se ** 2
    ok["pooled"] = (w1 * ok.confirm_EPL_diff + w2 * ok.confirm_unseen_diff) / (w1 + w2); ok["pooled_se"] = 1 / np.sqrt(w1 + w2)
    sgn = np.sign(ok["diff"])
    ok["p1"] = norm.sf(sgn * ok.pooled / ok.pooled_se)
    ok["same_both"] = (np.sign(ok.confirm_EPL_diff) == sgn) & (np.sign(ok.confirm_unseen_diff) == sgn)
    k = len(ok)
    ok = ok.sort_values("p1").reset_index(drop=True)
    ok["holm"] = [0.05 / (k - i) for i in range(k)]
    conf = []
    for _, r in ok.iterrows():
        if r.p1 < r.holm and r.same_both:
            conf.append(True)
        elif r.p1 >= r.holm:
            break
        else:
            conf.append(False)
    ok["confirmed"] = conf + [False] * (k - len(conf))
    print("testable in both confirmation sets:", k, "| confirmed:", int(ok.confirmed.sum()), "| same sign in both (any p):", int(ok.same_both.sum()))
    names = {0: "own third", 1: "own half", 2: "mid", 3: "att third", 4: "edge of box", 5: "box"}
    def describe(r):
        cell = r.cell; zx, zy, pr, gs, sc, per, stt = int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3]) - 1, int(cell[4]), int(cell[5]), int(cell[6])
        return (f"{names[zx]}/{['left','central','right'][zy]}, {'pressed' if pr else 'free'}, {['trailing','level','leading'][gs+1]}, "
                f"{['<5s','5-15s','>15s'][sc]} into possession, {['1st half','2nd half','75+'][per]}, {['underdog','even','favourite'][stt]}")
    ok["situation"] = ok.apply(describe, axis=1)
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 120)
    show = ok[ok.confirmed][["situation", "action", "ref_action", "diff", "pooled", "use_share", "size", "confirm_EPL_diff", "confirm_unseen_diff"]]
    print(show.to_string())
    res = dict(n_cells=int(st.cell.nunique()), n_searched=int(len(c)), n_bh=int(len(short)), n_testable=int(k), n_confirmed=int(ok.confirmed.sum()),
               n_same_sign_both=int(ok.same_both.sum()), expected_false_bh=float(0.05 * len(short)),
               sign_agreement_rate=float(ok.same_both.mean()) if k else None,
               confirmed=ok[ok.confirmed].to_dict(orient="records"))
    ok.to_parquet(os.path.join(PROC, "r11_rules.parquet"))
    json.dump(res, open(os.path.join(ROOT, "results", "r11_miner.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    {"build": build, "mine": mine}[sys.argv[1]]()
