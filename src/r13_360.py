"""Rounds 13 + 14A on StatsBomb 360 data (PREREGISTRATION_13.md, PREREGISTRATION_14.md).
  build  -> data/proc/r13_actions.parquet (open-play on-ball actions with 360 space features + net xG 20 s; regain flag)
  mine   -> round 13 (space-conditioned Tactic Miner)
  trigger-> round 14A (defenders back at regain)
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm, statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "sb360")
PROC = os.path.join(ROOT, "data", "proc")
RESTART = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
DISC = ("FIFA World Cup 2022", "UEFA Euro 2020", "UEFA Euro 2024", "African Cup of Nations 2023")
WOMEN = ("UEFA Women's Euro 2022", "UEFA Women's Euro 2025", "Women's World Cup 2023")


def tsec(e):
    h, m, s = e["timestamp"].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def one(m):
    mid = m["match_id"]
    pe, pf = os.path.join(RAW, f"{mid}_events.json.gz"), os.path.join(RAW, f"{mid}_three-sixty.json.gz")
    if not (os.path.exists(pe) and os.path.exists(pf)):
        return None
    try:
        ev = [e for e in json.load(gzip.open(pe)) if e["period"] <= 2]
        fr = {f["event_uuid"]: f for f in json.load(gzip.open(pf))}
    except Exception:
        return None
    shots = [(e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
             if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] not in ("Penalty",)]
    goals = [(e["period"], tsec(e), e["team"]["name"]) for e in ev
             if (e["type"]["name"] == "Shot" and e["shot"]["outcome"]["name"] == "Goal") or e["type"]["name"] == "Own Goal For"]
    # possession start times and previous possession team
    pstart, pteam = {}, {}
    for e in ev:
        k = e["possession"]
        if k not in pstart:
            pstart[k] = (e["period"], tsec(e)); pteam[k] = (e["possession_team"]["name"], e["play_pattern"]["name"])
    rows = []
    for e in ev:
        typ = e["type"]["name"]
        if typ not in ("Pass", "Carry", "Dribble", "Shot") or e["team"]["name"] != e["possession_team"]["name"]:
            continue
        if typ == "Pass" and e["pass"].get("type", {}).get("name") in RESTART:
            continue
        if typ == "Shot" and e["shot"]["type"]["name"] != "Open Play":
            continue
        f = fr.get(e["id"])
        if f is None or "location" not in e:
            continue
        x, y = e["location"][:2]
        opp = [p["location"] for p in f["freeze_frame"] if not p["teammate"] and not p.get("keeper")]
        mates = [p["location"] for p in f["freeze_frame"] if p["teammate"] and not p.get("actor") and not p.get("keeper")]
        d_opp = [np.hypot(o[0] - x, o[1] - y) for o in opp]
        from r4_structure import in_tri
        lane = sum(in_tri(o, (x, y), (120, 36), (120, 44)) for o in opp)
        if typ == "Pass":
            ex, ey = e["pass"]["end_location"][:2]; length = e["pass"].get("length", np.hypot(ex - x, ey - y))
        elif typ == "Carry":
            ex, ey = e["carry"]["end_location"][:2]; length = np.hypot(ex - x, ey - y)
        else:
            ex, ey, length = x, y, 0.0
        fwd = ex - x
        inbox = ex >= 102 and 18 <= ey <= 62
        if typ == "Shot":
            act = "shot"
        elif typ == "Dribble":
            act = "dribble"
        elif typ == "Pass" and abs(y - 40) > 22 and x >= 96 and inbox:
            act = "cross"
        elif typ == "Pass" and length >= 30 and fwd >= 10:
            act = "long_fwd"
        elif typ == "Pass" and length < 30 and fwd >= 5:
            act = "short_fwd"
        elif typ == "Pass":
            act = "recycle"
        elif fwd >= 5:
            act = "carry_fwd"
        else:
            if abs(fwd) + abs(ey - y) < 1:
                continue
            act = "carry_other"
        p0 = pstart[e["possession"]]; k = e["possession"]
        prev_team = pteam.get(k - 1, (None, None))[0]
        is_regain_start = (pteam[k][1] == "Regular Play" and prev_team is not None and prev_team != pteam[k][0] and p0 == (e["period"], tsec(e)))
        rows.append(dict(match_id=mid, comp=m["_comp"], team=e["team"]["name"], period=e["period"], t=tsec(e), x=x, y=y, action=act,
                         opp_goalside=sum(o[0] > x for o in opp), nearest_opp=min(d_opp) if d_opp else 30.0,
                         mates_ahead=sum(mm[0] > x for mm in mates), n_visible_opp=len(opp), lane=lane,
                         since=tsec(e) - p0[1] if p0[0] == e["period"] else 99, regain_start=is_regain_start))
    if not rows:
        return None
    a = pd.DataFrame(rows)
    S = pd.DataFrame(shots, columns=["period", "t", "team", "xg"]); Gl = pd.DataFrame(goals, columns=["period", "t", "team"])
    own = np.zeros(len(a)); opp_ = np.zeros(len(a)); gd = np.zeros(len(a), int)
    for i, r in enumerate(a.itertuples()):
        s = S[(S.period == r.period) & (S.t >= r.t - 0.01) & (S.t <= r.t + 20)]
        own[i] = s.xg[s.team == r.team].sum(); opp_[i] = s.xg[(s.team != r.team) & (s.t > r.t)].sum()
        g = Gl[(Gl.period < r.period) | ((Gl.period == r.period) & (Gl.t < r.t))]
        gd[i] = (g.team == r.team).sum() - (g.team != r.team).sum()
    a["own20"] = own; a["opp20"] = opp_; a["net"] = own - opp_; a["gd"] = gd
    return a


def build():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    out = []
    with ProcessPoolExecutor(8) as ex:
        for a in ex.map(one, ms, chunksize=4):
            if a is not None:
                out.append(a)
    A = pd.concat(out, ignore_index=True)
    A["pop"] = np.select([A.comp.str.startswith(DISC), A.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
    A["zx"] = pd.cut(A.x, [-1, 40, 60, 80, 96, 108, 121], labels=False)
    A["zy"] = np.select([(A.y - 40).abs() <= 13, A.y < 40], [1, 0], 2)
    A["og"] = np.select([A.opp_goalside <= 3, A.opp_goalside <= 6], [0, 1], 2)
    A["no"] = np.select([A.nearest_opp < 3, A.nearest_opp <= 7], [0, 1], 2)
    A["ma"] = np.select([A.mates_ahead <= 1, A.mates_ahead <= 3], [0, 1], 2)
    A["cell_zone"] = A.zx.astype(int).astype(str) + A.zy.astype(str)
    A["cell_space"] = A.cell_zone + A.og.astype(str) + A["no"].astype(str) + A.ma.astype(str)
    A.to_parquet(os.path.join(PROC, "r13_actions.parquet"))
    print(A.groupby(["pop", "action"]).size().unstack(0)); print(A.groupby("pop").match_id.nunique())


def rule_table(D, cellcol):
    st = D.groupby([cellcol, "action"]).net.agg(["mean", "var", "size"]).reset_index().rename(columns={cellcol: "cell"})
    ref = st.sort_values("size", ascending=False).drop_duplicates("cell").set_index("cell")
    c = st.merge(ref[["action", "mean", "var", "size"]].rename(columns=lambda x: "ref_" + x), left_on="cell", right_index=True)
    c = c[c.action != c.ref_action].copy()
    c["diff"] = c["mean"] - c.ref_mean; c["se"] = np.sqrt(c["var"] / c["size"] + c.ref_var / c.ref_size)
    return c


def forward_premium(D, cellcol):
    """Cell-weighted mean advantage of forward actions over recycle."""
    st = D.groupby([cellcol, "action"]).net.agg(["mean", "size"]).unstack("action")
    out = {}
    for a in ("carry_fwd", "short_fwd", "long_fwd"):
        m = st["mean"][[a, "recycle"]].dropna(); n = st["size"][[a, "recycle"]].dropna()
        ok = (n[a] >= 20) & (n["recycle"] >= 20)
        w = n[a][ok]
        out[a] = float(((m[a] - m["recycle"])[ok] * w).sum() / w.sum())
    return out


def mine():
    A = pd.read_parquet(os.path.join(PROC, "r13_actions.parquet"))
    D = A[A["pop"] == "discovery"]
    res = {"H13_1": {"zone_only": forward_premium(D, "cell_zone"), "zone_x_space": forward_premium(D, "cell_space")}}
    for popn in ("confirm_club", "confirm_women"):
        C = A[A["pop"] == popn]
        res["H13_1"][popn] = {"zone_only": forward_premium(C, "cell_zone"), "zone_x_space": forward_premium(C, "cell_space")}
    print("H13-1 forward premium (net xG per action vs recycle):", json.dumps(res["H13_1"], indent=1))
    c = rule_table(D, "cell_space"); c = c[(c["size"] >= 30) & (c.ref_size >= 30)]
    c["p"] = 2 * norm.sf(np.abs(c["diff"] / c.se)); c["q"] = multipletests(c.p, method="fdr_bh")[1]
    res["n_rules"] = int(len(c)); res["n_nominal"] = int((c.p < 0.05).sum()); res["n_bh"] = int((c.q < 0.05).sum())
    print("rules searched", len(c), "nominal", res["n_nominal"], "BH5%", res["n_bh"])
    agg = {}
    for popn in ("confirm_club", "confirm_women"):
        C = rule_table(A[A["pop"] == popn], "cell_space").set_index(["cell", "action"])
        for label, sub in (("nominal", c[c.p < 0.05]), ("strong", c[c.p < 0.001])):
            j = sub.join(C[["diff", "se", "size", "ref_size"]], on=["cell", "action"], rsuffix="_c").dropna(subset=["diff_c"])
            j = j[(j.size_c >= 10) & (j.ref_size_c >= 10)]
            ag = (np.sign(j["diff"]) == np.sign(j.diff_c))
            agg[f"{popn}_{label}"] = dict(n=int(len(j)), sign_agreement=float(ag.mean()) if len(j) else None,
                                         corr=float(np.corrcoef(j["diff"], j.diff_c)[0, 1]) if len(j) > 2 else None)
        # confirmation of BH shortlist
    res["H13_3"] = agg; print("H13-3", json.dumps(agg, indent=1))
    S = c[c.q < 0.05].copy(); conf = []
    if len(S):
        Cs = {p: rule_table(A[A["pop"] == p], "cell_space").set_index(["cell", "action"]) for p in ("confirm_club", "confirm_women")}
        for r in S.itertuples():
            ests = []
            for p in Cs:
                try:
                    v = Cs[p].loc[(r.cell, r.action)]
                    if v["size"] >= 10 and v.ref_size >= 10 and v.ref_action == r.ref_action:
                        ests.append((v["diff"], v.se))
                except KeyError:
                    pass
            if len(ests) == 2:
                w = np.array([1 / e[1] ** 2 for e in ests]); pooled = sum(w[i] * ests[i][0] for i in range(2)) / w.sum(); pse = 1 / np.sqrt(w.sum())
                conf.append(dict(cell=r.cell, action=r.action, ref=r.ref_action, disc=r.diff, pooled=pooled, p1=float(norm.sf(np.sign(r.diff) * pooled / pse)),
                                 same=bool(all(np.sign(e[0]) == np.sign(r.diff) for e in ests))))
        k = len(conf); conf = sorted(conf, key=lambda d: d["p1"]); flag = True
        for i, d in enumerate(conf):
            flag = flag and d["p1"] < 0.05 / (k - i); d["confirmed"] = bool(flag and d["same"])
        print("H13-2 shortlist testable:", k, "confirmed:", sum(d["confirmed"] for d in conf))
        for d in conf:
            if d["confirmed"]:
                print("  ", d)
    res["H13_2"] = conf
    json.dump(res, open(os.path.join(ROOT, "results", "r13_miner360.json"), "w"), indent=1, default=float)


def trigger():
    A = pd.read_parquet(os.path.join(PROC, "r13_actions.parquet"))
    R = A[A.regain_start & (A.x < 80)].copy()
    R["rx"] = R.x / 120; R["cent"] = 1 - (R.y - 40).abs() / 40; R["clk"] = (R.t / 60 + np.where(R.period == 2, 45, 0)) / 90
    R["gpos"] = (R.gd > 0).astype(float); R["gneg"] = (R.gd < 0).astype(float)
    out = {}
    for popn in ("discovery", "confirm_club", "confirm_women"):
        d = R[R["pop"] == popn]
        m = smf.glm("own20 ~ rx + I(rx**2) + I(rx**3) + cent + gpos + gneg + clk + opp_goalside + n_visible_opp", data=d,
                    family=sm.families.Poisson()).fit(scale="X2", cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
        b, se = float(m.params["opp_goalside"]), float(m.bse["opp_goalside"])
        out[popn] = dict(n=len(d), coef_per_defender=b, se=se, p_one=float(norm.cdf(b / se)), mult_per_defender=float(np.exp(b)))
        print(popn, out[popn])
    # trigger table (pooled): xG within 20 s by zone and defenders back
    R["zone"] = pd.cut(R.x, [-1, 40, 60, 80], labels=["own third", "own half", "middle"])
    R["back"] = pd.cut(R.opp_goalside, [-1, 3, 5, 7, 11], labels=["≤3", "4-5", "6-7", "8+"])
    tab = R.groupby(["zone", "back"]).own20.agg(["mean", "size"]).round(4)
    print(tab.unstack())
    out["table"] = {f"{z}|{b}": dict(xg=float(v["mean"]), n=int(v["size"])) for (z, b), v in tab.iterrows()}
    json.dump(out, open(os.path.join(ROOT, "results", "r14a_trigger.json"), "w"), indent=1)


if __name__ == "__main__":
    {"build": build, "mine": mine, "trigger": trigger}[sys.argv[1]]()
