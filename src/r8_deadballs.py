"""Round 8: dead-ball decision audit (PREREGISTRATION_8.md).
  build   -> data/proc/r8_deadballs.parquet (SB 2015/16, SB unseen, Wyscout 2017/18)
  audit   -> discovery on SB 2015/16 non-EPL, confirmation on SB EPL / WY / SB unseen
Coordinates in StatsBomb yards (120 x 80); Wyscout converted (x*1.2, y*0.8).
Zone clarification (fixed before outcomes were computed): 'short' = corner ending outside the box within 20 yds of the
corner spot; 'edge' = ending outside the box further than 20 yds from the spot (delivered to the edge)."""
import gzip, json, os, pickle, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")


def tsec(e):
    h, m, s = e["timestamp"].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def sb_one(args):
    path, comp = args
    mid = int(os.path.basename(path).split(".")[0])
    ev = json.load(gzip.open(path))
    shots, dbs = [], []
    for e in ev:
        if e["period"] > 2:
            continue
        typ = e["type"]["name"]; t = tsec(e)
        if typ == "Shot":
            st = e["shot"]["type"]["name"]
            shots.append((e["period"], t, e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0), st))
            if st == "Free Kick":
                x, y = e["location"][:2]
                dbs.append(dict(kind="fk", option="shot", x=x, y=y, ex=120, ey=40, tech=None, period=e["period"], t=t, team=e["team"]["name"]))
        elif typ == "Pass":
            pt = e["pass"].get("type", {}).get("name")
            if pt in ("Corner", "Free Kick", "Throw-in"):
                x, y = e["location"][:2]; ex, ey = e["pass"]["end_location"][:2]
                dbs.append(dict(kind={"Corner": "corner", "Free Kick": "fk", "Throw-in": "throw"}[pt], option=None, x=x, y=y, ex=ex, ey=ey,
                                tech=e["pass"].get("technique", {}).get("name"), height=e["pass"].get("height", {}).get("name"),
                                period=e["period"], t=t, team=e["team"]["name"]))
    return mid, comp, shots, dbs


def wy_tables():
    from xg_model import feats
    XG = pickle.load(open(os.path.join(PROC, "xg_model.pkl"), "rb"))
    WY = os.path.join(ROOT, "data", "raw", "wy")
    out = []
    for country, lg in {"England": "EPL", "Spain": "LaLiga", "Germany": "Bundesliga", "Italy": "SerieA", "France": "Ligue1"}.items():
        ev = [e for e in json.load(open(os.path.join(WY, f"events_{country}.json"))) if e["matchPeriod"] in ("1H", "2H")]
        sh = [e for e in ev if e["eventName"] == "Shot" or (e["eventName"] == "Free Kick" and e["subEventName"] == "Free kick shot")]
        df = pd.DataFrame(dict(x=[s["positions"][0]["x"] / 100 * 105 for s in sh], y=[s["positions"][0]["y"] / 100 * 68 for s in sh],
                               head=[any(t["id"] == 403 for t in s["tags"]) for s in sh], fk=[s["subEventName"] == "Free kick shot" for s in sh]))
        p = XG.predict_proba(feats(df))[:, 1]
        S = pd.DataFrame(dict(match_id=[s["matchId"] for s in sh], period=[1 if s["matchPeriod"] == "1H" else 2 for s in sh],
                              t=[s["eventSec"] for s in sh], team=[s["teamId"] for s in sh], xg=p))
        D = []
        for e in ev:
            if e["eventName"] != "Free Kick" or e["subEventName"] not in ("Corner", "Free Kick", "Free kick cross", "Free kick shot", "Throw in"):
                continue
            ps = e["positions"]; x, y = ps[0]["x"] * 1.2, ps[0]["y"] * 0.8
            ex, ey = (ps[1]["x"] * 1.2, ps[1]["y"] * 0.8) if len(ps) > 1 else (np.nan, np.nan)
            kind = {"Corner": "corner", "Throw in": "throw"}.get(e["subEventName"], "fk")
            opt = {"Free kick shot": "shot", "Free kick cross": "cross_wy"}.get(e["subEventName"])
            if e["subEventName"] == "Free kick shot":
                ex, ey = 120, 40
            D.append(dict(match_id=e["matchId"], comp="WY_" + lg, kind=kind, option=opt, x=x, y=y, ex=ex, ey=ey, tech=None,
                          height="High Pass" if any(t["id"] == 801 for t in e["tags"]) else None,
                          period=1 if e["matchPeriod"] == "1H" else 2, t=e["eventSec"], team=e["teamId"]))
        out.append(windows(pd.DataFrame(D), S))
    return pd.concat(out, ignore_index=True)


def windows(D, S):
    """Attach xg20 (own) and xga30 (opponent) to each dead ball."""
    D = D.reset_index(drop=True); xf = np.zeros(len(D)); xa = np.zeros(len(D))
    Sg = {k: g for k, g in S.groupby(["match_id", "period"])}
    for i, r in enumerate(D.itertuples()):
        g = Sg.get((r.match_id, r.period))
        if g is None:
            continue
        w = g[(g.t >= r.t - 0.01) & (g.t <= r.t + 20)]
        xf[i] = w.xg[w.team == r.team].sum()
        w2 = g[(g.t > r.t) & (g.t <= r.t + 30)]
        xa[i] = w2.xg[w2.team != r.team].sum()
    D["xg20"] = xf; D["xga30"] = xa
    return D


def classify(D):
    D = D.copy()
    in_box = (D.ex >= 102) & (D.ey >= 18) & (D.ey <= 62)
    # corners
    c = D.kind == "corner"
    side = np.where(D.y < 40, 0.0, 80.0)
    d_side = np.abs(D.ey - side)
    dist_spot = np.hypot(D.ex - D.x, D.ey - D.y)
    zone = np.full(len(D), None, dtype=object)
    zone[(c & ~in_box & (dist_spot <= 20)).values] = "short"
    zone[(c & ~in_box & (dist_spot > 20)).values] = "edge"
    deep = D.ex >= 108
    zone[(c & in_box & deep & (d_side <= 39)).values] = "near"
    zone[(c & in_box & (D.ex >= 114) & (d_side > 39) & (d_side <= 44)).values] = "central"
    zone[(c & in_box & deep & (d_side > 44)).values] = "far"
    zone[(c & in_box & pd.isna(zone)).values] = "penalty_area"
    D["zone"] = zone
    D["tech_c"] = np.where(D.tech.isin(["Inswinging", "Outswinging"]), D.tech.str[:3].str.lower(), "other")
    # free kicks
    f = D.kind == "fk"
    dist = np.hypot(120 - D.x, 40 - D.y); ang = np.degrees(np.arctan2(np.abs(D.y - 40), 120 - D.x))
    D["fk_cell"] = None
    dbin = pd.cut(dist, [15, 20, 25, 30, 40], labels=["15-20", "20-25", "25-30", "30-40"])
    D.loc[f & dist.between(15, 40), "fk_cell"] = (dbin.astype(str) + np.where(ang < 25, "_central", "_wide"))[f & dist.between(15, 40)]
    opt = D.option.copy()
    fo = f & opt.isna()
    opt[fo & in_box & ((D.height == "High Pass") | D.height.isna())] = "cross"
    opt[fo & opt.isna()] = "pass"
    opt[f & (opt == "cross_wy")] = "cross"
    D["fk_opt"] = np.where(f, opt, None)
    # throws
    th = (D.kind == "throw") & (D.x >= 85)
    D["throw_opt"] = np.where(th, np.where(in_box, "long_box", "short"), None)
    return D


def build():
    sbm = json.load(open(os.path.join(ROOT, "data", "raw", "sb_matches.json")))
    sb2 = json.load(open(os.path.join(ROOT, "data", "raw", "sb2_matches.json")))
    args = [(os.path.join(ROOT, "data", "raw", "sb", f"{m['match_id']}.json.gz"), "SB_" + m["_league"]) for m in sbm] + \
           [(os.path.join(ROOT, "data", "raw", "sb2", f"{m['match_id']}.json.gz"), "SB2_" + m["_comp"]) for m in sb2]
    Ds, Ss = [], []
    with ProcessPoolExecutor(8) as ex:
        for mid, comp, shots, dbs in ex.map(sb_one, args, chunksize=8):
            for d in dbs:
                d.update(match_id=mid, comp=comp)
            Ds += dbs
            Ss += [dict(match_id=mid, period=p, t=t, team=tm, xg=xg) for p, t, tm, xg, st in shots if st != "Penalty"]
    D = windows(pd.DataFrame(Ds), pd.DataFrame(Ss))
    W = wy_tables()
    W["team"] = W.team.astype(str); D["team"] = D.team.astype(str)
    A = classify(pd.concat([D, W], ignore_index=True))
    A["pop"] = np.select([A.comp == "SB_EPL", A.comp.str.startswith("SB2_"), A.comp.str.startswith("WY_")], ["confirm_SB_EPL", "confirm_SB2", "confirm_WY"], "discovery")
    A.to_parquet(os.path.join(PROC, "r8_deadballs.parquet"))
    print(A.groupby(["pop", "kind"]).size())


def compare(d, col, a, b, extra=""):
    """Mean xg20 of option a vs b (Poisson GLM, robust SE). Returns log ratio, se."""
    d = d[d[col].isin([a, b])].copy()
    d["is_a"] = (d[col] == a).astype(float)
    f = "xg20 ~ is_a" + extra
    m = smf.glm(f, data=d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
    return float(m.params["is_a"]), float(m.bse["is_a"]), int((d[col] == a).sum()), int((d[col] == b).sum()), float(d[d[col] == a].xg20.mean()), float(d[d[col] == b].xg20.mean())


def audit():
    A = pd.read_parquet(os.path.join(PROC, "r8_deadballs.parquet"))
    disc = A[A["pop"] == "discovery"]
    short = []
    # corners: zones vs most used zone; technique (SB only)
    c = disc[disc.kind == "corner"].copy(); c["side"] = (c.y < 40).astype(int)
    top = c.zone.value_counts().idxmax()
    rows = []
    for z in c.zone.dropna().unique():
        if z == top:
            continue
        lr, se, na, nb, ma, mb = compare(c, "zone", z, top, " + side + C(comp)")
        rows.append(dict(item=f"corner zone {z} vs {top}", col="zone", a=z, b=top, kind="corner", cell=None, lr=lr, se=se, na=na, nb=nb, ma=ma, mb=mb))
    lr, se, na, nb, ma, mb = compare(c[c.zone != "short"], "tech_c", "ins", "out", " + side + C(comp)")
    rows.append(dict(item="corner inswing vs outswing (delivered)", col="tech_c", a="ins", b="out", kind="corner", cell=None, lr=lr, se=se, na=na, nb=nb, ma=ma, mb=mb))
    # free kicks by cell
    f = disc[(disc.kind == "fk") & disc.fk_cell.notna()]
    for cell, g in f.groupby("fk_cell"):
        top = g.fk_opt.value_counts().idxmax()
        for o in g.fk_opt.unique():
            if o == top or (g.fk_opt == o).sum() < 30:
                continue
            lr, se, na, nb, ma, mb = compare(g, "fk_opt", o, top)
            rows.append(dict(item=f"FK {cell}: {o} vs {top}", col="fk_opt", a=o, b=top, kind="fk", cell=cell, lr=lr, se=se, na=na, nb=nb, ma=ma, mb=mb))
    t = disc[disc.throw_opt.notna()]
    top = t.throw_opt.value_counts().idxmax(); o = [x for x in t.throw_opt.unique() if x != top][0]
    lr, se, na, nb, ma, mb = compare(t, "throw_opt", o, top)
    rows.append(dict(item=f"throw {o} vs {top}", col="throw_opt", a=o, b=top, kind="throw", cell=None, lr=lr, se=se, na=na, nb=nb, ma=ma, mb=mb))
    R = pd.DataFrame(rows)
    R["p"] = 2 * norm.sf(np.abs(R.lr / R.se))
    R["misallocation"] = (R.lr > np.log(1.2)) & (R.ma - R.mb >= 0.005) & (R.p < 0.05)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string())
    sl = R[R.misallocation].copy()
    cz = R[R.item.str.startswith("corner zone")].sort_values("lr", ascending=False).head(1)
    sl = pd.concat([sl, cz]).drop_duplicates("item")
    print("SHORTLIST:", list(sl.item))
    json.dump(dict(all=R.to_dict(orient="records"), shortlist=sl.to_dict(orient="records")), open(os.path.join(ROOT, "results", "r8_discovery.json"), "w"), indent=1, default=float)
    # confirmation
    conf = []
    for s in sl.to_dict(orient="records"):
        ests = {}
        for popn in ("confirm_SB_EPL", "confirm_WY", "confirm_SB2"):
            d = A[(A["pop"] == popn) & (A.kind == s["kind"])].copy()
            if s["cell"]:
                d = d[d.fk_cell == s["cell"]]
            if s["col"] == "tech_c":
                d = d[d.zone != "short"]
                if d.tech.isna().all():
                    continue
            if (d[s["col"]] == s["a"]).sum() < 20 or (d[s["col"]] == s["b"]).sum() < 20:
                continue
            extra = ""
            if s["kind"] == "corner":
                d["side"] = (d.y < 40).astype(int); extra = " + side" + (" + C(comp)" if d.comp.nunique() > 1 else "")
            try:
                lr, se, na, nb, ma, mb = compare(d, s["col"], s["a"], s["b"], extra)
                ests[popn] = dict(lr=lr, se=se, na=na, nb=nb, ma=ma, mb=mb)
            except Exception as ex:
                ests[popn] = dict(error=str(ex))
        good = {k: v for k, v in ests.items() if "lr" in v}
        if good:
            w = np.array([1 / v["se"] ** 2 for v in good.values()]); l = np.array([v["lr"] for v in good.values()])
            pooled = float((w * l).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
            sgn = np.sign(s["lr"])
            p1 = float(norm.sf(sgn * pooled / pse))
            n_same = int(sum(np.sign(v["lr"]) == sgn for v in good.values()))
            confirmed = (p1 < 0.05) and (n_same >= min(2, len(good)))
        else:
            pooled = pse = p1 = np.nan; n_same = 0; confirmed = False
        conf.append(dict(item=s["item"], disc_ratio=float(np.exp(s["lr"])), pooled_ratio=float(np.exp(pooled)) if good else None, pooled_p_one=p1,
                         n_sets=len(good), n_same_sign=n_same, confirmed=bool(confirmed), per_set=ests))
        print(conf[-1]["item"], "| disc x%.2f | conf x%s p=%.4f sets %d same %d -> %s" % (np.exp(s["lr"]), round(np.exp(pooled), 2) if good else None, p1, len(good), n_same, confirmed))
    json.dump(conf, open(os.path.join(ROOT, "results", "r8_confirm.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    {"build": build, "audit": audit}[sys.argv[1]]()
