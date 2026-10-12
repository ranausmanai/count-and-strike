"""Round 4: defensive structure from shot freeze frames (PREREGISTRATION_4.md)."""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "sb")
PROC = os.path.join(ROOT, "data", "proc")


def in_tri(p, a, b, c):
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def shots(fname):
    mid = int(fname.split(".")[0])
    ev = json.load(gzip.open(os.path.join(RAW, fname)))
    out = []
    for e in ev:
        if e["type"]["name"] != "Shot" or e["period"] > 4:
            continue
        s = e["shot"]
        if s["type"]["name"] == "Penalty" or "freeze_frame" not in s:
            continue
        x, y = e["location"][:2]
        opp = [f for f in s["freeze_frame"] if not f["teammate"]]
        gk = [f for f in opp if f["position"]["name"] == "Goalkeeper"]
        de = [f["location"] for f in opp if f["position"]["name"] != "Goalkeeper"]
        cone = sum(in_tri(l, (x, y), (120, 36), (120, 44)) for l in de)
        press = sum(np.hypot(l[0] - x, l[1] - y) <= 3 for l in de)
        if gk:
            g = gk[0]["location"]
            # distance from shooter-goal-centre line
            v = np.array([120 - x, 40 - y]); w = np.array([g[0] - x, g[1] - y])
            gk_off = abs(v[0] * w[1] - v[1] * w[0]) / (np.linalg.norm(v) + 1e-9)
            gk_depth = 120 - g[0]
        else:
            gk_off, gk_depth = np.nan, np.nan
        out.append(dict(match_id=mid, team=e["team"]["name"], period=e["period"], x=x, y=y, xg=s.get("statsbomb_xg"),
                        goal=int(s["outcome"]["name"] == "Goal"), cone_def=cone, press_3y=press, gk_off=gk_off, gk_depth=gk_depth,
                        n_def_ff=len(de)))
    return out


def build():
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(shots, sorted(os.listdir(RAW)), chunksize=16):
            rows += r
    d = pd.DataFrame(rows)
    seg = pd.read_parquet(os.path.join(PROC, "segments.parquet"), columns=["match_id", "team", "opp", "league", "home_team", "away_team"]).drop_duplicates(["match_id", "team"])
    # event team names may differ from canonical; map by match
    names = seg.groupby("match_id").team.unique().to_dict()
    import difflib
    d["team"] = [t if t in names.get(m, []) else difflib.get_close_matches(t, list(names.get(m, [t])), 1, 0)[0] for m, t in zip(d.match_id, d.team)]
    d = d.merge(seg[["match_id", "team", "opp", "league"]], on=["match_id", "team"])
    d.to_parquet(os.path.join(PROC, "r4_shots.parquet"))
    return d


def split_half_team(df, team_col, val_fn, min_n=20):
    rows = []
    for tm, g in df.groupby(team_col):
        ms = sorted(g.match_id.unique())
        if len(ms) < min_n:
            continue
        a, b = g[g.match_id.isin(ms[0::2])], g[g.match_id.isin(ms[1::2])]
        rows.append((tm, val_fn(a), val_fn(b), val_fn(g)))
    r = pd.DataFrame(rows, columns=["team", "a", "b", "full"])
    c = r[["a", "b"]].corr().iloc[0, 1]
    return float(2 * c / (1 + c)), r


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    d = build() if not os.path.exists(os.path.join(PROC, "r4_shots.parquet")) else pd.read_parquet(os.path.join(PROC, "r4_shots.parquet"))
    d = d.dropna(subset=["xg", "gk_off", "gk_depth"])
    d["lx"] = np.log(d.xg.clip(1e-4, 1 - 1e-4) / (1 - d.xg.clip(1e-4, 1 - 1e-4)))
    feats = ["lx", "cone_def", "press_3y", "gk_off", "gk_depth"]
    tr, te = d[d.league != "EPL"], d[d.league == "EPL"]
    m0 = LogisticRegression(C=1e6, max_iter=1000).fit(tr[["lx"]], tr.goal)
    m1 = LogisticRegression(C=1e6, max_iter=1000).fit(tr[feats], tr.goal)
    p0, p1 = m0.predict_proba(te[["lx"]])[:, 1], m1.predict_proba(te[feats])[:, 1]
    l0 = -(te.goal * np.log(p0) + (1 - te.goal) * np.log(1 - p0)); l1 = -(te.goal * np.log(p1) + (1 - te.goal) * np.log(1 - p1))
    per = pd.DataFrame({"m": te.match_id, "g": (l0 - l1).values}).groupby("m").g.sum()
    rng = np.random.default_rng(20261012)
    bs = [per.values[rng.integers(0, len(per), len(per))].sum() for _ in range(2000)]
    out = {"H4_1": dict(n_train=len(tr), n_test=len(te), logloss_gain_total=float(per.sum()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        coefs=dict(zip(feats, map(float, m1.coef_[0]))))}
    print("H4-1", out["H4_1"])
    # H4-2: defending team = opp
    tmrat = pd.read_parquet(os.path.join(PROC, "team_match_channels.parquet"))
    rating = tmrat.groupby("team").apply(lambda g: np.mean(np.log(g.lam_own / g.lam_opp)))
    h42 = {}
    for f in ("cone_def", "press_3y"):
        rel, r = split_half_team(d, "opp", lambda g, f=f: g[f].mean())
        r = r.merge(rating.rename("rating"), left_on="team", right_index=True)
        h42[f] = dict(reliability=rel, wealth_r=float(r[["full", "rating"]].corr().iloc[0, 1]), n_teams=len(r))
    # does structure lower xG per shot conceded? (descriptive) and how much variation does it explain at team level
    rel_xgps, r = split_half_team(d, "opp", lambda g: g.xg.mean())
    h42["xg_per_shot_conceded"] = dict(reliability=rel_xgps)
    out["H4_2"] = h42; print("H4-2", h42)
    # H4-3: goals - xG conceded per match, SB and WY
    sb_tm = tmrat.copy()
    gl = pd.read_parquet(os.path.join(PROC, "segments.parquet"), columns=["match_id", "team", "home", "home_score", "away_score"]).drop_duplicates(["match_id", "team"])
    gl["ga"] = np.where(gl.home == 1, gl.away_score, gl.home_score)
    sb_tm = sb_tm.merge(gl[["match_id", "team", "ga"]], on=["match_id", "team"])
    sb_tm["over"] = sb_tm.ga - (sb_tm.xg_total_ag)
    rel_sb, _ = split_half_team(sb_tm, "team", lambda g: g.over.mean())
    # WY: need goals; recompute from events quickly via team_match table merged with matches json
    wy = pd.read_parquet(os.path.join(PROC, "wy_team_match_channels.parquet"))
    WY = os.path.join(ROOT, "data", "raw", "wy"); sc = {}
    for c in ("England", "Spain", "Germany", "Italy", "France"):
        for m in json.load(open(os.path.join(WY, f"matches_{c}.json"))):
            lab, res_ = m["label"].split(",")
            h, a = [x.strip() for x in lab.split(" - ")]; hg, ag = [int(x) for x in res_.strip().split(" - ")]
            sc[(m["wyId"], h)] = ag; sc[(m["wyId"], a)] = hg
    wy["ga"] = [sc.get((m, t), np.nan) for m, t in zip(wy.match_id, wy.team)]
    wy["over"] = wy.ga - wy.xg_total_ag
    wy = wy.dropna(subset=["ga"])
    # WY xG is location-only and under-predicts; demean by league so scale differences cancel
    wy["over"] = wy.over - wy.groupby("league").over.transform("mean")
    rel_wy, _ = split_half_team(wy.assign(team=wy.league + "|" + wy.team), "team", lambda g: g.over.mean())
    out["H4_3"] = dict(reliability_sb=rel_sb, reliability_wy=rel_wy)
    print("H4-3", out["H4_3"])
    json.dump(out, open(os.path.join(ROOT, "results", "r4_structure.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
