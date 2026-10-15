"""Round 18A: play into the overload (PREREGISTRATION_18.md)."""
import gzip, json, os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import tsec, windows, RAW, DISC, WOMEN, RESTART
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")


def lane(y):
    return 0 if y < 26.7 else (2 if y > 53.3 else 1)


def one(m):
    mid = m["match_id"]
    try:
        ev = [e for e in json.load(gzip.open(os.path.join(RAW, f"{mid}_events.json.gz"))) if e["period"] <= 2]
        fr = {f["event_uuid"]: f for f in json.load(gzip.open(os.path.join(RAW, f"{mid}_three-sixty.json.gz")))}
    except Exception:
        return None
    S = pd.DataFrame([(e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
                      if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"], columns=["period", "t", "team", "xg"])
    rows = []
    for e in ev:
        if e["type"]["name"] != "Pass" or e["team"]["name"] != e["possession_team"]["name"]:
            continue
        p = e["pass"]
        if p.get("type", {}).get("name") in RESTART or "location" not in e:
            continue
        x, y = e["location"][:2]
        if x < 40:
            continue
        f = fr.get(e["id"])
        if f is None:
            continue
        ex, ey = p["end_location"][:2]
        mates = [q["location"] for q in f["freeze_frame"] if q["teammate"] and not q.get("actor") and not q.get("keeper")]
        opps = [q["location"] for q in f["freeze_frame"] if not q["teammate"] and not q.get("keeper")]
        ov = [0, 0, 0]
        for q in mates:
            if q[0] > x - 5:
                ov[lane(q[1])] += 1
        for q in opps:
            if q[0] > x - 5:
                ov[lane(q[1])] -= 1
        d_opp = [np.hypot(q[0] - x, q[1] - y) for q in opps]
        lt, ls = lane(ey), lane(y)
        rows.append(dict(match_id=mid, comp=m["_comp"], period=e["period"], t=tsec(e), team=e["team"]["name"], x=x, y=y,
                         target_ov=ov[lt], start_ov=ov[ls], max_ov=max(ov), min_ov=min(ov), lane_change=abs(lt - ls),
                         opp_goalside=sum(q[0] > x for q in opps), nearest_opp=min(d_opp) if d_opp else 30.0,
                         complete="outcome" not in p))
    return windows(rows, S)


def main():
    cache = os.path.join(PROC, "r18a_passes.parquet")
    if not os.path.exists(cache):
        ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
        out = []
        with ProcessPoolExecutor(8) as ex:
            for r in ex.map(one, ms, chunksize=4):
                if r is not None and len(r):
                    out.append(r)
        D = pd.concat(out, ignore_index=True)
        D["pop"] = np.select([D.comp.str.startswith(DISC), D.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
        D["zx"] = pd.cut(D.x, [39, 60, 80, 96, 108, 121], labels=False)
        D["zy"] = np.select([(D.y - 40).abs() <= 13, D.y < 40], [1, 0], 2)
        D["cell"] = (D.zx.astype(int).astype(str) + D.zy.astype(str) + np.select([D.opp_goalside <= 3, D.opp_goalside <= 6], ["0", "1"], "2")
                     + np.select([D.nearest_opp < 3, D.nearest_opp <= 7], ["0", "1"], "2"))
        D.to_parquet(cache)
    D = pd.read_parquet(cache)
    print(D.groupby("pop").size().to_dict(), "| target_ov distribution:", D.target_ov.clip(-3, 3).value_counts().sort_index().to_dict())
    res = {}
    def within_cell_diff(d, mask_a, mask_b):
        """Cell-matched difference in mean net xG (a − b), weighted by min count; SE from cell-level variances."""
        a = d[mask_a].groupby("cell").net.agg(["mean", "var", "size"]); b = d[mask_b].groupby("cell").net.agg(["mean", "var", "size"])
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner"); j = j[(j.size_a >= 10) & (j.size_b >= 10)]
        w = np.minimum(j.size_a, j.size_b)
        diff = ((j.mean_a - j.mean_b) * w).sum() / w.sum()
        var = ((w / w.sum()) ** 2 * (j.var_a / j.size_a + j.var_b / j.size_b)).sum()
        return float(diff), float(np.sqrt(var)), int(len(j))
    tests = {
        "H18_1_overload_vs_underload": lambda d: within_cell_diff(d, d.target_ov >= 1, d.target_ov <= -1),
        "H18_2_switch_to_overload_vs_stay_in_underload": lambda d: within_cell_diff(d, (d.lane_change == 2) & (d.target_ov >= 1), (d.lane_change == 0) & (d.target_ov <= -1)),
    }
    for name, fn in tests.items():
        r = {p: fn(D[D["pop"] == p]) for p in ("discovery", "confirm_club", "confirm_women")}
        (dd, sd, _), c1, c2 = r["discovery"], r["confirm_club"], r["confirm_women"]
        w = np.array([1 / c1[1] ** 2, 1 / c2[1] ** 2]); pooled = (w[0] * c1[0] + w[1] * c2[0]) / w.sum(); pse = 1 / np.sqrt(w.sum())
        res[name] = dict(per_pop={p: dict(diff=v[0], se=v[1], cells=v[2]) for p, v in r.items()}, disc_p_one=float(norm.sf(dd / sd)),
                         pooled_confirm=float(pooled), pooled_se=float(pse), confirm_p_one=float(norm.sf(pooled / pse)),
                         confirmed=bool(dd > 0 and norm.sf(dd / sd) < 0.05 and pooled > 0 and norm.sf(pooled / pse) < 0.05))
        print(name, json.dumps(res[name], indent=0, default=float))
    # H18-3 dose-response with cell FE
    dr = {}
    for p in ("discovery", "confirm_club", "confirm_women"):
        d = D[D["pop"] == p].copy(); d["tov"] = d.target_ov.clip(-4, 4)
        d["net_dm"] = d.net - d.groupby("cell").net.transform("mean"); d["tov_dm"] = d.tov - d.groupby("cell").tov.transform("mean")
        m = smf.ols("net_dm ~ tov_dm - 1", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
        dr[p] = dict(per_player=float(m.params["tov_dm"]), se=float(m.bse["tov_dm"]), p_one=float(norm.sf(m.params["tov_dm"] / m.bse["tov_dm"])))
    res["H18_3_dose_response"] = dr
    print("H18-3", json.dumps(dr, indent=0, default=float))
    # descriptive: how often teams pass into overload vs underload when both exist
    both = D[(D.max_ov >= 1) & (D.min_ov <= -1)]
    res["usage_when_choice_exists"] = dict(share_into_overload=float((both.target_ov >= 1).mean()), share_into_underload=float((both.target_ov <= -1).mean()), n=int(len(both)))
    print("usage", res["usage_when_choice_exists"])
    json.dump(res, open(os.path.join(ROOT, "results", "r18a_overload.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
