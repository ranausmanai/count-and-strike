"""Test the locked Count & Strike predictions (PREDICTIONS.md) on NEW StatsBomb 360 data.

usage: python3 src/test_predictions.py <competition_id> <season_id> [<competition_id> <season_id> ...]
Downloads events + 360 frames for those competitions into data/raw/sb360_test/ and evaluates every prediction.
"""
import gzip, json, os, sys, urllib.request
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import r13_360, r15_360
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "raw", "sb360_test")
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
PRED = json.load(open(os.path.join(ROOT, "predictions.json")))


def fetch(pairs):
    os.makedirs(OUT, exist_ok=True)
    ms = []
    for c, s in pairs:
        for m in json.loads(urllib.request.urlopen(f"{BASE}/matches/{c}/{s}.json").read()):
            m["_comp"] = f"test {c}/{s}"; ms.append(m)
    def one(m):
        for kind in ("events", "three-sixty"):
            p = os.path.join(OUT, f"{m['match_id']}_{kind}.json.gz")
            if not os.path.exists(p):
                try:
                    raw = urllib.request.urlopen(f"{BASE}/{kind}/{m['match_id']}.json", timeout=90).read()
                    with gzip.open(p, "wb") as f: f.write(raw)
                except Exception:
                    pass
    with ThreadPoolExecutor(16) as ex: list(ex.map(one, ms))
    return ms


def main():
    a = sys.argv[1:]; pairs = list(zip(a[0::2], a[1::2]))
    ms = fetch(pairs)
    r13_360.RAW = OUT; r15_360.RAW = OUT
    att = [r for r in map(r15_360.one, ms) if r is not None]
    A = r15_360.cellify(pd.concat([x[0] for x in att], ignore_index=True))
    A["cell"] = (A.zx.astype(int).astype(str) + A.zy.astype(str) + np.select([A.opp_goalside <= 3, A.opp_goalside <= 6], ["0", "1"], "2")
                 + np.select([A.nearest_opp < 3, A.nearest_opp <= 7], ["0", "1"], "2") + np.select([A.mates_ahead <= 1, A.mates_ahead <= 3], ["0", "1"], "2"))
    R = pd.concat([x for x in map(r13_360.one, ms) if x is not None], ignore_index=True)
    print(f"test data: {len(ms)} matches, {len(A)} decisions")
    results = []
    for p in PRED["predictions"]:
        if p["type"] == "count":
            d = R[R.regain_start & (R.x < 80)].copy()
            d["rx"] = d.x / 120; d["cent"] = 1 - (d.y - 40).abs() / 40; d["clk"] = (d.t / 60 + np.where(d.period == 2, 45, 0)) / 90
            d["gpos"] = (d.gd > 0).astype(float); d["gneg"] = (d.gd < 0).astype(float)
            m = smf.glm("own20 ~ rx + I(rx**2) + I(rx**3) + cent + gpos + gneg + clk + opp_goalside + n_visible_opp", data=d,
                        family=sm.families.Poisson()).fit(scale="X2", cov_type="cluster", cov_kwds={"groups": pd.factorize(d.match_id)[0]})
            est = float(np.exp(m.params["opp_goalside"])); p1 = float(norm.cdf(m.params["opp_goalside"] / m.bse["opp_goalside"]))
            ok = p1 < 0.05 and p["range"][0] <= est <= p["range"][1]
        else:
            d = A[A.cell == p["cell"]]; x, y = d[d.action == p["better"]].net, d[d.action == p["worse"]].net
            if len(x) < 10 or len(y) < 10:
                results.append(dict(id=p["id"], verdict="INSUFFICIENT DATA", n=[len(x), len(y)])); continue
            est = float(x.mean() - y.mean()); se = float(np.sqrt(x.var() / len(x) + y.var() / len(y))); p1 = float(norm.sf(est / se))
            ok = p1 < 0.05 and p["range"][0] <= est <= p["range"][1]
        results.append(dict(id=p["id"], estimate=est, p_one_sided=p1, predicted_range=p["range"], verdict="HELD" if ok else "FAILED"))
        print(f"{p['id']:4s} {p['rule'][:70]:70s} est {est:+.4f}  p={p1:.4f}  range {p['range']} -> {results[-1]['verdict']}")
    held = sum(r["verdict"] == "HELD" for r in results)
    print(f"\n{held}/{len(results)} predictions held. Success criterion: {PRED['success_criterion']}")
    json.dump(results, open(os.path.join(ROOT, "results", "prediction_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
