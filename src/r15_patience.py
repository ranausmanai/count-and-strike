"""Robustness: does patience pay later? Recompute shot-vs-recycle rules with 45 s and 60 s windows (own − opp xG)."""
import gzip, json, os, sys, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import tsec, RAW
ROOT = os.path.join(os.path.dirname(__file__), ".."); PROC = os.path.join(ROOT, "data", "proc")
def shots(m):
    try:
        ev = json.load(gzip.open(os.path.join(RAW, f"{m}_events.json.gz")))
    except Exception:
        return None
    return pd.DataFrame([(m, e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
                         if e["type"]["name"] == "Shot" and e["period"] <= 2 and e["shot"]["type"]["name"] != "Penalty"], columns=["match_id", "period", "t", "team", "xg"])
def main():
    A = pd.read_parquet(os.path.join(PROC, "r15_attack.parquet"))
    R = [r for r in json.load(open(os.path.join(ROOT, "results", "r15_rules.json")))["confirmed_playbook"] if r["family"] == "attack" and "shot" in (r["better"],)]
    cells = {r["cell"] for r in R}
    D = A[A.cell.isin(cells)].copy()
    with ProcessPoolExecutor(8) as ex:
        S = pd.concat([s for s in ex.map(shots, D.match_id.unique()) if s is not None])
    Sg = {k: g for k, g in S.groupby(["match_id", "period"])}
    out = []
    for W in (20, 45, 60):
        net = np.zeros(len(D))
        for i, r in enumerate(D.itertuples()):
            g = Sg.get((r.match_id, r.period))
            if g is None: continue
            w = g[(g.t >= r.t - 0.01) & (g.t <= r.t + W)]
            net[i] = w.xg[w.team == r.team].sum() - w.xg[(w.team != r.team) & (w.t > r.t)].sum()
        D[f"net{W}"] = net
    for r in R:
        row = dict(rule=f"{r['situation']} -> SHOT not {r['worse']}")
        for W in (20, 45, 60):
            ests = []
            for p in ("discovery", "confirm_club", "confirm_women"):
                d = D[(D["pop"] == p) & (D.cell == r["cell"])]
                a, b = d[d.action == "shot"][f"net{W}"], d[d.action == r["worse"]][f"net{W}"]
                ests.append((a.mean() - b.mean(), np.sqrt(a.var() / len(a) + b.var() / len(b))))
            w = np.array([1 / e[1] ** 2 for e in ests]); est = float((w * np.array([e[0] for e in ests])).sum() / w.sum()); se = float(1 / np.sqrt(w.sum()))
            row[f"W{W}"] = dict(adv=round(est, 4), se=round(se, 4), all_positive=bool(all(e[0] > 0 for e in ests)))
        out.append(row); print(json.dumps(row))
    json.dump(out, open(os.path.join(ROOT, "results", "r15_patience.json"), "w"), indent=1)


if __name__ == '__main__':
    main()
