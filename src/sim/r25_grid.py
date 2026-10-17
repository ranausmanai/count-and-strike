"""Round 25 (PREREGISTRATION_25.md): grid search for an underdog in the realism-calibrated simulator (engine4)."""
import itertools, json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import engine4 as e

HERE = os.path.dirname(__file__)
CAL = np.array(json.load(open(os.path.join(HERE, "calibration4.json")))["cal"])
STRONG = np.array([0.97, 1.0, 1.0, 1.0, 1.0])
NAMES = ["def_line", "compact", "press_n", "press_zone", "runners", "width", "direct", "shoot_thr", "tempo", "rest", "asym", "release", "escape", "far_cover"]
LEVELS = [[25, 38, 50], [30, 40, 52], [0, 1, 2, 3], [45, 70, 105], [0, 1, 2], [40, 52, 64], [0.2, 0.7, 1.3], [0.03, 0.06, 0.11],
          [0.5, 1.0, 1.6], [1, 2, 3], [0.0, 0.4, 0.8], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]


def tac(v):
    return np.array(v, float)


D = e.default_tactic(); D[13] = 1.0


def t14(a):
    return np.concatenate([np.asarray(a, float), [0, 0, 0, 1.0]])


OPP = {"default": D.copy(), "high_press": t14([50, 44, 3, 105, 1, 54, 0.6, 0.06, 0.7, 2]), "deep_block": t14([22, 32, 1, 40, 1, 50, 0.6, 0.06, 1.2, 2]),
       "direct": t14([38, 40, 1, 60, 2, 56, 1.4, 0.05, 0.6, 2]), "possession": t14([45, 46, 2, 80, 0, 62, 0.2, 0.08, 1.6, 3])}
rng0 = np.random.default_rng(25)
PHYS = []
for k in range(6):
    c = CAL.copy()
    for i in (0, 2, 3, 6):
        c[i] *= rng0.uniform(0.8, 1.2)
    c[4] += rng0.uniform(-0.2, 0.2)
    PHYS.append(c)


def weak(s):
    return np.array([0.97 - 0.015 * (1 - s) / 0.15, s, s, s, 0.95 + 0.05 * (s - 0.85) / 0.15])


def play(args):
    sk, t, opp, cal, n, seed = args
    r = e.run(sk, STRONG, t, opp, n, seed, cal)
    gf, ga = r[:, 0], r[:, 1]
    return np.where(gf > ga, 3, np.where(gf == ga, 1, 0)).astype(float), gf, ga, (r[:, 8] / np.maximum(r[:, 18] * 0.5, 1))


def summary(R):
    p = np.concatenate([x[0] for x in R]); gf = np.concatenate([x[1] for x in R]); ga = np.concatenate([x[2] for x in R])
    return dict(pts=float(p.mean()), se=float(p.std() / np.sqrt(len(p))), win=float((gf > ga).mean()), gf=float(gf.mean()), ga=float(ga.mean()))


def main():
    t0 = time.time(); out = {}
    with Pool(8) as pool:
        # 1. calibrate underdog strength to ~0.9 pts/match vs default
        lo, hi = 0.6, 1.0
        for _ in range(7):
            s = (lo + hi) / 2
            R = pool.map(play, [(weak(s), D, OPP["default"], CAL, 100, 500_000 + k * 100) for k in range(8)])
            pts = summary(R)["pts"]
            lo, hi = (s, hi) if pts < 0.9 else (lo, s)
            print(f"skill {s:.3f} -> {pts:.3f}", flush=True)
        SK = weak((lo + hi) / 2); out["underdog_skills"] = SK.tolist()
        base = summary(pool.map(play, [(SK, D, OPP["default"], CAL, 250, 600_000 + k * 250) for k in range(8)])); out["baseline"] = base
        print("baseline", base, flush=True)
        # 2. grid sample
        full = list(itertools.product(*[range(len(l)) for l in LEVELS]))
        idx = np.random.default_rng(2501).choice(len(full), 3000, replace=False)
        grid = [tac([LEVELS[j][full[i][j]] for j in range(14)]) for i in idx]
        s1 = pool.map(play, [(SK, g, OPP["default"], CAL, 100, 1_000_000 + k * 100) for k, g in enumerate(grid)], chunksize=8)
        p1 = np.array([x[0].mean() for x in s1]); print(f"stage1 {time.time()-t0:.0f}s mean {p1.mean():.3f} max {p1.max():.3f}", flush=True)
        # knob importance (partial dependence)
        X = np.array([[full[i][j] for j in range(14)] for i in idx])
        out["partial_dependence"] = {NAMES[j]: {str(LEVELS[j][l]): float(p1[X[:, j] == l].mean()) for l in range(len(LEVELS[j]))} for j in range(14)}
        top = np.argsort(-p1)[:50]
        s2 = pool.map(play, [(SK, grid[i], OPP["default"], CAL, 1000, 5_000_000 + k * 1000) for k, i in enumerate(top)], chunksize=1)
        p2 = np.array([x[0].mean() for x in s2]); order = np.argsort(-p2)
        print(f"stage2 {time.time()-t0:.0f}s top50 stage1 {p1[top].mean():.3f} -> fresh {p2.mean():.3f}", flush=True)
        best10 = [top[j] for j in order[:10]]
        # 3. robustness: baseline and top 10 across opponents x physics, common seeds
        keys = [(o, p) for o in OPP for p in range(6)]
        def grid_eval(t):
            return pool.map(play, [(SK, t, OPP[o], PHYS[p], 300, 9_000_000 + 10_000 * pi + 100_000 * oi) for oi, o in enumerate(OPP) for pi, p in enumerate(range(6))], chunksize=1)
        B = grid_eval(D)
        bgoals = np.mean([x[1].mean() + x[2].mean() for x in B])
        res = []
        for i in best10:
            R = grid_eval(grid[i])
            gain = np.array([R[k][0].mean() - B[k][0].mean() for k in range(len(keys))])
            diff = np.concatenate([R[k][0] - B[k][0] for k in range(len(keys))])
            by_opp = {o: float(gain[[k for k, kk in enumerate(keys) if kk[0] == o]].mean()) for o in OPP}
            by_phys = [float(gain[[k for k, kk in enumerate(keys) if kk[1] == p]].mean()) for p in range(6)]
            sm = summary(R); goals = sm["gf"] + sm["ga"]
            d = dict(tactic=dict(zip(NAMES, grid[i].tolist())), stage1=float(p1[i]), fresh=float(p2[list(top).index(i)]), robust=sm,
                     gain=float(diff.mean()), gain_ci=[float(diff.mean() - 1.96 * diff.std() / np.sqrt(len(diff))), float(diff.mean() + 1.96 * diff.std() / np.sqrt(len(diff)))],
                     by_opp=by_opp, by_phys=by_phys, opp_pos=int(sum(v > 0 for v in by_opp.values())), phys_pos=int(sum(v > 0 for v in by_phys)),
                     goals_ratio=float(goals / bgoals))
            d["exploit"] = bool(d["gain"] > 0.5 or sm["win"] > 0.55 or not (0.6 <= d["goals_ratio"] <= 1.4))
            d["supported"] = bool(d["phys_pos"] >= 5 and d["opp_pos"] >= 4 and d["gain_ci"][0] > 0 and not d["exploit"])
            res.append(d)
            print(f"gain {d['gain']:+.3f} [{d['gain_ci'][0]:+.3f},{d['gain_ci'][1]:+.3f}] win {sm['win']:.2f} goals x{d['goals_ratio']:.2f} opp+{d['opp_pos']} phys+{d['phys_pos']} exploit={d['exploit']} supported={d['supported']} | " +
                  ", ".join(f"{k}={v:g}" for k, v in d["tactic"].items()), flush=True)
        out["baseline_robust"] = summary(B); out["top10"] = res
    json.dump(out, open(os.path.join(HERE, "..", "..", "results", "r25_grid.json"), "w"), indent=1)
    print("total", round(time.time() - t0))


if __name__ == "__main__":
    main()
