"""Round 24 (PREREGISTRATION_24.md): The Slingshot, evaluated across 8 physics variants x 5 opponent styles."""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import engine2

HERE = os.path.dirname(__file__)
CAL = np.array(json.load(open(os.path.join(HERE, "calibration.json")))["cal"])
STRONG = np.array([0.97, 1.00, 1.00, 1.00, 1.00]); WEAK = np.array([0.955, 0.85, 0.85, 0.85, 0.95])
D = engine2.default_tactic()


def t13(a):
    return np.concatenate([np.asarray(a, float), np.zeros(3)])


OPP = {"default": D.copy(), "high_press": t13([50, 44, 3, 105, 1, 54, 0.6, 0.06, 0.7, 2]), "deep_block": t13([22, 32, 1, 40, 1, 50, 0.6, 0.06, 1.2, 2]),
       "direct": t13([38, 40, 1, 60, 2, 56, 1.4, 0.05, 0.6, 2]), "possession": t13([45, 46, 2, 80, 0, 62, 0.2, 0.08, 1.6, 3])}
rng = np.random.default_rng(24)
PHYS = []
for k in range(8):
    c = CAL.copy()
    for i in (0, 2, 3, 6):
        c[i] *= rng.uniform(0.7, 1.3)
    c[4] += rng.uniform(-0.3, 0.3)
    PHYS.append(c)


def arm(**kw):
    t = D.copy()
    for k, v in kw.items():
        t[{"asym": 10, "release": 11, "escape": 12}[k]] = v
    return t


ARMS = {"A0_baseline": D.copy(), "A1_pull": arm(asym=0.8), "A2_pull_release": arm(asym=0.8, release=1.0),
        "A3_escape": arm(escape=1), "A4_slingshot": arm(asym=0.8, release=1.0, escape=1)}
LO = np.array([15, 24, 0, 30, 0, 30, 0.0, 0.02, 0.3, 1, 0, 0, 0]); HI = np.array([55, 56, 3.99, 105, 3.99, 64, 1.5, 0.15, 2.0, 4.99, 1, 1.5, 1.99])


def play(args):
    tac, opp, cal, n, seed = args
    r = engine2.run(WEAK, STRONG, tac, opp, n, seed, cal)
    gf, ga = r[:, 0], r[:, 1]
    pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0)).astype(float)
    return pts, gf, ga, r[:, 8]


def evaluate(tac, pool, n=600):
    jobs = [(tac, OPP[o], PHYS[p], n, 7_000_000 + 100_000 * p + 10_000 * oi) for p in range(8) for oi, o in enumerate(OPP)]
    return pool.map(play, jobs, chunksize=1)


def main():
    t0 = time.time(); res = {}
    with Pool(8) as pool:
        out = {a: evaluate(t, pool) for a, t in ARMS.items()}
        print("arms done", round(time.time() - t0), flush=True)
        # A5 free search
        cands = LO + np.random.default_rng(2401).random((600, 13)) * (HI - LO)
        s1 = pool.map(play, [(c, OPP["default"], CAL, 120, 3_000_000 + i * 120) for i, c in enumerate(cands)], chunksize=4)
        p1 = np.array([s[0].mean() for s in s1]); top = np.argsort(-p1)[:20]
        s2 = pool.map(play, [(cands[i], OPP["default"], CAL, 1000, 4_000_000 + k * 1000) for k, i in enumerate(top)], chunksize=1)
        best = cands[top[int(np.argmax([s[0].mean() for s in s2]))]]
        out["A5_search_best"] = evaluate(best, pool)
        res["A5_tactic"] = dict(zip(["def_line", "compact", "press_n", "press_zone", "runners", "width", "direct", "shoot_thr", "tempo", "rest", "asym", "release", "escape"], np.round(best, 2).tolist()))
        print("search done", round(time.time() - t0), flush=True)
    base = out["A0_baseline"]; keys = [(p, o) for p in range(8) for o in OPP]
    for a, R in out.items():
        g = np.array([R[k][0].mean() - base[k][0].mean() for k in range(len(keys))])
        diff = np.concatenate([R[k][0] - base[k][0] for k in range(len(keys))])
        by_phys = [float(g[[i for i, kk in enumerate(keys) if kk[0] == p]].mean()) for p in range(8)]
        by_opp = {o: float(g[[i for i, kk in enumerate(keys) if kk[1] == o]].mean()) for o in OPP}
        pts = np.concatenate([r[0] for r in R]); gf = np.concatenate([r[1] for r in R]); ga = np.concatenate([r[2] for r in R])
        bgf = np.concatenate([r[1] for r in base])
        d = dict(pts=float(pts.mean()), gain=float(diff.mean()), gain_ci=[float(diff.mean() - 1.96 * diff.std() / np.sqrt(len(diff))), float(diff.mean() + 1.96 * diff.std() / np.sqrt(len(diff)))],
                 win=float((gf > ga).mean()), gf=float(gf.mean()), ga=float(ga.mean()), goals_vs_base=float((gf.mean() + ga.mean()) / (bgf.mean() + np.concatenate([r[2] for r in base]).mean())),
                 phys_positive=int(sum(x > 0 for x in by_phys)), opp_positive=int(sum(v > 0 for v in by_opp.values())), by_phys=by_phys, by_opp=by_opp)
        d["exploit_flag"] = bool(d["win"] > 0.6 or d["gain"] > 0.5)
        if a == "A4_slingshot":
            d["passed"] = bool(d["phys_positive"] >= 7 and d["opp_positive"] >= 4 and d["gain"] >= 0.05 and d["gain_ci"][0] > 0 and 0.75 <= d["goals_vs_base"] <= 1.25)
        res[a] = d
        print(f"{a:16s} pts {d['pts']:.3f} gain {d['gain']:+.3f} win {d['win']:.2f} gf {d['gf']:.2f} ga {d['ga']:.2f} phys+ {d['phys_positive']}/8 opp+ {d['opp_positive']}/5 exploit={d['exploit_flag']}", flush=True)
    json.dump(res, open(os.path.join(HERE, "..", "..", "results", "r24_slingshot.json"), "w"), indent=1)
    print("total", round(time.time() - t0))


if __name__ == "__main__":
    main()
