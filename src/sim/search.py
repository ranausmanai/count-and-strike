"""Tactic search for an underdog in the agent-based simulator.
Stage 1: 2,000 random tactics × 150 matches vs a strong team (default tactic).
Stage 2: top 40 re-evaluated on 2,000 fresh matches each (optimiser's-curse guard).
Stage 3: robustness of the top 10 against 4 alternative strong-team tactics."""
import json, os, time
import numpy as np
from multiprocessing import Pool
import engine

CAL = np.array(json.load(open(os.path.join(os.path.dirname(__file__), "calibration.json")))["cal"])
STRONG = np.array([0.97, 1.00, 1.00, 1.00, 1.00])
WEAK = np.array([0.955, 0.85, 0.85, 0.85, 0.95])
LO = np.array([15, 24, 0, 30, 0, 30, 0.0, 0.02, 0.3, 1]); HI = np.array([55, 56, 3.99, 105, 3.99, 64, 1.5, 0.15, 2.0, 4.99])
NAMES = ["def_line", "compact", "press_n", "press_zone", "runners", "width", "direct", "shoot_thr", "tempo", "rest"]
OPP = {"default": engine.default_tactic(),
       "high_press": np.array([50, 44, 3, 105, 1, 54, 0.6, 0.06, 0.7, 2]),
       "deep_block": np.array([22, 32, 1, 40, 1, 50, 0.6, 0.06, 1.2, 2]),
       "direct": np.array([38, 40, 1, 60, 2, 56, 1.4, 0.05, 0.6, 2]),
       "possession": np.array([45, 46, 2, 80, 0, 62, 0.2, 0.08, 1.6, 3])}


def evaluate(args):
    tac, opp, n, seed = args
    tac = np.floor(tac * 1) if False else tac
    r = engine.run(WEAK, STRONG, tac, opp, n, seed, CAL)
    gf, ga = r[:, 0], r[:, 1]
    pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
    return dict(pts=float(pts.mean()), pts_se=float(pts.std() / np.sqrt(n)), gd=float((gf - ga).mean()), win=float((gf > ga).mean()),
                draw=float((gf == ga).mean()), gf=float(gf.mean()), ga=float(ga.mean()), shots=float(r[:, 2].mean()), shots_ag=float(r[:, 3].mean()),
                counter_goals=float(r[:, 10].mean()), high_regains=float(r[:, 12].mean()))


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261015)
    base = evaluate((engine.default_tactic(), OPP["default"], 3000, 10_000_000))
    print("baseline (underdog plays the default tactic):", base, flush=True)
    tactics = LO + rng.random((2000, 10)) * (HI - LO)
    with Pool(8) as pool:
        s1 = pool.map(evaluate, [(tactics[i], OPP["default"], 150, 1_000_000 + i * 150) for i in range(len(tactics))], chunksize=4)
    pts1 = np.array([r["pts"] for r in s1])
    print(f"stage 1 done in {time.time()-t0:.0f}s; mean pts {pts1.mean():.3f}, best {pts1.max():.3f}", flush=True)
    top = np.argsort(-pts1)[:40]
    with Pool(8) as pool:
        s2 = pool.map(evaluate, [(tactics[i], OPP["default"], 2000, 50_000_000 + k * 2000) for k, i in enumerate(top)], chunksize=1)
    pts2 = np.array([r["pts"] for r in s2])
    order = np.argsort(-pts2)
    print(f"stage 2 done in {time.time()-t0:.0f}s; top-40 stage-1 mean {pts1[top].mean():.3f} -> fresh {pts2.mean():.3f} (shrinkage)", flush=True)
    best10 = [top[j] for j in order[:10]]
    with Pool(8) as pool:
        jobs = [(tactics[i], OPP[o], 1000, 90_000_000 + k * 10_000 + oi * 1000) for k, i in enumerate(best10) for oi, o in enumerate(OPP)]
        s3 = pool.map(evaluate, jobs, chunksize=1)
    rob = {}
    for k, i in enumerate(best10):
        rob[int(i)] = {o: s3[k * len(OPP) + oi]["pts"] for oi, o in enumerate(OPP)}
    base_rob = {}
    with Pool(5) as pool:
        bres = pool.map(evaluate, [(engine.default_tactic(), OPP[o], 2000, 95_000_000 + oi * 2000) for oi, o in enumerate(OPP)])
    base_rob = {o: bres[oi]["pts"] for oi, o in enumerate(OPP)}
    out = dict(baseline=base, baseline_vs_opponents=base_rob, names=NAMES,
               stage1=[dict(tactic=tactics[i].tolist(), **s1[i]) for i in range(len(tactics))],
               stage2=[dict(idx=int(top[j]), tactic=tactics[top[j]].tolist(), stage1_pts=float(pts1[top[j]]), **s2[j]) for j in order],
               robustness=rob)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "sim_search.json"), "w"), indent=1)
    print(f"total {time.time()-t0:.0f}s")
    print("baseline vs opponents:", {k: round(v, 3) for k, v in base_rob.items()})
    for j in order[:10]:
        i = top[j]
        print(round(s2[j]["pts"], 3), "±", round(s2[j]["pts_se"], 3), "gd", round(s2[j]["gd"], 2), dict(zip(NAMES, np.round(tactics[i], 2))), "| vs opps:", {k: round(v, 2) for k, v in rob[int(i)].items()})


if __name__ == "__main__":
    main()
