"""Round 25b (post-hoc, labelled): Escape alone vs baseline in the calibrated engine4, 5 opponents x 6 physics x 600 matches, common seeds."""
import json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from r25_grid import play, OPP, PHYS, D
HERE = os.path.dirname(__file__)
SK = np.array(json.load(open(os.path.join(HERE, "..", "..", "results", "r25_grid.json")))["underdog_skills"])
E = D.copy(); E[12] = 1.0
if __name__ == "__main__":
    keys = [(o, p) for o in OPP for p in range(6)]
    with Pool(8) as pool:
        R = {a: pool.map(play, [(SK, t, OPP[o], PHYS[p], 600, 20_000_000 + 10_000 * p + 100_000 * oi) for oi, o in enumerate(OPP) for p in range(6)], chunksize=1) for a, t in (("base", D), ("escape", E))}
    g = np.array([R["escape"][k][0].mean() - R["base"][k][0].mean() for k in range(len(keys))])
    d = np.concatenate([R["escape"][k][0] - R["base"][k][0] for k in range(len(keys))])
    by_opp = {o: float(g[[i for i, kk in enumerate(keys) if kk[0] == o]].mean()) for o in OPP}
    by_phys = [float(g[[i for i, kk in enumerate(keys) if kk[1] == p]].mean()) for p in range(6)]
    win = lambda a: float(np.mean(np.concatenate([(x[1] > x[2]) for x in R[a]])))
    gl = lambda a: float(np.mean(np.concatenate([x[1] + x[2] for x in R[a]])))
    out = dict(gain=float(d.mean()), ci=[float(d.mean() - 1.96 * d.std() / np.sqrt(len(d))), float(d.mean() + 1.96 * d.std() / np.sqrt(len(d)))],
               by_opp=by_opp, by_phys=by_phys, win_base=win("base"), win_escape=win("escape"), goals_base=gl("base"), goals_escape=gl("escape"),
               pts_base=float(np.mean(np.concatenate([x[0] for x in R["base"]]))))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "..", "..", "results", "r25b_escape.json"), "w"), indent=1)
