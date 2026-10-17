"""Round 25 calibration, stage 2: local refinement from calibration4.json with bigger samples (80 matches/eval), 4 parallel candidates per step."""
import json, time
import numpy as np
from multiprocessing import Pool
from calibrate4 import measure, loss
lo = np.array([0.1, 0.6, 0.5, 0.005, -5.5, 0.4, 0.0, 0.4]); hi = np.array([0.8, 2.5, 8.0, 0.10, -2.0, 1.2, 25.0, 2.2])


def ev(args):
    cal, seed = args
    r = measure(cal, n=80, seed=seed); return loss(r), r


if __name__ == "__main__":
    J = json.load(open("calibration4.json")); best_cal = np.array(J["cal"]); rng = np.random.default_rng(251)
    with Pool(6) as pool:
        bl, br = ev((best_cal, 11)); print("start", round(bl, 4), {k: round(v, 3) for k, v in br.items()}, flush=True)
        for it in range(40):
            cands = [np.clip(best_cal + rng.normal(0, 0.03, 8) * (hi - lo), lo, hi) for _ in range(6)]
            res = pool.map(ev, [(c, 11) for c in cands])
            i = int(np.argmin([r[0] for r in res]))
            if res[i][0] < bl:
                bl, br, best_cal = res[i][0], res[i][1], cands[i]; print(it, round(bl, 4), {k: round(v, 3) for k, v in br.items()}, flush=True)
        chk = pool.map(ev, [(best_cal, 1000 + s) for s in range(6)])
    m = {k: float(np.mean([c[1][k] for c in chk])) for k in chk[0][1]}
    print("check 480 fresh matches:", {k: round(v, 3) for k, v in m.items()})
    json.dump(dict(cal=best_cal.tolist(), stats=br, check=m, loss=bl, target=J["target"], note="engine4, both teams far_cover=1"), open("calibration4.json", "w"), indent=1)
