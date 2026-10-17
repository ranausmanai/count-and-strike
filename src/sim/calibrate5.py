"""Round 26: effect-calibrated simulator. Fit 10 params to averages + null-effect anchors (PREREGISTRATION_26.md)."""
import json, time
import numpy as np
from multiprocessing import Pool
import engine5 as e
TARGET = dict(goals=1.35, shots=12.5, completion=0.83, tackles=45, counter_ratio=2.0)
SK = np.array([0.97, 1.0, 1.0, 1.0, 1.0])
B = e.default_tactic(); B[13] = 1.0
P0 = B.copy(); P0[2] = 0; P2 = B.copy(); P2[2] = 2; P2[3] = 105; P0[3] = 105
BAIT = B.copy(); BAIT[10] = 0.8; BAIT[11] = 1.0
NOFC = B.copy(); NOFC[13] = 0.0
lo = np.array([0.1, 0.6, 0.5, 0.005, -5.5, 0.4, 0.0, 0.4, 1.0, 4.0]); hi = np.array([0.8, 2.5, 8.0, 0.10, -2.0, 3.0, 25.0, 2.2, 1.5, 20.0])


def pts(r):
    return np.where(r[:, 0] > r[:, 1], 3, np.where(r[:, 0] == r[:, 1], 1, 0)).astype(float)


def chunk(args):
    cal, tA, n, seed = args
    return e.run(SK, SK, tA, B, n, seed, cal)


def measure(cal, pool, n=200, seed=7):
    jobs = []
    for arm in (B, P0, P2, BAIT, NOFC):
        jobs += [(cal, arm, n // 4, seed + k * 1000) for k in range(4)]
    R = pool.map(chunk, jobs, chunksize=1)
    arms = [np.concatenate(R[i * 4:(i + 1) * 4]) for i in range(5)]
    m = arms[0].mean(0)
    lo3 = m[23] / max(m[20], 1e-9); hi8 = m[25] / max(m[22], 1e-9)
    avg = dict(goals=(m[0] + m[1]) / 2, shots=(m[2] + m[3]) / 2, completion=(m[8] + m[9]) / max(m[18], 1), tackles=m[19] / 2, counter_ratio=lo3 / max(hi8, 1e-4))
    anc = dict(N1_press=float(pts(arms[2]).mean() - pts(arms[1]).mean()), N2_bait=float(pts(arms[3]).mean() - pts(arms[0]).mean()),
               N3_farcover=float(pts(arms[0]).mean() - pts(arms[4]).mean()))
    return avg, anc


HARD = ("goals", "shots", "completion", "tackles")


def loss(avg, anc):
    """Hard requirement: goals, shots, completion and tackles within +-20% of target; only then do the null anchors count."""
    a = sum(((avg[k] - v) / v) ** 2 for k, v in TARGET.items() if k != "counter_ratio") + 0.25 * ((avg["counter_ratio"] - 2.0) / 2.0) ** 2
    if any(abs(avg[k] / TARGET[k] - 1) > 0.2 for k in HARD):
        return 100.0 + a
    return a + sum((v / 0.12) ** 2 for v in anc.values())


if __name__ == "__main__":
    rng = np.random.default_rng(26); t0 = time.time()
    best_cal = np.array(json.load(open("calibration4.json"))["cal"] + [1.25, 12.0]); best_cal[5] = 1.9
    with Pool(8) as pool:
        avg, anc = measure(best_cal, pool); bl = loss(avg, anc)
        print("start", round(bl, 3), {k: round(v, 3) for k, v in {**avg, **anc}.items()}, flush=True)
        for it in range(400):
            sd = 0.2 if it < 150 else 0.06
            cal = np.clip(best_cal + rng.normal(0, sd, 10) * (hi - lo), lo, hi)
            avg, anc = measure(cal, pool); l = loss(avg, anc)
            if l < bl:
                bl, best_cal = l, cal; print(it, round(l, 3), {k: round(v, 3) for k, v in {**avg, **anc}.items()}, f"{time.time()-t0:.0f}s", flush=True)
        avg, anc = measure(best_cal, pool, n=1200, seed=999)
    print("CHECK (1200 fresh per arm):", {k: round(v, 3) for k, v in {**avg, **anc}.items()}, "loss", round(loss(avg, anc), 3))
    json.dump(dict(cal=best_cal.tolist(), check_avg=avg, check_anchors=anc, loss=loss(avg, anc), target=TARGET), open("calibration5.json", "w"), indent=1)
