"""Round 25 realism calibration for engine4 (both teams mark the far side). Targets come from real data:
goals/shots/completion/tackles per team-match, and the counter gradient (xG within 15 s per middle-third regain: <=3 back vs 8+ back = 2.0x, r19)."""
import json, time
import numpy as np
import engine4 as engine
TARGET = dict(goals=1.35, shots=12.5, completion=0.83, tackles=45, counter_ratio=2.0)
sk = np.array([0.97, 1.0, 1.0, 1.0, 1.0]); t = engine.default_tactic(); t[13] = 1.0


def measure(cal, n=40, seed=3):
    r = engine.run(sk, sk, t, t, n, seed, cal); m = r.mean(0)
    lo = m[23] / max(m[20], 1e-9); hi = m[25] / max(m[22], 1e-9)
    return dict(goals=(m[0] + m[1]) / 2, shots=(m[2] + m[3]) / 2, completion=(m[8] + m[9]) / max(m[18], 1), tackles=m[19] / 2,
                counter_ratio=lo / max(hi, 1e-4), regains_le3=m[20], regains_8p=m[22])


def loss(r):
    return sum(((r[k] - v) / v) ** 2 for k, v in TARGET.items())


if __name__ == "__main__":
    rng = np.random.default_rng(25)
    lo = np.array([0.1, 0.6, 0.5, 0.005, -5.5, 0.4, 0.0, 0.4]); hi = np.array([0.8, 2.5, 8.0, 0.10, -2.0, 1.2, 25.0, 2.2])
    start = np.array(json.load(open("calibration.json"))["cal"])
    best = (loss(measure(start)), start, measure(start)); print("start", round(best[0], 3), {k: round(v, 3) for k, v in best[2].items()}, flush=True)
    t0 = time.time()
    for it in range(400):
        cal = lo + rng.random(8) * (hi - lo) if it < 150 else np.clip(best[1] + rng.normal(0, 0.05, 8) * (hi - lo), lo, hi)
        r = measure(cal); l = loss(r)
        if l < best[0]:
            best = (l, cal, r); print(it, round(l, 3), {k: round(v, 3) for k, v in r.items()}, flush=True)
    chk = measure(best[1], n=300, seed=99)
    print("check on 300 fresh matches:", {k: round(v, 3) for k, v in chk.items()}, "time", round(time.time() - t0))
    json.dump(dict(cal=best[1].tolist(), stats=best[2], check=chk, loss=best[0], target=TARGET), open("calibration4.json", "w"), indent=1)
