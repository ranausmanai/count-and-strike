"""Random search over physics constants to match real-football targets for two equal teams (default tactic)."""
import numpy as np, json, time
import engine
TARGET = dict(goals=1.40, shots=12.5, completion=0.82, passes=430, tackles=45)
sk = np.array([0.97, 1.0, 1.0, 1.0, 1.0]); t = engine.default_tactic()
def measure(cal, n=24, seed=3):
    m = engine.run(sk, sk, t, t, n, seed, cal).mean(0)
    return dict(goals=(m[0]+m[1])/2, shots=(m[2]+m[3])/2, completion=(m[8]+m[9])/max(m[18],1), passes=(m[8]+m[9])/2, tackles=m[19]/2)
def loss(r):
    return sum(((r[k]-v)/v)**2 for k, v in TARGET.items())
rng = np.random.default_rng(0)
lo = np.array([0.2, 0.6, 0.5, 0.005, -5.5, 0.4, 0.0, 0.4]); hi = np.array([0.8, 2.5, 8.0, 0.10, -2.0, 1.2, 25.0, 2.2])
best = None
engine.run(sk, sk, t, t, 1, 0, engine.CAL0)
t0 = time.time()
for it in range(360):
    cal = lo + rng.random(8) * (hi - lo) if (best is None or it < 160) else np.clip(best[1] + rng.normal(0, 0.06, 8) * (hi - lo), lo, hi)
    r = measure(cal); l = loss(r)
    if best is None or l < best[0]:
        best = (l, cal, r); print(it, round(l, 3), {k: round(v, 2) for k, v in r.items()}, np.round(cal, 3).tolist(), flush=True)
print("time", time.time() - t0)
json.dump(dict(cal=best[1].tolist(), stats=best[2], loss=best[0]), open("calibration.json", "w"), indent=1)
