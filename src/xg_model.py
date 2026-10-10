"""Location-based xG model trained on StatsBomb 2015/16 shots (goal outcome), for use on Wyscout 2017/18 shots.
Features (pitch normalised to 105x68 m): distance, angle, header, direct free kick; penalties fixed at 0.76."""
import gzip, json, os, pickle
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "sb")

def shots(f):
    ev = json.load(gzip.open(os.path.join(RAW, f)))
    out = []
    for e in ev:
        if e["type"]["name"] == "Shot" and e["period"] <= 4:
            s = e["shot"]
            out.append(dict(x=e["location"][0] / 120 * 105, y=e["location"][1] / 80 * 68, head=s["body_part"]["name"] == "Head",
                            fk=s["type"]["name"] == "Free Kick", pen=s["type"]["name"] == "Penalty",
                            goal=s["outcome"]["name"] == "Goal", sbxg=s.get("statsbomb_xg", np.nan)))
    return out

def feats(d):
    dx = 105 - d.x; dy = (d.y - 34).abs()
    dist = np.hypot(dx, dy)
    # angle subtended by the goal mouth (7.32 m)
    a = np.arctan2(7.32 * dx, dx ** 2 + dy ** 2 - (7.32 / 2) ** 2); a = np.where(a < 0, a + np.pi, a)
    return np.column_stack([dist, np.log(dist + 1), a, d["head"].astype(float), d.fk.astype(float), dist * d["head"].astype(float)])

if __name__ == "__main__":
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(shots, sorted(os.listdir(RAW)), chunksize=16):
            rows += r
    d = pd.DataFrame(rows); d = d[~d.pen]
    X = feats(d); y = d.goal.values
    m = LogisticRegression(C=10, max_iter=2000)
    p = cross_val_predict(m, X, y, cv=5, method="predict_proba")[:, 1]
    from sklearn.metrics import log_loss, roc_auc_score
    print("n", len(d), "cv logloss", log_loss(y, p), "auc", roc_auc_score(y, p), "| SB xG logloss", log_loss(y, d.sbxg.clip(1e-4, 1 - 1e-4)), "auc", roc_auc_score(y, d.sbxg))
    print("corr with SB xG", np.corrcoef(p, d.sbxg)[0, 1], "sum p", p.sum(), "goals", y.sum(), "sum sbxg", d.sbxg.sum())
    m.fit(X, y)
    pickle.dump(m, open(os.path.join(ROOT, "data", "proc", "xg_model.pkl"), "wb"))
