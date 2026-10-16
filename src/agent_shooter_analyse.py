"""Agent shooter: role-specific shot-vs-recycle advantage (PREREGISTRATION_AGENT_SHOOTER.md)."""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
POPS = ("discovery", "confirm_club", "confirm_women")
RECYCLE = {"lateral", "back_pass", "short_fwd_ground", "short_fwd_air"}
ROLE = {}
for p in ("Center Forward", "Left Center Forward", "Right Center Forward", "Secondary Striker"):
    ROLE[p] = "FW"
for p in ("Center Attacking Midfield", "Left Attacking Midfield", "Right Attacking Midfield", "Left Wing", "Right Wing",
          "Left Midfield", "Right Midfield"):
    ROLE[p] = "AMW"
for p in ("Center Midfield", "Left Center Midfield", "Right Center Midfield", "Center Defensive Midfield",
          "Left Defensive Midfield", "Right Defensive Midfield"):
    ROLE[p] = "CMDM"
for p in ("Center Back", "Left Center Back", "Right Center Back", "Left Back", "Right Back", "Left Wing Back", "Right Wing Back"):
    ROLE[p] = "DEF"
K = 50.0


def load():
    A = pd.read_parquet(os.path.join(PROC, "r15_attack.parquet"))
    A = A[A.action.isin(RECYCLE | {"shot"})]
    E = pd.read_parquet(os.path.join(PROC, "agent_shooter_ev.parquet"))
    A = A.merge(E, on=["match_id", "idx"], how="inner")
    A["role"] = A.position.map(ROLE)
    A = A[A.role.notna()].copy()
    A["shot"] = A.action == "shot"
    base = (pd.Series(np.select([A.opp_goalside <= 3, A.opp_goalside <= 6], ["0", "1"], "2"), index=A.index)
            + np.select([A.nearest_opp < 3, A.nearest_opp <= 7], ["0", "1"], "2").astype(object)
            + np.where(A.lane >= 2, "2", A.lane.astype(str)).astype(object))
    A["sit"] = base
    A["cell"] = pd.Series(np.where(A.x >= 108, "b", "e"), index=A.index) + base
    # prior finishing skill
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    A["date"] = pd.to_datetime(A.match_id.map({m["match_id"]: m["match_date"] for m in ms}))
    Sh = pd.read_parquet(os.path.join(PROC, "agent_shooter_shots.parquet"))
    Sh["date"] = pd.to_datetime(Sh.date)
    g = Sh.groupby(["player_id", "date"]).agg(n=("xg", "size"), xg=("xg", "sum"), goal=("goal", "sum")).reset_index()
    g = g.sort_values(["player_id", "date"])
    for c in ("n", "xg", "goal"):
        g["c" + c] = g.groupby("player_id")[c].cumsum()
    keys = A[["player_id", "date"]].drop_duplicates().sort_values("date")
    m = pd.merge_asof(keys, g[["player_id", "date", "cn", "cxg", "cgoal"]].sort_values("date"), on="date", by="player_id",
                      allow_exact_matches=False)
    m = m.fillna(0.0)
    m["skill"] = (m.cgoal - m.cxg) / (m.cn + K)
    A = A.merge(m, on=["player_id", "date"], how="left")
    return A


def pooled_adv(D, cellcol="cell", y="net"):
    st = D.groupby([cellcol, "shot"])[y].agg(["mean", "var", "size"]).unstack("shot")
    st = st.dropna()
    st = st[(st[("size", True)] >= 5) & (st[("size", False)] >= 5)]
    if st.empty:
        return dict(est=np.nan, se=np.nan, cells=0, n_shot=0, n_rec=0)
    d = st[("mean", True)] - st[("mean", False)]
    se = np.sqrt(st[("var", True)] / st[("size", True)] + st[("var", False)] / st[("size", False)])
    w = st[("size", True)].astype(float)
    return dict(est=float((w * d).sum() / w.sum()), se=float(np.sqrt((w ** 2 * se ** 2).sum()) / w.sum()), cells=int(len(st)),
                n_shot=int(st[("size", True)].sum()), n_rec=int(st[("size", False)].sum()))


def diff(a, b):
    return dict(est=a["est"] - b["est"], se=float(np.hypot(a["se"], b["se"])))


def tests(A, pop, y="net"):
    P = A[A["pop"] == pop]
    R = P[(P.x >= 96) & ((P.y - 40).abs() <= 20)]
    out = {}
    adv = {r: pooled_adv(R[R.role == r], y=y) for r in ("FW", "AMW", "CMDM", "DEF")}
    for r, v in adv.items():
        out[f"H1_{r}"] = v
    out["H2_FW-DEF"] = diff(adv["FW"], adv["DEF"])
    out["H2_FW-CMDM"] = diff(adv["FW"], adv["CMDM"])
    R3 = P[(P.x >= 90) & (P.x <= 102) & ((P.y - 40).abs() <= 20) & (P.role == "CMDM")]
    out["H3_CMDM_20-30yd"] = pooled_adv(R3, cellcol="sit", y=y)
    Rk = R[R.cn > 0].copy()
    pm = Rk.groupby("player_id").skill.mean().median()
    Rk["rc"] = Rk.role + Rk.cell
    hi, lo = pooled_adv(Rk[Rk.skill > pm], "rc", y), pooled_adv(Rk[Rk.skill <= pm], "rc", y)
    out["H4_hi-lo_skill"] = dict(diff(hi, lo), hi=hi, lo=lo, median_skill=float(pm),
                                 players=int(Rk.player_id.nunique()), rows_with_prior=int(len(Rk)), rows=int(len(R)))
    return out


def main():
    A = load()
    R = A[(A.x >= 96) & ((A.y - 40).abs() <= 20)]
    sizes = {f"{a}|{b}|{'shot' if c else 'recycle'}": int(v) for (a, b, c), v in R.groupby(["pop", "role", "shot"]).size().items()}
    res = {p: tests(A, p) for p in POPS}
    names = list(res["discovery"].keys())
    D = res["discovery"]
    p2 = np.array([2 * norm.sf(abs(D[n]["est"] / D[n]["se"])) for n in names])
    q = multipletests(p2, method="fdr_bh")[1]
    summary = []
    for n, pp, qq in zip(names, p2, q):
        d = D[n]
        row = dict(test=n, disc=d["est"], disc_se=d["se"], disc_p=float(pp), disc_q=float(qq),
                   disc_pass=bool(qq < 0.05 and d["est"] > 0))
        ests = [(res[p][n]["est"], res[p][n]["se"]) for p in ("confirm_club", "confirm_women")]
        row.update(club=ests[0][0], club_se=ests[0][1], women=ests[1][0], women_se=ests[1][1])
        w = np.array([1 / s ** 2 for _, s in ests]); e = np.array([v for v, _ in ests])
        row["pooled"] = float((w * e).sum() / w.sum()); row["pooled_se"] = float(1 / np.sqrt(w.sum()))
        row["pooled_p1"] = float(norm.sf(row["pooled"] / row["pooled_se"]))
        row["same_sign"] = bool((e > 0).all())
        summary.append(row)
    carried = sorted([r for r in summary if r["disc_pass"]], key=lambda r: r["pooled_p1"])
    k = len(carried); ok = True
    for i, r in enumerate(carried):
        ok = ok and r["pooled_p1"] < 0.05 / (k - i)
        r["confirmed"] = bool(ok and r["same_sign"])
    for r in summary:
        r.setdefault("confirmed", False)
    # robustness: 60 s horizon for confirmed tests (pooled confirmation populations)
    rob = {}
    conf_names = [r["test"] for r in summary if r["confirmed"]]
    if conf_names:
        r60 = {p: tests(A, p, y="net60") for p in POPS}
        for n in conf_names:
            ests = [(r60[p][n]["est"], r60[p][n]["se"]) for p in ("confirm_club", "confirm_women")]
            w = np.array([1 / s ** 2 for _, s in ests]); e = np.array([v for v, _ in ests])
            pe, ps = float((w * e).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
            rob[n] = dict(disc=r60["discovery"][n]["est"], disc_se=r60["discovery"][n]["se"], club=ests[0], women=ests[1],
                          pooled=pe, pooled_se=ps, p1=float(norm.sf(pe / ps)))
    out = dict(sizes=sizes, per_population=res, summary=summary, robustness_60s=rob)
    json.dump(out, open(os.path.join(ROOT, "results", "agent_shooter.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary).round(4).to_string())
    print(json.dumps(rob, indent=1, default=float))
    for p in POPS:
        print(p, {k: (round(v.get("cells", 0)), v.get("n_shot"), v.get("n_rec")) for k, v in res[p].items() if "cells" in v})
        print(p, {k: v for k, v in res[p]["H4_hi-lo_skill"].items() if k not in ("hi", "lo")})
    print(R.groupby(["pop", "role"]).shot.agg(["size", "sum"]))


if __name__ == "__main__":
    main()
