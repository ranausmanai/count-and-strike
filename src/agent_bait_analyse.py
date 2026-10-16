"""Agent bait: analysis per PREREGISTRATION_AGENT_bait.md -> results/agent_bait.json"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import PROC, ROOT
from r22a_strike import aipw, verdict, POPS
from r22c_stress_all import add_pre, shots, crossfit

F2 = ["x", "ady", "opp_goalside", "nearest_opp", "far_vis", "poss_age", "minute"]
F3 = F2 + ["far_ov", "ball_ov", "mid_ov", "opp_bh", "mate_bh"]


def horizon(D, S, h):
    own = np.zeros(len(D)); opp = np.zeros(len(D))
    Sg = {k: g.sort_values("t") for k, g in S.groupby(["match_id", "period"])}
    for (mid, per), idx in D.groupby(["match_id", "period"]).indices.items():
        s = Sg.get((mid, per))
        if s is None:
            continue
        st = s.t.values
        for team in np.unique(D.team.values[idx]):
            sel = idx[D.team.values[idx] == team]; mine = s.team.values == team
            cm = np.concatenate([[0], np.cumsum(np.where(mine, s.xg.values, 0))]); co = np.concatenate([[0], np.cumsum(np.where(~mine, s.xg.values, 0))])
            tt = D.t.values[sel]
            lo = np.searchsorted(st, tt - 0.01, "left"); hi = np.searchsorted(st, tt + h, "right"); lo2 = np.searchsorted(st, tt + 0.01, "right")
            own[sel] = cm[hi] - cm[lo]; opp[sel] = co[hi] - co[lo2]
    return own - opp


def within(d, a, b, y, cell="cell", mn=5):
    A = d[a].groupby(cell)[y].agg(["mean", "var", "size"]); B = d[b].groupby(cell)[y].agg(["mean", "var", "size"])
    j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner"); j = j[(j.size_a >= mn) & (j.size_b >= mn)]
    if j.empty:
        return np.nan, np.nan, 0, 0, 0
    w = np.minimum(j.size_a, j.size_b)
    diff = ((j.mean_a - j.mean_b) * w).sum() / w.sum()
    var = ((w / w.sum()) ** 2 * (j.var_a.fillna(0) / j.size_a + j.var_b.fillna(0) / j.size_b)).sum()
    return float(diff), float(np.sqrt(var)), int(len(j)), int(j.size_a.sum()), int(j.size_b.sum())


def run_test(D, a_fn, b_fn, y, cell="cell"):
    out = {}
    for p in POPS:
        d = D[D["pop"] == p]
        out[p] = within(d, a_fn(d), b_fn(d), y, cell)
    return out


def confirm(r):
    (d0, s0) = r["discovery"][:2]; c = [r["confirm_club"][:2], r["confirm_women"][:2]]
    w = np.array([1 / s ** 2 for _, s in c]); pooled = float((w * np.array([m for m, _ in c])).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
    return dict(per_pop={k: dict(est=v[0], se=v[1], cells=v[2], n_a=v[3], n_b=v[4]) for k, v in r.items()},
                disc_z=float(d0 / s0), disc_p2=float(2 * norm.sf(abs(d0 / s0))), pooled=pooled, pooled_se=pse,
                pooled_p1=float(norm.sf(pooled / pse)), same_sign=bool(d0 > 0 and all(m > 0 for m, _ in c)))


def battery(D, feat, y="net20"):
    eff, plc, cnt = {}, {}, {}
    for p in POPS:
        d = D[D["pop"] == p].reset_index(drop=True)
        eff[p] = aipw(d, *crossfit(d, feat, y), y=y); plc[p] = aipw(d, *crossfit(d, feat, "pre_net"), y="pre_net")
        cnt[p] = (int(d["T"].sum()), int((1 - d["T"]).sum()))
    v = verdict(eff); pl = verdict(plc); pl["fails"] = bool(pl["pooled"] > 0 and pl["pooled_p"] < .05)
    return dict(effect=v, placebo=pl, n=cnt)


def main():
    S = shots()
    D = pd.read_parquet(os.path.join(PROC, "agent_bait_passes.parquet"))
    D = D[D.poss_age >= 8].reset_index(drop=True)
    D["net20"] = horizon(D, S, 20); D["net45"] = horizon(D, S, 45)
    D["Bp"] = D.bait_loose & (D.draw >= 1); D["B0"] = ~D.bait_loose
    D["fvbin"] = np.where(D.far_vis >= 0.8, "h", "m")
    D["cell_v"] = D.cell + D.fvbin
    D["fobin"] = np.select([D.far_ov <= -1, D.far_ov >= 1], ["n", "p"], "z")
    D["cell_f"] = D.cell + D.fobin
    D = add_pre(D, S)
    res = {"n_decisions": D.groupby("pop").size().to_dict()}
    # descriptives
    desc = {}
    for p, d in D.groupby("pop"):
        desc[p] = dict(share_Bp=float(d.Bp.mean()), share_loose=float(d.bait_loose.mean()), switch_rate_Bp=float(d[d.Bp].switch.mean()),
                       switch_rate_B0=float(d[d.B0].switch.mean()), stay_rate_Bp=float(d[d.Bp].stay.mean()),
                       far_ov_ge1_Bp=float((d[d.Bp].far_ov >= 1).mean()), far_ov_ge1_B0=float((d[d.B0].far_ov >= 1).mean()),
                       far_vis_Bp=float(d[d.Bp].far_vis.mean()), far_vis_B0=float(d[d.B0].far_vis.mean()),
                       switch_rate_Bp_when_far_ov_ge1=float(d[d.Bp & (d.far_ov >= 1)].switch.mean()),
                       n_switch_Bp=int((d.Bp & d.switch).sum()), n_switch_B0=int((d.B0 & d.switch).sum()))
    res["descriptive"] = desc; print(json.dumps(desc, indent=0))
    # ---- primary family ----
    V = D[D.far_vis >= 0.5]
    t1 = run_test(V, lambda d: d.Bp, lambda d: d.B0, "far_ov", "cell_v")
    SW = D[D.switch]
    t2 = run_test(SW, lambda d: d.Bp, lambda d: d.B0, "net20")
    BP = D[D.Bp & (D.switch | D.stay)]
    t3 = run_test(BP, lambda d: d.switch, lambda d: d.stay, "net20")
    # H4 possession level
    D["pk"] = D.match_id.astype(str) + "_" + D.period.astype(str) + "_" + D.possession.astype(str)
    D["sw_from_Bp"] = D.switch & D.Bp
    first = D[D.Bp].groupby("pk").t.min().rename("t_anchor")
    Q = D.merge(first, left_on="pk", right_index=True)
    after = Q[Q.t >= Q.t_anchor]
    ag = after.groupby("pk").agg(any_sw_bp=("sw_from_Bp", "any"), any_sw=("switch", "any"))
    A = Q[Q.t == Q.t_anchor].drop_duplicates("pk").merge(ag, left_on="pk", right_index=True)
    A = A[A.any_sw_bp | ~A.any_sw].reset_index(drop=True); A["T"] = A.any_sw_bp.astype(int)
    t4 = run_test(A, lambda d: d["T"] == 1, lambda d: d["T"] == 0, "net20")
    fam = {"H1_mechanism_far_ov": t1, "H2_switch_after_bait_vs_none": t2, "H3_after_bait_switch_vs_stay": t3, "H4_possession_baitswitch_vs_baitonly": t4}
    C = {k: confirm(v) for k, v in fam.items()}
    pz = [C[k]["disc_p2"] if C[k]["per_pop"]["discovery"]["est"] > 0 else 1.0 for k in fam]
    bh = multipletests(pz, alpha=0.05, method="fdr_bh")[0]
    carried = [k for k, b in zip(fam, bh) if b]
    holm = dict(zip(carried, multipletests([C[k]["pooled_p1"] for k in carried], alpha=0.05, method="holm")[0])) if carried else {}
    for k, b in zip(fam, bh):
        C[k]["bh_disc"] = bool(b); C[k]["confirmed"] = bool(b and C[k]["same_sign"] and holm.get(k, False))
    res["family"] = C
    for k, v in C.items():
        print(k, {p: (round(x["est"], 4), round(x["se"], 4), x["n_a"], x["n_b"]) for p, x in v["per_pop"].items()}, "pooled", round(v["pooled"], 4), round(v["pooled_se"], 4), "conf", v["confirmed"], flush=True)
    # ---- secondary / robustness ----
    sec = {}
    sec["H1_all_frames"] = confirm(run_test(D, lambda d: d.Bp, lambda d: d.B0, "far_ov"))
    ch = D[D.Bp & D.start_far_ov.notna()]
    sec["H1_change_far_ov_within_Bp"] = {p: dict(mean=float((g.far_ov - g.start_far_ov).mean()), se=float((g.far_ov - g.start_far_ov).std() / np.sqrt(len(g))), n=len(g)) for p, g in ch.groupby("pop")}
    chn = D[D.B0 & D.start_far_ov.notna() & (D.n_prior >= 2)]
    sec["H1_change_far_ov_B0_ref"] = {p: dict(mean=float((g.far_ov - g.start_far_ov).mean()), n=len(g)) for p, g in chn.groupby("pop")}
    sec["H2_net45"] = confirm(run_test(SW, lambda d: d.Bp, lambda d: d.B0, "net45"))
    sec["H3_net45"] = confirm(run_test(BP, lambda d: d.switch, lambda d: d.stay, "net45"))
    sec["H4_net45"] = confirm(run_test(A, lambda d: d["T"] == 1, lambda d: d["T"] == 0, "net45"))
    sec["H2_loose"] = confirm(run_test(SW, lambda d: d.bait_loose, lambda d: d.B0, "net20"))
    BL = D[D.bait_loose & (D.switch | D.stay)]
    sec["H3_loose"] = confirm(run_test(BL, lambda d: d.switch, lambda d: d.stay, "net20"))
    sec["H2_finer_cell"] = confirm(run_test(SW, lambda d: d.Bp, lambda d: d.B0, "net20", "cell_f"))
    sec["H3_finer_cell"] = confirm(run_test(BP, lambda d: d.switch, lambda d: d.stay, "net20", "cell_f"))
    sec["H2_farvis50"] = confirm(run_test(SW[SW.far_vis >= .5], lambda d: d.Bp, lambda d: d.B0, "net20"))
    sec["H3_farvis50"] = confirm(run_test(BP[BP.far_vis >= .5], lambda d: d.switch, lambda d: d.stay, "net20"))
    # B0 reference: switch vs stay without bait (is any H3 effect bait-specific?)
    B0s = D[D.B0 & (D.switch | D.stay)]
    sec["H3_ref_no_bait_switch_vs_stay"] = confirm(run_test(B0s, lambda d: d.switch, lambda d: d.stay, "net20"))
    res["secondary"] = sec
    for k, v in sec.items():
        if "per_pop" in v:
            print(k, {p: (round(x["est"], 4), round(x["se"], 4)) for p, x in v["per_pop"].items()}, "pooled", round(v["pooled"], 4), round(v["pooled_se"], 4))
        else:
            print(k, v)
    # ---- stress battery ----
    st = {}
    s2 = SW[SW.Bp | SW.B0].copy(); s2["T"] = s2.Bp.astype(int)
    st["H2"] = battery(s2, F2)
    s3 = BP.copy(); s3["T"] = s3.switch.astype(int)
    st["H3"] = battery(s3, F3)
    st["H4"] = battery(A, F3)
    for k, v in st.items():
        v["survives"] = bool(C[{"H2": "H2_switch_after_bait_vs_none", "H3": "H3_after_bait_switch_vs_stay", "H4": "H4_possession_baitswitch_vs_baitonly"}[k]]["confirmed"]
                             and v["effect"]["passed"] and not v["placebo"]["fails"])
        print(k, "AIPW", {p: tuple(round(z, 4) for z in (x["est"], x["se"])) for p, x in v["effect"]["per_pop"].items()}, "pooled", round(v["effect"]["pooled"], 4),
              round(v["effect"]["pooled_se"], 4), "pass", v["effect"]["passed"], "| placebo pooled", round(v["placebo"]["pooled"], 4), round(v["placebo"]["pooled_se"], 4),
              "fails", v["placebo"]["fails"], "| survives", v["survives"], flush=True)
    res["stress"] = st
    res["n_H4_possessions"] = A.groupby(["pop", "T"]).size().unstack().to_dict()
    json.dump(res, open(os.path.join(ROOT, "results", "agent_bait.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
