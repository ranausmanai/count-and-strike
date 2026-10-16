"""Agent avail: tests per PREREGISTRATION_AGENT_avail.md."""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import PROC, ROOT
from r22a_strike import aipw, verdict, POPS
from r22c_stress_all import add_pre, shots, crossfit

FEAT = ["x", "y", "opp_goalside", "nearest_opp", "start_ov", "max_ov", "min_ov", "n_on_ov", "crowded_on", "full_width", "n_mates_vis", "n_opps_vis"]


def wcd(d, y="net", cell="cell"):
    """r18a within_cell_diff, T=1 minus T=0."""
    a = d[d["T"] == 1].groupby(cell)[y].agg(["mean", "var", "size"]); b = d[d["T"] == 0].groupby(cell)[y].agg(["mean", "var", "size"])
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner"); j = j[(j.size_a >= 10) & (j.size_b >= 10)]
    w = np.minimum(j.size_a, j.size_b)
    diff = ((j.mean_a - j.mean_b) * w).sum() / w.sum()
    var = ((w / w.sum()) ** 2 * (j.var_a / j.size_a + j.var_b / j.size_b)).sum()
    return float(diff), float(np.sqrt(var))


def per_pop(D, fn):
    return {p: fn(D[D["pop"] == p].reset_index(drop=True)) for p in POPS}


def share_ci(d, B=1000, seed=0):
    g = d.groupby("match_id")["T"].agg(["sum", "size"]); rng = np.random.default_rng(seed)
    s = 1 - g["sum"].sum() / g["size"].sum(); idx = rng.integers(0, len(g), (B, len(g)))
    bs = 1 - g["sum"].values[idx].sum(1) / g["size"].values[idx].sum(1)
    return dict(share_crowded=float(s), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=int(len(d)))


def main():
    A = pd.read_parquet(os.path.join(PROC, "agent_avail_passes.parquet"))
    M = pd.read_parquet(os.path.join(PROC, "agent_avail_matches.parquet"))
    A["overload_on"] = A.n_on_ov > 0; A["crowded_on"] = (A.n_on_cr > 0).astype(int); A["full_width"] = A.full_width.astype(int)
    res = {}
    # frequency per team per match (all choice situations, before dropping target_ov == 0)
    fq = {}
    for p in POPS:
        nm = int((M["pop"] == p).sum()); a = A[A["pop"] == p]
        c = a[a.overload_on].groupby(["match_id", "team"]).size()
        cf = a[a.overload_on & (a.full_width == 1)].groupby(["match_id", "team"]).size()
        fq[p] = dict(team_matches=2 * nm, mean_on=float(c.sum() / (2 * nm)), median_on=float(c.reindex(range(0)).median()) if False else float(np.median(np.r_[c.values, np.zeros(max(0, 2 * nm - len(c)))])),
                     mean_on_fullwidth=float(cf.sum() / (2 * nm)), choice_situations_per_team_match=float(len(a) / (2 * nm)),
                     share_choice_with_overload_on=float(a.overload_on.mean()), share_fullwidth=float(a.full_width.mean()))
    res["frequency"] = fq
    D = A[A.target_ov != 0].copy(); D["T"] = (D.target_ov > 0).astype(int)
    D["complete"] = D.complete.astype(float)
    D = add_pre(D, shots())
    on = D[D.overload_on].reset_index(drop=True); off = D[~D.overload_on].reset_index(drop=True)
    print({p: (int((on["pop"] == p).sum()), int((off["pop"] == p).sum())) for p in POPS})
    res["n"] = {p: dict(on=int((on["pop"] == p).sum()), on_T1=int(((on["pop"] == p) & (on["T"] == 1)).sum()),
                        on_fullwidth=int(((on["pop"] == p) & (on.full_width == 1)).sum()), off=int((off["pop"] == p).sum()),
                        on_both=int(((on["pop"] == p) & (on.crowded_on == 1)).sum())) for p in POPS}
    # H1 descriptive
    res["H1_share_crowded_when_on"] = per_pop(on, share_ci)
    res["H1_share_crowded_when_on_fullwidth"] = per_pop(on[on.full_width == 1], share_ci)
    res["H1_share_crowded_when_not_on"] = per_pop(off, share_ci)
    res["H1_holds"] = bool(all(v["lo"] > 0.5 for v in res["H1_share_crowded_when_on"].values()))
    # raw rates
    res["raw_when_on"] = {p: dict(compl_T1=float(d[d["T"] == 1].complete.mean()), compl_T0=float(d[d["T"] == 0].complete.mean()),
                                   net_T1=float(d[d["T"] == 1].net.mean()), net_T0=float(d[d["T"] == 0].net.mean()))
                          for p, d in ((p, on[on["pop"] == p]) for p in POPS)}
    # primary
    H2 = per_pop(on, wcd); H3 = per_pop(on[on.full_width == 1], wcd)
    H4, H4p = {}, {}
    for p in POPS:
        d = on[on["pop"] == p].reset_index(drop=True)
        H4[p] = aipw(d, *crossfit(d, FEAT, "net"), y="net"); H4p[p] = aipw(d, *crossfit(d, FEAT, "pre_net"), y="pre_net")
        print(p, "H2", H2[p], "H3", H3[p], "H4", H4[p], "placebo", H4p[p], flush=True)
    prim = {"H2_cell_net_on": verdict(H2), "H3_cell_net_on_fullwidth": verdict(H3), "H4_aipw_net_on": verdict(H4)}
    pl = verdict(H4p); pl["fails"] = bool(pl["pooled"] > 0 and pl["pooled_p"] < .05); prim["H4_aipw_net_on"]["placebo_pre20"] = pl
    # BH in discovery (two-sided), then Holm in confirmation
    names = list(prim); z = np.array([prim[k]["per_pop"]["discovery"]["est"] / prim[k]["per_pop"]["discovery"]["se"] for k in names])
    rej, q, _, _ = multipletests(2 * norm.sf(np.abs(z)), alpha=.05, method="fdr_bh")
    carried = [k for k, r, zz in zip(names, rej, z) if r and zz > 0]
    for k, r, qq in zip(names, rej, q):
        prim[k]["disc_bh_q"] = float(qq); prim[k]["disc_bh_pass"] = bool(r and prim[k]["per_pop"]["discovery"]["est"] > 0)
    if carried:
        hr, hp, _, _ = multipletests([prim[k]["pooled_p"] for k in carried], alpha=.05, method="holm")
    for k in names:
        same = all(prim[k]["per_pop"][c]["est"] > 0 for c in ("confirm_club", "confirm_women"))
        prim[k]["confirmed"] = bool(k in carried and same and hr[carried.index(k)])
        if k in carried:
            prim[k]["holm_p"] = float(hp[carried.index(k)])
    prim["H4_aipw_net_on"]["survives_placebo"] = bool(prim["H4_aipw_net_on"]["confirmed"] and not pl["fails"])
    res["primary"] = prim
    # secondary
    sec = {"H2c_completion_on": verdict(per_pop(on, lambda d: wcd(d, "complete"))),
           "H2b_both_on": verdict(per_pop(on[on.crowded_on == 1], wcd)),
           "H2b_both_on_completion": verdict(per_pop(on[on.crowded_on == 1], lambda d: wcd(d, "complete"))),
           "contrast_not_on": verdict(per_pop(off, wcd)),
           "contrast_not_on_completion": verdict(per_pop(off, lambda d: wcd(d, "complete"))),
           "H3c_completion_on_fullwidth": verdict(per_pop(on[on.full_width == 1], lambda d: wcd(d, "complete")))}
    res["secondary"] = sec
    # robustness (reported for any confirmed or for all primary, net45 + finer cells)
    on["cell2"] = on.cell + np.where(on.n_on_ov >= 2, "b", "a") + on.crowded_on.astype(str)
    rob = {"net45_cell_on": verdict(per_pop(on, lambda d: wcd(d, "net45"))),
           "finer_cell_on": verdict(per_pop(on, lambda d: wcd(d, "net", "cell2"))),
           "net45_cell_on_fullwidth": verdict(per_pop(on[on.full_width == 1], lambda d: wcd(d, "net45")))}
    R45 = {}
    for p in POPS:
        d = on[on["pop"] == p].reset_index(drop=True); R45[p] = aipw(d, *crossfit(d, FEAT, "net45"), y="net45")
    rob["net45_aipw_on"] = verdict(R45)
    res["robustness"] = rob
    c = [prim[k]["confirmed"] for k in ("H2_cell_net_on", "H3_cell_net_on_fullwidth", "H4_aipw_net_on")]
    res["verdict"] = ("CONFIRMED" if prim["H2_cell_net_on"]["confirmed"] and prim["H4_aipw_net_on"]["survives_placebo"]
                      else "MIXED" if any(c) else "NOT CONFIRMED")
    for k, v in res.items():
        print(k, json.dumps(v, default=float)[:900])
    json.dump(res, open(os.path.join(ROOT, "results", "agent_avail.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
