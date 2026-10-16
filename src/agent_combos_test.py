"""Agent 'combos': pre-registered tests -> results/agent_combos.json"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(__file__))
from r15_360 import PROC, ROOT

TYPES = ["solo_carry", "one_two", "third_man", "layoff_fwd", "switch_attack"]
POPS = ["discovery", "confirm_club", "confirm_women"]


def rule_table(D, y):
    D = D[D.cat != "shot"]
    st = D.groupby(["cell", "cat"])[y].agg(["mean", "var", "size"]).reset_index()
    ref = st.sort_values("size", ascending=False).drop_duplicates("cell").set_index("cell")
    c = st.merge(ref[["cat", "mean", "var", "size"]].rename(columns=lambda x: "ref_" + x), left_on="cell", right_index=True)
    c = c[c.cat != c.ref_cat].copy()
    c["diff"] = c["mean"] - c.ref_mean; c["se"] = np.sqrt(c["var"].fillna(0) / c["size"] + c.ref_var.fillna(0) / c.ref_size)
    return c


def pool(ests):
    w = np.array([1 / s ** 2 for _, s in ests]); m = float((w * np.array([d for d, _ in ests])).sum() / w.sum())
    return m, float(1 / np.sqrt(w.sum()))


def holm(conf):
    k = len(conf); conf.sort(key=lambda d: d["p1"]); flag = True
    for i, d in enumerate(conf):
        flag = flag and d["p1"] < 0.05 / (k - i); d["confirmed"] = bool(flag and d["same"])
    return conf


def cell_level(D, y="net20"):
    disc = rule_table(D[D["pop"] == "discovery"], y); disc = disc[(disc["size"] >= 30) & (disc.ref_size >= 30)].copy()
    disc["p"] = 2 * norm.sf(np.abs(disc["diff"] / disc.se)); disc["q"] = multipletests(disc.p, method="fdr_bh")[1]
    S = disc[disc.q < 0.05]
    C = {p: rule_table(D[D["pop"] == p], y).set_index(["cell", "cat"]) for p in POPS[1:]}
    conf = []
    for r in S.itertuples():
        ests = {}
        for p, T in C.items():
            if (r.cell, r.cat) in T.index:
                v = T.loc[(r.cell, r.cat)]
                if v["size"] >= 10 and v.ref_size >= 10 and v.ref_cat == r.ref_cat:
                    ests[p] = (float(v["diff"]), float(v.se))
        d = dict(cell=r.cell, cat=r.cat, ref=r.ref_cat, n=int(r.size), ref_n=int(r.ref_size), disc=float(r.diff), disc_se=float(r.se), q=float(r.q),
                 conf=ests)
        if len(ests) == 2:
            m, s = pool(list(ests.values()))
            d.update(pooled=m, pooled_se=s, p1=float(norm.sf(np.sign(r.diff) * m / s)),
                     same=bool(all(np.sign(e[0]) == np.sign(r.diff) for e in ests.values())))
            conf.append(d)
    conf = holm(conf)
    untestable = [dict(cell=r.cell, cat=r.cat, ref=r.ref_cat, disc=float(r.diff), q=float(r.q)) for r in S.itertuples()
                  if not any(c["cell"] == r.cell and c["cat"] == r.cat for c in conf)]
    dist = disc.groupby("cat").agg(rules=("diff", "size"), mean_diff=("diff", "mean"), n_pos=("diff", lambda s: int((s > 0).sum())))
    return dict(searched=int(len(disc)), bh=int(len(S)), testable=len(conf), confirmed=int(sum(c["confirmed"] for c in conf)),
                rules=conf, untestable=untestable, disc_rule_summary=dist.reset_index().to_dict("records"))


def type_level_pop(D, y, typ, min_n=5):
    T = rule_table(D, y); T = T[(T.cat == typ) & (T["size"] >= min_n) & (T.ref_size >= min_n) & (T.se > 0)]
    if T.empty:
        return None
    m, s = pool(list(zip(T["diff"], T.se)))
    return dict(est=m, se=s, cells=int(len(T)), n=int(T["size"].sum()))


def type_level(D, y="net20"):
    out = {}
    for typ in TYPES:
        out[typ] = {p: type_level_pop(D[D["pop"] == p], y, typ) for p in POPS}
        e = out[typ]["discovery"]; out[typ]["p_disc"] = float(2 * norm.sf(abs(e["est"] / e["se"])))
    q = multipletests([out[t]["p_disc"] for t in TYPES], method="fdr_bh")[1]
    conf = []
    for t, qq in zip(TYPES, q):
        out[t]["q_disc"] = float(qq)
        if qq < 0.05:
            d0 = out[t]["discovery"]["est"]; ests = [(out[t][p]["est"], out[t][p]["se"]) for p in POPS[1:]]
            m, s = pool(ests)
            d = dict(type=t, disc=d0, pooled=m, pooled_se=s, p1=float(norm.sf(np.sign(d0) * m / s)), same=bool(all(np.sign(e[0]) == np.sign(d0) for e in ests)))
            conf.append(d)
    conf = holm(conf)
    for d in conf:
        out[d["type"]].update(pooled=d["pooled"], pooled_se=d["pooled_se"], p1=d["p1"], same=d["same"], confirmed=d["confirmed"])
    return out


def usage(D):
    u = D.groupby(["pop", "cat"]).size().unstack(0).fillna(0)
    share = (u / u.sum()).round(4)
    ntm = D.groupby("pop").apply(lambda g: g.groupby(["match_id", "team"]).ngroups)
    per_tm = (u / ntm).round(2)
    return dict(counts=u.astype(int).to_dict(), share=share.to_dict(), per_team_match=per_tm.to_dict())


def main():
    D = pd.read_parquet(os.path.join(PROC, "agent_combos_P.parquet"))
    res = dict(n_P={p: int((D["pop"] == p).sum()) for p in POPS}, n_matches={p: int(D[D["pop"] == p].match_id.nunique()) for p in POPS},
               usage=usage(D))
    res["cell_level"] = cell_level(D)
    res["type_level"] = type_level(D)
    # robustness for anything confirmed (also reported for type-level estimates as descriptive)
    conf_rules = [r for r in res["cell_level"]["rules"] if r["confirmed"]]
    conf_types = [t for t in TYPES if res["type_level"][t].get("confirmed")]
    rob = {}
    variants = {"h45": (D, "net45"), "no_shot3": (D[~D.shot3], "net20"),
                "completion_matched": (D[D.a1_complete & ~D.cat.isin(["solo_carry", "shot"])], "net20")}
    for name, (DD, y) in variants.items():
        rob[name] = {}
        for t in TYPES:
            rob[name][t] = {p: type_level_pop(DD[DD["pop"] == p], y, t) for p in POPS}
        for r in conf_rules:
            key = f"{r['cell']}|{r['cat']}"
            rob[name][key] = {}
            for p in POPS:
                T = rule_table(DD[DD["pop"] == p], y).set_index(["cell", "cat"])
                if (r["cell"], r["cat"]) in T.index:
                    v = T.loc[(r["cell"], r["cat"])]
                    rob[name][key][p] = dict(diff=float(v["diff"]), se=float(v.se), ref=v.ref_cat, n=int(v["size"]))
    res["robustness"] = rob
    res["confirmed_cell_rules"] = conf_rules; res["confirmed_types"] = conf_types
    json.dump(res, open(os.path.join(ROOT, "results", "agent_combos.json"), "w"), indent=1, default=float)
    cl = res["cell_level"]
    print("cell-level: searched", cl["searched"], "BH", cl["bh"], "testable", cl["testable"], "confirmed", cl["confirmed"])
    for r in cl["rules"]:
        print(r["cell"], r["cat"], "vs", r["ref"], f"disc {r['disc']:+.4f}±{r['disc_se']:.4f} pooled {r['pooled']:+.4f}±{r['pooled_se']:.4f} p1 {r['p1']:.3g}", r["conf"], r["confirmed"])
    print("untestable", cl["untestable"])
    print(pd.DataFrame(cl["disc_rule_summary"]))
    for t in TYPES:
        o = res["type_level"][t]
        print(t, {p: (round(o[p]["est"], 4), round(o[p]["se"], 4), o[p]["cells"], o[p]["n"]) for p in POPS}, "q", round(o["q_disc"], 4), o.get("confirmed"))
    for name in rob:
        print(name, {t: {p: (round(rob[name][t][p]["est"], 4), round(rob[name][t][p]["se"], 4)) for p in POPS if rob[name][t][p]} for t in TYPES})
    print(pd.DataFrame(res["usage"]["share"]))


if __name__ == "__main__":
    main()
