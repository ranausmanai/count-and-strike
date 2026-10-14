"""Round 16: possession simulator / constrained policy iteration (PREREGISTRATION_16.md)."""
import json, os, sys
import numpy as np
import pandas as pd
ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
GAP = 30.0
A = pd.read_parquet(os.path.join(PROC, "r15_attack.parquet")).sort_values(["match_id", "period", "idx"]).reset_index(drop=True)
TAG = ""
if os.environ.get("NO_ADJUST") == "1":   # robustness: drop short adjustment carries (not genuine decisions)
    A = A[A.action != "carry_other"].reset_index(drop=True); TAG = "_noadjust"
CELLS = sorted(A.cell.unique()); CI = {c: i for i, c in enumerate(CELLS)}
ACTS = sorted(A.action.unique()); AI = {a: i for i, a in enumerate(ACTS)}
S, NA = len(CELLS), len(ACTS)
nxt = A.groupby(["match_id", "period"]).shift(-1)
A["s"] = A.cell.map(CI); A["a"] = A.action.map(AI)
A["ns"] = nxt.cell.map(CI); A["dt"] = nxt.t - A.t
A["kind"] = np.where(A.ns.isna() | (A.dt > GAP), 0, np.where(nxt.team == A.team, 1, 2))  # 0 terminal, 1 continue, 2 loss
A["start"] = (A.groupby(["match_id", "period"]).team.shift(1) != A.team) | (A.groupby(["match_id", "period"]).t.diff() > GAP)


def model(D):
    n = np.zeros((S, NA)); R = np.zeros((S, NA)); Pc = np.zeros((NA, S, S)); Pl = np.zeros((NA, S, S))
    s, a, k, ns, r = D.s.values, D.a.values, D.kind.values, D.ns.fillna(0).astype(int).values, D.sxg.values
    np.add.at(n, (s, a), 1); np.add.at(R, (s, a), r)
    m = k == 1; np.add.at(Pc, (a[m], s[m], ns[m]), 1)
    m = k == 2; np.add.at(Pl, (a[m], s[m], ns[m]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(n > 0, R / n, 0.0)
        Pc = np.where(n.T[:, :, None] > 0, Pc / n.T[:, :, None], 0.0); Pl = np.where(n.T[:, :, None] > 0, Pl / n.T[:, :, None], 0.0)
    pib = n / np.maximum(n.sum(1, keepdims=True), 1)
    return dict(n=n, R=R, Pc=Pc, Pl=Pl, pib=pib)


def Q(M, Vs, Vo):
    return M["R"] + np.einsum("ast,t->sa", M["Pc"], Vs) - np.einsum("ast,t->sa", M["Pl"], Vo)


def solve(M, pi_us, it=400):
    Vu = np.zeros(S); Vo = np.zeros(S)
    for _ in range(it):
        Vu_new = (pi_us * Q(M, Vu, Vo)).sum(1); Vo_new = (M["pib"] * Q(M, Vo, Vu)).sum(1)
        if np.max(np.abs(Vu_new - Vu)) + np.max(np.abs(Vo_new - Vo)) < 1e-10:
            Vu, Vo = Vu_new, Vo_new; break
        Vu, Vo = Vu_new, Vo_new
    return Vu, Vo


def policy_iteration(M, min_n=30, rounds=20):
    pi = M["pib"].copy(); elig = M["n"] >= min_n
    for _ in range(rounds):
        Vu, Vo = solve(M, pi)
        q = np.where(elig, Q(M, Vu, Vo), -np.inf)
        best = q.argmax(1); has = elig.any(1)
        new = M["pib"].copy(); new[has] = 0; new[np.where(has)[0], best[has]] = 1
        if np.allclose(new, pi):
            break
        pi = new
    return pi


def gain(M, pi, starts):
    """Value gain per possession start of policy pi vs behaviour (opponent = behaviour)."""
    # cells where pi chooses an action unobserved in this model fall back to behaviour
    pi = pi.copy(); bad = ((pi > 0) & (M["n"] < 1)).any(1); pi[bad] = M["pib"][bad]
    Vp, _ = solve(M, pi); Vb, _ = solve(M, M["pib"])
    w = np.bincount(starts, minlength=S) / len(starts)
    return float((w * (Vp - Vb)).sum()), Vb


def main():
    out = {}
    pops = {p: A[A["pop"] == p] for p in ("discovery", "confirm_club", "confirm_women")}
    Md = model(pops["discovery"])
    pistar = policy_iteration(Md)
    st_d = pops["discovery"][pops["discovery"].start].s.values
    out["naive_gain_per_possession"], _ = gain(Md, pistar, st_d)
    out["cells_changed"] = int(((pistar.argmax(1) != Md["pib"].argmax(1)) & (Md["n"].sum(1) >= 30)).sum())
    print("naive (discovery) gain per possession start:", out["naive_gain_per_possession"], "| cells changed:", out["cells_changed"])
    rng = np.random.default_rng(20261014)
    for p in ("confirm_club", "confirm_women"):
        D = pops[p]; M = model(D); st = D[D.start].s.values
        g, Vb = gain(M, pistar, st)
        mids = D.match_id.unique(); grp = D.groupby("match_id").indices
        bs = []
        for _ in range(100):
            pick = rng.choice(mids, len(mids)); Db = D.iloc[np.concatenate([grp[m] for m in pick])]
            Mb = model(Db); bs.append(gain(Mb, pistar, Db[Db.start].s.values)[0])
        starts_per_tm = len(st) / D.groupby(["match_id", "team"]).ngroups
        out[p] = dict(gain_per_possession=g, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      possessions_per_team_match=starts_per_tm, xg_per_team_match=g * starts_per_tm, xg_per_season=g * starts_per_tm * 38,
                      baseline_value_per_possession=float((np.bincount(st, minlength=S) / len(st) * Vb).sum()))
        print(p, out[p])
    # plays: one-step deviation advantages in each confirmation model (behaviour values), split immediate/downstream
    plays = []
    changed = np.where((pistar.argmax(1) != Md["pib"].argmax(1)) & (Md["n"].sum(1) >= 30))[0]
    conf_models = {}
    for p in ("confirm_club", "confirm_women"):
        M = model(pops[p]); Vu, Vo = solve(M, M["pib"]); conf_models[p] = (M, Vu, Vo)
    boot = {p: [] for p in conf_models}
    for p in conf_models:
        D = pops[p]; mids = D.match_id.unique(); grp = D.groupby("match_id").indices
        for _ in range(60):
            pick = rng.choice(mids, len(mids)); Mb = model(D.iloc[np.concatenate([grp[m] for m in pick])]); Vu, Vo = solve(Mb, Mb["pib"])
            boot[p].append((Mb, Q(Mb, Vu, Vo)))
    for s in changed:
        a_star, a_usual = int(pistar[s].argmax()), int(Md["pib"][s].argmax())
        row = dict(cell=CELLS[s], play=ACTS[a_star], instead_of=ACTS[a_usual], disc_n=int(Md["n"][s, a_star]))
        ok = True
        for p, (M, Vu, Vo) in conf_models.items():
            if M["n"][s, a_star] < 10 or M["n"][s, a_usual] < 10:
                ok = False; break
            q = Q(M, Vu, Vo); adv = q[s, a_star] - q[s, a_usual]; imm = M["R"][s, a_star] - M["R"][s, a_usual]
            bs = [b[1][s, a_star] - b[1][s, a_usual] for b in boot[p] if b[0]["n"][s, a_star] > 0 and b[0]["n"][s, a_usual] > 0]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row[p] = dict(adv=float(adv), immediate=float(imm), downstream=float(adv - imm), ci=[float(lo), float(hi)], n=int(M["n"][s, a_star]))
            ok = ok and lo > 0
        row["confirmed"] = bool(ok)
        if all(p in row for p in conf_models):
            plays.append(row)
    conf = [r for r in plays if r["confirmed"]]
    print(f"plays changed: {len(changed)} | testable in both: {len(plays)} | confirmed: {len(conf)}")
    def desc(c):
        zx, zy, og, no, ma = map(int, c)
        return (f"{['own third','own half','middle third','attacking third','edge of box','in the box'][zx]}/{['left','central','right'][zy]}, "
                f"{['≤3','4–6','7+'][og]} ahead, {['pressed','3–7 yds','free'][no]}, {['0–1','2–3','4+'][ma]} mates ahead")
    for r in sorted(conf, key=lambda r: -np.mean([r[p]["adv"] for p in conf_models])):
        ds = np.mean([r[p]["downstream"] for p in conf_models]); ad = np.mean([r[p]["adv"] for p in conf_models])
        print(f"  {desc(r['cell'])}: {r['play'].upper()} instead of {r['instead_of']}  adv {ad:+.4f} (downstream share {ds/ad:.0%})")
        r["situation"] = desc(r["cell"])
    out["plays"] = plays; out["n_confirmed_plays"] = len(conf)
    json.dump(out, open(os.path.join(ROOT, "results", f"r16_sim{TAG}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
