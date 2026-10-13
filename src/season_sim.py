"""Season simulator: Brentford in a 2025/26-strength Premier League.

Ratings: log lam = mu + home + att[team] - def[opp], fitted by least squares to the 2025/26 closing-market rates.
Non-Brentford matches: Dixon-Coles Poisson scorelines. Brentford matches: the outcome distribution of the in-match DP
under a chosen intervention and policy.

Interventions (each restricted to an empirically observed range):
  setpiece   cheap-skill upgrade to the 90th percentile of *true* team skill (observed SD × sqrt(reliability)) in
             corner xG/corner, attacking-throw xG/throw and corner xG conceded (channels.py), applied as Δ xG/match
  tempo      an intensity mode menu {×(1-c), ×1, ×(1+c)} on both teams' rates (share unchanged). c = 1 SD of the
             between-team habitual-tempo spread (P4 data). The in-match DP chooses the mode by (minute, goal diff)
  risk dial  DP terminal utility = points + κ·1[win] (P5: does the title-optimal κ differ from 0?)
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import dp

ROOT = os.path.join(os.path.dirname(__file__), "..")
K = 95; G = 2 * dp.GMAX + 1
TEAM = "Brentford"


def ratings(season=2025, league="EPL"):
    mk = pd.read_parquet(os.path.join(ROOT, "data", "proc", "market.parquet"))
    d = mk[(mk.league == league) & (mk.season == season)]
    teams = sorted(set(d.home) | set(d.away)); ix = {t: i for i, t in enumerate(teams)}; n = len(teams)
    rows, y = [], []
    for r in d.itertuples():
        a = np.zeros(2 * n + 2); a[0] = 1; a[1] = 1; a[2 + ix[r.home]] = 1; a[2 + n + ix[r.away]] = -1; rows.append(a); y.append(np.log(r.lam_h))
        b = np.zeros(2 * n + 2); b[0] = 1; b[2 + ix[r.away]] = 1; b[2 + n + ix[r.home]] = -1; rows.append(b); y.append(np.log(r.lam_a))
    A = np.array(rows); A = np.vstack([A, np.r_[0, 0, np.ones(n), np.zeros(n)], np.r_[0, 0, np.zeros(n), np.ones(n)]])
    y = np.r_[y, 0, 0]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    mu, home = coef[0], coef[1]; att = dict(zip(teams, coef[2:2 + n])); dfn = dict(zip(teams, coef[2 + n:]))
    actual = d
    return teams, mu, home, att, dfn, actual


def rate(mu, home, att, dfn, t, o, is_home):
    return np.exp(mu + home * is_home + att[t] - dfn[o])


def dc_sample(lh, la, n, rng, rho=-0.06):
    g = np.arange(11)
    from scipy.stats import poisson
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M[0, 0] *= 1 - lh * la * rho; M[0, 1] *= 1 + lh * rho; M[1, 0] *= 1 + la * rho; M[1, 1] *= 1 - rho
    M /= M.sum()
    k = rng.choice(M.size, n, p=M.ravel())
    return k // 11, k % 11


def match_dist(lf, la, c=0.0, kappa=0.0, policy="opt"):
    """Final goal-difference distribution for Brentford under the mode menu and utility."""
    if c == 0:
        RF = np.full((K, G, 1), lf / K); RA = np.full((K, G, 1), la / K)
    else:
        m = np.array([1 - c, 1.0, 1 + c])
        RF = np.broadcast_to(lf / K * m, (K, G, 3)).copy(); RA = np.broadcast_to(la / K * m, (K, G, 3)).copy()
    g = np.arange(G) - dp.GMAX
    U = np.where(g > 0, 3.0 + kappa, np.where(g == 0, 1.0, 0.0))
    if policy == "opt":
        V, pol = dp.solve(RF, RA, utility=U)
    else:  # fixed mode index
        pol = np.full((K, G), policy)
    return dp.outcome_dist(RF, RA, pol)


def simulate(n_sims=200_000, sp_for=0.0, sp_against=0.0, c=0.0, kappa=0.0, att_boost=0.0, seed=7, policy="opt", return_all=False, team=None):
    TEAM = team or globals()["TEAM"]
    teams, mu, home, att, dfn, actual = ratings()
    rng = np.random.default_rng(seed)
    n = len(teams); ix = {t: i for i, t in enumerate(teams)}
    pts = np.zeros((n_sims, n)); gd = np.zeros((n_sims, n))
    att = dict(att); att[TEAM] += att_boost
    for h in teams:
        for a in teams:
            if h == a:
                continue
            lh = rate(mu, home, att, dfn, h, a, 1); la = rate(mu, home, att, dfn, a, h, 0)
            if TEAM in (h, a):
                us_home = h == TEAM
                lf, lagn = (lh, la) if us_home else (la, lh)
                lf = max(lf + sp_for, 0.05); lagn = max(lagn - sp_against, 0.05)
                p = match_dist(lf, lagn, c, kappa, policy)
                gds = rng.choice(np.arange(G) - dp.GMAX, n_sims, p=p / p.sum())
                us, them = ix[TEAM], ix[a if us_home else h]
                pts[:, us] += np.where(gds > 0, 3, np.where(gds == 0, 1, 0)); pts[:, them] += np.where(gds < 0, 3, np.where(gds == 0, 1, 0))
                gd[:, us] += gds; gd[:, them] -= gds
            else:
                hg, ag = dc_sample(lh, la, n_sims, rng)
                pts[:, ix[h]] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0)); pts[:, ix[a]] += np.where(hg < ag, 3, np.where(hg == ag, 1, 0))
                gd[:, ix[h]] += hg - ag; gd[:, ix[a]] += ag - hg
    key = pts * 1000 + gd + rng.random(pts.shape) * 0.1
    rank = (-key).argsort(1).argsort(1) + 1
    b = ix[TEAM]
    res = dict(E_pts=float(pts[:, b].mean()), sd_pts=float(pts[:, b].std()), p_title=float((rank[:, b] == 1).mean()),
               p_top4=float((rank[:, b] <= 4).mean()), p_top6=float((rank[:, b] <= 6).mean()), mean_rank=float(rank[:, b].mean()),
               p_releg=float((rank[:, b] >= n - 2).mean()))
    if return_all:
        res["all_teams"] = {t: dict(E_pts=float(pts[:, ix[t]].mean()), p_title=float((rank[:, ix[t]] == 1).mean())) for t in teams}
    return res


def main():
    teams, mu, home, att, dfn, actual = ratings()
    out = {"ratings": {t: dict(att=att[t], dfn=dfn[t]) for t in teams}, "mu": mu, "home": home}
    ch = json.load(open(os.path.join(ROOT, "results", "channels.json")))["cheap_skill_map"]
    tm = pd.read_parquet(os.path.join(ROOT, "data", "proc", "team_match_channels.parquet"))
    z90 = 1.2816
    def true_sd(k):
        return ch[k]["spread"] * np.sqrt(max(ch[k]["reliability"], 0))
    corners_pm = tm.n_corner.mean(); throws_pm = tm.n_throw_in.mean()
    sp_for = z90 * (true_sd("corner_xg_per_corner") * corners_pm + true_sd("throw_xg_per_att_throw") * throws_pm)
    sp_against = z90 * true_sd("corner_xg_conceded_per_match")
    c = 0.09
    out["setpiece_deltas"] = dict(sp_for=sp_for, sp_against=sp_against, corners_pm=corners_pm, throws_pm=throws_pm)
    print("set-piece Δ xG/match: for +%.3f, against −%.3f" % (sp_for, sp_against))
    scen = {
        "baseline": dict(),
        "setpiece_p90": dict(sp_for=sp_for, sp_against=sp_against),
        "tempo_dp_c09": dict(c=c),
        "setpiece+tempo": dict(sp_for=sp_for, sp_against=sp_against, c=c),
    }
    for k in (0.5, 1, 2, 4, 8):
        scen[f"setpiece+tempo_kappa{k}"] = dict(sp_for=sp_for, sp_against=sp_against, c=c, kappa=k)
    for k, kw in scen.items():
        out[k] = simulate(**kw)
        print(k, {a: round(b, 4) for a, b in out[k].items()})
    # context: how much pure attacking quality would equal the playbook, and what title odds would take
    for boost in (0.05, 0.1, 0.2, 0.3, 0.4):
        out[f"att_boost_{boost}"] = simulate(att_boost=boost, n_sims=100_000)
        print("att boost", boost, {a: round(b, 4) for a, b in out[f"att_boost_{boost}"].items()})
    out["league_baseline"] = simulate(n_sims=100_000, return_all=True)["all_teams"]
    json.dump(out, open(os.path.join(ROOT, "results", "season_sim.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
