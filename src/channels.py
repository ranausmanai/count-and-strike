"""Where does quality matter less? (exploratory, post-hoc — see DEVIATIONS #3)

1. Quality gradient by chance channel: Poisson GLM of team-match xG in each channel on the market's
   log attacking strength (lam_own) and opponent strength (lam_opp). Elasticity < 1 means the channel is
   'flatter' than overall chance creation, so weak teams lose less ground there.
   Also: the underdog's share of each channel's xG in mismatches (underdog pwin < 0.25).
2. Cheap-skill map: team-season metrics. Persistence = split-half (odd/even match) reliability with
   Spearman-Brown correction. Wealth link = correlation with team market rating.
   'Cheap skills' are persistent but weakly tied to rating.
Replication: every quantity is computed per league; we report the leave-one-league-out spread.
"""
import json, os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, os.path.dirname(__file__))
from frontier import load_segments, ROOT, PROC

CH = ["build_up", "counter", "high_regain", "corner", "free_kick", "throw_in", "penalty"]


def team_match_table():
    p = pd.read_parquet(os.path.join(PROC, "possessions.parquet"))
    seg = load_segments(min_minutes=0.0)
    info = seg.drop_duplicates(["match_id", "team"])[["match_id", "team", "opp", "home", "lam_own", "lam_opp", "pwin", "ploss", "league", "date"]]
    # possession teams use canonical names already
    x = p.pivot_table(index=["match_id", "poss_team"], columns="channel", values="xg", aggfunc="sum", fill_value=0).reset_index()
    n = p.pivot_table(index=["match_id", "poss_team"], columns="channel", values="xg", aggfunc="size", fill_value=0).reset_index()
    x = x.rename(columns={"poss_team": "team"}); n = n.rename(columns={"poss_team": "team"})
    n.columns = [c if c in ("match_id", "team") else "n_" + c for c in n.columns]
    t = info.merge(x, on=["match_id", "team"], how="left").merge(n, on=["match_id", "team"], how="left").fillna(0)
    # opponent's channel xG = against
    opp = t[["match_id", "team"] + CH].rename(columns={"team": "opp", **{c: c + "_ag" for c in CH}})
    t = t.merge(opp, on=["match_id", "opp"], how="left")
    t["xg_total"] = t[CH].sum(1); t["xg_total_ag"] = t[[c + "_ag" for c in CH]].sum(1)
    t["open_play"] = t[["build_up", "counter", "high_regain"]].sum(1)
    t["set_piece"] = t[["corner", "free_kick", "throw_in"]].sum(1)
    t["open_play_ag"] = t[["build_up_ag", "counter_ag", "high_regain_ag"]].sum(1)
    t["set_piece_ag"] = t[["corner_ag", "free_kick_ag", "throw_in_ag"]].sum(1)
    return t.dropna(subset=["lam_own"])


def elasticities(t, cols):
    out = {}
    X = sm.add_constant(pd.DataFrame(dict(l_own=np.log(t.lam_own), l_opp=np.log(t.lam_opp), home=t.home)))
    for c in cols:
        m = sm.GLM(t[c].values, X.values, family=sm.families.Poisson()).fit(scale="X2", cov_type="cluster",
                                                                            cov_kwds={"groups": pd.factorize(t.team)[0]})
        out[c] = dict(e_own=float(m.params[1]), se_own=float(m.bse[1]), e_opp=float(m.params[2]), se_opp=float(m.bse[2]),
                      mean_xg=float(t[c].mean()))
    return out


def split_half(t, metric_fn, min_matches=30):
    rows = []
    for (lg, tm), d in t.sort_values("date").groupby(["league", "team"]):
        if len(d) < min_matches:
            continue
        a, b = d.iloc[0::2], d.iloc[1::2]
        rows.append((lg, tm, metric_fn(a), metric_fn(b), metric_fn(d), np.mean(np.log(d.lam_own / d.lam_opp))))
    r = pd.DataFrame(rows, columns=["league", "team", "h1", "h2", "full", "rating"]).replace([np.inf, -np.inf], np.nan).dropna()
    rho = r[["h1", "h2"]].corr().iloc[0, 1]
    rel = 2 * rho / (1 + rho) if rho > -1 else np.nan
    wealth = r[["full", "rating"]].corr().iloc[0, 1]
    by_league = {}
    for lg, d in r.groupby("league"):
        if len(d) >= 15:
            rr = d[["h1", "h2"]].corr().iloc[0, 1]
            by_league[lg] = dict(rel=float(2 * rr / (1 + rr)), wealth=float(d[["full", "rating"]].corr().iloc[0, 1]))
    return dict(reliability=float(rel), wealth_corr=float(wealth), n_teams=len(r), by_league=by_league,
                spread=float(r.full.std()))


def main():
    t = team_match_table()
    t.to_parquet(os.path.join(PROC, "team_match_channels.parquet"))
    cols = CH + ["open_play", "set_piece", "xg_total"]
    res = {"elasticity_all": elasticities(t, cols)}
    res["elasticity_by_league"] = {lg: elasticities(d, ["open_play", "set_piece", "counter", "corner", "xg_total"])
                                   for lg, d in t.groupby("league") if lg != "Bundesliga"}
    print("Elasticity of channel xG w.r.t. own market attack strength (e_own) and opponent (e_opp)")
    for c, v in res["elasticity_all"].items():
        print(f"  {c:12s} e_own {v['e_own']:+.2f} ({v['se_own']:.2f})  e_opp {v['e_opp']:+.2f} ({v['se_opp']:.2f})  mean {v['mean_xg']:.3f}")
    for lg, v in res["elasticity_by_league"].items():
        print("  ", lg, {k: round(x["e_own"], 2) for k, x in v.items()})
    # underdog share by channel in mismatches
    mm = t[t.pwin < 0.25]
    share = {c: float(mm[c].sum() / (mm[c].sum() + mm[c + "_ag"].sum())) for c in CH + ["open_play", "set_piece", "xg_total"]}
    res["underdog_share_mismatch"] = share; res["n_mismatch"] = len(mm)
    print("underdog xG share in mismatches (pwin<0.25, n=%d):" % len(mm), {k: round(v, 3) for k, v in share.items()})

    # cheap-skill map (team-season metrics)
    eps = 1e-9
    metrics = {
        # attacking efficiencies (per opportunity)
        "corner_xg_per_corner": lambda d: d.corner.sum() / (d.n_corner.sum() + eps),
        "fk_xg_per_att_fk": lambda d: d.free_kick.sum() / (d.n_free_kick.sum() + eps),
        "throw_xg_per_att_throw": lambda d: d.throw_in.sum() / (d.n_throw_in.sum() + eps),
        "counter_xg_per_match": lambda d: d.counter.mean(),
        "high_regain_xg_per_match": lambda d: d.high_regain.mean(),
        "buildup_xg_per_match": lambda d: d.build_up.mean(),
        "corners_won_per_match": lambda d: d.n_corner.mean(),
        # defensive (lower = better)
        "corner_xg_conceded_per_match": lambda d: d.corner_ag.mean(),
        "counter_xg_conceded_per_match": lambda d: d.counter_ag.mean(),
        "buildup_xg_conceded_per_match": lambda d: d.build_up_ag.mean(),
        "setpiece_share_of_own_xg": lambda d: d.set_piece.sum() / (d.xg_total.sum() + eps),
    }
    res["cheap_skill_map"] = {k: split_half(t, f) for k, f in metrics.items()}
    print("\nCheap-skill map: reliability (split-half, SB-corrected) vs correlation with market rating")
    for k, v in res["cheap_skill_map"].items():
        print(f"  {k:32s} rel {v['reliability']:+.2f}  wealth {v['wealth_corr']:+.2f}  sd {v['spread']:.4f}  n {v['n_teams']}")
    json.dump(res, open(os.path.join(ROOT, "results", "channels.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
