"""Out-of-season, out-of-provider replication of the channel analysis on Wyscout 2017/18 (Pappalardo et al. 2019).
Possessions are reconstructed from the event stream; xG from the StatsBomb-trained location model (xg_model.py).
Channel definitions mirror possessions.py as closely as the two schemas allow."""
import difflib, json, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from xg_model import feats
from channels import elasticities, split_half, CH

ROOT = os.path.join(os.path.dirname(__file__), "..")
WY = os.path.join(ROOT, "data", "raw", "wy")
LEAGUES = {"England": "EPL", "Spain": "LaLiga", "Germany": "Bundesliga", "Italy": "SerieA", "France": "Ligue1"}
CONTROL = {"Pass", "Free Kick", "Shot", "Others on the ball"}
RESTART = {"Corner", "Throw in", "Free Kick", "Free kick cross", "Free kick shot", "Goal kick", "Penalty"}
XG = pickle.load(open(os.path.join(ROOT, "data", "proc", "xg_model.pkl"), "rb"))


def league_table(country, lg):
    ms = json.load(open(os.path.join(WY, f"matches_{country}.json")))
    ev = json.load(open(os.path.join(WY, f"events_{country}.json")))
    names = {}
    minfo = {}
    for m in ms:
        lab = m["label"].split(",")[0]; h, a = [x.strip() for x in lab.split(" - ")]
        sides = {v["side"]: int(k) for k, v in m["teamsData"].items()}
        names[sides["home"]] = h; names[sides["away"]] = a
        minfo[m["wyId"]] = dict(home=sides["home"], away=sides["away"], date=m["dateutc"][:10])
    ev.sort(key=lambda e: (e["matchId"], e["matchPeriod"], e["eventSec"]))
    shots, counts = [], {}
    cur_team = None; start = None; poss_counter = False
    prev_match = None
    for e in ev:
        mid = e["matchId"]
        if mid != prev_match:
            cur_team = None; prev_match = mid
        if e["matchPeriod"] not in ("1H", "2H"):
            continue
        name, sub, tm = e["eventName"], e["subEventName"], e["teamId"]
        x = e["positions"][0]["x"] if e["positions"] else np.nan
        tags = {t["id"] for t in e["tags"]}
        key = (mid, tm)
        c = counts.setdefault(key, dict(n_corner=0, n_throw_in=0, n_free_kick=0))
        if name == "Free Kick":
            if sub == "Corner":
                c["n_corner"] += 1
            elif sub == "Throw in" and x >= 66.7:
                c["n_throw_in"] += 1
            elif sub in ("Free Kick", "Free kick cross", "Free kick shot") and x >= 50:
                c["n_free_kick"] += 1
        if name in CONTROL and (tm != cur_team or (name == "Free Kick")):
            cur_team = tm; start = dict(name=name, sub=sub, x=x, t=e["eventSec"], period=e["matchPeriod"]); poss_counter = False
        if 1901 in tags and tm == cur_team:
            poss_counter = True
        is_shot = name == "Shot" or (name == "Free Kick" and sub in ("Free kick shot", "Penalty"))
        if is_shot and start is not None:
            pen = sub == "Penalty"
            y = e["positions"][0]["y"]
            d = pd.DataFrame(dict(x=[x / 100 * 105], y=[y / 100 * 68], head=[403 in tags], fk=[sub == "Free kick shot"]))
            xg = 0.76 if pen else float(XG.predict_proba(feats(d))[0, 1])
            secs = e["eventSec"] - start["t"] if start["period"] == e["matchPeriod"] else 999
            st = start
            if pen:
                ch = "penalty"
            elif st["name"] == "Free Kick" and st["sub"] == "Corner" and secs <= 20:
                ch = "corner"
            elif st["name"] == "Free Kick" and st["sub"] == "Throw in" and st["x"] >= 66.7 and secs <= 20:
                ch = "throw_in"
            elif st["name"] == "Free Kick" and st["sub"] in ("Free Kick", "Free kick cross", "Free kick shot") and st["x"] >= 50 and secs <= 20:
                ch = "free_kick"
            elif poss_counter or 1901 in tags or (secs <= 15 and st["name"] != "Free Kick" and st["x"] < 66.7):
                ch = "counter"
            elif st["name"] != "Free Kick" and st["x"] >= 66.7:
                ch = "high_regain"
            else:
                ch = "build_up"
            shots.append(dict(match_id=mid, team_id=tm, channel=ch, xg=xg, goal=101 in tags))
    s = pd.DataFrame(shots)
    rows = []
    for mid, mi in minfo.items():
        for tid, oid, home in ((mi["home"], mi["away"], 1), (mi["away"], mi["home"], 0)):
            r = dict(match_id=mid, team=names[tid], opp=names[oid], home=home, date=mi["date"], league=lg, **counts.get((mid, tid), {}))
            d = s[(s.match_id == mid) & (s.team_id == tid)]
            for chn in CH:
                r[chn] = d.xg[d.channel == chn].sum()
            rows.append(r)
    t = pd.DataFrame(rows).fillna(0)
    return t, s


def main():
    mk = pd.read_parquet(os.path.join(ROOT, "data", "proc", "market.parquet")); mk = mk[mk.season == 2017]
    tabs = []
    for country, lg in LEAGUES.items():
        t, s = league_table(country, lg)
        fd = sorted(set(mk[mk.league == lg].home))
        manual = {"Internazionale": "Inter", "Milan": "Milan", "Deportivo Alavés": "Alaves", "Deportivo La Coruña": "La Coruna",
                  "Athletic Club": "Ath Bilbao", "Atlético Madrid": "Ath Madrid", "Real Betis": "Betis", "Celta de Vigo": "Celta",
                  "Bayern München": "Bayern Munich", "Borussia M'gladbach": "M'gladbach", "Bayer Leverkusen": "Leverkusen",
                  "Paris Saint-Germain": "Paris SG", "Saint-Étienne": "St Etienne", "Olympique Lyonnais": "Lyon", "Olympique Marseille": "Marseille",
                  "Manchester City": "Man City", "Manchester United": "Man United", "Tottenham Hotspur": "Tottenham", "AFC Bournemouth": "Bournemouth",
                  "West Bromwich Albion": "West Brom", "Brighton & Hove Albion": "Brighton", "Huddersfield Town": "Huddersfield",
                  "Real Sociedad": "Sociedad", "Rayo Vallecano": "Vallecano", "Hellas Verona": "Verona", "SPAL": "Spal",
                  "Eintracht Frankfurt": "Ein Frankfurt", "Köln": "FC Koln", "Hertha BSC": "Hertha", "Mainz 05": "Mainz",
                  "Schalke 04": "Schalke 04", "Hamburger SV": "Hamburg", "Borussia Dortmund": "Dortmund", "VfB Stuttgart": "Stuttgart"}
        teams = sorted(set(t.team))
        mp = {}
        for n in teams:
            u = n.encode().decode("unicode_escape") if "\\u" in n else n
            mp[n] = manual.get(u, manual.get(n)) or difflib.get_close_matches(u, fd, 1, 0)[0]
        t["team_fd"] = t.team.map(mp); t["opp_fd"] = t.opp.map(mp)
        print(lg, {k: v for k, v in mp.items() if k != v})
        t["home_fd"] = np.where(t.home == 1, t.team_fd, t.opp_fd); t["away_fd"] = np.where(t.home == 1, t.opp_fd, t.team_fd)
        t = t.merge(mk[mk.league == lg][["home", "away", "lam_h", "lam_a", "pH", "pA"]].rename(columns={"home": "home_fd", "away": "away_fd"}),
                    on=["home_fd", "away_fd"], how="left")
        t["lam_own"] = np.where(t.home == 1, t.lam_h, t.lam_a); t["lam_opp"] = np.where(t.home == 1, t.lam_a, t.lam_h)
        t["pwin"] = np.where(t.home == 1, t.pH, t.pA)
        print(lg, "unmatched", t.lam_own.isna().sum(), "shots", len(s), "xg", round(s.xg.sum(), 1), "goals", s.goal.sum())
        tabs.append(t)
    t = pd.concat(tabs, ignore_index=True).dropna(subset=["lam_own"])
    opp = t[["match_id", "team"] + CH].rename(columns={"team": "opp", **{c: c + "_ag" for c in CH}})
    t = t.merge(opp, on=["match_id", "opp"], how="left")
    t["xg_total"] = t[CH].sum(1); t["xg_total_ag"] = t[[c + "_ag" for c in CH]].sum(1)
    t["open_play"] = t[["build_up", "counter", "high_regain"]].sum(1); t["set_piece"] = t[["corner", "free_kick", "throw_in"]].sum(1)
    t["open_play_ag"] = t[["build_up_ag", "counter_ag", "high_regain_ag"]].sum(1); t["set_piece_ag"] = t[["corner_ag", "free_kick_ag", "throw_in_ag"]].sum(1)
    t.to_parquet(os.path.join(ROOT, "data", "proc", "wy_team_match_channels.parquet"))
    cols = CH + ["open_play", "set_piece", "xg_total"]
    res = {"elasticity_all": elasticities(t, cols)}
    res["elasticity_by_league"] = {lg: elasticities(d, ["open_play", "set_piece", "counter", "corner", "xg_total"]) for lg, d in t.groupby("league")}
    for c, v in res["elasticity_all"].items():
        print(f"  {c:12s} e_own {v['e_own']:+.2f} ({v['se_own']:.2f})  e_opp {v['e_opp']:+.2f}  mean {v['mean_xg']:.3f}")
    for lg, v in res["elasticity_by_league"].items():
        print("  ", lg, {k: round(x["e_own"], 2) for k, x in v.items()})
    mm = t[t.pwin < 0.25]
    res["underdog_share_mismatch"] = {c: float(mm[c].sum() / (mm[c].sum() + mm[c + "_ag"].sum())) for c in CH + ["open_play", "set_piece", "xg_total"]}
    res["n_mismatch"] = len(mm)
    print("underdog share", {k: round(v, 3) for k, v in res["underdog_share_mismatch"].items()})
    eps = 1e-9
    metrics = {"corner_xg_per_corner": lambda d: d.corner.sum() / (d.n_corner.sum() + eps),
               "fk_xg_per_att_fk": lambda d: d.free_kick.sum() / (d.n_free_kick.sum() + eps),
               "throw_xg_per_att_throw": lambda d: d.throw_in.sum() / (d.n_throw_in.sum() + eps),
               "counter_xg_per_match": lambda d: d.counter.mean(), "buildup_xg_per_match": lambda d: d.build_up.mean(),
               "corners_won_per_match": lambda d: d.n_corner.mean(), "corner_xg_conceded_per_match": lambda d: d.corner_ag.mean(),
               "counter_xg_conceded_per_match": lambda d: d.counter_ag.mean(), "buildup_xg_conceded_per_match": lambda d: d.build_up_ag.mean()}
    t["date"] = pd.to_datetime(t.date)
    res["cheap_skill_map"] = {k: split_half(t, f) for k, f in metrics.items()}
    for k, v in res["cheap_skill_map"].items():
        print(f"  {k:32s} rel {v['reliability']:+.2f}  wealth {v['wealth_corr']:+.2f}  n {v['n_teams']}")
    json.dump(res, open(os.path.join(ROOT, "results", "wyscout_replication.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
