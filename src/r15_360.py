"""Round 15: full advantage search on 360 data (PREREGISTRATION_15.md).
  build  -> data/proc/r15_attack.parquet, data/proc/r15_press.parquet
  mine   -> results/r15_rules.json
"""
import gzip, json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(__file__))
from r4_structure import in_tri
ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "data", "raw", "sb360")
PROC = os.path.join(ROOT, "data", "proc")
RESTART = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
DISC = ("FIFA World Cup 2022", "UEFA Euro 2020", "UEFA Euro 2024", "African Cup of Nations 2023")
WOMEN = ("UEFA Women's Euro 2022", "UEFA Women's Euro 2025", "Women's World Cup 2023")


def tsec(e):
    h, m, s = e["timestamp"].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def windows(rows, S, team_key="team", sign=1):
    """own/opp xG in the 20 s after each row (vectorised per period)."""
    a = pd.DataFrame(rows)
    if a.empty:
        return a
    own = np.zeros(len(a)); opp = np.zeros(len(a))
    for per, idx in a.groupby("period").indices.items():
        s = S[S.period == per].sort_values("t")
        if s.empty:
            continue
        st = s.t.values; sx = s.xg.values; steam = s.team.values
        for team in np.unique(a[team_key].values[idx]):
            sel = idx[a[team_key].values[idx] == team]
            mine = steam == team
            cm = np.concatenate([[0], np.cumsum(np.where(mine, sx, 0))]); co = np.concatenate([[0], np.cumsum(np.where(~mine, sx, 0))])
            tt = a.t.values[sel]
            lo = np.searchsorted(st, tt - 0.01, "left"); hi = np.searchsorted(st, tt + 20, "right"); lo2 = np.searchsorted(st, tt + 0.01, "right")
            own[sel] = cm[hi] - cm[lo]; opp[sel] = co[hi] - co[lo2]
    a["own20"] = own; a["opp20"] = opp; a["net"] = own - opp
    return a


def one(m):
    mid = m["match_id"]
    pe, pf = os.path.join(RAW, f"{mid}_events.json.gz"), os.path.join(RAW, f"{mid}_three-sixty.json.gz")
    try:
        ev = [e for e in json.load(gzip.open(pe)) if e["period"] <= 2]
        fr = {f["event_uuid"]: f for f in json.load(gzip.open(pf))}
    except Exception:
        return None
    S = pd.DataFrame([(e["period"], tsec(e), e["team"]["name"], e["shot"].get("statsbomb_xg", 0.0)) for e in ev
                      if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"], columns=["period", "t", "team", "xg"])
    press_t = {}
    for e in ev:
        if e["type"]["name"] == "Pressure":
            press_t.setdefault((e["period"], e["team"]["name"]), []).append(tsec(e))
    for k in press_t:
        press_t[k] = np.sort(press_t[k])
    pattern = {}
    for e in ev:
        pattern.setdefault(e["possession"], e["play_pattern"]["name"])
    att, prs = [], []
    teams = sorted({e["team"]["name"] for e in ev})
    for e in ev:
        typ = e["type"]["name"]; f = fr.get(e["id"])
        if f is None or "location" not in e:
            continue
        x, y = e["location"][:2]
        opp = [p["location"] for p in f["freeze_frame"] if not p["teammate"] and not p.get("keeper")]
        mates = [p["location"] for p in f["freeze_frame"] if p["teammate"] and not p.get("actor") and not p.get("keeper")]
        d_opp = [np.hypot(o[0] - x, o[1] - y) for o in opp]
        d_mate = [np.hypot(o[0] - x, o[1] - y) for o in mates]
        base = dict(match_id=mid, comp=m["_comp"], period=e["period"], t=tsec(e), x=x, y=y)
        # ---------- pressing decision: opponent receives the ball (we are the non-actor team) ----------
        if typ == "Ball Receipt*" and "outcome" not in e.get("ball_receipt", {}) and pattern.get(e["possession"]) in ("Regular Play", "From Counter"):
            receiver_team = e["team"]["name"]
            ours = [o for o in opp]  # 'opp' of the receiver = us
            us_team = [t for t in teams if t != receiver_team]
            if not us_team:
                continue
            us = us_team[0]
            pt = press_t.get((e["period"], us), np.array([]))
            t0 = tsec(e)
            pressed = bool(len(pt) and ((pt >= t0) & (pt <= t0 + 2)).any())
            prs.append(dict(base, team=us, receiver_team=receiver_team, press=pressed,
                            support=min(d_mate) if d_mate else 30.0, our_goalside=sum(o[0] > x for o in ours),
                            already=int(min(d_opp) < 3) if d_opp else 0))
            continue
        # ---------- attacking decision ----------
        if typ not in ("Pass", "Carry", "Dribble", "Shot") or e["team"]["name"] != e["possession_team"]["name"]:
            continue
        if typ == "Pass" and e["pass"].get("type", {}).get("name") in RESTART:
            continue
        if typ == "Shot" and e["shot"]["type"]["name"] != "Open Play":
            continue
        if typ == "Shot":
            act = "shot"
        elif typ == "Dribble":
            act = "dribble"
        elif typ == "Carry":
            ex, ey = e["carry"]["end_location"][:2]
            if abs(ex - x) + abs(ey - y) < 1:
                continue
            act = "carry_fwd" if ex - x >= 5 else "carry_other"
        else:
            p = e["pass"]; ex, ey = p["end_location"][:2]; fwd = ex - x; ln = p.get("length", np.hypot(ex - x, ey - y))
            h = p.get("height", {}).get("name", "Ground Pass")
            inbox = ex >= 102 and 18 <= ey <= 62
            if p.get("through_ball") or p.get("technique", {}).get("name") == "Through Ball":
                act = "through_ball"
            elif p.get("cut_back"):
                act = "cutback"
            elif p.get("cross") or (abs(y - 40) > 22 and x >= 96 and inbox):
                act = "cross_high" if h == "High Pass" else "cross_low"
            elif p.get("switch"):
                act = "switch"
            elif ln >= 30 and fwd >= 10:
                act = "long_fwd"
            elif fwd >= 5:
                act = "short_fwd_ground" if h == "Ground Pass" else "short_fwd_air"
            elif fwd <= -5:
                act = "back_pass"
            else:
                act = "lateral"
        lane = sum(in_tri(o, (x, y), (120, 36), (120, 44)) for o in opp)
        att.append(dict(base, team=e["team"]["name"], action=act, opp_goalside=sum(o[0] > x for o in opp),
                        nearest_opp=min(d_opp) if d_opp else 30.0, mates_ahead=sum(mm[0] > x for mm in mates), lane=lane))
    A = windows(att, S); P = windows(prs, S)
    return A, P


def cellify(A):
    A["zx"] = pd.cut(A.x, [-1, 40, 60, 80, 96, 108, 121], labels=False)
    A["zy"] = np.select([(A.y - 40).abs() <= 13, A.y < 40], [1, 0], 2)
    return A


def build():
    ms = json.load(open(os.path.join(ROOT, "data", "raw", "sb360_matches.json")))
    As, Ps = [], []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(one, ms, chunksize=4):
            if r is not None:
                As.append(r[0]); Ps.append(r[1])
    for name, L in (("attack", As), ("press", Ps)):
        D = cellify(pd.concat(L, ignore_index=True))
        D["pop"] = np.select([D.comp.str.startswith(DISC), D.comp.str.startswith(WOMEN)], ["discovery", "confirm_women"], "confirm_club")
        if name == "attack":
            D["cell"] = (D.zx.astype(int).astype(str) + D.zy.astype(str) + np.select([D.opp_goalside <= 3, D.opp_goalside <= 6], ["0", "1"], "2")
                         + np.select([D.nearest_opp < 3, D.nearest_opp <= 7], ["0", "1"], "2") + np.select([D.mates_ahead <= 1, D.mates_ahead <= 3], ["0", "1"], "2"))
        else:
            D["cell"] = (D.zx.astype(int).astype(str) + D.zy.astype(str) + np.select([D.support < 8, D.support <= 15], ["0", "1"], "2")
                         + np.select([D.our_goalside <= 5, D.our_goalside <= 8], ["0", "1"], "2") + D.already.astype(str))
            D["action"] = np.where(D.press, "press", "hold")
        D.to_parquet(os.path.join(PROC, f"r15_{name}.parquet"))
        print(name, D.groupby(["pop", "action"]).size().unstack(0))


def rule_table(D):
    st = D.groupby(["cell", "action"]).net.agg(["mean", "var", "size"]).reset_index()
    ref = st.sort_values("size", ascending=False).drop_duplicates("cell").set_index("cell")
    c = st.merge(ref[["action", "mean", "var", "size"]].rename(columns=lambda x: "ref_" + x), left_on="cell", right_index=True)
    c = c[c.action != c.ref_action].copy()
    c["diff"] = c["mean"] - c.ref_mean; c["se"] = np.sqrt(c["var"] / c["size"] + c.ref_var / c.ref_size)
    return c


def mine_family(D, fam):
    disc = rule_table(D[D["pop"] == "discovery"]); disc = disc[(disc["size"] >= 30) & (disc.ref_size >= 30)]
    disc["p"] = 2 * norm.sf(np.abs(disc["diff"] / disc.se)); disc["q"] = multipletests(disc.p, method="fdr_bh")[1] if len(disc) else []
    S = disc[disc.q < 0.05]
    C = {p: rule_table(D[D["pop"] == p]).set_index(["cell", "action"]) for p in ("confirm_club", "confirm_women")}
    conf = []
    for r in S.itertuples():
        ests = []
        for p, T in C.items():
            if (r.cell, r.action) in T.index:
                v = T.loc[(r.cell, r.action)]
                if v["size"] >= 10 and v.ref_size >= 10 and v.ref_action == r.ref_action:
                    ests.append((v["diff"], v.se))
        if len(ests) == 2:
            w = np.array([1 / e[1] ** 2 for e in ests]); pooled = float((w * np.array([e[0] for e in ests])).sum() / w.sum()); pse = float(1 / np.sqrt(w.sum()))
            conf.append(dict(family=fam, cell=r.cell, action=r.action, ref=r.ref_action, disc=float(r.diff), pooled=pooled, pooled_se=pse,
                             p1=float(norm.sf(np.sign(r.diff) * pooled / pse)), same=bool(all(np.sign(e[0]) == np.sign(r.diff) for e in ests))))
    k = len(conf); conf = sorted(conf, key=lambda d: d["p1"]); flag = True
    for i, d in enumerate(conf):
        flag = flag and d["p1"] < 0.05 / (k - i); d["confirmed"] = bool(flag and d["same"])
    print(f"[{fam}] rules searched {len(disc)} | BH5% {len(S)} | testable {k} | confirmed {sum(d['confirmed'] for d in conf)}")
    return conf, dict(searched=int(len(disc)), bh=int(len(S)), testable=k)


def describe_attack(c):
    zx, zy, og, no, ma = map(int, c)
    zn = ["own third", "own half", "middle third", "attacking third", "edge of box", "in the box"][zx]
    return (f"{zn}, {['left', 'central', 'right'][zy]}; {['≤3', '4–6', '7+'][og]} opponents ahead; "
            f"{['pressed (<3 yds)', 'loosely marked (3–7)', 'free (>7)'][no]}; {['0–1', '2–3', '4+'][ma]} teammates ahead")


def describe_press(c):
    zx, zy, su, gs, al = map(int, c)
    zn = ["their own third", "their half", "middle third", "our half (attacking third for them)", "edge of our box", "our box"][zx]
    return (f"opponent receives in {zn}, {['left', 'central', 'right'][zy]}; support {['close (<8 yds)', 'medium (8–15)', 'isolated (>15)'][su]}; "
            f"{['≤5', '6–8', '9+'][gs]} of us goal-side; {'already marked' if al else 'unmarked'}")


def mine():
    A = pd.read_parquet(os.path.join(PROC, "r15_attack.parquet")); P = pd.read_parquet(os.path.join(PROC, "r15_press.parquet"))
    ca, sa = mine_family(A, "attack"); cp, sp = mine_family(P, "press")
    ntm_a = A.groupby(["match_id", "team"]).ngroups; ntm_p = P.groupby(["match_id", "team"]).ngroups
    play = []
    for d, D, ntm, desc in [(x, A, ntm_a, describe_attack) for x in ca] + [(x, P, ntm_p, describe_press) for x in cp]:
        if not d["confirmed"]:
            continue
        better, worse = (d["action"], d["ref"]) if d["pooled"] > 0 else (d["ref"], d["action"])
        freq = ((D.cell == d["cell"]) & (D.action == worse)).sum() / ntm
        d.update(better=better, worse=worse, situation=desc(d["cell"]), worse_per_match=float(freq),
                 season_xg_50pct=float(abs(d["pooled"]) * freq * 0.5 * 38))
        play.append(d)
    play = sorted(play, key=lambda d: -d["season_xg_50pct"])
    for d in play:
        print(f"{d['season_xg_50pct']:5.2f} xG/season | {d['family']:6s} | {d['situation']} -> {d['better'].upper()} not {d['worse']} (+{abs(d['pooled']):.4f}/decision, {d['worse_per_match']:.2f}/match)")
    print("TOTAL xG/season (50% adoption):", round(sum(d["season_xg_50pct"] for d in play), 2))
    json.dump(dict(attack=sa, press=sp, confirmed_playbook=play, all_attack=ca, all_press=cp), open(os.path.join(ROOT, "results", "r15_rules.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    {"build": build, "mine": mine}[sys.argv[1]]()
