"""Match-plan generator: turns Project Underdog's findings into a plan for one fixture.

usage: python3 src/matchplan.py "<our team>" "<home team>" "<away team>" <div> <oddsH> <oddsD> <oddsA> [odds_over2.5 odds_under2.5]
       python3 src/matchplan.py Lincoln Lincoln "West Ham" E1 4.60 3.98 1.68 1.64 2.30

Every recommendation carries an evidence label:
  CONFIRMED   held in pre-registered tests on unseen leagues
  MODEL       exact optimisation on measured rates; the size of the tempo lever (±9%) is observed between teams, not proven in-game
  DESCRIPTIVE opponent facts from this season's data; our tests found no opponent-specific edge beyond strength
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import dp
from odds import implied_rates

ROOT = os.path.join(os.path.dirname(__file__), "..")
K = 95; G = 2 * dp.GMAX + 1
E_SP, E_OP = 0.47, 1.00            # mean set-piece / open-play strength elasticities across the three datasets
SP_SHARE_AVG = 0.29                # league-average share of non-penalty xG from set pieces (StatsBomb 2015/16)
LEAGUE_LAM = 1.35                  # average team goals per match
WINDOW_MULT = 1.5                  # confirmed 'break after they commit' multiplier (1.37–1.69)


def tempo_policy(lf, la, c=0.09):
    m = np.array([1 - c, 1.0, 1 + c])
    RF = np.broadcast_to(lf / K * m, (K, G, 3)).copy(); RA = np.broadcast_to(la / K * m, (K, G, 3)).copy()
    V, pol = dp.solve(RF, RA)
    Vn, _ = dp.solve(RF, RA, fixed_policy=np.ones((K, G), int))
    return pol, V[0, dp.GMAX] - Vn[0, dp.GMAX]


def team_profile(div, team):
    d = pd.read_csv(os.path.join(ROOT, "data", "raw", "fd", f"{div}_2627.csv"), encoding="latin-1")
    rows = []
    for _, r in d.iterrows():
        rows.append(dict(team=r.HomeTeam, sf=r.HS, sa=r.AS, cf=r.HC, ca=r.AC, gf=r.FTHG, ga=r.FTAG))
        rows.append(dict(team=r.AwayTeam, sf=r.AS, sa=r.HS, cf=r.AC, ca=r.HC, gf=r.FTAG, ga=r.FTHG))
    t = pd.DataFrame(rows).groupby("team").agg(n=("sf", "size"), sf=("sf", "mean"), sa=("sa", "mean"),
                                               cf=("cf", "mean"), ca=("ca", "mean"), gf=("gf", "mean"), ga=("ga", "mean"))
    t["tempo"] = t.sf + t.sa
    rk = lambda c, asc=False: int(t[c].rank(ascending=asc, method="min")[team])
    p = t.loc[team]
    return dict(n=int(p.n), shots_for=float(p.sf), shots_against=float(p.sa), corners_for=float(p.cf), corners_against=float(p.ca),
                tempo=float(p.tempo), tempo_rank=rk("tempo"), corners_for_rank=rk("cf"), corners_against_rank=rk("ca"),
                n_teams=len(t), league_tempo=float(t.tempo.mean()), league_corners=float(t.cf.mean()))


def plan(us, home, away, div, oH, oD, oA, oO=None, oU=None):
    inv = np.array([1 / oH, 1 / oD, 1 / oA]); pH, pD, pA = inv / inv.sum()
    pO = (1 / oO) / (1 / oO + 1 / oU) if oO else np.nan
    lh, la = implied_rates(pH, pD, pA, pO)
    is_home = us == home; opp = away if is_home else home
    lf, lagn = (lh, la) if is_home else (la, lh)
    pw, pl = (pH, pA) if is_home else (pA, pH)
    pol, gain = tempo_policy(lf, lagn)
    names = ["LOW", "NORMAL", "HIGH"]
    grid = {g: {k: names[pol[min(k, K - 1), g + dp.GMAX]] for k in (0, 15, 30, 45, 60, 70, 75, 80, 85, 89)} for g in (-2, -1, 0, 1, 2)}
    switch = next((k for k in range(K) if pol[k, dp.GMAX] == 2), None)
    rel = lf / LEAGUE_LAM
    sp_share = SP_SHARE_AVG * rel ** E_SP / (SP_SHARE_AVG * rel ** E_SP + (1 - SP_SHARE_AVG) * rel ** E_OP)
    rel_o = lagn / LEAGUE_LAM
    sp_share_o = SP_SHARE_AVG * rel_o ** E_SP / (SP_SHARE_AVG * rel_o ** E_SP + (1 - SP_SHARE_AVG) * rel_o ** E_OP)
    try:
        prof_o, prof_u = team_profile(div, opp), team_profile(div, us)
    except Exception:
        prof_o = prof_u = None
    return dict(us=us, opp=opp, venue="home" if is_home else "away", p_win=pw, p_draw=pD, p_loss=pl, xg_us=lf, xg_opp=lagn,
                share=lf / (lf + lagn), grid=grid, switch_minute_level=switch, tempo_gain=gain,
                sp_share_us=sp_share, sp_share_opp=sp_share_o, prof_opp=prof_o, prof_us=prof_u)


def html(P):
    def cell(v):
        col = {"LOW": "#2a78d6", "HIGH": "#eb6834", "NORMAL": "#9a9993"}[v]
        return f'<td style="background:{col};color:#fff;text-align:center;font-weight:600">{v}</td>'
    mins = list(next(iter(P["grid"].values())).keys())
    rows = "".join(f"<tr><th>{'+' if g > 0 else ''}{g}</th>" + "".join(cell(P['grid'][g][m]) for m in mins) + "</tr>" for g in (2, 1, 0, -1, -2))
    o = P["prof_opp"]; u = P["prof_us"]
    opp_block = ""
    if o:
        tempo_word = "high-tempo" if o["tempo"] > o["league_tempo"] * 1.05 else ("low-tempo" if o["tempo"] < o["league_tempo"] * 0.95 else "average-tempo")
        opp_block = f"""<h2>3. The opponent: {P['opp']} <span class="tag d">DESCRIPTIVE</span></h2>
<p>{o['n']} league matches this season. <b>{o['shots_for']:.1f}</b> shots for and <b>{o['shots_against']:.1f}</b> against per match, {"an" if tempo_word.startswith("a") else "a"} {tempo_word} side (tempo rank {o['tempo_rank']}/{o['n_teams']}).
They win <b>{o['corners_for']:.1f}</b> corners a match (rank {o['corners_for_rank']}/{o['n_teams']}). Their attacking set-piece volume is the one opponent trait our tests found to persist, so prepare the set-piece defence for it.</p>
<p class="note">What our research says <i>not</i> to do: build the plan around an opponent "weak spot" (a channel they concede from). Channel-specific concession profiles did not persist or predict in our tests. Plan against their strength level and their attacking habits, not their supposed holes.</p>"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Match plan: {P['us']} v {P['opp']}</title>
<style>:root{{--ink:#0b0b0b;--muted:#52514e;--bg:#fcfcfb}}body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;color:var(--ink);background:var(--bg);line-height:1.45}}
h1{{margin-bottom:2px}}h2{{margin-top:28px;font-size:18px}}table{{border-collapse:separate;border-spacing:3px;width:100%}}th{{font-size:12px;color:var(--muted);font-weight:600}}td{{font-size:11px;padding:6px 2px;border-radius:4px}}
.kpi{{display:flex;gap:12px;flex-wrap:wrap}}.k{{flex:1;min-width:120px;background:#fff;border:1px solid #e6e5e0;border-radius:8px;padding:10px}}.k b{{font-size:22px;display:block}}
.tag{{font-size:10px;padding:2px 6px;border-radius:10px;vertical-align:middle;color:#fff}}.c{{background:#1baf7a}}.m{{background:#2a78d6}}.d{{background:#9a9993}}.note{{color:var(--muted);font-size:13px}}
ol li{{margin-bottom:6px}}</style></head><body>
<h1>Match plan: {P['us']} v {P['opp']} ({P['venue']})</h1>
<p class="note">Generated by Project Underdog from the closing-market odds and this season's data. Every item carries its evidence level.</p>
<div class="kpi"><div class="k">Win<b>{P['p_win']*100:.0f}%</b></div><div class="k">Draw<b>{P['p_draw']*100:.0f}%</b></div><div class="k">Loss<b>{P['p_loss']*100:.0f}%</b></div>
<div class="k">Expected goals<b>{P['xg_us']:.2f} – {P['xg_opp']:.2f}</b></div><div class="k">Our share of chances<b>{P['share']*100:.0f}%</b></div></div>

<h2>1. Tempo by score and minute <span class="tag m">MODEL</span></h2>
<p>LOW = slow the game, protect the shape and the scoreline. HIGH = open it up, commit numbers. Rows show our goal difference, columns the minute.</p>
<table><tr><th></th>{''.join(f'<th>{m}′</th>' for m in mins)}</tr>{rows}</table>
<p><b>If level:</b> {"play LOW until the " + str(P['switch_minute_level']) + "th minute, then go HIGH." if P['switch_minute_level'] is not None else "stay LOW to the end; the draw is worth protecting."}
<b>If behind:</b> go HIGH. <b>If ahead:</b> LOW. Value of following this plan: about <b>+{P['tempo_gain']:.3f} points</b> over playing it straight. It is small, and it relies on the team being able to shift tempo by ~9%.</p>

<h2>2. The two confirmed rules <span class="tag c">CONFIRMED</span></h2>
<ol><li><b>Set pieces are your best channel.</b> At this strength gap, about <b>{P['sp_share_us']*100:.0f}%</b> of your expected chances should come from set pieces, compared with {P['sp_share_opp']*100:.0f}% for {P['opp']}. Quality matters least on dead balls (confirmed in 4 populations). Rehearse 2–3 corner and wide-free-kick routines, and attack every second ball.</li>
<li><b>Break after they commit.</b> When you win the ball within seconds of them having been in your box, go forward immediately: two runners, and the first pass forward. Those counters are worth about <b>{WINDOW_MULT}×</b> a normal regain (confirmed in 4 populations). When you win it while they are still in shape, keep it.</li></ol>

<h2>2b. Real-time rule card (what to do in the moment) <span class="tag c">CONFIRMED</span></h2>
<p class="note">Mined from 730k decisions using what the player could see (360 frames), confirmed in men's club and women's football. The comparison holds the number of defenders in the shooting lane fixed.</p>
<table style="border-spacing:0 4px"><tr><th style="text-align:left">When you see…</th><th style="text-align:left">Do</th><th>Gain per decision</th></tr>
<tr><td>Ball in the box, central, ≤3 defenders between you and goal, even if pressed</td><td><b>Shoot</b>, don't lay it back</td><td>+0.10 xG</td></tr>
<tr><td>Edge of the box, central, packed defence (7+ behind the ball)</td><td><b>Shoot</b>, don't recycle or turn</td><td>+0.03–0.05 xG</td></tr>
<tr><td>25–40 yds, central, packed defence, you are pressed, 4+ teammates ahead</td><td><b>Shoot</b> (teams recycle here 7× more than they shoot)</td><td>+0.013 xG</td></tr>
<tr><td>Attacking third, central, free (no one within 7 yds), ≤3 defenders back</td><td><b>Carry it forward into the space</b>; don't dribble or dither</td><td>+0.04 xG</td></tr>
<tr><td>Midfield, central, pressed, packed defence ahead</td><td><b>Short pass forward</b>, not backwards</td><td>+0.006 xG</td></tr>
<tr><td><b>The counter trigger:</b> you just won the ball, so count the opponents behind it</td><td><b>≤3 back: GO.</b> 8+ back: keep it</td><td>each extra defender back costs 11–19% of the counter's value</td></tr>
</table>
{opp_block}
<h2>4. What the plan deliberately does not include</h2>
<p class="note">No formation trick, pressing switch or substitution timing. In our tests, in-game style changes and coaching changes did not measurably move chance rates, and "chase earlier" did not survive. These rules are worth a few points over a season, not a guaranteed result in one match. One match is mostly noise.</p>
</body></html>"""


if __name__ == "__main__":
    a = sys.argv[1:]
    P = plan(a[0], a[1], a[2], a[3], *map(float, a[4:]))
    out = os.path.join(ROOT, "results", f"matchplan_{a[0].replace(' ', '_')}_v_{P['opp'].replace(' ', '_')}.html")
    open(out, "w").write(html(P))
    print(json.dumps({k: v for k, v in P.items() if k not in ("grid", "prof_opp", "prof_us")}, default=float, indent=1)); print(out)
