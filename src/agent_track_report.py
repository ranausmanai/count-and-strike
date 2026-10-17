"""Aggregate per-match tracking statistics into results/agent_track.json (agent TAG = track).
Match-level bootstrap (2,000 reps; stratified by dataset when pooling). EXPLORATORY."""
import json
import os
import pickle

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, 'data/proc')
RNG = np.random.default_rng(7)
B = 2000

res = pickle.load(open(f'{PROC}/agent_track_permatch.pkl', 'rb'))
DS = ['metrica', 'skillcorner', 'dfl']


def boot(df, fn, strat=True):
    """fn(df) -> float; resample matches (stratified by ds). returns est, lo, hi."""
    est = fn(df)
    groups = {d: g for d, g in df.groupby('ds')} if strat else {'all': df}
    mlists = {d: g['match'].unique() for d, g in groups.items()}
    by_match = {m: g for m, g in df.groupby('match')}
    vals = []
    for _ in range(B):
        parts = []
        for d, ms in mlists.items():
            pick = RNG.choice(ms, len(ms), replace=True)
            parts += [by_match[m] for m in pick]
        vals.append(fn(pd.concat(parts, ignore_index=True)))
    vals = np.array(vals, float)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return dict(est=None if not np.isfinite(est) else round(float(est), 4), lo=round(float(lo), 4), hi=round(float(hi), 4),
                n=int(len(df)), n_matches=int(df['match'].nunique()))


def per_ds(df, fn):
    out = {}
    for d in DS:
        s = df[df['ds'] == d]
        if len(s):
            out[d] = boot(s, fn)
    out['pooled'] = boot(df, fn)
    return out


J = dict(note='EXPLORATORY: descriptive statistics from few public matches; 95% CIs are match-level bootstraps. '
              'Coordinates 105x68 m; team-normalised x toward the opponent goal.',
         datasets={}, realism={}, E1={}, E2={}, E3={})

# ---------------- datasets
for d in DS:
    ms = [k for k in res if res[k]['ds'] == d]
    metas = [json.load(open(f'{PROC}/agent_track_{k}_meta.json')) for k in ms]
    J['datasets'][d] = dict(n_matches=len(ms), fps=metas[0]['fps'] if metas else None,
                            matches=[dict(match=m['match'], home=m.get('home'), away=m.get('away'), score=m.get('score'),
                                          competition=m.get('competition')) for m in metas])

# ---------------- speeds
sp_rows = []
for k, r in res.items():
    for p in r['speed']['players']:
        sp_rows.append(dict(match=k, ds=r['ds'], **p))
SP = pd.DataFrame(sp_rows)
J['realism']['speed_outfield_per_player'] = {}
for d in DS:
    s = SP[(SP.ds == d) & ~SP.gk]
    if not len(s):
        continue
    J['realism']['speed_outfield_per_player'][d] = dict(
        n_players=int(len(s)),
        median_of_player_median=round(float(s['median'].median()), 2),
        median_of_player_p90=round(float(s['p90'].median()), 2),
        median_of_player_max1s=round(float(s['max1s'].median()), 2),
        p90_of_player_max1s=round(float(s['max1s'].quantile(.9)), 2),
        top_player_max1s=round(float(s['max1s'].max()), 2),
        frac_time_over_5p5=round(float(s['frac_hi'].median()), 4),
        frac_time_over_7=round(float(s['frac_sprint'].median()), 4),
        all_frames_quantiles=pd.DataFrame([res[k]['speed']['all_q'] for k in res if res[k]['ds'] == d]).mean().round(2).to_dict(),
        artefact_frac_over_12=round(float(np.mean([res[k]['speed']['artefact_frac'] for k in res if res[k]['ds'] == d])), 5))
    gk = SP[(SP.ds == d) & SP.gk]
    J['realism']['speed_outfield_per_player'][d]['gk_median_of_median'] = round(float(gk['median'].median()), 2) if len(gk) else None

# ---------------- shape
sh = []
for k, r in res.items():
    for tm in ('home', 'away'):
        sh.append(dict(match=k, ds=r['ds'], **r['shape'][tm]))
SH = pd.DataFrame(sh)
J['realism']['team_shape_outfield_median_m'] = {
    c: per_ds(SH, lambda df, c=c: df[c].mean()) for c in ('width_in', 'width_out', 'length_in', 'length_out')}

# ---------------- block shift
shf = pd.DataFrame([dict(match=k, ds=r['ds'], slope=r['shift']['slope'], slope_lag=r['shift']['slope_at_lag'],
                         lag=r['shift']['lag_level_s'], lagv=r['shift']['lag_vel_s']) for k, r in res.items()])
J['realism']['block_shift'] = dict(
    definition='OLS of defending outfield centroid y on ball y (centred at 34) during opponent possession spells >=10 s; '
               'lag = argmax cross-correlation of demeaned levels (lag_level) or of 0.2 s velocities (lag_vel), 0-4 s, 0.2 s steps',
    slope=per_ds(shf, lambda df: df['slope'].mean()),
    slope_at_best_lag=per_ds(shf, lambda df: df['slope_lag'].mean()),
    lag_level_s=per_ds(shf, lambda df: df['lag'].mean()),
    lag_vel_s=per_ds(shf, lambda df: df['lagv'].median()))
# mean correlation curves
curves = {}
for d in DS:
    cs = [res[k]['shift']['corr_curve'] for k in res if res[k]['ds'] == d]
    cv = [res[k]['shift']['corrv_curve'] for k in res if res[k]['ds'] == d]
    if cs:
        curves[d] = dict(dt=0.2, level=np.round(np.mean(cs, 0), 3).tolist(), velocity=np.round(np.mean(cv, 0), 3).tolist())
J['realism']['block_shift']['mean_corr_curves'] = curves

# ---------------- passes
pas = {}
for d in DS:
    ks = [k for k in res if res[k]['ds'] == d]
    if not ks:
        continue
    n = sum(res[k]['passes']['n'] for k in ks)
    comp = np.average([res[k]['passes']['completion'] for k in ks], weights=[res[k]['passes']['n'] for k in ks])
    bins = {}
    for b in ('<15', '15-30', '>=30'):
        bb = [res[k]['passes']['by_bin'].get(b) for k in ks if res[k]['passes']['by_bin'].get(b)]
        nb = sum(x['n'] for x in bb)
        bins[b] = dict(n=nb, completion=round(float(np.average([x['completion'] for x in bb], weights=[x['n'] for x in bb])), 3),
                       flight_s_median=round(float(np.nanmedian([x['flight_med'] for x in bb if x['flight_med'] is not None])), 2),
                       ball_speed_ms_median=round(float(np.nanmedian([x['bspeed_med'] for x in bb if x['bspeed_med'] is not None])), 2))
    pas[d] = dict(n_passes=int(n), completion=round(float(comp), 3), by_length=bins,
                  flight_s_median=round(float(np.median([res[k]['passes']['flight_med'] for k in ks])), 2),
                  ball_speed_ms_median=round(float(np.median([res[k]['passes']['bspeed_med'] for k in ks])), 2))
J['realism']['passes'] = pas

# ---------------- long balls
LB = pd.DataFrame([dict(ds=r['ds'], **x) for k, r in res.items() for x in r['longballs']])
LBo = LB[~LB['setpiece']]
J['realism']['long_balls_local_counts'] = dict(
    definition='passes >=30 m (open play), arrival = reception/next touch frame (DFL: tracking-detected first touch); '
               'counts of players within 10 m of the arrival point at the kick frame and at the arrival frame',
    n=int(len(LBo)),
    defenders_at_kick=per_ds(LBo, lambda df: df['def0'].mean()),
    defenders_at_arrival=per_ds(LBo, lambda df: df['def1'].mean()),
    attackers_at_kick=per_ds(LBo, lambda df: df['att0'].mean()),
    attackers_at_arrival=per_ds(LBo, lambda df: df['att1'].mean()),
    flight_s_median={d: round(float(LBo[LBo.ds == d]['flight'].median()), 2) for d in DS if (LBo.ds == d).any()},
    nearest_defender_at_arrival_m=per_ds(LBo, lambda df: df['near_def'].median()))

# ---------------- E1 bait
SPL = pd.DataFrame([dict(ds=r['ds'], **x) for k, r in res.items() for x in r['e1_spells']])
CT = pd.DataFrame([dict(ds=r['ds'], **x) for k, r in res.items() for x in r['e1_ctrl']])
SPL['grow'] = SPL['space3'] - SPL['space0']
CT['grow'] = CT['space3'] - CT['space0']
SPL['dshift'] = SPL['dcy3'] - SPL['dcy0']
CT['dshift'] = CT['dcy3'] - CT['dcy0']
both = pd.concat([SPL[['match', 'ds', 'grow']].assign(g='flank'), CT[['match', 'ds', 'grow']].assign(g='central')])
J['E1']['definition'] = ('Flank spell = ball |y-34|>12 m on the same side, same team in possession, ball alive, >=3 s. '
                         'Far-side space = distance from the widest far-side outfield attacker to the nearest opponent. '
                         'Control = central-ball spells (|y-34|<=12) >=3 s, far side = side opposite the ball.')
J['E1']['n_flank_spells'] = int(SPL['space0'].notna().sum())
J['E1']['n_central_spells'] = int(CT['space0'].notna().sum())
J['E1']['far_space_at_0s'] = per_ds(SPL, lambda df: df['space0'].mean())
J['E1']['far_space_at_1s'] = per_ds(SPL, lambda df: df['space1'].mean())
J['E1']['far_space_at_2s'] = per_ds(SPL, lambda df: df['space2'].mean())
J['E1']['far_space_at_3s'] = per_ds(SPL, lambda df: df['space3'].mean())
J['E1']['growth_0_to_3s_flank'] = per_ds(SPL, lambda df: df['grow'].mean())
J['E1']['growth_0_to_3s_central_control'] = per_ds(CT, lambda df: df['grow'].mean())
J['E1']['growth_diff_flank_minus_control'] = per_ds(both, lambda df: df.loc[df.g == 'flank', 'grow'].mean() - df.loc[df.g == 'central', 'grow'].mean())
J['E1']['defending_centroid_shift_toward_ball_0_to_3s_flank'] = per_ds(SPL, lambda df: df['dshift'].mean())
J['E1']['defending_centroid_offset_toward_ball_at_3s'] = per_ds(SPL, lambda df: df['dcy3'].mean())

SW = pd.DataFrame([dict(ds=r['ds'], **x) for k, r in res.items() for x in r['switches']])
SW['bait'] = SW['bait'].astype(bool)
J['E1']['switches'] = dict(
    definition='open-play pass, |dy|>=30 m crossing the middle, start x<70 (team-normalised); bait = ball held on the starting flank >=3 s '
               'before the kick by the same team; outcomes within 10 s of the kick (final third = ball x>=70 in possession before any opponent possession)',
    n_total=int(len(SW)), n_bait=int(SW['bait'].sum()))
for out in ('final_third10', 'shot10', 'success'):
    J['E1']['switches'][out] = dict(
        bait=per_ds(SW, lambda df, o=out: df.loc[df.bait, o].astype(float).mean()),
        other=per_ds(SW, lambda df, o=out: df.loc[~df.bait, o].astype(float).mean()),
        diff=per_ds(SW, lambda df, o=out: df.loc[df.bait, o].astype(float).mean() - df.loc[~df.bait, o].astype(float).mean()))
J['E1']['switches']['space_at_arrival_bait_minus_other'] = per_ds(
    SW, lambda df: df.loc[df.bait, 'space_arr'].mean() - df.loc[~df.bait, 'space_arr'].mean())

# ---------------- E2 escape
E2 = LBo.dropna(subset=['retained5']).copy()


def bins(b):
    return np.select([b <= -2, b == -1, b == 0, b == 1, b >= 2], ['<=-2', '-1', '0', '+1', '>=+2'])


E2['bin1'] = bins(E2['bal1'])
E2['bin0'] = bins(E2['bal0'])
J['E2']['definition'] = ('open-play balls >=30 m; balance = attackers minus defenders within 10 m of the arrival point; '
                         'retained = passing team in possession at the first known possession frame 5-8 s after arrival')
J['E2']['n'] = int(len(E2))
J['E2']['retention_overall'] = per_ds(E2, lambda df: df['retained5'].mean())
tab = {}
for b in ['<=-2', '-1', '0', '+1', '>=+2']:
    s = E2[E2.bin1 == b]
    tab[b] = dict(n=int(len(s)), share=round(len(s) / len(E2), 3),
                  retention=boot(s, lambda df: df['retained5'].mean()) if len(s) >= 10 else None,
                  by_ds={d: dict(n=int((s.ds == d).sum()), ret=round(float(s.loc[s.ds == d, 'retained5'].mean()), 3)) for d in DS if (s.ds == d).any()})
J['E2']['retention_by_balance_at_arrival'] = tab
tab0 = {}
for b in ['<=-2', '-1', '0', '+1', '>=+2']:
    s = E2[E2.bin0 == b]
    tab0[b] = dict(n=int(len(s)), retention=round(float(s['retained5'].mean()), 3) if len(s) else None)
J['E2']['retention_by_balance_at_kick'] = tab0



def lpm_slope(df, col):
    x = df[col].clip(-3, 3).values.astype(float)
    y = df['retained5'].values.astype(float)
    if np.std(x) == 0:
        return np.nan
    return np.polyfit(x, y, 1)[0]


J['E2']['retention_per_unit_balance_arrival'] = per_ds(E2, lambda df: lpm_slope(df, 'bal1'))
J['E2']['retention_per_unit_balance_kick'] = per_ds(E2, lambda df: lpm_slope(df, 'bal0'))
J['E2']['balance_change_kick_to_arrival'] = per_ds(E2, lambda df: (df['bal1'] - df['bal0']).mean())
# support balance: exclude the first-touch player (nearest to the arrival point), split by who touched first
E2['binS'] = bins(E2['sup_bal'])
sup = {}
for who, s0 in (('attacker_first_touch', E2[E2.ft_att]), ('defender_first_touch', E2[~E2.ft_att])):
    sup[who] = dict(n=int(len(s0)), retention=boot(s0, lambda df: df['retained5'].mean()),
                    by_support_balance={b: dict(n=int((s0.binS == b).sum()),
                                                retention=boot(s0[s0.binS == b], lambda df: df['retained5'].mean()) if (s0.binS == b).sum() >= 15 else None)
                                        for b in ['<=-2', '-1', '0', '+1', '>=+2']},
                    per_unit_support=per_ds(s0, lambda df: lpm_slope(df.assign(bal1=df['sup_bal']), 'bal1')))
J['E2']['first_touch_share_attacker'] = per_ds(E2, lambda df: df['ft_att'].mean())
J['E2']['retention_by_support_balance'] = sup

# ---------------- E3 exploratory extras
# (a) switch flight time vs far-side space at arrival
SWf = SW.dropna(subset=['space_arr', 'flight']).copy()
SWf = SWf[(SWf.flight > 0.3) & (SWf.flight < 6)]
SWf['speed'] = SWf['dist'] / SWf['flight']
q = SWf['speed'].quantile([1 / 3, 2 / 3]).values
SWf['tier'] = np.select([SWf.speed < q[0], SWf.speed < q[1]], ['slow', 'mid'], 'fast')
J['E3']['switch_ball_speed_vs_space'] = dict(
    definition='switches (as E1) split into terciles of ball speed (distance/flight time); far-side space at arrival = nearest opponent to the widest far-side attacker',
    tercile_cuts_ms=np.round(q, 2).tolist(),
    space_at_arrival={t: boot(SWf[SWf.tier == t], lambda df: df['space_arr'].mean()) for t in ('slow', 'mid', 'fast')},
    space_change_kick_to_arrival={t: boot(SWf[SWf.tier == t].dropna(subset=['space_kick']), lambda df: (df['space_arr'] - df['space_kick']).mean()) for t in ('slow', 'mid', 'fast')},
    final_third10={t: boot(SWf[SWf.tier == t], lambda df: df['final_third10'].astype(float).mean()) for t in ('slow', 'mid', 'fast')},
    fast_minus_slow_space=boot(SWf, lambda df: df.loc[df.tier == 'fast', 'space_arr'].mean() - df.loc[df.tier == 'slow', 'space_arr'].mean()))
# (b) long-ball flight time vs defenders converging
LBf = LBo.dropna(subset=['flight']).copy()
LBf['ft'] = pd.cut(LBf['flight'], [0, 1.5, 2.5, 10], labels=['<1.5s', '1.5-2.5s', '>2.5s'])
J['E3']['long_ball_flight_vs_convergence'] = dict(
    definition='open-play long balls; change in defenders (and attackers) within 10 m of the arrival point between kick and arrival, by flight time',
    **{str(t): dict(n=int(len(s)), d_def=boot(s, lambda df: (df['def1'] - df['def0']).mean()),
                    d_att=boot(s, lambda df: (df['att1'] - df['att0']).mean()),
                    retained5=round(float(s['retained5'].mean()), 3))
       for t, s in LBf.groupby('ft', observed=True)})
# (c) retention by landing zone
LBz = E2.copy()
LBz['zone'] = pd.cut(LBz['land_x'], [-10, 52.5, 70, 120], labels=['own half', 'mid-third (opp half)', 'final third'])
J['E3']['long_ball_retention_by_landing_zone'] = {str(z): dict(n=int(len(s)), retained5=boot(s, lambda df: df['retained5'].mean()),
                                                                mean_balance=round(float(s['bal1'].mean()), 2))
                                                   for z, s in LBz.groupby('zone', observed=True)}

os.makedirs(f'{ROOT}/results', exist_ok=True)
json.dump(J, open(f'{ROOT}/results/agent_track.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps({k: J[k] for k in ('realism', 'E1', 'E2', 'E3')}, indent=1, default=str)[:30000])
