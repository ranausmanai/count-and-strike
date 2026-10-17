"""Realism targets + exploratory movement tests on public tracking data (agent TAG = track).

Reads data/proc/agent_track_<match>_{long,frames,events}.parquet written by agent_track_load.py.
Per-match statistics are computed on a 5 Hz grid (speeds at native rate) and saved to
data/proc/agent_track_permatch.pkl; the summariser (agent_track_report.py) builds CIs.
EXPLORATORY ONLY: few matches; CIs are match-level bootstraps.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, 'data/proc')
L, W = 105.0, 68.0
SIDE = 12.0          # |y-34| > 12 m = ball on a flank
BAIT_S = 3.0         # spell length for E1
LONG = 30.0          # long ball threshold (m)
R = 10.0             # local radius (m)
VMAX = 12.0          # speeds above this are tracking artefacts


def grid(frames, fps):
    step = 5 if fps == 25 else 2
    f0 = frames.groupby('period')['frame'].transform('min')
    g = frames[((frames['frame'] - f0) % step) == 0].reset_index(drop=True)
    return g, step / fps


def team_arrays(long, gframes):
    """Return dict team -> (X, Y, GK) arrays of shape (T, P) aligned with gframes."""
    out = {}
    lg = long[long['frame'].isin(gframes) & (long['team'] != 'ball')]
    for tm in ('home', 'away'):
        s = lg[lg['team'] == tm]
        X = s.pivot_table(index='frame', columns='player_id', values='x', aggfunc='first').reindex(gframes)
        Y = s.pivot_table(index='frame', columns='player_id', values='y', aggfunc='first').reindex(gframes)[X.columns]
        G = s.pivot_table(index='frame', columns='player_id', values='gk', aggfunc='first').reindex(gframes)[X.columns]
        out[tm] = (X.values.astype(float), Y.values.astype(float), G.fillna(False).values.astype(bool), list(X.columns))
    return out


def speeds(long, fps, alive_frames):
    p = long[(long['team'] != 'ball')]
    p = p[p['frame'].isin(alive_frames)]
    v = p['speed'].values
    art = float(np.mean(v[np.isfinite(v)] > VMAX))
    p = p[np.isfinite(p['speed']) & (p['speed'] <= VMAX)]
    res = []
    for pid, s in p.groupby('player_id'):
        n = len(s)
        if n < 30 * 60 * fps * 0.85:      # at least ~30 min of in-play samples
            continue
        sp = s.sort_values('frame')['speed']
        # max = highest speed sustained for 1 s (rolling median), robust to single-frame spikes
        mx = sp.rolling(int(fps), center=True, min_periods=int(fps)).median().max()
        res.append(dict(player=pid, gk=bool(s['gk'].any()), median=float(sp.median()), p90=float(sp.quantile(.9)),
                        max1s=float(mx), max_raw=float(sp.max()), frac_hi=float((sp > 5.5).mean()),
                        frac_sprint=float((sp > 7.0).mean()), minutes=n / fps / 60))
    allv = p.loc[~p['gk'], 'speed'].values
    return dict(players=res, artefact_frac=art,
                all_q=dict(zip(['p50', 'p75', 'p90', 'p99'], np.percentile(allv, [50, 75, 90, 99]).round(3).tolist())))


def nearest(xa, ya, xb, yb):
    """For each t and each a-player, distance to nearest b-player. shapes (T,Pa),(T,Pb)."""
    d = np.hypot(xa[:, :, None] - xb[:, None, :], ya[:, :, None] - yb[:, None, :])
    return np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2)


def runs(mask):
    """start/end indices (inclusive start, exclusive end) of True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return np.where(d == 1)[0], np.where(d == -1)[0]


def analyse(tag):
    meta = json.load(open(f'{PROC}/agent_track_{tag}_meta.json'))
    fps = meta['fps']
    long = pd.read_parquet(f'{PROC}/agent_track_{tag}_long.parquet')
    frames = pd.read_parquet(f'{PROC}/agent_track_{tag}_frames.parquet')
    ev = pd.read_parquet(f'{PROC}/agent_track_{tag}_events.parquet')
    out = dict(match=tag, ds=meta['ds'], fps=fps)
    alive_frames = frames.loc[frames['alive'] & frames['poss'].notna(), 'frame'].values
    out['speed'] = speeds(long, fps, alive_frames)

    g, dt = grid(frames, fps)
    gf = g['frame'].values
    T = len(g)
    arr = team_arrays(long, gf)
    bx, by = g['bx'].values.astype(float), g['by'].values.astype(float)
    poss = g['poss'].values
    alive = g['alive'].values.astype(bool) & pd.notna(poss)
    per = g['period'].values
    tsec = g['time_s'].values

    # ---------------- shape in / out of possession (outfield only)
    shape = {}
    cen = {}
    for tm in ('home', 'away'):
        X, Y, G, _ = arr[tm]
        Xo, Yo = np.where(G, np.nan, X), np.where(G, np.nan, Y)
        n = np.sum(np.isfinite(Xo), axis=1)
        ok = alive & (n >= 9)
        width = np.nanmax(Yo, 1) - np.nanmin(Yo, 1)
        length = np.nanmax(Xo, 1) - np.nanmin(Xo, 1)
        cen[tm] = (np.nanmean(Xo, 1), np.nanmean(Yo, 1))
        inp = ok & (poss == tm)
        oup = ok & (poss != tm)
        shape[tm] = dict(width_in=float(np.nanmedian(width[inp])), width_out=float(np.nanmedian(width[oup])),
                         length_in=float(np.nanmedian(length[inp])), length_out=float(np.nanmedian(length[oup])),
                         n_in=int(inp.sum()), n_out=int(oup.sum()))
    out['shape'] = shape

    # ---------------- defensive lateral shift: defending outfield centroid y vs ball y
    rows = []
    for tm in ('home', 'away'):
        att = 'away' if tm == 'home' else 'home'
        m = alive & (poss == att) & np.isfinite(by) & np.isfinite(cen[tm][1])
        # contiguous spells (same period, consecutive grid frames)
        st, en = runs(m)
        for a, b in zip(st, en):
            if (b - a) * dt < 10:
                continue
            if per[a] != per[b - 1]:
                continue
            rows.append((tm, by[a:b].copy(), cen[tm][1][a:b].copy()))
    lags = np.arange(0, int(4 / dt) + 1)
    num = np.zeros(len(lags)); den1 = np.zeros(len(lags)); den2 = np.zeros(len(lags))
    numv = np.zeros(len(lags)); dv1 = np.zeros(len(lags)); dv2 = np.zeros(len(lags))
    xs, ys = [], []
    for _, b_, c_ in rows:
        xs.append(b_ - 34); ys.append(c_ - 34)
        bb, cc = b_ - b_.mean(), c_ - c_.mean()
        vb, vc = np.diff(b_), np.diff(c_)
        for i, k in enumerate(lags):
            if len(bb) - k < 5:
                continue
            u, v = bb[:len(bb) - k], cc[k:]
            num[i] += np.sum(u * v); den1[i] += np.sum(u * u); den2[i] += np.sum(v * v)
            if len(vb) - k >= 5:
                u, v = vb[:len(vb) - k], vc[k:]
                numv[i] += np.sum(u * v); dv1[i] += np.sum(u * u); dv2[i] += np.sum(v * v)
    corr = num / np.sqrt(den1 * den2)
    corrv = numv / np.sqrt(dv1 * dv2)
    xx, yy = np.concatenate(xs), np.concatenate(ys)
    slope = float(np.polyfit(xx, yy, 1)[0])
    kbest = int(lags[np.nanargmax(corr)])
    # slope at best lag
    xl, yl = [], []
    for _, b_, c_ in rows:
        if len(b_) > kbest:
            xl.append(b_[:len(b_) - kbest] - 34); yl.append(c_[kbest:] - 34)
    slope_lag = float(np.polyfit(np.concatenate(xl), np.concatenate(yl), 1)[0])
    out['shift'] = dict(slope=slope, slope_at_lag=slope_lag, lag_level_s=float(kbest * dt),
                        lag_vel_s=float(lags[np.nanargmax(corrv)] * dt),
                        corr_curve=corr.round(4).tolist(), corrv_curve=corrv.round(4).tolist(), dt=dt,
                        n_spells=len(rows), n_frames=int(len(xx)))

    # ---------------- passes
    gidx = pd.Series(np.arange(T), index=gf)

    def gi(f):
        """nearest grid index for frame f"""
        j = np.searchsorted(gf, f)
        j = np.clip(j, 1, T - 1)
        return np.where(np.abs(gf[j - 1] - f) <= np.abs(gf[j] - f), j - 1, j)

    P = ev[ev['kind'] == 'pass'].copy()
    P = P[P['team'].isin(['home', 'away'])]
    P['dist'] = np.hypot(P['x1'] - P['x0'], P['y1'] - P['y0'])
    P['flight'] = P['t1'] - P['t0']
    okf = P['f1'].notna() & (P['flight'] > 0.15) & (P['flight'] < 8) & P['dist'].notna()
    P['bspeed'] = np.where(okf, P['dist'] / P['flight'], np.nan)
    P['bin'] = pd.cut(P['dist'], [0, 15, 30, 200], labels=['<15', '15-30', '>=30'], right=False)
    out['passes'] = dict(n=int(len(P)), completion=float(P['success'].mean()),
                         by_bin={str(k): dict(n=int(len(s)), completion=float(s['success'].mean()),
                                              flight_med=float(s['flight'][okf].median()) if okf[s.index].any() else None,
                                              bspeed_med=float(s['bspeed'].median()) if s['bspeed'].notna().any() else None)
                                 for k, s in P.groupby('bin', observed=True)},
                         bspeed_med=float(P['bspeed'].median()), flight_med=float(P.loc[okf, 'flight'].median()))

    # team-normalised helpers (attacking toward x=105)
    def norm_x(x, tm):
        return x if tm == 'home' else L - x

    # positions at a grid index
    def players_at(i, tm):
        X, Y, G, _ = arr[tm]
        ok = np.isfinite(X[i])
        return X[i][ok], Y[i][ok], G[i][ok]

    shots = ev[ev['kind'] == 'shot'][['team', 'period', 't0']]

    # ---------------- long balls: local counts at kick vs arrival, E2 retention
    lb = []
    for r in P[(P['dist'] >= LONG) & P['f1'].notna() & (P['flight'] < 6)].itertuples():
        i0, i1 = int(gi(r.f0)), int(gi(r.f1))
        if per[i0] != per[i1] or i1 <= i0:
            continue
        opp = 'away' if r.team == 'home' else 'home'
        ax1, ay1, _ = players_at(i1, r.team)
        dx1, dy1, _ = players_at(i1, opp)
        ax0, ay0, _ = players_at(i0, r.team)
        dx0, dy0, _ = players_at(i0, opp)
        lx, ly = r.x1, r.y1
        att1 = int(np.sum(np.hypot(ax1 - lx, ay1 - ly) <= R))
        def1 = int(np.sum(np.hypot(dx1 - lx, dy1 - ly) <= R))
        att0 = int(np.sum(np.hypot(ax0 - lx, ay0 - ly) <= R))
        def0 = int(np.sum(np.hypot(dx0 - lx, dy0 - ly) <= R))
        # nearest players to landing at arrival
        na = float(np.min(np.hypot(ax1 - lx, ay1 - ly))) if len(ax1) else np.nan
        nd = float(np.min(np.hypot(dx1 - lx, dy1 - ly))) if len(dx1) else np.nan
        # first touch = nearest player to the arrival point; support balance excludes that player
        ft_att = bool(na <= nd)
        sup_bal = (att1 - (1 if ft_att else 0)) - (def1 - (0 if ft_att else 1))
        # retention 5 s after arrival: first known possession in [t1+5, t1+8]
        j = i1 + int(round(5 / dt))
        ret = np.nan
        for k in range(j, min(j + int(round(3 / dt)), T)):
            if per[k] != per[i1]:
                break
            if pd.notna(poss[k]):
                ret = float(poss[k] == r.team)
                break
        shot10 = bool(((shots['team'] == r.team) & (shots['period'] == r.period) & (shots['t0'] > r.t0) & (shots['t0'] <= r.t0 + 10)).any())
        lb.append(dict(match=tag, team=r.team, setpiece=bool(r.setpiece), dist=r.dist, flight=r.flight,
                       land_x=norm_x(lx, r.team), att0=att0, def0=def0, att1=att1, def1=def1,
                       bal0=att0 - def0, bal1=att1 - def1, ft_att=ft_att, sup_bal=sup_bal, near_att=na, near_def=nd, success=r.success,
                       retained5=ret, shot10=shot10))
    out['longballs'] = lb

    # ---------------- E1 bait: flank spells and far-side space
    # per grid frame: attacking team, ball side
    side = np.where(np.abs(by - 34) > SIDE, np.sign(by - 34), 0)
    # far-side space for attacker team tm, given ball side s (+1/-1): widest attacker on the opposite flank
    def far_space(i, tm, s):
        opp = 'away' if tm == 'home' else 'home'
        ax, ay, ag = players_at(i, tm)
        dx, dy, _ = players_at(i, opp)
        ax, ay = ax[~ag], ay[~ag]
        if len(ax) < 8 or len(dx) < 8:
            return np.nan, np.nan, np.nan
        far = -s * (ay - 34)
        k = int(np.argmax(far))
        if far[k] <= 0:
            return np.nan, np.nan, np.nan
        sp = float(np.min(np.hypot(dx - ax[k], dy - ay[k])))
        # defending outfield centroid offset toward ball side
        dcy = float(np.mean(dy) - 34) * s
        return sp, float(far[k]), dcy

    spells = []
    curves = []
    key = np.array([f'{p}|{q}|{s}' for p, q, s in zip(per, poss, side)])
    m = alive & (side != 0)
    st, en = runs(m)
    # split runs where key changes
    segs = []
    for a, b in zip(st, en):
        cut = a
        for k in range(a + 1, b):
            if key[k] != key[k - 1]:
                segs.append((cut, k)); cut = k
        segs.append((cut, b))
    n3 = int(round(BAIT_S / dt))
    for a, b in segs:
        tm, s = poss[a], side[a]
        dur = (b - a) * dt
        rec = dict(match=tag, team=tm, start=a, end=b, dur=dur, side=s)
        if dur >= BAIT_S:
            pts = {}
            for sec in (0, 1, 2, 3):
                i = a + int(round(sec / dt))
                sp, wide, dcy = far_space(i, tm, s)
                pts[sec] = (sp, wide, dcy)
            rec.update(space0=pts[0][0], space3=pts[3][0], wide0=pts[0][1], wide3=pts[3][1],
                       dcy0=pts[0][2], dcy3=pts[3][2],
                       space1=pts[1][0], space2=pts[2][0])
        spells.append(rec)
    # control: central-ball spells >=3 s (ball |y-34|<=12): widest attacker on the flank away from ball
    mc = alive & (side == 0)
    st, en = runs(mc)
    ctrl = []
    for a, b in zip(st, en):
        if (b - a) * dt < BAIT_S:
            continue
        if per[a] != per[b - 1] or len(set(poss[a:b])) != 1:
            continue
        tm = poss[a]
        s0 = np.sign(by[a] - 34) or 1.0
        r0 = far_space(a, tm, s0)
        r3 = far_space(a + n3, tm, s0)
        ctrl.append(dict(match=tag, space0=r0[0], space3=r3[0], dcy0=r0[2], dcy3=r3[2]))
    out['e1_spells'] = [s for s in spells if s['dur'] >= BAIT_S]
    out['e1_ctrl'] = ctrl

    # switches: passes with |dy| >= 30 m crossing the middle, open play, start x (team-norm) < 70
    sw = []
    spell_end_side = {}
    for a, b in segs:
        pass
    for r in P.itertuples():
        if r.setpiece or not np.isfinite(r.y0) or not np.isfinite(r.y1):
            continue
        if abs(r.y1 - r.y0) < 30 or (r.y0 - 34) * (r.y1 - 34) >= 0:
            continue
        x0n = norm_x(r.x0, r.team)
        if x0n >= 70:
            continue
        i0 = int(gi(r.f0))
        s0 = np.sign(r.y0 - 34)
        # time ball had been on the y0 flank (|y-34|>12, same sign) with same team in possession, before kick
        k = i0
        while k - 1 >= 0 and per[k - 1] == per[i0] and side[k - 1] == s0 and poss[k - 1] == r.team and alive[k - 1]:
            k -= 1
        flank_t = (i0 - k) * dt
        # outcomes within 10 s of kick
        j_end = min(T, i0 + int(round(10 / dt)) + 1)
        ft = False; lost = False
        for q in range(i0, j_end):
            if per[q] != per[i0]:
                break
            if pd.notna(poss[q]) and poss[q] != r.team:
                lost = True; break
            if poss[q] == r.team and norm_x(bx[q], r.team) >= 70:
                ft = True; break
        shot10 = bool(((shots['team'] == r.team) & (shots['period'] == r.period) & (shots['t0'] > r.t0) & (shots['t0'] <= r.t0 + 10)).any())
        # far-side space at arrival: receiver-side widest attacker's nearest defender at arrival
        sp_arr = np.nan
        if pd.notna(r.f1):
            i1 = int(gi(r.f1))
            if per[i1] == per[i0]:
                sp_arr = far_space(i1, r.team, s0)[0]
        sp_kick = far_space(i0, r.team, s0)[0]
        sw.append(dict(match=tag, team=r.team, flank_t=flank_t, bait=flank_t >= BAIT_S, success=r.success,
                       final_third10=ft, shot10=shot10, flight=r.flight, dist=r.dist,
                       space_kick=sp_kick, space_arr=sp_arr))
    out['switches'] = sw
    return out


if __name__ == '__main__':
    tags = sorted(os.path.basename(p)[len('agent_track_'):-len('_meta.json')] for p in glob.glob(f'{PROC}/agent_track_*_meta.json'))
    if len(sys.argv) > 1:
        tags = [t for t in tags if t.startswith(sys.argv[1])]
    store = f'{PROC}/agent_track_permatch.pkl'
    res = pickle.load(open(store, 'rb')) if os.path.exists(store) else {}
    for t in tags:
        if t in res and '--force' not in sys.argv:
            continue
        res[t] = analyse(t)
        print(t, 'shape', res[t]['shape']['home'], 'shift', {k: res[t]['shift'][k] for k in ('slope', 'lag_level_s', 'lag_vel_s')},
              'lb', len(res[t]['longballs']), 'sw', len(res[t]['switches']), 'spells', len(res[t]['e1_spells']), flush=True)
        pickle.dump(res, open(store, 'wb'))
