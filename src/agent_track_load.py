"""Loaders for public tracking datasets -> common long format (agent TAG = track).

Common output per match (written to data/proc/agent_track_<ds>_<match>_{long,frames,events}.parquet):
  long  : match, frame, time_s, period, team ('home'/'away'/'ball'), player_id, x, y, speed, gk
          metres on 105x68, origin bottom-left, HOME ATTACKS TOWARD x=105 IN BOTH PERIODS
          (away team therefore attacks toward x=0; analyses flip per team as needed).
  frames: frame, period, time_s, bx, by, bz, poss ('home'/'away'/None), alive
  events: team, kind ('pass'/'shot'/'other'), setpiece, period, f0, f1, t0, t1, x0, y0, x1, y1, success, player
          (f0 = release frame, f1 = arrival/next-touch frame; positions from event data if
          given, else from ball tracking; normalised like the tracking).
Datasets: Metrica sample (3, 25 Hz), SkillCorner opendata (broadcast, 10 Hz, extrapolated),
DFL/IDSSE open Bundesliga (7, 25 Hz).
"""
import glob
import gzip
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data/raw/tracking')
PROC = os.path.join(ROOT, 'data/proc')
L, W = 105.0, 68.0


# ---------------------------------------------------------------- helpers
def add_speed(long, fps):
    """Per-player speed (m/s): positions smoothed with a 0.2 s centred mean, then
    central difference over +-0.2 s. Gaps (non-consecutive frames) give NaN."""
    long = long.sort_values(['player_id', 'frame']).reset_index(drop=True)
    h = max(1, int(round(0.2 * fps)))
    w = max(1, int(round(0.2 * fps)))
    sp = np.full(len(long), np.nan, dtype=np.float32)
    for pid, idx in long.groupby('player_id', sort=False).indices.items():
        if pid == 'ball':
            continue
        f = long['frame'].values[idx]
        x = pd.Series(long['x'].values[idx]).rolling(w, center=True, min_periods=1).mean().values
        y = pd.Series(long['y'].values[idx]).rolling(w, center=True, min_periods=1).mean().values
        n = len(idx)
        if n <= 2 * h:
            continue
        df_ = f[2 * h:] - f[:-2 * h]
        d = np.hypot(x[2 * h:] - x[:-2 * h], y[2 * h:] - y[:-2 * h])
        v = d / (df_ / fps)
        v[df_ != 2 * h] = np.nan
        sp[idx[h:n - h]] = v
    long['speed'] = sp
    return long


def normalise(long, frames, events):
    """Rotate per period so the home team attacks toward x=105. GK = player whose mean x is
    farthest from the halfway line (per team & period)."""
    long['gk'] = False
    for per in sorted(long['period'].unique()):
        m = (long['period'] == per) & (long['team'] != 'ball')
        sub = long[m]
        cnt = sub.groupby('player_id').size()
        keep = cnt[cnt > 0.3 * cnt.max()].index
        mx = sub[sub.player_id.isin(keep)].groupby(['team', 'player_id'])['x'].mean()
        gks = {}
        for tm in ('home', 'away'):
            s = mx.loc[tm]
            gks[tm] = (s - L / 2).abs().idxmax()
        home_gk_x = mx.loc[('home', gks['home'])]
        flip = home_gk_x > L / 2
        long.loc[m & long.player_id.isin(list(gks.values())), 'gk'] = True
        if flip:
            mm = long['period'] == per
            long.loc[mm, 'x'] = L - long.loc[mm, 'x']
            long.loc[mm, 'y'] = W - long.loc[mm, 'y']
            fm = frames['period'] == per
            frames.loc[fm, 'bx'] = L - frames.loc[fm, 'bx']
            frames.loc[fm, 'by'] = W - frames.loc[fm, 'by']
            if events is not None and len(events):
                em = events['period'] == per
                for c in ('x0', 'x1'):
                    events.loc[em, c] = L - events.loc[em, c]
                for c in ('y0', 'y1'):
                    events.loc[em, c] = W - events.loc[em, c]
    return long, frames, events


def fill_event_xy_from_ball(events, frames):
    fr = frames.set_index('frame')
    for a, b, fcol in (('x0', 'y0', 'f0'), ('x1', 'y1', 'f1')):
        miss = events[a].isna() & events[fcol].notna()
        if miss.any():
            ff = events.loc[miss, fcol].astype(int)
            ff = ff.clip(fr.index.min(), fr.index.max())
            idx = fr.index.get_indexer(ff, method='nearest')
            events.loc[miss, a] = fr['bx'].values[idx]
            events.loc[miss, b] = fr['by'].values[idx]
    return events


def finish(ds, match, long, frames, events, fps, meta):
    long['match'] = f'{ds}_{match}'
    long = add_speed(long, fps)
    events = fill_event_xy_from_ball(events, frames) if len(events) else events
    long, frames, events = normalise(long, frames, events)
    tmap = frames.set_index('frame')['time_s']
    if len(events):
        idx0 = tmap.index.get_indexer(events['f0'].astype(int), method='nearest')
        events['t0'] = tmap.values[idx0]
        f1 = events['f1'].fillna(-1).astype(int)
        idx1 = tmap.index.get_indexer(f1.clip(lower=tmap.index.min()), method='nearest')
        events['t1'] = np.where(events['f1'].notna(), tmap.values[idx1], np.nan)
    for c in ('x', 'y', 'speed', 'time_s'):
        long[c] = long[c].astype(np.float32)
    long['frame'] = long['frame'].astype(np.int32)
    long['period'] = long['period'].astype(np.int8)
    long = long[['match', 'frame', 'time_s', 'period', 'team', 'player_id', 'x', 'y', 'speed', 'gk']]
    tag = f'{ds}_{match}'
    long.to_parquet(f'{PROC}/agent_track_{tag}_long.parquet', index=False)
    frames.to_parquet(f'{PROC}/agent_track_{tag}_frames.parquet', index=False)
    events.to_parquet(f'{PROC}/agent_track_{tag}_events.parquet', index=False)
    meta.update(dict(ds=ds, match=tag, fps=fps, n_frames=int(len(frames)),
                     n_events=int(len(events)), n_players=int(long.player_id.nunique() - 1)))
    with open(f'{PROC}/agent_track_{tag}_meta.json', 'w') as fh:
        json.dump(meta, fh)
    print(tag, meta, flush=True)


def poss_from_events(frames, ev_rows):
    """ev_rows: DataFrame(frame, team, kind) with kind in {'own','dead','restart'}; forward-fill."""
    s = pd.Series(index=frames['frame'].values, dtype=object)
    alive = pd.Series(index=frames['frame'].values, dtype=float)
    ev_rows = ev_rows.sort_values('frame')
    for fr_, tm, kind in ev_rows[['frame', 'team', 'kind']].itertuples(index=False):
        if fr_ not in s.index:
            continue
        if kind == 'dead':
            s.loc[fr_] = 'NONE'
            alive.loc[fr_] = 0.0
        else:
            s.loc[fr_] = tm
            alive.loc[fr_] = 1.0
    s = s.ffill()
    alive = alive.ffill().fillna(0)
    s[s == 'NONE'] = None
    return s.values, alive.values.astype(bool)


# ---------------------------------------------------------------- Metrica
def _metrica_csv(path, team):
    raw = pd.read_csv(path, skiprows=2, low_memory=False)
    cols = list(raw.columns)
    rows = []
    out = {'period': raw['Period'].values, 'frame': raw['Frame'].values, 'time': raw['Time [s]'].values}
    players = {}
    for i, c in enumerate(cols):
        if c.startswith('Player') or c == 'Ball':
            players[c] = (raw.iloc[:, i].values.astype(float), raw.iloc[:, i + 1].values.astype(float))
    return out, players


def load_metrica(g):
    d = f'{RAW}/metrica/data/Sample_Game_{g}'
    fps = 25
    if g in (1, 2):
        base, ph = _metrica_csv(f'{d}/Sample_Game_{g}_RawTrackingData_Home_Team.csv', 'home')
        _, pa = _metrica_csv(f'{d}/Sample_Game_{g}_RawTrackingData_Away_Team.csv', 'away')
        frame, period, time = base['frame'], base['period'], base['time']
        parts = []
        for team, pl in (('home', ph), ('away', pa)):
            for name, (x, y) in pl.items():
                if name == 'Ball':
                    if team == 'home':
                        bx, by = x, y
                    continue
                ok = ~np.isnan(x)
                parts.append(pd.DataFrame({'frame': frame[ok], 'period': period[ok], 'team': team,
                                           'player_id': f'{team[0]}{name}', 'x': x[ok] * L, 'y': (1 - y[ok]) * W}))
        long = pd.concat(parts, ignore_index=True)
        ev = pd.read_csv(f'{d}/Sample_Game_{g}_RawEventsData.csv')
        ev = ev.rename(columns={'Team': 'team', 'Type': 'type', 'Subtype': 'sub', 'Period': 'period',
                                'Start Frame': 'f0', 'End Frame': 'f1', 'Start X': 'x0', 'Start Y': 'y0',
                                'End X': 'x1', 'End Y': 'y1', 'From': 'player'})
        ev['team'] = ev['team'].str.lower()
        ev['sub'] = ev['sub'].fillna('').astype(str)
    else:
        tree = ET.parse(f'{d}/Sample_Game_3_metadata.xml')
        r = tree.getroot()
        pteam = {p.get('id'): ('home' if p.get('teamId') == 'FIFATMA' else 'away') for p in r.iter('Player')}
        chan = {c.get('id'): c.get('playerId') for c in r.iter('PlayerChannel')}
        specs = []
        for sp in r.iter('DataFormatSpecification'):
            ids = [c.get('playerChannelId') for c in sp.iter('PlayerChannelRef')]
            pids = [chan[i] for i in ids if i.endswith('_x')]
            specs.append((int(sp.get('startFrame')), int(sp.get('endFrame')), pids))
        fr_l, p_l, x_l, y_l, bxs, bys, frames_all = [], [], [], [], [], [], []
        with open(f'{d}/Sample_Game_3_tracking.txt') as fh:
            for line in fh:
                fs, rest = line.strip().split(':', 1)
                pl, ball = rest.split(':')
                f = int(fs)
                pids = next(s[2] for s in specs if s[0] <= f <= s[1])
                for pid, xy in zip(pids, pl.split(';')):
                    a, b = xy.split(',')
                    if a != 'NaN':
                        fr_l.append(f); p_l.append(pid); x_l.append(float(a)); y_l.append(float(b))
                a, b = ball.split(',')
                frames_all.append(f); bxs.append(float(a)); bys.append(float(b))
        frame = np.array(frames_all)
        h1end = specs[0][1]
        # first spec ends at first-half end per metadata (69661)
        half_end = int([p.find('Value').text for p in r.iter('ProviderParameter') if p.find('Name').text == 'first_half_end'][0])
        period = np.where(frame <= half_end, 1, 2)
        long = pd.DataFrame({'frame': fr_l, 'player_id': p_l, 'x': np.array(x_l) * L, 'y': (1 - np.array(y_l)) * W})
        long['team'] = long['player_id'].map(pteam)
        long['period'] = np.where(long['frame'] <= half_end, 1, 2)
        bx, by = np.array(bxs), np.array(bys)
        time = np.zeros(len(frame))
        evj = json.load(open(f'{d}/Sample_Game_3_events.json'))['data']
        rows = []
        for e in evj:
            sub = e['subtypes']
            if isinstance(sub, list):
                sub = '-'.join(s['name'] for s in sub)
            elif isinstance(sub, dict):
                sub = sub['name']
            else:
                sub = ''
            rows.append(dict(team='home' if e['team']['id'] == 'FIFATMA' else 'away', type=e['type']['name'], sub=sub,
                             period=e['period'], f0=e['start']['frame'], f1=e['end']['frame'],
                             x0=e['start']['x'], y0=e['start']['y'], x1=e['end']['x'], y1=e['end']['y'],
                             player=(e['from'] or {}).get('id')))
        ev = pd.DataFrame(rows)
    # time within period
    frames = pd.DataFrame({'frame': frame, 'period': period})
    frames['time_s'] = (frames['frame'] - frames.groupby('period')['frame'].transform('min')) / fps
    frames['bx'] = bx * L
    frames['by'] = (1 - by) * W
    frames['bz'] = np.nan
    ball = frames[['frame', 'period']].copy()
    ball['team'] = 'ball'; ball['player_id'] = 'ball'; ball['x'] = frames['bx']; ball['y'] = frames['by']
    long = pd.concat([long, ball.dropna()], ignore_index=True)
    long['time_s'] = long['frame'].map(frames.set_index('frame')['time_s'])
    # events
    for c in ('x0', 'x1'):
        ev[c] = ev[c].astype(float) * L
    for c in ('y0', 'y1'):
        ev[c] = (1 - ev[c].astype(float)) * W
    ev['kind'] = np.where(ev['type'] == 'PASS', 'pass', np.where(ev['type'] == 'SHOT', 'shot', 'other'))
    # set piece passes: a SET PIECE event at the same frame
    sp_frames = set(ev.loc[ev['type'] == 'SET PIECE', 'f0'])
    ev['setpiece'] = ev['f0'].isin(sp_frames) | ev['sub'].str.contains('GOAL KICK|KICK OFF', regex=True)
    # pass success: next event by same team is the reception (Metrica PASS rows are completed passes;
    # failed passes appear as BALL LOST with a pass-like subtype)
    ev['success'] = np.where(ev['type'] == 'PASS', 1.0, np.nan)
    lost_pass = (ev['type'] == 'BALL LOST') & ~ev['sub'].str.contains('THEFT|FORCED|HEAD$|CLEARANCE|OFFSIDE', regex=True) \
        & ev['x1'].notna() & (ev['f1'] > ev['f0'])
    ev.loc[lost_pass, 'kind'] = 'pass'
    ev.loc[lost_pass, 'success'] = 0.0
    ev.loc[ev['f1'] <= ev['f0'], 'f1'] = np.nan
    # possession
    own = ev[ev['type'].isin(['PASS', 'SET PIECE', 'SHOT', 'RECOVERY', 'CARRY', 'BALL LOST'])][['f0', 'team']]
    own = own.rename(columns={'f0': 'frame'}).assign(kind='own')
    dead = ev[ev['type'] == 'BALL OUT'][['f0', 'team']].rename(columns={'f0': 'frame'}).assign(kind='dead')
    shots = ev[ev['type'] == 'SHOT']
    # goals: shot followed by a kick-off -> dead until then
    rows = pd.concat([own, dead], ignore_index=True)
    frames['poss'], frames['alive'] = poss_from_events(frames, rows)
    events = ev[['team', 'kind', 'setpiece', 'period', 'f0', 'f1', 'x0', 'y0', 'x1', 'y1', 'success', 'player', 'type', 'sub']].copy()
    finish('metrica', g, long, frames, events, fps, dict(source='Metrica sample-data', home='Home', away='Away'))


# ---------------------------------------------------------------- SkillCorner
def load_skillcorner(mid):
    d = f'{RAW}/skillcorner/data/matches/{mid}'
    m = json.load(open(f'{d}/{mid}_match.json'))
    Lsc, Wsc = m['pitch_length'], m['pitch_width']
    home_id = m['home_team']['id']
    pteam = {p['id']: ('home' if p['team_id'] == home_id else 'away') for p in m['players']}
    fps = 10
    fr_l, p_l, x_l, y_l, det_l = [], [], [], [], []
    frows = []
    with open(f'{d}/{mid}_tracking_extrapolated.jsonl') as fh:
        for line in fh:
            o = json.loads(line)
            if o['period'] is None or not o['player_data']:
                continue
            f = o['frame']
            b = o['ball_data']
            g = (o.get('possession') or {}).get('group')
            frows.append((f, o['period'], o['timestamp'], b['x'], b['y'], b['z'], b['is_detected'],
                          None if g is None else ('home' if g.startswith('home') else 'away')))
            for pd_ in o['player_data']:
                fr_l.append(f); p_l.append(pd_['player_id']); x_l.append(pd_['x']); y_l.append(pd_['y'])
                det_l.append(pd_['is_detected'])
    frames = pd.DataFrame(frows, columns=['frame', 'period', 'ts', 'bx', 'by', 'bz', 'bdet', 'poss'])
    ts = frames['ts'].str.split(':', expand=True).astype(float)
    frames['time_s'] = ts[0] * 3600 + ts[1] * 60 + ts[2]
    frames['time_s'] -= frames.groupby('period')['time_s'].transform('min')
    for c in ('bx', 'by', 'bz'):
        frames[c] = frames[c].astype(float)
    frames['bx'] = (frames['bx'] + Lsc / 2) * L / Lsc
    frames['by'] = (frames['by'] + Wsc / 2) * W / Wsc
    long = pd.DataFrame({'frame': fr_l, 'player_id': p_l, 'x': (np.array(x_l, float) + Lsc / 2) * L / Lsc,
                         'y': (np.array(y_l, float) + Wsc / 2) * W / Wsc, 'det': det_l})
    long['team'] = long['player_id'].map(pteam)
    long = long[long['team'].notna()]
    long['player_id'] = long['player_id'].astype(str)
    fp = frames.set_index('frame')
    long['period'] = long['frame'].map(fp['period'])
    long['time_s'] = long['frame'].map(fp['time_s'])
    ball = frames[['frame', 'period', 'time_s', 'bx', 'by']].rename(columns={'bx': 'x', 'by': 'y'}).dropna()
    ball['team'] = 'ball'; ball['player_id'] = 'ball'; ball['det'] = True
    long = pd.concat([long, ball], ignore_index=True)
    # possession: SkillCorner group; forward-fill gaps of up to 3 s inside the same period
    frames['poss'] = frames.groupby('period')['poss'].ffill(limit=30)
    # events from dynamic_events: player_possession rows
    de = pd.read_csv(f'{d}/{mid}_dynamic_events.csv', low_memory=False)
    pp = de[de['event_type'] == 'player_possession'].sort_values('frame_start').reset_index(drop=True)
    pp['team'] = np.where(pp['team_id'] == home_id, 'home', 'away')
    nxt_start = pp['frame_start'].shift(-1)
    rows = []
    for i, r in pp.iterrows():
        et = r['end_type']
        st = str(r.get('start_type', ''))
        if et == 'pass':
            rows.append(dict(team=r['team'], kind='pass', setpiece=str(r.get('game_interruption_before', '')) not in ('nan', '', 'None'),
                             period=r['period'], f0=r['frame_end'], f1=nxt_start[i] if not np.isnan(nxt_start[i]) else np.nan,
                             success=1.0 if r['pass_outcome'] == 'successful' else 0.0, player=str(r['player_id']),
                             type='pass', sub=str(r['pass_outcome'])))
        elif et == 'shot':
            rows.append(dict(team=r['team'], kind='shot', setpiece=False, period=r['period'], f0=r['frame_end'], f1=np.nan,
                             success=np.nan, player=str(r['player_id']), type='shot', sub=''))
    events = pd.DataFrame(rows)
    for c in ('x0', 'y0', 'x1', 'y1'):
        events[c] = np.nan
    # SkillCorner pass outcome 'successful'/'unsuccessful'/'offside' etc.
    frames['alive'] = frames['poss'].notna() | frames['bdet'].fillna(False).astype(bool)
    frames = frames[['frame', 'period', 'time_s', 'bx', 'by', 'bz', 'poss', 'alive']]
    det = long[['frame', 'player_id', 'det']].copy()
    finish_sc(mid, long, frames, events, fps, m)


def finish_sc(mid, long, frames, events, fps, m):
    # speed only where detected on both ends (extrapolated positions are model output)
    det = long['det'].fillna(True).astype(bool).values
    long = long.drop(columns=['det'])
    long['_det'] = det
    tag = f'skillcorner_{mid}'
    long['match'] = tag
    long = add_speed(long, fps)
    long.loc[~long['_det'], 'speed'] = np.nan
    long = long.drop(columns=['_det'])
    events = fill_event_xy_from_ball(events, frames)
    long, frames, events = normalise(long, frames, events)
    tmap = frames.set_index('frame')['time_s']
    idx0 = tmap.index.get_indexer(events['f0'].astype(int), method='nearest')
    events['t0'] = tmap.values[idx0]
    f1 = events['f1'].fillna(-1).astype(int)
    idx1 = tmap.index.get_indexer(f1.clip(lower=tmap.index.min()), method='nearest')
    events['t1'] = np.where(events['f1'].notna(), tmap.values[idx1], np.nan)
    for c in ('x', 'y', 'speed', 'time_s'):
        long[c] = long[c].astype(np.float32)
    long['frame'] = long['frame'].astype(np.int32)
    long['period'] = long['period'].astype(np.int8)
    long = long[['match', 'frame', 'time_s', 'period', 'team', 'player_id', 'x', 'y', 'speed', 'gk']]
    long.to_parquet(f'{PROC}/agent_track_{tag}_long.parquet', index=False)
    frames.to_parquet(f'{PROC}/agent_track_{tag}_frames.parquet', index=False)
    events.to_parquet(f'{PROC}/agent_track_{tag}_events.parquet', index=False)
    meta = dict(ds='skillcorner', match=tag, fps=fps, n_frames=int(len(frames)), n_events=int(len(events)),
                home=m['home_team']['name'], away=m['away_team']['name'],
                competition=m['competition_edition'].get('name'), score=f"{m['home_team_score']}-{m['away_team_score']}",
                n_players=int(long.player_id.nunique() - 1))
    json.dump(meta, open(f'{PROC}/agent_track_{tag}_meta.json', 'w'))
    print(tag, meta, flush=True)


# ---------------------------------------------------------------- DFL / IDSSE
FRAME_RE = re.compile(r'<Frame N="(\d+)" T="([^"]+)" X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")?[^>]*?(?: BallPossession="(\d)")?(?: BallStatus="(\d)")?\s*/>')
FS_RE = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]+" TeamId="([^"]+)" PersonId="([^"]+)"')


def _ts(s):
    return datetime.fromisoformat(s).timestamp()


def load_dfl(mid):
    d = f'{RAW}/dfl'
    info = ET.parse(glob.glob(f'{d}/DFL_02_01_matchinformation_*_{mid}.xml')[0]).getroot()
    gen = info.find('.//General')
    home_id, away_id = gen.get('HomeTeamId'), gen.get('GuestTeamId')
    tside = {home_id: 'home', away_id: 'away'}
    fps = 25
    pos = glob.glob(f'{d}/DFL_04_03_positions_raw_observed_*_{mid}.xml*')[0]
    op = gzip.open if pos.endswith('.gz') else open
    fr_l, p_l, x_l, y_l, per_l = [], [], [], [], []
    ball = []
    cur = None
    with op(pos, 'rt') as fh:
        for line in fh:
            if line.startswith('<Frame '):
                if cur is None:
                    continue
                mm = FRAME_RE.match(line)
                if cur[1] == 'BALL':
                    ball.append((int(mm.group(1)), mm.group(2), float(mm.group(3)), float(mm.group(4)),
                                 float(mm.group(5) or 'nan'), mm.group(6), mm.group(7), cur[0]))
                else:
                    fr_l.append(int(mm.group(1))); x_l.append(float(mm.group(3))); y_l.append(float(mm.group(4)))
                    p_l.append(cur[2]); per_l.append(cur[0])
            elif line.startswith('<FrameSet'):
                g = FS_RE.match(line)
                if g.group(2) == 'referee':
                    cur = None
                else:
                    per = {'firstHalf': 1, 'secondHalf': 2}.get(g.group(1), 3)
                    cur = (per, g.group(2), g.group(3))
                    if g.group(2) != 'BALL':
                        pteam_cur = tside[g.group(2)]
                        teams_of = load_dfl.__dict__.setdefault('pt', {})
                        teams_of[g.group(3)] = pteam_cur
    pteam = load_dfl.pt
    frames = pd.DataFrame(ball, columns=['frame', 'T', 'bx', 'by', 'bz', 'bp', 'bs', 'period'])
    frames['epoch'] = frames['T'].map(_ts)
    frames['time_s'] = (frames['frame'] - frames.groupby('period')['frame'].transform('min')) / fps
    frames['bx'] = frames['bx'] + L / 2
    frames['by'] = frames['by'] + W / 2
    frames['alive'] = frames['bs'] == '1'
    frames['poss'] = np.where(frames['bp'] == '1', 'home', np.where(frames['bp'] == '2', 'away', None))
    frames.loc[~frames['alive'], 'poss'] = None
    long = pd.DataFrame({'frame': fr_l, 'player_id': p_l, 'x': np.array(x_l) + L / 2, 'y': np.array(y_l) + W / 2,
                         'period': per_l})
    long['team'] = long['player_id'].map(pteam)
    # DFL keeps a FrameSet for every squad player who featured; drop frames when a player is off the pitch
    # (positions are then outside the pitch area far from play, flagged by M=0 in some files). Keep frames
    # only where the player lies within 6 m of the pitch.
    long = long[(long.x > -6) & (long.x < L + 6) & (long.y > -6) & (long.y < W + 6)]
    fp = frames.set_index('frame')
    long['time_s'] = long['frame'].map(fp['time_s'])
    b = frames[['frame', 'period', 'time_s', 'bx', 'by']].rename(columns={'bx': 'x', 'by': 'y'})
    b['team'] = 'ball'; b['player_id'] = 'ball'
    long = pd.concat([long, b], ignore_index=True)
    # events: sync each on-ball event to tracking (closest ball-player approach within [-3, +1] s of EventTime)
    ev_root = ET.parse(glob.glob(f'{d}/DFL_03_02_events_raw_*_{mid}.xml')[0]).getroot()
    ep = frames['epoch'].values
    fr_arr = frames['frame'].values
    bxy = frames[['bx', 'by']].values
    # player position lookup
    lp = long[long.team != 'ball'].set_index(['player_id', 'frame'])[['x', 'y']]
    rows = []
    for e in ev_root.iter('Event'):
        t = _ts(e.get('EventTime'))
        play = e.find('.//Play')
        shot = e.find('.//ShotAtGoal')
        node, kind = (play, 'pass') if play is not None else ((shot, 'shot') if shot is not None else (None, 'other'))
        if node is None:
            # other ball actions for 'next touch' timing
            oth = None
            for tag in ('OtherBallAction', 'TacklingGame', 'BallClaiming', 'BallDeflection'):
                oth = e.find('.//' + tag)
                if oth is not None:
                    break
            if oth is None:
                continue
            node = oth
        tm = node.get('Team') or node.get('WinnerTeam')
        player = node.get('Player') or node.get('Winner')
        setpiece = any(e.find('.//' + s) is not None for s in ('KickOff', 'ThrowIn', 'FreeKick', 'CornerKick', 'GoalKick', 'Penalty'))
        if e.get('CalculatedFrame'):
            f0 = int(e.get('CalculatedFrame'))
        else:
            lo, hi = np.searchsorted(ep, t - 3.0), np.searchsorted(ep, t + 1.0)
            f0 = np.nan
            if hi > lo and player is not None:
                cand = fr_arr[lo:hi]
                try:
                    pxy = lp.loc[player].reindex(cand).values
                    dist = np.hypot(pxy[:, 0] - bxy[lo:hi, 0], pxy[:, 1] - bxy[lo:hi, 1])
                    if np.isfinite(dist).any():
                        # release = LAST frame in the window with the ball at the passer's feet (<2 m);
                        # the closest approach alone can pick the passer's own reception
                        close = np.where(dist < 2.0)[0]
                        f0 = int(cand[close[-1]]) if len(close) else int(cand[np.nanargmin(dist)])
                except KeyError:
                    pass
            if np.isnan(f0):
                f0 = int(fr_arr[min(np.searchsorted(ep, t), len(ep) - 1)])
        ev_ = node.get('Evaluation')
        success = np.nan
        if kind == 'pass':
            success = 1.0 if ev_ in ('successfullyCompleted', 'successful') else 0.0
        rows.append(dict(team=tside.get(tm), kind=kind, setpiece=setpiece, f0=f0, success=success, player=player,
                         type=node.tag, sub=(play.find('Cross') is not None and 'cross') if play is not None else '',
                         ex=float(e.get('X-Position') or 'nan'), ey=float(e.get('Y-Position') or 'nan')))
    ev = pd.DataFrame(rows).dropna(subset=['f0']).sort_values('f0').reset_index(drop=True)
    ev['f0'] = ev['f0'].astype(int)
    ev['period'] = ev['f0'].map(fp['period'])
    ev = ev.dropna(subset=['period'])
    # arrival = next on-ball event's synced frame (same period)
    nxt = ev.groupby('period')['f0'].shift(-1)
    ev['f1'] = np.where(ev['kind'] == 'pass', nxt, np.nan)
    ev.loc[ev['f1'] <= ev['f0'], 'f1'] = np.nan
    ev.loc[(ev['f1'] - ev['f0']) > 8 * fps, 'f1'] = np.nan
    # DFL has no reception events: refine arrival from tracking = first frame >=0.2 s after release where
    # the ball (z < 2.5 m) is within 1.5 m of a player other than the passer, or the ball goes dead;
    # bounded by the next event (fallback when nothing is found).
    PX = long[long.team != 'ball'].pivot_table(index='frame', columns='player_id', values='x', aggfunc='first').reindex(fr_arr)
    PY = long[long.team != 'ball'].pivot_table(index='frame', columns='player_id', values='y', aggfunc='first').reindex(fr_arr)[PX.columns]
    pcols = {c: i for i, c in enumerate(PX.columns)}
    PXv, PYv = PX.values, PY.values
    bz = frames['bz'].values
    alive_arr = frames['alive'].values
    fpos = pd.Series(np.arange(len(fr_arr)), index=fr_arr)
    newf1, how = [], []
    for r in ev.itertuples():
        if r.kind != 'pass':
            newf1.append(np.nan); how.append(''); continue
        i0 = fpos.get(r.f0)
        lim = int(r.f1) if np.isfinite(r.f1) else r.f0 + 6 * fps
        i_end = fpos.get(lim, None)
        if i0 is None or i_end is None:
            newf1.append(r.f1); how.append('next_event'); continue
        a, b = i0 + 5, min(i_end + 1, i0 + 8 * fps)
        if b <= a:
            newf1.append(r.f1); how.append('next_event'); continue
        dx = PXv[a:b] - bxy[a:b, 0:1]
        dy = PYv[a:b] - bxy[a:b, 1:2]
        dd = np.hypot(dx, dy)
        if r.player in pcols:
            dd[:, pcols[r.player]] = np.inf
        md = np.nanmin(np.where(np.isnan(dd), np.inf, dd), axis=1)
        hit = (md < 1.5) & ~(bz[a:b] > 2.5)
        dead = ~alive_arr[a:b]
        k = np.where(hit | dead)[0]
        if len(k):
            newf1.append(int(fr_arr[a + k[0]])); how.append('dead' if dead[k[0]] else 'touch')
        else:
            newf1.append(r.f1); how.append('next_event')
    ev['f1'] = newf1
    ev['arrival'] = how
    for c in ('x0', 'y0', 'x1', 'y1'):
        ev[c] = np.nan
    ev['sub'] = ev['sub'].astype(str)
    events = ev[ev['kind'] != 'other'][['team', 'kind', 'setpiece', 'period', 'f0', 'f1', 'x0', 'y0', 'x1', 'y1', 'success', 'player', 'type', 'sub', 'arrival']].copy()
    events['setpiece'] = events['setpiece'].astype(bool)
    frames = frames[['frame', 'period', 'time_s', 'bx', 'by', 'bz', 'poss', 'alive']].copy()
    finish('dfl', mid, long, frames, events, fps, dict(source='DFL/IDSSE (Bassek et al. 2025)', home=info.find('.//General').get('HomeTeamName'),
                                                        away=gen.get('GuestTeamName'), competition=gen.get('CompetitionName'),
                                                        season=gen.get('Season'), score=gen.get('Result')))


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'metrica':
        for g in (1, 2, 3):
            load_metrica(g)
    elif which == 'skillcorner':
        ids = sorted(os.listdir(f'{RAW}/skillcorner/data/matches'))
        for mid in ids:
            if os.path.exists(f'{PROC}/agent_track_skillcorner_{mid}_long.parquet'):
                continue
            p = f'{RAW}/skillcorner/data/matches/{mid}/{mid}_tracking_extrapolated.jsonl'
            if os.path.getsize(p) < 10000:
                print('skip (LFS pointer)', mid); continue
            load_skillcorner(mid)
    elif which == 'dfl':
        mids = sorted({re.search(r'(DFL-MAT-\w+)', p).group(1) for p in glob.glob(f'{RAW}/dfl/DFL_04_*')})
        for mid in mids:
            if os.path.exists(f'{PROC}/agent_track_dfl_{mid}_long.parquet'):
                continue
            p = glob.glob(f'{RAW}/dfl/DFL_04_03_positions_raw_observed_*_{mid}.xml*')[0]
            if not p.endswith('.gz'):
                print('skip (download in progress)', mid); continue
            load_dfl(mid)
