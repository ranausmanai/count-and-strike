"""engine2 (round 24): engine.py + Slingshot knobs 10 asym, 11 release, 12 escape, and long-ball success depending on local balance.
Agent-based 11v11 football simulator (numba). Team 0 attacks toward x = 105, team 1 toward x = 0.

Each team: 10 outfield players + an implicit goalkeeper (shot saving). Time step 0.2 s, 90 minutes.
Skills (per team): pass, shot, tackle, dribble, speed.
Tactics (per team, 10 knobs):
  0 def_line     x (own frame, metres from own goal) of the back line when defending      [15, 55]
  1 compact      team width when defending (m)                                             [24, 56]
  2 press_n      players who press the ball carrier                                       [0, 3]
  3 press_zone   press only when ball is within this distance of OUR goal... (own-frame x of ball <= press_zone) [30, 105]
  4 runners      forwards who stay high when defending (counter outlets)                   [0, 3]
  5 width        attacking width (m)                                                       [30, 64]
  6 direct       weight on forward progress when choosing passes                           [0, 1.5]
  7 shoot_thr    minimum goal probability to shoot (lower = shoot more)                    [0.02, 0.15]
  8 tempo        seconds on the ball before deciding when not pressed                      [0.3, 2.0]
  9 rest         players kept back when attacking                                          [1, 4]
"""
import numpy as np
from numba import njit

L, W = 105.0, 68.0
DT = 0.2
NP = 10
CAL0 = np.array([0.5, 1.2, 1.0, 0.10, -2.3, 1.0, 18.0, 0.6])  # lane_k, lane_s, risk_k, tackle_k, shot_b, tackle_r, engage_r, min_hold


@njit(cache=True)
def own(x, team):
    return x if team == 0 else L - x


@njit(cache=True)
def xthreat(xo, y):
    """Value of having the ball at own-frame x (toward goal at x=105)."""
    d = np.sqrt((L - xo) ** 2 + (y - W / 2) ** 2)
    return 0.004 + 0.22 * np.exp(-d / 11.0)


@njit(cache=True)
def shot_prob(xo, y, n_block, skill, shot_b):
    d = np.sqrt((L - xo) ** 2 + (y - W / 2) ** 2)
    ang = np.arctan2(7.32 * (L - xo), (L - xo) ** 2 + (y - W / 2) ** 2 - 13.4)
    if ang < 0:
        ang += np.pi
    p = 1.0 / (1.0 + np.exp(-(shot_b + 2.6 * ang - 0.075 * d)))
    return min(0.95, p * skill * (0.5 ** n_block))


@njit(cache=True)
def seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c = vx * vx + vy * vy
    t = 0.0 if c == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / c))
    qx, qy = ax + t * vx - px, ay + t * vy - py
    return np.sqrt(qx * qx + qy * qy), t


@njit(cache=True)
def targets(pos, team, has_ball, ball_x, ball_y, tac, out):
    """Write target positions for the 10 outfield players of `team` into out (10x2)."""
    bxo = own(ball_x, team)
    # 4-4-2 template: rows = defenders(4), mids(4), forwards(2); lateral slots in [-1,1]
    lat = np.array([-0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.3, 0.3])
    row = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    if has_ball:
        width = tac[5]
        anchor = min(max(bxo - 28.0, 12.0), 60.0)
        rest = int(tac[9])
        for i in range(NP):
            xo = anchor + 18.0 * row[i]
            if row[i] == 2:
                xo = min(max(bxo + 12.0, xo), 100.0)  # forwards make runs ahead of the ball
            if row[i] == 1:
                xo = min(max(bxo - 2.0, xo), 95.0)
            y = W / 2 + lat[i] * width / 2 + (0.25 + 0.5 * tac[10]) * (ball_y - W / 2)
            out[i, 0] = xo; out[i, 1] = min(max(y, 2.0), W - 2.0)
        # rest defence: the first `rest` defenders stay behind
        for i in range(min(rest, 4)):
            out[i, 0] = min(out[i, 0], max(bxo - 35.0, 18.0))
        if tac[10] > 0:  # PULL: one isolated far-side outlet, high and wide
            far = 4 if ball_y > W / 2 else 7
            out[far, 1] = 4.0 if ball_y > W / 2 else W - 4.0
            out[far, 0] = min(max(bxo + 6.0, out[far, 0]), 95.0)
        if tac[12] > 0 and bxo < 35:  # ESCAPE: forwards move to the landing zone
            out[8, 0] = 60.0; out[8, 1] = W / 2 - 8.0
            out[9, 0] = 60.0; out[9, 1] = W / 2 + 8.0
    else:
        line = tac[0]
        comp = tac[1]
        shift = 0.45 * (ball_y - W / 2)
        line_x = min(line, max(bxo - 8.0, 8.0))  # never deeper than needed, never behind the ball by much
        runners = int(tac[4])
        for i in range(NP):
            xo = line_x + 11.0 * row[i]
            y = W / 2 + lat[i] * comp / 2 + shift
            out[i, 0] = xo; out[i, 1] = min(max(y, 2.0), W - 2.0)
        for k in range(runners):  # forwards (8, 9) then a wide mid (4) stay high
            idx = 8 + k if k < 2 else 4
            out[idx, 0] = max(out[idx, 0], min(bxo + 10.0, 75.0))
        # pressers: nearest press_n players chase the ball if ball is within press_zone of our goal (own frame of ball distance)
        if (L - bxo) >= (L - tac[3]):
            pass
    # convert own-frame x to absolute
    for i in range(NP):
        if team == 1:
            out[i, 0] = L - out[i, 0]


@njit(cache=True)
def restart(pos, tac, own_team, bx, by, owner, tgt):
    for t in range(2):
        targets(pos[t], t, t == own_team, bx, by, tac[t], tgt)
        for i in range(NP):
            pos[t, i, 0] = tgt[i, 0]; pos[t, i, 1] = tgt[i, 1]
    # opponents at least 12 m from the ball
    o = 1 - own_team
    for i in range(NP):
        dx, dy = pos[o, i, 0] - bx, pos[o, i, 1] - by
        d = np.sqrt(dx * dx + dy * dy)
        if d < 12.0:
            if d < 1e-6:
                dx, dy, d = 1.0, 0.0, 1.0
            pos[o, i, 0] = bx + dx / d * 12.0; pos[o, i, 1] = min(max(by + dy / d * 12.0, 1.0), W - 1.0)
    pos[own_team, owner, 0] = bx; pos[own_team, owner, 1] = by


@njit(cache=True)
def play_match(skA, skB, tacA, tacB, seed, cal):
    """Simulate one match. Returns stats array:
    [goalsA, goalsB, shotsA, shotsB, xgA, xgB, possA_secs, possB_secs, passesA, passesB, counter_goalsA, counter_goalsB, regains_highA, regains_highB]"""
    np.random.seed(seed)
    sk = np.zeros((2, 5)); sk[0] = skA; sk[1] = skB
    tac = np.zeros((2, 13)); tac[0, :len(tacA)] = tacA; tac[1, :len(tacB)] = tacB
    pos = np.zeros((2, NP, 2))
    tgt = np.zeros((NP, 2))
    stats = np.zeros(20)
    # kickoff
    owner_team = 0; owner = 9
    ball = np.array([L / 2, W / 2])
    for t in range(2):
        targets(pos[t], t, t == owner_team, ball[0], ball[1], tac[t], tgt)
        for i in range(NP):
            pos[t, i, 0] = min(tgt[i, 0], L / 2 - 1) if t == 0 else max(tgt[i, 0], L / 2 + 1)
            pos[t, i, 1] = tgt[i, 1]
    pos[0, 9, 0] = L / 2; pos[0, 9, 1] = W / 2
    hold = 0.0
    flight = 0.0  # seconds until a pass arrives
    flight_to_team = 0; flight_to = 0
    regain_time = -100.0
    regain_x = 0.0
    side_team = -1; side_sign = 0; side_time = 0.0
    n_steps = int(90 * 60 / DT)
    for step in range(n_steps):
        tsec = step * DT
        # ---- movement ----
        for t in range(2):
            has = (owner_team == t) and flight <= 0
            if flight > 0:
                has = flight_to_team == t
            targets(pos[t], t, has, ball[0], ball[1], tac[t], tgt)
            if not has:  # pressers
                bxo = own(ball[0], t)
                if True:
                    npress = int(tac[t, 2]) if bxo <= tac[t, 3] else 0
                    d = np.zeros(NP)
                    for i in range(NP):
                        d[i] = (pos[t, i, 0] - ball[0]) ** 2 + (pos[t, i, 1] - ball[1]) ** 2
                    order = np.argsort(d)
                    for k in range(npress):
                        tgt[order[k], 0] = ball[0]; tgt[order[k], 1] = ball[1]
                    # the nearest player always closes the carrier if close
                    if npress == 0 and d[order[0]] < cal[6] ** 2:
                        # jockey: hold 2.5 m goal-side of the ball
                        gx0 = 0.0 if t == 0 else L
                        vx, vy = gx0 - ball[0], W / 2 - ball[1]
                        vn = np.sqrt(vx * vx + vy * vy) + 1e-9
                        tgt[order[0], 0] = ball[0] + vx / vn * 2.5; tgt[order[0], 1] = ball[1] + vy / vn * 2.5
                    if d[order[1]] < 10.0 ** 2:
                        gx = 0.0 if t == 0 else L
                        tgt[order[1], 0] = 0.7 * ball[0] + 0.3 * gx; tgt[order[1], 1] = 0.8 * ball[1] + 0.2 * W / 2
            vmax = 6.8 * sk[t, 4] * DT
            for i in range(NP):
                if flight <= 0 and owner_team == t and i == owner:
                    continue
                dx, dy = tgt[i, 0] - pos[t, i, 0], tgt[i, 1] - pos[t, i, 1]
                dd = np.sqrt(dx * dx + dy * dy)
                if dd > 1e-6:
                    s = min(vmax, dd) / dd
                    pos[t, i, 0] += dx * s; pos[t, i, 1] += dy * s
        # ---- ball in flight ----
        if flight > 0:
            flight -= DT
            if flight <= 0:
                owner_team = flight_to_team; owner = flight_to
                ball[0] = pos[owner_team, owner, 0]; ball[1] = pos[owner_team, owner, 1]
                hold = 0.0
            continue
        T = owner_team; O = 1 - T
        stats[6 + T] += DT
        cx, cy = pos[T, owner, 0], pos[T, owner, 1]
        ball[0], ball[1] = cx, cy
        sgn = 1 if cy > W / 2 else -1
        if side_team == T and sgn == side_sign:
            side_time += DT
        else:
            side_team = T; side_sign = sgn; side_time = 0.0
        # ---- tackles ----
        near_d = 1e9; near_i = 0
        for j in range(NP):
            d = np.sqrt((pos[O, j, 0] - cx) ** 2 + (pos[O, j, 1] - cy) ** 2)
            if d < near_d:
                near_d = d; near_i = j
        if near_d < cal[5]:
            pwin = cal[3] * sk[O, 2] / (sk[O, 2] + sk[T, 3])
            if np.random.random() < pwin:
                owner_team = O; owner = near_i; hold = 0.0; regain_time = tsec; regain_x = own(cx, O); stats[19] += 1
                if own(cx, O) >= 70:
                    stats[12 + O] += 1
                continue
        hold += DT
        pressed = near_d < 3.0
        if hold < tac[T, 8] and not pressed:
            ahead = False
            for j in range(NP):
                ddx = (pos[O, j, 0] - cx) * (1.0 if T == 0 else -1.0)
                if ddx > 0 and ddx < 6.0 and abs(pos[O, j, 1] - cy) < 4.0:
                    ahead = True
            if not ahead:
                step_x = (1.0 if T == 0 else -1.0) * 4.5 * sk[T, 4] * DT * 0.6
                pos[T, owner, 0] = min(max(cx + step_x, 1.0), L - 1.0)
            continue
        if hold < cal[7]:
            continue
        # ---- decide ----
        cxo = own(cx, T)
        base_v = xthreat(cxo, cy)
        risk = cal[2] * xthreat(L - cxo, cy)  # value to the opponent if we lose it here
        best_ev = -1e9; best_kind = 0; best_j = -1; best_p = 0.0
        # shoot
        if L - cxo < 34:
            nb = 0
            gx = L if T == 0 else 0.0
            for j in range(NP):
                d, tt = seg_dist(pos[O, j, 0], pos[O, j, 1], cx, cy, gx, W / 2)
                if d < 1.4 and tt > 0.05:
                    nb += 1
            ps = shot_prob(cxo, cy, nb, sk[T, 1], cal[4])
            if ps >= tac[T, 7]:
                ev = ps - 0.15 * risk
                if ev > best_ev:
                    best_ev = ev; best_kind = 1; best_p = ps
        # pass options
        for j in range(NP):
            if j == owner:
                continue
            rx, ry = pos[T, j, 0], pos[T, j, 1]
            ln = np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
            if ln < 3 or ln > 55:
                continue
            p = sk[T, 0] ** (ln / 15.0) * (1 - 0.25 * np.exp(-near_d / 2.0))
            rd = 1e9
            for k in range(NP):
                d, tt = seg_dist(pos[O, k, 0], pos[O, k, 1], cx, cy, rx, ry)
                if tt > 0.05:
                    p *= 1.0 - cal[0] * np.exp(-(d * d) / (2 * cal[1] * cal[1]))
                dr = np.sqrt((pos[O, k, 0] - rx) ** 2 + (pos[O, k, 1] - ry) ** 2)
                rd = min(rd, dr)
            rxo = own(rx, T)
            v = xthreat(rxo, ry) * (0.55 + 0.45 * (1 - np.exp(-rd / 4.0))) + tac[T, 6] * 0.012 * (rxo - cxo) / 10.0
            if tac[T, 11] > 0 and side_time >= 3.0 and abs(ry - cy) > 25.0 and rd > 5.0:  # RELEASE
                v += tac[T, 11] * 0.012
            ev = p * v - (1 - p) * risk
            if ev > best_ev:
                best_ev = ev; best_kind = 2; best_j = j; best_p = p
        # dribble
        dx = (1.0 if T == 0 else -1.0) * 6.0
        nx = min(max(cx + dx, 1.0), L - 1.0)
        pd = sk[T, 3] * (1 - np.exp(-near_d / 2.5)) * 0.95
        ev = pd * xthreat(own(nx, T), cy) - (1 - pd) * risk
        if ev > best_ev:
            best_ev = ev; best_kind = 3; best_p = pd
        # clearance: long kick upfield when in own third
        if cxo < 35:
            evc = 0.45 * xthreat(65.0, W / 2) - 0.55 * 0.35 * xthreat(L - 65.0, W / 2)
            if evc > best_ev:
                best_ev = evc; best_kind = 4; best_p = 0.45
        if best_kind == 1:
            stats[2 + T] += 1; stats[4 + T] += best_p
            stats[14] += L - cxo
            if tsec - regain_time < 10 and regain_time > 0:
                stats[15] += 1; stats[17] += regain_x
            stats[16] += hold
            if np.random.random() < best_p:
                stats[T] += 1
                if tsec - regain_time < 15 and regain_time > 0:
                    stats[10 + T] += 1
                # kickoff for the other team
                owner_team = O; owner = 9; hold = 0.0
                for t in range(2):
                    targets(pos[t], t, t == owner_team, L / 2, W / 2, tac[t], tgt)
                    for i in range(NP):
                        pos[t, i, 0] = min(tgt[i, 0], L / 2 - 1) if t == 0 else max(tgt[i, 0], L / 2 + 1)
                        pos[t, i, 1] = tgt[i, 1]
                pos[O, 9, 0] = L / 2; pos[O, 9, 1] = W / 2
            else:
                # goal kick: defending team restarts from its own 18 m
                owner_team = O; owner = 1; hold = 0.0
                gkx = 18.0 if O == 0 else L - 18.0
                restart(pos, tac, O, gkx, W / 2, 1, tgt)
            regain_time = -100.0
        elif best_kind == 2:
            rx, ry = pos[T, best_j, 0], pos[T, best_j, 1]
            ln = np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
            flight = ln / 17.0
            stats[18] += 1
            if np.random.random() < best_p:
                flight_to_team = T; flight_to = best_j; stats[8 + T] += 1
            else:
                # intercepted by the defender closest to the passing lane
                bd = 1e9; bi = 0
                for k in range(NP):
                    d, tt = seg_dist(pos[O, k, 0], pos[O, k, 1], cx, cy, rx, ry)
                    if d < bd:
                        bd = d; bi = k
                flight_to_team = O; flight_to = bi; regain_time = tsec + flight; regain_x = own(pos[O, bi, 0], O)
                if own(pos[O, bi, 0], O) >= 70:
                    stats[12 + O] += 1
            hold = 0.0
        elif best_kind == 4:
            # long ball: nearest teammate to the landing point vs nearest opponent
            lx = 65.0 if T == 0 else L - 65.0
            bsup = -99; bly = W / 2
            for cand in range(3):
                cy2 = W / 2 + (cand - 1) * 15.0
                sup = 0
                for j in range(NP):
                    if (pos[T, j, 0] - lx) ** 2 + (pos[T, j, 1] - cy2) ** 2 < 100.0:
                        sup += 1
                    if (pos[O, j, 0] - lx) ** 2 + (pos[O, j, 1] - cy2) ** 2 < 100.0:
                        sup -= 1
                if sup > bsup:
                    bsup = sup; bly = cy2
            if tac[T, 12] > 0:  # ESCAPE: aim at the best local balance
                ly = bly + (np.random.random() - 0.5) * 6.0
            else:
                ly = W / 2 + (np.random.random() - 0.5) * 30.0
            sup = 0
            for j in range(NP):
                if (pos[T, j, 0] - lx) ** 2 + (pos[T, j, 1] - ly) ** 2 < 100.0:
                    sup += 1
                if (pos[O, j, 0] - lx) ** 2 + (pos[O, j, 1] - ly) ** 2 < 100.0:
                    sup -= 1
            best_p = 0.45 + 0.17 * min(max(sup, -2), 1)
            bt, bti = 1e9, 0
            for j in range(NP):
                d = (pos[T, j, 0] - lx) ** 2 + (pos[T, j, 1] - ly) ** 2
                if d < bt:
                    bt = d; bti = j
            bo, boi = 1e9, 0
            for j in range(NP):
                d = (pos[O, j, 0] - lx) ** 2 + (pos[O, j, 1] - ly) ** 2
                if d < bo:
                    bo = d; boi = j
            flight = 2.6
            if np.random.random() < best_p:
                flight_to_team = T; flight_to = bti
            else:
                flight_to_team = O; flight_to = boi; regain_time = tsec + flight; regain_x = own(lx, O)
            hold = 0.0
        else:
            if np.random.random() < best_p:
                pos[T, owner, 0] = nx; hold = tac[T, 8] * 0.5
            else:
                owner_team = O; owner = near_i; hold = 0.0; regain_time = tsec; regain_x = own(cx, O); stats[19] += 1
                if own(cx, O) >= 70:
                    stats[12 + O] += 1
    return stats


def default_tactic():
    return np.array([38.0, 40.0, 1.0, 60.0, 1.0, 50.0, 0.6, 0.06, 0.9, 2.0, 0.0, 0.0, 0.0])


def run(skA, skB, tacA, tacB, n, seed0=0, cal=None):
    cal = CAL0 if cal is None else np.asarray(cal, float)
    out = np.zeros((n, 20))
    for k in range(n):
        out[k] = play_match(np.asarray(skA, float), np.asarray(skB, float), np.asarray(tacA, float), np.asarray(tacB, float), seed0 + k, cal)
    return out
