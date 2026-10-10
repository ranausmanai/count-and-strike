"""Frontier model: segment-level xG rates as a function of strength, game state and (lagged) style.

Shared helpers used by P1 (held-out test), the DP policy (P2), and the season simulator.
"""
import json, os
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "proc")
STYLE = ["def_height", "press_rate", "long_share", "poss_share"]       # P1 registered set
POLICY_KNOBS = ["def_height", "press_rate", "long_share"]               # own-team controls (no poss share)


def load_segments(min_minutes=3.0):
    s = pd.read_parquet(os.path.join(PROC, "segments.parquet"))
    tmap = json.load(open(os.path.join(PROC, "team_map.json")))
    mk = pd.read_parquet(os.path.join(PROC, "market.parquet"))
    mk = mk[mk.season == 2015]
    s["home_fd"] = s.home_team.map(tmap); s["away_fd"] = s.away_team.map(tmap)
    s = s.merge(mk[["home", "away", "lam_h", "lam_a", "pH", "pD", "pA"]].rename(columns={"home": "home_fd", "away": "away_fd"}),
                on=["home_fd", "away_fd"], how="left")
    s["lam_own"] = np.where(s.home == 1, s.lam_h, s.lam_a)
    s["lam_opp"] = np.where(s.home == 1, s.lam_a, s.lam_h)
    s["pwin"] = np.where(s.home == 1, s.pH, s.pA)
    s["ploss"] = np.where(s.home == 1, s.pA, s.pH)
    # style descriptors
    s["def_height"] = s.def_height_sum / s.n_def.replace(0, np.nan)
    s["press_rate"] = s.n_press / s.n_pass_opp.replace(0, np.nan)
    s["long_share"] = s.n_long / s.n_open_pass.replace(0, np.nan)
    s["poss_share"] = s.n_pass / (s.n_pass + s.n_pass_opp).replace(0, np.nan)
    s["ppda"] = s.opp_build / s.our_high_def.replace(0, np.nan)
    s = s.sort_values(["match_id", "team", "seg"]).reset_index(drop=True)
    # lagged style (previous segment, same team, same match); weight the lag by events to reduce noise
    for c in STYLE:
        s[c + "_lag"] = s.groupby(["match_id", "team"])[c].shift(1)
    s["lag_minutes"] = s.groupby(["match_id", "team"])["minutes"].shift(1)
    # opponent's lagged style on the same row
    opp = s[["match_id", "seg", "team"] + [c + "_lag" for c in STYLE] + STYLE].rename(
        columns={"team": "opp", **{c + "_lag": c + "_lag_opp" for c in STYLE}, **{c: c + "_opp" for c in STYLE}})
    s = s.merge(opp, on=["match_id", "seg", "opp"], how="left")
    s["gd_c"] = s.gd.clip(-2, 2)
    s["men_c"] = s.men_diff.clip(-1, 1)
    s["tmin"] = s.clock_mid / 90.0
    s["log_ratio"] = np.log(s.lam_own / s.lam_opp)
    s = s[s.minutes >= min_minutes].copy()
    return s


class StyleScaler:
    def fit(self, d, cols):
        self.mu = d[cols].mean(); self.sd = d[cols].std(); self.cols = cols; return self

    def z(self, d, suffix=""):
        out = {}
        for c in self.cols:
            out[c] = (d[c + suffix] - self.mu[c]) / self.sd[c]
        return pd.DataFrame(out, index=d.index)


def design(d, sc, kind="full", lag=True):
    """Build design matrix. Offset = log(minutes * lam_own / 90)."""
    X = pd.DataFrame(index=d.index)
    X["const"] = 1.0
    for g in (-2, -1, 1, 2):
        X[f"gd{g}"] = (d.gd_c == g).astype(float)
    X["men"] = d.men_c
    X["tmin"] = d.tmin
    X["tmin2"] = d.tmin ** 2
    X["late_x_gdpos"] = d.tmin * (d.gd_c > 0)
    X["late_x_gdneg"] = d.tmin * (d.gd_c < 0)
    X["log_ratio"] = d.log_ratio
    if kind == "full":
        suf = "_lag" if lag else ""
        zo = sc.z(d, suf); zp = sc.z(d, suf + "_opp")
        for c in sc.cols:
            X[c + "_own"] = zo[c].values
            X[c + "_opp"] = zp[c].values
            X[c + "_own_x_ratio"] = zo[c].values * d.log_ratio.values
            X[c + "_opp_x_ratio"] = zp[c].values * d.log_ratio.values
    off = np.log(d.minutes * d.lam_own / 90.0)
    return X, off


def fit_glm(d, X, off, y="xg_for", alpha=0.0):
    m = sm.GLM(d[y].values, X.values, family=sm.families.Poisson(), offset=off.values)
    r = m.fit_regularized(alpha=alpha, L1_wt=0.0) if alpha > 0 else m.fit(scale="X2")
    return r


def poisson_dev(y, mu):
    y = np.asarray(y); mu = np.asarray(mu)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (t - (y - mu))
