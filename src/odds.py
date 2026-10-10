"""Market-implied strength for every big-five league match 2000/01-2025/26 (football-data.co.uk).

For each match: de-vigged closing 1X2 probabilities (Pinnacle closing > Pinnacle > market average > B365),
market expected points for both sides, and the implied Poisson goal rates (lam_h, lam_a) that
reproduce the 1X2 probabilities (and P(over 2.5) when available).
"""
import glob, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

ROOT = os.path.join(os.path.dirname(__file__), "..")
FD = os.path.join(ROOT, "data", "raw", "fd")
LEAGUE = {"E0": "EPL", "SP1": "LaLiga", "D1": "Bundesliga", "I1": "SerieA", "F1": "Ligue1"}

ODDS_SETS = [("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA"), ("AvgCH", "AvgCD", "AvgCA"),
             ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A"),
             ("WHH", "WHD", "WHA"), ("GBH", "GBD", "GBA"), ("IWH", "IWD", "IWA")]
OU_SETS = [("PC>2.5", "PC<2.5"), ("P>2.5", "P<2.5"), ("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"),
           ("B365>2.5", "B365<2.5")]


def load():
    frames = []
    for f in sorted(glob.glob(os.path.join(FD, "*.csv"))):
        code, s = os.path.basename(f)[:-4].split("_")
        try:
            d = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")
        except Exception as e:
            print("skip", f, e)
            continue
        d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        d["league"] = LEAGUE[code]
        d["season"] = int("20" + s[:2]) if int(s[:2]) < 50 else int("19" + s[:2])
        oh = od = oa = None
        H = np.full(len(d), np.nan); D = H.copy(); A = H.copy(); src = np.array([""] * len(d), dtype=object)
        for h, dd, a in ODDS_SETS:
            if h in d:
                m = np.isnan(H) & pd.to_numeric(d[h], errors="coerce").notna().values & (pd.to_numeric(d[h], errors="coerce") > 1).values
                H[m] = pd.to_numeric(d[h], errors="coerce").values[m]
                D[m] = pd.to_numeric(d[dd], errors="coerce").values[m]
                A[m] = pd.to_numeric(d[a], errors="coerce").values[m]
                src[m] = h
        O = np.full(len(d), np.nan); U = O.copy()
        for o, u in OU_SETS:
            if o in d:
                ov = pd.to_numeric(d[o], errors="coerce").values
                m = np.isnan(O) & (ov > 1)
                O[m] = ov[m]
                U[m] = pd.to_numeric(d[u], errors="coerce").values[m]
        out = pd.DataFrame(dict(league=d["league"].values, season=d["season"].values,
                                date=pd.to_datetime(d["Date"], dayfirst=True, errors="coerce").values,
                                home=d["HomeTeam"].values, away=d["AwayTeam"].values,
                                hg=d["FTHG"].astype(int).values, ag=d["FTAG"].astype(int).values,
                                hs=pd.to_numeric(d.get("HS"), errors="coerce").values if "HS" in d else np.nan,
                                as_=pd.to_numeric(d.get("AS"), errors="coerce").values if "AS" in d else np.nan,
                                hst=pd.to_numeric(d.get("HST"), errors="coerce").values if "HST" in d else np.nan,
                                ast=pd.to_numeric(d.get("AST"), errors="coerce").values if "AST" in d else np.nan,
                                oH=H, oD=D, oA=A, src=src, oO=O, oU=U))
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    inv = 1 / df[["oH", "oD", "oA"]].values
    p = inv / inv.sum(1, keepdims=True)
    df["pH"], df["pD"], df["pA"] = p.T
    io = 1 / df[["oO", "oU"]].values
    df["pOver"] = (io[:, 0] / io.sum(1))
    df["xpts_h"] = 3 * df.pH + df.pD
    df["xpts_a"] = 3 * df.pA + df.pD
    df["pts_h"] = np.where(df.hg > df.ag, 3, np.where(df.hg == df.ag, 1, 0))
    df["pts_a"] = np.where(df.hg < df.ag, 3, np.where(df.hg == df.ag, 1, 0))
    return df


def _probs(lh, la, rho=-0.06, n=11):
    g = np.arange(n)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    # Dixon-Coles low-score correction
    M[0, 0] *= 1 - lh * la * rho; M[0, 1] *= 1 + lh * rho; M[1, 0] *= 1 + la * rho; M[1, 1] *= 1 - rho
    M /= M.sum()
    pH = np.tril(M, -1).sum(); pA = np.triu(M, 1).sum(); pD = np.trace(M)
    tot = np.add.outer(g, g)
    pO = M[tot > 2.5].sum()
    return pH, pD, pA, pO


def implied_rates(pH, pD, pA, pO=np.nan):
    def loss(z):
        lh, la = np.exp(z)
        h, d, a, o = _probs(lh, la)
        l = (h - pH) ** 2 + (a - pA) ** 2 + (d - pD) ** 2
        if not np.isnan(pO):
            l += (o - pO) ** 2
        return l
    r = minimize(loss, np.log([1.5, 1.1]), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-10))
    return np.exp(r.x)


if __name__ == "__main__":
    df = load()
    df = df.dropna(subset=["pH"]).reset_index(drop=True)
    print(df.groupby("league").size(), df.src.value_counts())
    # implied rates (cache)
    from joblib import Parallel, delayed
    rates = Parallel(n_jobs=8)(delayed(implied_rates)(a, b, c, d) for a, b, c, d in df[["pH", "pD", "pA", "pOver"]].values)
    df[["lam_h", "lam_a"]] = np.array(rates)
    df.to_parquet(os.path.join(ROOT, "data", "proc", "market.parquet"))
    print(df.tail())
