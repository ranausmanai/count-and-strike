# Project Underdog

**Can tactics make player quality matter less?** An autonomous, pre-registered research run (11 Oct 2026) looking for a tactical policy that lets a weak team (Brentford) consistently outperform stronger squads.

📄 Paper: [`paper/underdog_paper.pdf`](paper/underdog_paper.pdf) (source [`paper/paper.md`](paper/paper.md)) · 🧾 [`PREREGISTRATION.md`](PREREGISTRATION.md) · [`DEVIATIONS.md`](DEVIATIONS.md)

## Results in one table

| | Finding | Status |
|---|---|---|
| P1 | In-match style does not predict next-segment xG beyond strength and score | pre-registered, **not supported** |
| P2 | Optimal in-match style policy: +0.015 pts/match when cross-fitted (in-sample looks 3× better) | pre-registered, **not supported** |
| P3 | Habitual style vs market (EPL) | pre-registered, **null** |
| P4 | High-tempo underdogs underperform the closing market (−0.0069 pts/shot, p=.008, 36k matches); fading since 2015 | pre-registered, **supported (modest)** |
| P5 | "Win-hungry" risk-seeking does not raise title odds | pre-registered, **not supported** |
| — | Coaching changes (6,789 subs/formation shifts) barely move xG | post-hoc |
| — | **Set pieces are ~⅓ as quality-sensitive as open play**; replicated on Wyscout 2017/18 | post-hoc, replicated |
| — | Set-piece efficiency is the only persistent, non-wealth skill, but modest and already priced | post-hoc |
| — | Brentford 2025/26: full playbook +1.8 to +4.5 pts/season; top-4 3% → 5–9%; title ≤ 0.15% | simulation |
| — | Leicester 2015/16: +28 pts vs market, 18 of them above their own xG | case study |

## Reproduce
Python 3.12 + numpy/pandas/scipy/statsmodels/scikit-learn/matplotlib/pyarrow. Run the scripts in `src/` in the order listed in §10 of the paper. Raw data (~1 GB) is downloaded by `src/fetch_statsbomb.py`, the football-data loop in `PROGRESS.md`, and the Wyscout figshare files; it is git-ignored.

Data: StatsBomb Open Data · Pappalardo et al. 2019 (Wyscout) · football-data.co.uk.
