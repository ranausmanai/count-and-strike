# Project Underdog — Football's Tactics Problem Is a Power Problem

**Can tactics make player quality matter less?** An autonomous research run (11–12 Oct 2026): six pre-registered rounds, 21 tests, 4 independent data populations looking for a tactical policy that lets a weak team (Brentford) consistently outperform stronger squads.

📄 Paper: [`paper/underdog_paper.pdf`](paper/underdog_paper.pdf) (source [`paper/paper.md`](paper/paper.md)) · 🧾 [`PREREGISTRATION.md`](PREREGISTRATION.md) · [`DEVIATIONS.md`](DEVIATIONS.md)

## Main contribution (round 10)

Tactical effects worth points are an order of magnitude below what any club's history can detect. A +20% corner routine needs 15 club-seasons; a +0.1 pts/match tactic needs 57. That is why 26 of 28 pre-registered tactical hypotheses failed. Validated **surrogate endpoints** fix part of this: 'final third within 10 s of a regain' cuts the sample needed tenfold. With them, a club can run randomised tactical trials (protocol in §10.3 of the paper).

## Confirmed on unseen leagues (round 6)

1. **Set pieces are where quality matters least.** Set-piece xG elasticity to team strength is 0.41 / 0.28 / 0.72 vs open play 1.09 / 0.89 / 1.02 (StatsBomb 15/16, Wyscout 17/18, unseen WSL/NWSL/ISL).
2. **Break after they commit.** At a fixed regain spot, score and minute, a regain made after the opponent had pushed into your box is worth ×1.37–1.69 the counter xG, in all four populations.

Everything else in rounds 2–5 failed confirmation (box-score style vs market, ML, opponent weak spots, cone density). See Fig 8 in the paper.

## Round 1 results

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
