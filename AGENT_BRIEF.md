# Brief for research agents (Project Underdog / Count & Strike)

Repository: /Users/usman/Documents/research/football/underdog. Python 3.12 (anaconda), with pandas, numpy, scipy, statsmodels, sklearn, numba and lightgbm.

## Mission
Test ONE family of coach-style tactical hypotheses for football: is there a CHOICE a team (especially an underdog) can make that gives a measurable advantage? Use the established protocol. We value honest nulls as much as positives.

## Data you can use (do not re-download)
- `data/raw/sb360_matches.json` lists 477 matches. Each has `data/raw/sb360/{match_id}_events.json.gz` and, for most, `{match_id}_three-sixty.json.gz`: StatsBomb 360 freeze frames with visible player positions at every event, in the same coordinate frame as the event (actor attacking toward x=120).
  - **Discovery** = competitions starting with `src/r15_360.py:DISC` (men's international).
  - **Confirmation** = women's (`WOMEN`) and men's club (everything else). Both are required.
- `data/raw/sb/{match_id}.json.gz` (StatsBomb big-five 2015/16, 1,551 matches; list in `data/raw/sb_matches.json`, league in `_league`).
- `data/raw/sb2/{match_id}.json.gz` (577 matches: WSL/NWSL/ISL; `_comp`).
- `data/raw/wy/events_<Country>.json` and `matches_<Country>.json` (Wyscout 2017/18; tags 101 goal, 1901 counter, 403 head; positions 0–100). Wyscout xG: `pickle data/proc/xg_model.pkl` with `src/xg_model.py:feats(df[x,y in metres 105x68, head, fk])`.
- Helpers:
  - `src/r15_360.py`: `tsec(e)` gives event seconds within the period; `windows(rows, S)` adds own20/opp20/net (net xG over the next 20 s) given shots `S=DataFrame(period,t,team,xg)`; `RAW`, `DISC`, `WOMEN`, `RESTART`.
  - `src/possessions.py:build_possessions`, `src/build_segments.py:compact_events`.
- Processed tables already exist in `data/proc/`, e.g. `r15_attack.parquet` (360 on-ball actions with situation cell, net), `r15_press.parquet`, `r18a_passes.parquet` (lane overloads), `r19_tempo.parquet`, `r19_press.parquet`, `r21_restarts.parquet`, `r21_goals.parquet`, `events.parquet` (StatsBomb 2015/16 compact events). Inspect schemas with pandas before use.

## Protocol (mandatory)
1. Write `PREREGISTRATION_AGENT_<TAG>.md` BEFORE looking at outcomes: hypotheses, direction, cells, thresholds, success rule.
2. Compare choices WITHIN matched situations (zone, defenders goal-side, nearest-opponent distance, etc.). The outcome is usually net xG over the next 20 s (own − opponent).
3. Success rule:
   - 360-based work: Benjamini–Hochberg 5% in discovery, then the same sign in BOTH confirmation populations, with pooled one-sided p < 0.05 (Holm-corrected if several).
   - Event-based work across populations: discovery = StatsBomb 2015/16. A rule is confirmed with the same sign in ≥ 2 of the other populations and pooled one-sided p < 0.05.
4. For any confirmed result, run at least one robustness check against selection, e.g. finer matching or a horizon of 45–60 s.
5. Never claim causality beyond what the design supports. Report nulls plainly.

## Constraints
- Use at most 2 worker processes (other agents run in parallel on a 10-core machine).
- Do NOT git commit. Do NOT edit existing files. Write only new files with your TAG in the name:
  - `src/agent_<tag>_*.py`
  - `results/agent_<tag>.json`
  - `results/agents/<tag>_summary.md`
- Aim to finish within about 2 hours of compute. Prefer vectorised pandas over per-event Python loops on big tables.
- The summary must be ≤ 300 words and contain: question, data and sample sizes, a results table (estimates with SE/CI per population), verdict (CONFIRMED / NOT CONFIRMED / MIXED), the key caveat, and one plain-English coaching sentence if confirmed.
