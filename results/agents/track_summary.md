# Agent TRACK: tracking data, sim targets, exploratory tests

**Data (30 matches, no sign-up):** Metrica (3, 25 Hz), SkillCorner (20 A-League 2024/25, 10 Hz broadcast, extrapolated), DFL/IDSSE (7 Bundesliga/2.BL 2022/23, 25 Hz). Loader: `src/agent_track_load.py`.

**Realism targets**

| Target | Metrica | SkillCorner | DFL |
|---|---|---|---|
| Speed median/p90/max-1s (m/s) | 1.5 / 3.9 / 8.1 | 2.2 / 4.6 / 8.5 | 2.2 / 4.5 / 8.4 |
| Width in/out poss (m) | 45.8 / 37.0 | 44.2 / 35.5 | 46.0 / 37.1 |
| Length in/out poss (m) | 34.0 / 30.2 | 34.0 / 29.5 | 35.3 / 32.4 |
| Block shift slope; lag | 0.34; 1.7 s | 0.40; 1.7 s | 0.35; 1.7 s |
| Pass completion (≥30 m) | 0.83 (0.62) | 0.84 (0.50) | 0.78 (0.54) |
| Ball speed <15/15-30/≥30 m | 10 / 13 / 16 | 8 / 11 / 13 | 8 / 10 / 15 |
| Defenders ≤10 m of long-ball arrival: kick → arrival | 0.50 → 1.29 | 0.58 → 1.67 | 0.60 → 1.58 |

**E1 bait (exploratory; 95% match-bootstrap CIs).** In 7,904 flank spells (≥3 s), far-side space grew +1.03 m (0.91–1.15) vs −0.30 m in central spells: +1.33 m (1.06–1.59); block slid 3.6 m. Switches after bait (206/465) arrived with +1.9 m (1.2–2.7) more space but did **not** reach the final third more often: 0.49 vs 0.49 (diff −0.09 to +0.09). Shots ≤10 s 3.9% vs 2.3% (CI includes 0).

**E2 escape (1,721 open-play long balls), retention 5 s after arrival by balance:**

| Balance | ≤−2 | −1 | 0 | +1 | ≥+2 |
|---|---|---|---|---|---|
| n | 186 | 419 | 648 | 438 | 30 |
| Retained | 0.35 | 0.37 | 0.63 | 0.84 | 0.87 |

Balance partly encodes who wins first touch; excluding that player, each extra supporter adds +0.14 retention (0.11–0.18) if we win it, +0.06 (0.02–0.09) if we lose it.

**E3 (exploratory).** Defenders out-converge attackers during flight (balance −0.26 per ball); flights >2.5 s draw +1.09 defenders vs +0.84 attackers and retain less (0.57 vs 0.65). Fast switches (>15.6 m/s) reached the final third 59% vs 39% for slow, with equal space (likely confounded).

**Caveats.** Few matches; descriptive only. SkillCorner speeds use detected samples; Metrica in-play status is event-derived; DFL release/arrival synced from tracking; arrival = first touch, not bounce. No confirmation claims.
