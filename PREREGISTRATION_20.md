# Project Underdog — Round 20 pre-registration: does the simulator's discovery hold in real football?

*Written 2026-10-15, after the agent-based simulator search (src/sim/) and before any real-data test of its claims.*

The simulator's top underdog tactics (2,000 tactics × 150 matches; re-tested on fresh matches and against 5 opponent styles) point to: (a) narrow defending (team width ≈ 25 m), (b) moderate attacking width (≈ 40 m, not the full pitch), (c) no extra pressers, (d) shooting at any decent chance. (d) is already confirmed in real data (Strike Through). We test (a)–(c) on StatsBomb 360 frames **whose visible area spans the full pitch width** (visible y range covers ≤ 5 to ≥ 75), so lateral spreads are measurable.

* **H20-1 (attacking width).** For our passes in x ∈ [40, 96], *our width* = lateral range of our visible outfield players. Within situation cells (zone × opponents goal-side × nearest-opponent distance), net xG over 20 s is higher when our width is 32–48 m than when it is > 48 m. Same discovery/confirmation rule as rounds 13–19: one-sided p < 0.05 in discovery, same sign with pooled one-sided p < 0.05 in both confirmations.
* **H20-2 (defensive compactness).** At opponent receipts in our half (the opponent attacking), *our compactness* = lateral range of our visible outfield players. Our net xG over 20 s is higher when our range is ≤ 32 m than when it is > 40 m, within cells (zone × receiver support). Same criterion.
* **H20-3 (pressing, average).** Pressing within 2 s vs holding, averaged within cells (round-19 press table), is ≤ 0 in all three populations. The simulator predicts extra pressers do not pay on average; this is a directional replication.

If (a)–(c) fail in real data, the simulator's sweet spot is classified as a simulator artifact.
