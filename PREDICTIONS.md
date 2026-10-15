# Count & Strike: locked predictions

**Locked on 14 October 2026, before any of the data that will test them exists publicly.** The git commit hash and date of this file are the timestamp.

These six predictions come from the rules in `paper/count_and_strike.pdf`. They will be tested on the **next StatsBomb open-data release with 360 frames** (for example the FIFA World Cup 2026), which was not used to build the rules. Anyone can run the test:

```bash
python3 src/test_predictions.py <competition_id> <season_id>
```

| ID | Rule | What we predict on new data |
|---|---|---|
| C1 | **Count:** at a regain, each extra opponent behind the ball lowers counter value | multiplier per defender between **0.75 and 0.97** (now 0.81–0.89) |
| S1 | **Strike:** in the box, central, ≤3 ahead, pressed → shooting beats a sideways pass | **+0.04 to +0.20** xG per decision (now +0.11) |
| S2 | **Strike:** edge of the box, central, 4–6 ahead, pressed → shooting beats a sideways pass | **+0.015 to +0.08** (now +0.047) |
| S3 | **Strike:** edge of the box, central, packed (7+), pressed → shooting beats a sideways pass | **+0.01 to +0.07** (now +0.039) |
| S4 | **Strike (weak):** 25–40 yds, central, packed, pressed → shooting beats a sideways pass | **0 to +0.035** (now +0.015) |
| R1 | **Run:** attacking third, wide, free, ≤3 ahead → carrying forward beats dribbling | **+0.005 to +0.07** (now +0.029) |

**Count & Strike is supported if at least 5 of 6 hold.** "Holds" means the right sign with one-sided p < 0.05 *and* an estimate inside the predicted range. We will publish the result whatever it is.

## Addendum, locked 15 October 2026: the overload rule

| ID | Rule | Predicted on new data |
|---|---|---|
| O1 | **Overload:** a pass's value rises with how many more players we have than them, ahead of the ball, in its target lane (same situation and lane change) | **+0.0003 to +0.0025** net xG per extra player (now +0.0010 to +0.0013) |
| O2 | **Overload (behaviour):** when both an overloaded and a crowded lane exist, players choose the overloaded one less than 40% of the time | **< 40%** (now 23%) |
