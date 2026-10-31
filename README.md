# Battleship, counted — code and numbers

Code and numbers behind the humandecoded computation video on Battleship
strategy: every legal fleet placement counted, search strategies measured
on sampled fleets, and a human-playable rule built from the results.

## Headline results

Board 10x10, fleet Carrier 5, Battleship 4, Cruiser 3, Submarine 3,
Destroyer 2. Ships may touch. Metric everywhere: single-player average
shots to sink the whole fleet. Lower shots = better SEARCH. Higher
shots = better HIDING.

### The count (exact, not sampled)

- Labelled fleet placements (two 3-ships counted as different ships):
  **30,093,975,536**
- Unlabelled board configurations (two 3-ships identical):
  **15,046,987,768** (exactly half)
- Naive labelled product ignoring overlap: 120 x 140 x 160 x 160 x 180
  = 77,414,400,000. Exact / naive = 0.3887 (61.13% overlap).
- Method: exact backtracking count in C/OpenMP (`count_fleets.c`,
  `count_fleets.log`), recursion over labelled ships in order,
  52.6 seconds. This count was reproduced by humandecoded here; it is
  not merely quoted from another source.
- Placement-count per labelled ship on an empty board:
  5: 120, 4: 140, 3: 160, 3: 160, 2: 180 (total 760).
- Marginal heatmap (exact single-ship counts): corner square (0,0) = 10,
  centre square (4,4) = 34, ratio 3.4x.

### Tournament (production, 20,000 paired fleets, seed 20291029)

Same fleets across all arms. 95% CI = mean +/- 1.96 x SD / sqrt(n).

| Strategy | Mean | Median | SD | 95% CI |
| --- | --- | --- | --- | --- |
| Random (analytic baseline, exact) | 95.39 | — | — | — |
| Random (sampled check) | 95.45 | 97.0 | 4.77 | +/-0.07 |
| Hunt + Target | 66.36 | 66.0 | 12.87 | +/-0.18 |
| Parity hunt + Target | 61.34 | 62.0 | 8.69 | +/-0.12 |
| Density, group-all marginal placement-count | 45.17 | 45.0 | 8.76 | +/-0.12 |

Random analytic baseline: expected maximum position of 17 fleet cells in
a random permutation of 100 = 17 x 101 / 18 = 95.39 shots.
Density here means highest placement-count (see below), NOT joint
probability and NOT "most probable square".

### Best human rule (verified, 10,000 paired fleets, seed 20291110)

Centre-out checkerboard hunt (black squares only, sorted by squared
distance from centre (4.5, 4.5), ties by lowest row-major) + line-only
targeting after two collinear hits + drop-on-sink:

- Baseline Parity + Target in this sample: 61.18 (CI +/-0.17)
- Best human rule, clear-adjacent drop: **54.45** (CI +/-0.21),
  paired saving vs baseline 6.72 shots (CI +/-0.27)
- Best human rule, origin-only drop (the variant the video quotes):
  **54.40**. Difference from clear-adjacent: 0.05 shots, not material.
- Density in this sample: 44.97. The human rule stays about 9.5 shots
  behind the computer. It does NOT get 45 shots.
- Line-bonus density (best computer variant): 44.50, paired -0.47 vs
  density. A real half-shot refinement, NOT a headline.
- Memory-friendly spacing version of the human rule: 55.77, about 1.3
  shots worse than the centre version. Present the two as different
  rules, never as the same rule.

### Hiding (10,000 placements per strategy; higher = better hiding)

Against plain density (production sample):

| Placement | Mean shots for the searcher |
| --- | --- |
| Random | 45.11 |
| Edge (every ship touches an edge) | 51.98 (best) |
| Corner (at least one ship in a corner) | 49.15 |
| Spread (no two ships touch, even diagonally) | 46.45 |
| Interior (no ship touches an edge) | 37.54 (worst) |

Against parity, the ranking flips (Spread best) — there is no single
placement winner. Every hiding claim must name its opponent.
Edge collapses against an edge-aware counter-searcher (about 32.97 in
the round-1 counter test, 2,000 placements): edge is best only against
a searcher that does not know you hide on edges.

## Model and limits

- Placement distribution: uniform random legal placements, touching
  allowed. The count is exhaustive; strategy averages are over sampled
  fleets (20,000 tournament, 10,000 verification, 10,000 per placement
  strategy), always labelled with their sample.
- Density definition: group-all marginal placement-count, sinking-aware.
  Score(square) = sum over remaining (unsunk) ships of placements
  covering the square, avoiding all miss cells and sunk cells, and, when
  unsunk hits exist, covering ALL unsunk hits if any placement of that
  ship does so, otherwise covering at least one unsunk hit for that
  ship. Sunk lengths are removed. Tie-break: smallest distance to board
  centre (4.5, 4.5), then lowest row-major index. Deterministic.
- Sunk announcement: true-ship announcement (the defender names the
  sunk ship, standard play). A ship is sunk when all its cells are shot.
  Sunk cells are blocked for other ships; hits from a sunk ship leave
  the unsunk hit set.
- Target stack (Parity + Target, Hunt + Target): LIFO, neighbours
  pushed up, right, down, left; line extensions pushed after side
  neighbours so they pop first; most recent hit group finished first;
  never reshoot; no diagonal targeting.
- No two-player win rate is claimed. First-player rule, opponent
  adaptation across games, and the salvo variant are out of scope.
- No BPK comparison is paired: BPK's fleets, seeds and exact density
  formula are not public in the material used here. Any level comparison
  is labelled unpaired. Never claim this work "beats BPK", is "optimal",
  or gives "perfect" or "unexploitable" hiding.

## Corrections to the benchmark definition sheet

`definition-sheet.txt` in this repo is the benchmark-stage sheet. Two of
its wordings are superseded by the production logs and by this README,
and must not be quoted literally:

1. Sheet section 6 phrases sunk length as "inferred from the size of
   the contiguous hit group". The code uses true-ship sunk detection
   (see above). Group-size inference is ambiguous when two touching
   ships merge into one connected hit group, so it is not used.
2. Sheet section 10 says "lower = better placement". That line is
   INVERTED for hiding and is not followed anywhere in this repo:
   higher shots = better hiding (the searcher takes longer),
   lower shots = better search.

Both corrections are flagged in `numbers-log.txt` ("Flagged deviations /
corrections") and in `verified-core.txt`.

## Files

- `engine.py` — production tournament + placement engine.
- `count_fleets.c`, `count_fleets.log` — exact labelled count.
- `results.json`, `numbers-log.txt`, `production_run.log` — production
  tournament and placement results (20,000 fleets, seed 20291029).
- `verified-core.txt`, `verified_core.json`, `verified_core_run.log`,
  `verify_core.py` — 10,000-fleet verification run (seed 20291110).
- `advanced-investigation.txt`, `advanced-investigation-2.txt`,
  `advanced2.py`, `advanced2_rollout_dp.py`, `advanced2_core.json`,
  `rollout100.json`, `placement_vs_best.json` — round-1/round-2
  investigations: line-bonus density, entropy (dead end), rollout
  lookahead (dead end), joint-count feasibility, placement vs best
  searchers.
- `example_counter_run.py`, `example_counter_run.json`,
  `example-and-counter-log.txt` — the video's example fleet (chosen as
  closest to the arm means, not cherry-picked), the edge-aware
  counter-search, and the origin-only drop variant.
- `definition-sheet.txt` — benchmark-stage locked definitions (see the
  corrections above before quoting sections 6 or 10).

## How to run

`engine.py` is the production engine; `numbers-log.txt` records the exact
seeds and sample sizes for every headline number. `count_fleets.c` is the
exact counter (compile and run to reproduce `LABELLED_FLEETS=30093975536`
in `count_fleets.log`). Benchmark-stage numbers (N=300) in the definition
sheet are not production numbers and are not used for any headline above.
