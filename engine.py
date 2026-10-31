#!/usr/bin/env python3
"""Production engine — humandecoded Battleship computation test.

Follows ~/workspace/sim-lab/battleship/definition-sheet.txt exactly,
except for one flagged clarification (see notes in numbers-log.txt):
  * Sunk detection in code is by a ship's cells all being shot
    (the standard sunk announcement: defender names the sunk ship).
    The sheet (sec. 6) phrases this as "length inferred from the
    size of the contiguous hit group". Those coincide whenever a
    hit group is a single ship. When two ships touch and their
    hit cells merge into one connected group, group-size
    inference is ambiguous (group size != sunk ship length),
    so true-ship sunk detection is used — the same method as
    the benchmark prototype, which produced the benchmark numbers.
    This is reported, not hidden.

All other definitions are as locked:
  10x10, fleet [5,4,3,3,2] labelled (two 3-ships distinct),
  touching allowed, uniform random legal placements by
  independent-uniform rejection (conditioned on disjointness),
  paired fleets across tournament arms, group-all marginal
  placement-count density (sinking-aware), parity (row+col)
  even, target stack LIFO neighbours up/right/down/left with
  line extensions pushed after side neighbours so they pop
  first, tie-break centre distance then row-major index,
  single-player shots to sink.
No script/voiceover/video/thumbnail work is done by this file.
"""
import argparse, json, math, random, sys, time
import numpy as np

N = 10
FLEET = [5, 4, 3, 3, 2]
CENTRE = 4.5

# ---------------------------------------------------------------- placements
def placements_for_length(L):
    out = []
    for r in range(N):
        for c in range(N - L + 1):
            out.append(tuple(r * N + c + i for i in range(L)))
    for r in range(N - L + 1):
        for c in range(N):
            out.append(tuple((r + i) * N + c for i in range(L)))
    return out

PL_INT = {}   # length -> np array (P, L) of cell indices
PL_BOOL = {}  # length -> np bool array (P, 100)
PL_MASK = {}  # length -> list of int bitmasks
for L in set(FLEET):
    pls = placements_for_length(L)
    PL_INT[L] = np.array(pls, dtype=np.int64)
    b = np.zeros((len(pls), N * N), dtype=bool)
    for i, cells in enumerate(pls):
        for c in cells:
            b[i, c] = True
    PL_BOOL[L] = b
    masks = []
    for cells in pls:
        m = 0
        for c in cells:
            m |= (1 << c)
        masks.append(m)
    PL_MASK[L] = masks

DIST2 = np.array([ (r - CENTRE) ** 2 + (c - CENTRE) ** 2
                   for r in range(N) for c in range(N)], dtype=float)

def sample_fleet(rng):
    """Uniform labelled fleet by independent-uniform rejection.
    Each ship sampled uniformly among its placements, whole fleet
    rejected on any overlap. Hence uniform over labelled legal
    fleets (product conditioned on disjointness). Acceptance
    ~0.389, matching BPK count / naive product."""
    while True:
        used = 0
        ships = []
        ok = True
        for L in FLEET:
            masks = PL_MASK[L]
            idx = rng.randrange(len(masks))
            m = masks[idx]
            if m & used:
                ok = False
                break
            used |= m
            # recover cells from the placement index
            ships.append(set(int(x) for x in PL_INT[L][idx]))
        if ok:
            return ships

# ------------------------------------------------------------- placement rules
def touches_edge(ship):
    return any(c // N in (0, N - 1) or c % N in (0, N - 1) for c in ship)

def is_corner(c):
    return c in (0, N - 1, N * (N - 1), N * N - 1)

def ships_adjacent(a, b):
    for x in a:
        r, c = divmod(x, N)
        for y in b:
            r2, c2 = divmod(y, N)
            if max(abs(r - r2), abs(c - c2)) == 1:
                return True
    return False

PLACEMENT_STRATEGIES = {
    # name: (predicate, description)
    "P-random": (lambda ships: True,
                 "Uniform random legal placement (section 1)."),
    "P-edge": (lambda ships: all(touches_edge(s) for s in ships),
               "Every ship touches at least one board edge "
               "(row 0/9 or column 0/9)."),
    "P-corner": (lambda ships: any(is_corner(c) for s in ships for c in s),
                 "At least one ship occupies a corner cell "
                 "(0,0), (0,9), (9,0) or (9,9)."),
    "P-interior": (lambda ships: not any(touches_edge(s) for s in ships),
                   "No ship touches any board edge "
                   "(all ships fully interior)."),
    "P-spread": (lambda ships: not any(
                     ships_adjacent(ships[i], ships[j])
                     for i in range(len(ships)) for j in range(i + 1, len(ships))),
                 "No two ships touch each other, even diagonally "
                 "(Chebyshev distance >= 2 between cells of different ships)."),
}

def sample_placement(rng, predicate):
    while True:
        ships = sample_fleet(rng)
        if predicate(ships):
            return ships

# ------------------------------------------------------------------- density
def density_scores(remaining_lengths, misses, sunk_cells, unsunk_hits):
    """Group-all marginal placement-count scores, sinking-aware.
    Returns int array length 100. Shot squares are masked by caller."""
    blocked = misses | sunk_cells
    blocked_arr = np.zeros(N * N, dtype=bool)
    for c in blocked:
        blocked_arr[c] = True
    scores = np.zeros(N * N, dtype=np.int64)
    hits = sorted(unsunk_hits)
    for L in remaining_lengths:
        arr = PL_INT[L]
        barr = PL_BOOL[L]
        valid = ~barr[:, blocked_arr].any(axis=1) if blocked else np.ones(len(arr), dtype=bool)
        if hits:
            cover_all = valid & barr[:, hits].all(axis=1)
            if cover_all.any():
                counted = cover_all
            else:
                counted = valid & barr[:, hits].any(axis=1)
        else:
            counted = valid
        if counted.any():
            scores += np.bincount(arr[counted].ravel(), minlength=N * N)
    return scores

def play_density(ships):
    ship_cells = [set(s) for s in ships]
    remaining = list(range(len(FLEET)))  # labelled indices
    misses = set(); sunk_cells = set(); shot = set(); unsunk_hits = set()
    shots = 0
    while remaining:
        rem_lengths = [FLEET[i] for i in remaining]
        scores = density_scores(rem_lengths, misses, sunk_cells, unsunk_hits)
        for s in shot:
            scores[s] = -1
        mx = scores.max()
        cand = np.flatnonzero(scores == mx)
        cand = cand[np.lexsort((cand, DIST2[cand]))]
        sq = int(cand[0])
        shots += 1; shot.add(sq)
        hit_idx = None
        for idx in remaining:
            if sq in ship_cells[idx]:
                hit_idx = idx; break
        if hit_idx is None:
            misses.add(sq)
        else:
            unsunk_hits.add(sq)
            if ship_cells[hit_idx] <= shot:
                remaining.remove(hit_idx)
                sunk_cells |= ship_cells[hit_idx]
                unsunk_hits -= ship_cells[hit_idx]
    return shots

# -------------------------------------------------------------------- target
def neighbours(sq):
    r, c = divmod(sq, N)
    out = []
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):  # up, right, down, left
        rr, cc = r + dr, c + dc
        if 0 <= rr < N and 0 <= cc < N:
            out.append(rr * N + cc)
    return out

def play_target(ships, rng, parity):
    ship_cells = [set(s) for s in ships]
    cell_to_ship = {}
    for idx, s in enumerate(ship_cells):
        for c in s:
            cell_to_ship[c] = idx
    remaining = set(range(len(FLEET)))
    shot = set()
    if parity:
        hunt = [i for i in range(N * N) if (i // N + i % N) % 2 == 0]
    else:
        hunt = list(range(N * N))
    rng.shuffle(hunt)
    hunt_i = 0
    stack = []
    unsunk = set()
    shots = 0
    def group_of(sq):
        seen = {sq}; frontier = [sq]
        while frontier:
            cur = frontier.pop()
            for nb in neighbours(cur):
                if nb in unsunk and nb not in seen:
                    seen.add(nb); frontier.append(nb)
        return seen
    while remaining:
        while stack and stack[-1] in shot:
            stack.pop()
        if stack:
            sq = stack.pop()
        else:
            while hunt_i < len(hunt) and hunt[hunt_i] in shot:
                hunt_i += 1
            if hunt_i < len(hunt):
                sq = hunt[hunt_i]; hunt_i += 1
            else:
                sq = next(i for i in range(N * N) if i not in shot)
        shots += 1; shot.add(sq)
        idx = cell_to_ship.get(sq)
        if idx is None or idx not in remaining:
            continue
        unsunk.add(sq)
        grp = group_of(sq)
        if len(grp) >= 2:
            rows = {g // N for g in grp}; cols = {g % N for g in grp}
            ext = []
            if len(rows) == 1:
                r = next(iter(rows)); cs = sorted(g % N for g in grp)
                for cc in (min(cs) - 1, max(cs) + 1):
                    if 0 <= cc < N:
                        ext.append(r * N + cc)
            elif len(cols) == 1:
                c = next(iter(cols)); rs = sorted(g // N for g in grp)
                for rr in (min(rs) - 1, max(rs) + 1):
                    if 0 <= rr < N:
                        ext.append(rr * N + c)
            for nb in neighbours(sq):
                if nb not in shot and nb not in stack:
                    stack.append(nb)
            for e in reversed(ext):
                if e not in shot and e not in stack:
                    stack.append(e)
        else:
            for nb in neighbours(sq):
                if nb not in shot and nb not in stack:
                    stack.append(nb)
        if ship_cells[idx] <= shot:
            remaining.remove(idx)
            unsunk -= ship_cells[idx]
    return shots

def play_random(ships, rng):
    order = list(range(N * N)); rng.shuffle(order)
    cells = set().union(*ships)
    pos = {cell: i for i, cell in enumerate(order)}
    return max(pos[c] for c in cells) + 1

# --------------------------------------------------------------------- stats
def stats(xs):
    xs = np.array(xs, dtype=float)
    n = len(xs)
    mean = float(xs.mean()); med = float(np.median(xs))
    sd = float(xs.std(ddof=1)) if n > 1 else 0.0
    hw = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "median": med, "sd": sd,
            "ci_half_width": hw, "ci_low": mean - hw, "ci_high": mean + hw}

def paired_diff_stats(a, b):
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    s = stats(d)
    return {"mean_diff": s["mean"], "sd_diff": s["sd"],
            "ci_half_width": s["ci_half_width"],
            "ci_low": s["ci_low"], "ci_high": s["ci_high"]}

def unpaired_diff_stats(a, b):
    sa, sb = stats(a), stats(b)
    diff = sa["mean"] - sb["mean"]
    se = math.sqrt((sa["sd"] ** 2 / sa["n"]) + (sb["sd"] ** 2 / sb["n"]))
    hw = 1.96 * se
    return {"mean_diff": diff, "ci_half_width": hw,
            "ci_low": diff - hw, "ci_high": diff + hw}

# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tournament-n", type=int, default=20000)
    ap.add_argument("--placement-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20291029)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-tournament", action="store_true")
    ap.add_argument("--skip-placement", action="store_true")
    args = ap.parse_args()
    base = args.seed
    t0 = time.perf_counter()
    out = {
        "definitions": {
            "board": "10x10", "fleet_labelled": FLEET,
            "placement_distribution": "uniform random legal placements, touching allowed, "
                "independent-uniform rejection conditioned on disjointness",
            "density": "group-all marginal placement-count, sinking-aware; "
                "tie-break centre distance (4.5,4.5) then row-major index",
            "parity_colour": "(row+col) even, no switch after sinking",
            "target_stack": "LIFO, neighbours up/right/down/left, line extensions "
                "pushed after side neighbours so they pop first, most recent group first",
            "sunk": "sunk length from sunk announcement; in code, a ship is sunk when "
                "all its cells are shot (see engine docstring clarification vs sheet sec.6 group-size phrasing)",
            "metric": "single-player shots to sink whole fleet",
            "seed": base,
            "headline_count_labelled": 30093975536,
            "headline_count_unlabelled": 15046987768,
            "headline_count_note": "BPK figure, plausibility-checked (Monte Carlo disjoint ratio "
                "0.3883 vs BPK count / naive product 0.3887 in feasibility work); "
                "NOT reproduced by full enumeration in this run — see numbers-log.txt",
        },
        "tournament": {}, "placement": {},
    }
    # exact marginal heatmap (single-ship counts)
    heat = density_scores(FLEET, set(), set(), set())
    out["marginal_heatmap"] = {
        "corner_0_0": int(heat[0]), "centre_4_4": int(heat[4 * N + 4]),
        "ratio_centre_corner": float(heat[4 * N + 4] / heat[0]),
        "note": "Exact marginal heatmap: sum over labelled ships of placements "
                "covering each square. Not a joint heatmap.",
    }
    if not args.skip_tournament:
        print(f"[tournament] sampling {args.tournament_n} fleets, seed {base}", flush=True)
        frng = random.Random(base)
        fleets = [sample_fleet(frng) for _ in range(args.tournament_n)]
        print(f"[tournament] fleets sampled in {time.perf_counter()-t0:.1f}s", flush=True)
        arms = {}
        # Random (sampled check + analytic)
        t = time.perf_counter()
        arms["Random"] = [play_random(f, random.Random(base + 1000000 + i))
                          for i, f in enumerate(fleets)]
        print(f"[tournament] Random done {time.perf_counter()-t:.1f}s", flush=True)
        t = time.perf_counter()
        arms["Hunt+Target"] = [play_target(f, random.Random(base + 2000000 + i), parity=False)
                               for i, f in enumerate(fleets)]
        print(f"[tournament] Hunt+Target done {time.perf_counter()-t:.1f}s", flush=True)
        t = time.perf_counter()
        arms["Parity+Target"] = [play_target(f, random.Random(base + 3000000 + i), parity=True)
                                 for i, f in enumerate(fleets)]
        print(f"[tournament] Parity+Target done {time.perf_counter()-t:.1f}s", flush=True)
        t = time.perf_counter()
        arms["Density-group-all"] = [play_density(f) for f in fleets]
        print(f"[tournament] Density done {time.perf_counter()-t:.1f}s", flush=True)
        out["tournament"]["n_paired_fleets"] = args.tournament_n
        out["tournament"]["analytic_random_baseline"] = 17 * 101 / 18
        out["tournament"]["arms"] = {k: stats(v) for k, v in arms.items()}
        out["tournament"]["paired_differences_vs_density"] = {
            k: paired_diff_stats(v, arms["Density-group-all"])
            for k, v in arms.items() if k != "Density-group-all"}
        out["tournament"]["paired_differences"] = {}
        keys = list(arms)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                out["tournament"]["paired_differences"][f"{keys[i]} - {keys[j]}"] = \
                    paired_diff_stats(arms[keys[i]], arms[keys[j]])
        # keep raw shots for audit? 20k x 4 ints is small; store summary only
        # plus first-100 raw values as a spot-check artefact
        out["tournament"]["raw_first_100"] = {k: v[:100] for k, v in arms.items()}
    if not args.skip_placement:
        print(f"[placement] {args.placement_n} placements per strategy", flush=True)
        for name, (pred, desc) in PLACEMENT_STRATEGIES.items():
            prng = random.Random(base + 5000000 + abs(hash(name)) % 100000)
            # hash() is salted per process in Python; use a stable offset instead
            prng = random.Random(base + 5000000 + sum(ord(c) for c in name) * 1000)
            t = time.perf_counter()
            placements = [sample_placement(prng, pred) for _ in range(args.placement_n)]
            t_sample = time.perf_counter() - t
            t = time.perf_counter()
            dens = [play_density(f) for f in placements]
            t_dens = time.perf_counter() - t
            t = time.perf_counter()
            par = [play_target(f, random.Random(base + 6000000 + sum(ord(c) for c in name) * 1000 + i),
                               parity=True) for i, f in enumerate(placements)]
            t_par = time.perf_counter() - t
            out["placement"][name] = {
                "definition": desc,
                "n": args.placement_n,
                "vs_density": stats(dens),
                "vs_parity": stats(par),
                "paired_density_minus_parity": paired_diff_stats(dens, par),
                "sampling_seconds": t_sample, "density_seconds": t_dens,
                "parity_seconds": t_par,
            }
            print(f"[placement] {name} done dens {t_dens:.1f}s parity {t_par:.1f}s", flush=True)
        # unpaired strategy differences vs P-random, per opponent
        for opp in ("vs_density", "vs_parity"):
            base_stats = out["placement"]["P-random"][opp]
            # recompute from stored? we did not store raw placement shots;
            # unpaired diff needs SDs only, which stats() provides
            out["placement"][f"unpaired_vs_random_{opp}"] = {}
            for name in PLACEMENT_STRATEGIES:
                if name == "P-random":
                    continue
                s = out["placement"][name][opp]
                diff = s["mean"] - base_stats["mean"]
                se = math.sqrt((s["sd"] ** 2 / s["n"]) + (base_stats["sd"] ** 2 / base_stats["n"]))
                hw = 1.96 * se
                out["placement"][f"unpaired_vs_random_{opp}"][name] = {
                    "mean_diff_vs_random": diff, "ci_half_width": hw,
                    "ci_low": diff - hw, "ci_high": diff + hw}
    out["runtime_seconds_total"] = time.perf_counter() - t0
    out_path = args.out or "/home/hatch/workspace/sim-lab/battleship/results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {out_path} in {out['runtime_seconds_total']:.1f}s", flush=True)
    if "arms" in out.get("tournament", {}):
        for k, v in out["tournament"]["arms"].items():
            print(f"  {k}: mean {v['mean']:.2f} median {v['median']:.1f} SD {v['sd']:.2f} CI +/-{v['ci_half_width']:.2f}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
