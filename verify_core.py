#!/usr/bin/env python3
"""Production-scale verification of round-2 core claims.
Fresh seed, 10,000 paired tournament fleets, 10,000 placements per strategy.
Uses engine.py (production) + advanced2.py (round-2 arms) unchanged.
Writes verified_core.json in same folder; verified-core.txt is written by hand from it.
"""
import json, math, random, sys, time
import numpy as np
sys.path.insert(0, '/home/hatch/workspace/sim-lab/battleship')
import engine as E
import advanced2 as A2

def stats(xs):
    xs = np.array(xs, float); n = len(xs)
    m = float(xs.mean()); med = float(np.median(xs)); sd = float(xs.std(ddof=1)) if n > 1 else 0.0
    hw = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": m, "median": med, "sd": sd, "ci_half_width": hw, "ci_low": m-hw, "ci_high": m+hw}

def paired(a, b):
    d = np.array(a, float) - np.array(b, float)
    return stats(d.tolist())

SEED = 20291110
N = 10000
t0 = time.time()
frng = random.Random(SEED)
fleets = [E.sample_fleet(frng) for _ in range(N)]
print(f"fleets sampled {time.time()-t0:.1f}s", flush=True)

def run(name, fn):
    t = time.time(); vals = [fn(f) for f in fleets]
    print(f"{name}: mean {stats(vals)['mean']:.2f}  time {time.time()-t:.1f}s", flush=True)
    return vals

arms = {}
arms["Density-group-all"] = run("Density", E.play_density)
arms["LineBonusDensity"] = run("LineBonus", lambda f: A2.play_density_generic(f, A2.line_bonus_score_fn))
# Baseline parity+target: production engine, per-game rng like production (offset 3000000)
arms["Parity+Target-production"] = run("ParityTarget-prod",
    lambda f, i=[0]: None) if False else [E.play_target(f, random.Random(SEED + 3000000 + i), parity=True) for i, f in enumerate(fleets)]
print(f"ParityTarget-prod: mean {stats(arms['Parity+Target-production'])['mean']:.2f}", flush=True)
# Round-2 repro baseline (same code as round 2, random hunt)
arms["Parity+Target-round2repro"] = []
for i, f in enumerate(fleets):
    arms["Parity+Target-round2repro"].append(A2.play_human(f, random.Random(SEED + 3000000 + i), hunt_mode='random_parity', clear_on_sink='none', line_only=False))
print(f"ParityTarget-r2: mean {stats(arms['Parity+Target-round2repro'])['mean']:.2f}", flush=True)
# Best human: centre parity, clear-adjacent, line-only (deterministic hunt order)
arms["BestHuman-centre-lineonly-clearAdj"] = [A2.play_human(f, random.Random(SEED + 3600000 + i), hunt_mode='centre_parity', clear_on_sink='adjacent', line_only=True) for i, f in enumerate(fleets)]
print(f"BestHuman: mean {stats(arms['BestHuman-centre-lineonly-clearAdj'])['mean']:.2f}", flush=True)
# Memory-friendly spacing version, for reference in script
arms["Human-spacing-lineonly-clearAdj"] = [A2.play_human(f, random.Random(SEED + 3400000 + i), hunt_mode='spacing_random', clear_on_sink='adjacent', line_only=True) for i, f in enumerate(fleets)]
print(f"SpacingHuman: mean {stats(arms['Human-spacing-lineonly-clearAdj'])['mean']:.2f}", flush=True)

out = {"seed": SEED, "n_paired_fleets": N, "tournament": {}, "paired_vs_density": {}, "paired_vs_baseline": {}}
for k, v in arms.items():
    out["tournament"][k] = stats(v)
for k, v in arms.items():
    if k != "Density-group-all":
        out["paired_vs_density"][k] = paired(v, arms["Density-group-all"])
for base in ("Parity+Target-production", "Parity+Target-round2repro"):
    out["paired_vs_baseline"][f"BestHuman - {base}"] = paired(arms["BestHuman-centre-lineonly-clearAdj"], arms[base])
    out["paired_vs_baseline"][f"LineBonus - Density"] = paired(arms["LineBonusDensity"], arms["Density-group-all"])

# Placement vs best searchers, 10,000 per strategy
placement = {}
for name, (pred, desc) in E.PLACEMENT_STRATEGIES.items():
    prng = random.Random(SEED + 5000000 + sum(ord(c) for c in name) * 1000)
    t = time.time()
    placements = [E.sample_placement(prng, pred) for _ in range(10000)]
    t_s = time.time() - t
    lb = [A2.play_density_generic(f, A2.line_bonus_score_fn) for f in placements]
    bh = [A2.play_human(f, random.Random(SEED + 3600000 + i), hunt_mode='centre_parity', clear_on_sink='adjacent', line_only=True) for i, f in enumerate(placements)]
    dens = [E.play_density(f) for f in placements]
    placement[name] = {"definition": desc, "n": 10000, "sampling_seconds": t_s,
                       "vs_density": stats(dens), "vs_linebonus": stats(lb), "vs_besthuman": stats(bh)}
    print(f"placement {name}: linebonus {stats(lb)['mean']:.2f} besthuman {stats(bh)['mean']:.2f} density {stats(dens)['mean']:.2f}", flush=True)

def unpaired(a, b):
    diff = a["mean"] - b["mean"]
    se = math.sqrt(a["sd"]**2 / a["n"] + b["sd"]**2 / b["n"]); hw = 1.96 * se
    return {"mean_diff": diff, "ci_half_width": hw, "ci_low": diff-hw, "ci_high": diff+hw}

out["placement"] = placement
out["placement_unpaired_vs_random"] = {}
for opp in ("vs_linebonus", "vs_besthuman", "vs_density"):
    out["placement_unpaired_vs_random"][opp] = {}
    for name in E.PLACEMENT_STRATEGIES:
        if name == "P-random": continue
        out["placement_unpaired_vs_random"][opp][name] = unpaired(placement[name][opp], placement["P-random"][opp])
out["runtime_seconds"] = time.time() - t0
with open('/home/hatch/workspace/sim-lab/battleship/verified_core.json', 'w') as fh:
    json.dump(out, fh, indent=2)
print(f"Wrote verified_core.json in {out['runtime_seconds']:.1f}s", flush=True)
for k, v in out["tournament"].items():
    print(f"  {k}: {v['mean']:.2f} median {v['median']:.1f} CI +/-{v['ci_half_width']:.2f}", flush=True)
print("paired besthuman - prod baseline:", out["paired_vs_baseline"]["BestHuman - Parity+Target-production"])
print("paired besthuman - r2 baseline:", out["paired_vs_baseline"]["BestHuman - Parity+Target-round2repro"])
print("paired linebonus - density:", out["paired_vs_baseline"]["LineBonus - Density"])
