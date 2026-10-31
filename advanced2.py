#!/usr/bin/env python3
"""Round 2 core: best computer search, best human rule, best hiding vs best search.
Bonus arms only. Primary production numbers untouched.
State model shared with engine.py: 10x10, fleet [5,4,3,3,2] labelled,
touching allowed, sunk announcement (ship sunk when all its cells shot).
"""
import math, random, sys, time
import numpy as np
sys.path.insert(0, '/home/hatch/workspace/sim-lab/battleship')
import engine as E

N=10; FLEET=E.FLEET; PL_INT=E.PL_INT; PL_BOOL=E.PL_BOOL; DIST2=E.DIST2

def stats(xs):
    xs=np.array(xs,float); n=len(xs); m=float(xs.mean()); sd=float(xs.std(ddof=1)) if n>1 else 0.0
    hw=1.96*sd/math.sqrt(n) if n>1 else 0.0
    return m, float(np.median(xs)), sd, hw
def paired(a,b):
    d=np.array(a,float)-np.array(b,float); m=float(d.mean()); sd=float(d.std(ddof=1)) if len(d)>1 else 0.0; hw=1.96*sd/math.sqrt(len(d))
    return m, hw

# ---------- generic density scorer (returns scores, per-ship counted) ----------
def density_detail(rem_lengths, misses, sunk_cells, unsunk_hits):
    blocked=misses|sunk_cells
    barr=np.zeros(N*N,bool)
    for c in blocked: barr[c]=True
    scores=np.zeros(N*N,dtype=np.int64); per=[]; total=0
    hits=sorted(unsunk_hits)
    for L in rem_lengths:
        arr=PL_INT[L]; b=PL_BOOL[L]
        valid=~b[:,barr].any(axis=1) if blocked else np.ones(len(arr),bool)
        if hits:
            ca=valid & b[:,hits].all(axis=1)
            counted=ca if ca.any() else (valid & b[:,hits].any(axis=1))
        else:
            counted=valid
        per.append(counted)
        total+=int(counted.sum())
        if counted.any():
            scores+=np.bincount(arr[counted].ravel(),minlength=N*N)
    return scores, per, total

def argmax_centre(scores, shot):
    s=scores.astype(float).copy()
    for q in shot: s[q]=-np.inf
    mx=s.max(); cand=np.flatnonzero(s==mx)
    cand=cand[np.lexsort((cand, DIST2[cand]))]
    return int(cand[0])

# ---------- computer players (stateful, given true ships) ----------
def play_density_generic(ships, score_fn):
    """score_fn(rem_lengths, misses, sunk_cells, unsunk_hits, shot) -> scores array"""
    cells=[set(s) for s in ships]; remaining=list(range(len(FLEET)))
    misses=set(); sunk=set(); shot=set(); hits=set(); n=0
    while remaining:
        rem=[FLEET[i] for i in remaining]
        scores=score_fn(rem, misses, sunk, hits, shot)
        sq=argmax_centre(scores, shot)
        n+=1; shot.add(sq); hit=None
        for idx in remaining:
            if sq in cells[idx]: hit=idx; break
        if hit is None: misses.add(sq)
        else:
            hits.add(sq)
            if cells[hit]<=shot:
                remaining.remove(hit); sunk|=cells[hit]; hits-=cells[hit]
    return n

def density_score_fn(rem, misses, sunk, hits, shot):
    s,_,_=density_detail(rem, misses, sunk, hits); return s

def line_bonus_score_fn(rem, misses, sunk, hits, shot):
    s,_,_=density_detail(rem, misses, sunk, hits)
    # line bonus: after >=2 collinear unsunk hits, bonus the two extensions
    if len(hits)>=2:
        # find collinear groups (rows/cols)
        hs=sorted(hits)
        # check any row with >=2 or col with >=2
        from collections import defaultdict
        rows=defaultdict(list); cols=defaultdict(list)
        for q in hs: rows[q//N].append(q%N); cols[q%N].append(q//N)
        bonus=set()
        for r,cs in rows.items():
            if len(cs)>=2:
                for cc in (min(cs)-1, max(cs)+1):
                    if 0<=cc<N: bonus.add(r*N+cc)
        for c,rs in cols.items():
            if len(rs)>=2:
                for rr in (min(rs)-1, max(rs)+1):
                    if 0<=rr<N: bonus.add(rr*N+c)
        mx=int(s.max()) if s.max()>0 else 0
        for q in bonus:
            if q not in shot: s[q]+=mx+1000  # force extension first, ranked by density among extensions via tie? use large bonus
        # Actually to rank extensions by density, we need bonus large but equal; argmax_centre then picks centre-first among max. Acceptable: round1 line-bonus used large bonus.
    return s

def entropy_scores(rem, misses, sunk, hits):
    scores, per, total = density_detail(rem, misses, sunk, hits)
    if total<=0: return np.zeros(N*N)
    # per-ship hit counts
    hitcount=np.zeros(N*N)
    # recompute per ship bincount of counted placements
    for L, counted in zip(rem, per):
        if counted.any():
            hitcount+=np.bincount(PL_INT[L][counted].ravel(), minlength=N*N)
    # hitcount == scores (sum) by construction; S_hit=scores, S_miss=total-scores
    # p(occupied) approx: 1 - prod_L (1 - hitcount_L/total_L) ; need per-ship hitcount
    p=np.ones(N*N)
    for L, counted in zip(rem, per):
        tot=int(counted.sum())
        if tot==0: continue
        hc=np.bincount(PL_INT[L][counted].ravel(), minlength=N*N)/tot
        p*= (1-hc)
    p=1-p
    p=np.clip(p, 1e-9, 1-1e-9)
    Sh=np.maximum(scores,1); Sm=np.maximum(total-scores,1)
    gain=np.log(total) - (p*np.log(Sh) + (1-p)*np.log(Sm))
    return gain

def entropy_score_fn(rem, misses, sunk, hits, shot):
    return entropy_scores(rem, misses, sunk, hits)

def blend_score_fn(rem, misses, sunk, hits, shot):
    d,_ ,_=density_detail(rem, misses, sunk, hits)
    g=entropy_scores(rem, misses, sunk, hits)
    # normalise by max over unshot
    unshot=[q for q in range(N*N) if q not in shot]
    dm=float(d[unshot].max()) if unshot else 1.0
    gm=float(g[unshot].max()) if unshot else 1.0
    if dm<=0: dm=1.0
    if gm<=0: gm=1.0
    return d/dm + g/gm

def entropy_line_score_fn(rem, misses, sunk, hits, shot):
    # blend, then line bonus on top (same large bonus logic as line_bonus, applied to blended score)
    s=blend_score_fn(rem, misses, sunk, hits, shot)
    if len(hits)>=2:
        from collections import defaultdict
        rows=defaultdict(list); cols=defaultdict(list)
        for q in sorted(hits): rows[q//N].append(q%N); cols[q%N].append(q//N)
        bonus=set()
        for r,cs in rows.items():
            if len(cs)>=2:
                for cc in (min(cs)-1, max(cs)+1):
                    if 0<=cc<N: bonus.add(r*N+cc)
        for c,rs in cols.items():
            if len(rs)>=2:
                for rr in (min(rs)-1, max(rs)+1):
                    if 0<=rr<N: bonus.add(rr*N+c)
        mx=float(s.max())
        for q in bonus:
            if q not in shot: s[q]=mx+10.0
    return s

# ---------- human players ----------
def play_human(ships, rng, hunt_mode='random_parity', clear_on_sink='adjacent', line_only=False):
    """hunt_mode: random_parity | centre_parity | spacing_random | spacing_centre | centre_spacing
       spacing: hunt squares (r+c)%m==0 where m=min remaining length; order random or centre.
       centre_parity: parity squares in centre-distance order.
       clear_on_sink: 'adjacent' (drop stack entries adjacent to sunk cells), 'all', 'none'
    """
    cells=[set(s) for s in ships]
    cell_to={}
    for i,s in enumerate(cells):
        for c in s: cell_to[c]=i
    remaining=set(range(len(FLEET))); shot=set(); stack=[]; unsunk=set(); n=0
    # hunt order management: we generate a hunt list per mode and advance a pointer,
    # but for spacing the list changes when m changes, so regenerate lazily.
    hunt=[]; hi=0
    def build_hunt():
        nonlocal hunt, hi
        m=min(FLEET[i] for i in remaining) if remaining else 2
        if hunt_mode in ('spacing_random','spacing_centre'):
            cand=[q for q in range(N*N) if (q//N+q%N)%m==0 and q not in shot]
        elif hunt_mode=='centre_parity':
            cand=[q for q in range(N*N) if (q//N+q%N)%2==0 and q not in shot]
            cand.sort(key=lambda q:(DIST2[q], q)); return cand, 0
        else: # random_parity
            cand=[q for q in range(N*N) if (q//N+q%N)%2==0 and q not in shot]
            rng.shuffle(cand); return cand, 0
        if hunt_mode=='spacing_centre':
            cand.sort(key=lambda q:(DIST2[q], q))
        else:
            rng.shuffle(cand)
        return cand, 0
    hunt, hi = build_hunt()
    last_m = min(FLEET[i] for i in remaining) if remaining else 2
    def group_of(sq):
        seen={sq}; fr=[sq]
        while fr:
            cur=fr.pop()
            for nb in E.neighbours(cur):
                if nb in unsunk and nb not in seen: seen.add(nb); fr.append(nb)
        return seen
    while remaining:
        # spacing change -> rebuild hunt (keep shot exclusion)
        m=min(FLEET[i] for i in remaining)
        if hunt_mode.startswith('spacing') and m!=last_m:
            hunt, hi = build_hunt(); last_m=m
        while stack and stack[-1] in shot: stack.pop()
        if stack: sq=stack.pop()
        else:
            while hi < len(hunt) and hunt[hi] in shot: hi+=1
            if hi < len(hunt): sq=hunt[hi]; hi+=1
            else:
                # fallback: any unshot, centre first (for spacing this should not happen before sink, but after hunt exhausts)
                cand=[q for q in range(N*N) if q not in shot]
                if not cand: break
                # if spacing, rebuild with m (should be exhausted -> means ships should have been found; fallback centre)
                cand.sort(key=lambda q:(DIST2[q], q)); sq=cand[0]
        n+=1; shot.add(sq); idx=cell_to.get(sq)
        if idx is None or idx not in remaining: continue
        unsunk.add(sq); grp=group_of(sq)
        # target pushes
        if line_only and len(grp)>=2:
            rows={g//N for g in grp}; cols={g%N for g in grp}; ext=[]
            if len(rows)==1:
                r=next(iter(rows)); cs=sorted(g%N for g in grp)
                for cc in (min(cs)-1, max(cs)+1):
                    if 0<=cc<N: ext.append(r*N+cc)
            elif len(cols)==1:
                c=next(iter(cols)); rs=sorted(g//N for g in grp)
                for rr in (min(rs)-1, max(rs)+1):
                    if 0<=rr<N: ext.append(rr*N+c)
            for e in reversed(ext):
                if e not in shot and e not in stack: stack.append(e)
        else:
            # engine stack: neighbours, then line extensions after side neighbours so they pop first
            ext=[]
            if len(grp)>=2:
                rows={g//N for g in grp}; cols={g%N for g in grp}
                if len(rows)==1:
                    r=next(iter(rows)); cs=sorted(g%N for g in grp)
                    for cc in (min(cs)-1, max(cs)+1):
                        if 0<=cc<N: ext.append(r*N+cc)
                elif len(cols)==1:
                    c=next(iter(cols)); rs=sorted(g//N for g in grp)
                    for rr in (min(rs)-1, max(rs)+1):
                        if 0<=rr<N: ext.append(rr*N+c)
            for nb in E.neighbours(sq):
                if nb not in shot and nb not in stack: stack.append(nb)
            for e in reversed(ext):
                if e not in shot and e not in stack: stack.append(e)
        if cells[idx]<=shot:
            remaining.remove(idx); unsunk-=cells[idx]
            if clear_on_sink=='all': stack=[]
            elif clear_on_sink=='adjacent':
                sunkcells=cells[idx]
                def adj_to_sunk(q):
                    r,c=divmod(q,N)
                    for sc in sunkcells:
                        r2,c2=divmod(sc,N)
                        if abs(r-r2)+abs(c-c2)==1: return True
                    return False
                stack=[q for q in stack if not adj_to_sunk(q)]
            # after sink, spacing may change on next loop
    return n

def main():
    seed=20291102; n=3000
    print(f"seed {seed} n {n}", flush=True)
    frng=random.Random(seed)
    fleets=[E.sample_fleet(frng) for _ in range(n)]
    print("fleets sampled", flush=True)
    results={}
    def run(name, fn):
        t=time.time(); vals=[fn(f) for f in fleets]; dt=time.time()-t
        m,med,sd,hw=stats(vals); results[name]=vals
        print(f"{name}: mean {m:.2f} median {med:.1f} SD {sd:.2f} CI +/-{hw:.2f}  time {dt:.1f}s", flush=True)
        return vals
    dens=run("Density", lambda f: play_density_generic(f, density_score_fn))
    run("LineBonusDensity", lambda f: play_density_generic(f, line_bonus_score_fn))
    run("Entropy", lambda f: play_density_generic(f, entropy_score_fn))
    run("BlendDensityEntropy", lambda f: play_density_generic(f, blend_score_fn))
    run("BlendEntropyLine", lambda f: play_density_generic(f, entropy_line_score_fn))
    # human: use per-game rngs like production (base+offset+i) for random arms
    run("Human_ParityTarget_baseline", lambda f, i=[0]: None) if False else None
    # do human arms with explicit loop to pass rng per game
    def run_human(name, mode, clear, line_only, seedoff):
        t=time.time(); vals=[]
        for i,f in enumerate(fleets):
            vals.append(play_human(f, random.Random(seed+seedoff+i), hunt_mode=mode, clear_on_sink=clear, line_only=line_only))
        dt=time.time()-t; m,med,sd,hw=stats(vals); results[name]=vals
        print(f"{name}: mean {m:.2f} median {med:.1f} SD {sd:.2f} CI +/-{hw:.2f}  time {dt:.1f}s", flush=True)
        return vals
    run_human("Human_ParityTarget_repro", 'random_parity', 'none', False, 3000000)
    run_human("Human_ParityTarget_clearAdj", 'random_parity', 'adjacent', False, 3100000)
    run_human("Human_CentreParity_clearAdj", 'centre_parity', 'adjacent', False, 3200000)
    run_human("Human_SpacingRandom_clearAdj", 'spacing_random', 'adjacent', False, 3300000)
    run_human("Human_SpacingRandom_clearAdj_lineOnly", 'spacing_random', 'adjacent', True, 3400000)
    run_human("Human_SpacingCentre_clearAdj_lineOnly", 'spacing_centre', 'adjacent', True, 3500000)
    run_human("Human_CentreParity_clearAdj_lineOnly", 'centre_parity', 'adjacent', True, 3600000)
    print("\nPAIRED DIFFS vs Density (negative = fewer shots = better):", flush=True)
    for name, vals in results.items():
        if name=="Density": continue
        dm, hw = paired(vals, dens)
        print(f"  {name} - Density: {dm:+.2f} +/-{hw:.2f}", flush=True)
    # save
    import json
    out={k: {"mean": stats(v)[0], "median": stats(v)[1], "sd": stats(v)[2], "ci_hw": stats(v)[3],
             "paired_vs_density": paired(v, dens) if k!="Density" else None} for k,v in results.items()}
    with open('/home/hatch/workspace/sim-lab/battleship/advanced2_core.json','w') as fh: json.dump(out, fh, indent=2)
    print("wrote advanced2_core.json", flush=True)

if __name__=='__main__': main()
