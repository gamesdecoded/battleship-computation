#!/usr/bin/env python3
"""Round 2: rollout/lookahead + profile-DP joint-count prototype.
Rollout: first 10 shots only, top-3 density squares, M sampled fleets
consistent with shots so far (rejection from uniform fleets), each
candidate rolled out with greedy density to completion; pick lowest
expected remaining shots. After shot 10, pure density.
DP: row-by-row profile DP counting fleets consistent with a given
miss/hit/sunk state. Prototype + timing only; no tournament unless fast.
"""
import math, random, sys, time
import numpy as np
sys.path.insert(0, '/home/hatch/workspace/sim-lab/battleship')
import engine as E
from advanced2 import density_detail, argmax_centre, play_density_generic, density_score_fn, stats, paired

N=10; FLEET=E.FLEET

# ---------- stateful density continuation ----------
def density_continue(ships, shot, misses, sunk, hits, remaining, len_by_idx=None):
    """len_by_idx: optional dict idx->length, used when remaining indices are
    positions in a sampled fleet whose ship lengths differ from FLEET[idx].
    If omitted, lengths are FLEET[idx] (the true-fleet case)."""
    cells=[set(s) for s in ships]
    shot=set(shot); misses=set(misses); sunk=set(sunk); hits=set(hits); remaining=list(remaining)
    n=0
    while remaining:
        if n>=100:
            # Safety cap: should never happen for a valid fleet (max 100 squares).
            # Returning 100 flags a bug instead of looping forever.
            return 100
        if len_by_idx is not None:
            rem=[len_by_idx[i] for i in remaining]
        else:
            rem=[FLEET[i] for i in remaining]
        scores,_ ,_=density_detail(rem, misses, sunk, hits)
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

def sample_consistent(rng, misses, sunk, hits, remaining_lengths, sunk_ships_cells, max_tries=20000):
    """Fast consistent-fleet sampler for rollout only.
    Remaining ships are sampled from their valid placements (numpy-filtered
    to avoid misses+sunk), disjointness enforced per ship, fleet rejected
    unless every unsunk hit is covered. Sunk ships are fixed exactly.
    The sampled fleet is a valid fleet, so a rollout on it is valid;
    the sampling distribution is uniform over valid placements per ship
    conditioned on disjointness + hit coverage (not exactly uniform over
    consistent fleets, because per-ship sequential sampling weights differ
    slightly - flagged in the report as an approximation, not exact posterior).
    """
    blocked = misses | sunk
    barr = np.zeros(N*N, dtype=bool)
    for c in blocked: barr[c]=True
    valid_idx={}
    for L in set(remaining_lengths):
        b=E.PL_BOOL[L]
        valid_idx[L]=np.flatnonzero(~b[:, barr].any(axis=1)) if blocked else np.arange(len(b))
    # Hit-guided attempts first: force ships to cover hits, then fill rest.
    # This avoids the blind-rejection blowup when hits are hard to cover jointly.
    hits_list=sorted(hits)
    # bitmask helpers: PL_MASK[L][idx] is an int mask over 100 cells
    def mask_of_cells(s):
        m=0
        for c in s: m|=1<<c
        return m
    sunk_mask=0
    for sc in sunk_ships_cells: sunk_mask|=mask_of_cells(sc)
    for _attempt in range(60):
        if not hits_list:
            break
        used_mask=sunk_mask; used=set()
        for sc in sunk_ships_cells: used|=sc
        ships_by_len={}
        remaining=list(remaining_lengths); ok=True; covered=set()
        for h in hits_list:
            if h in covered: continue
            order=list(range(len(remaining))); rng.shuffle(order)
            placed=False
            for oi in order:
                if remaining[oi] is None: continue
                L=remaining[oi]
                cand=[idx for idx in valid_idx[L] if (E.PL_MASK[L][idx]>>h)&1 and not (E.PL_MASK[L][idx]&used_mask)]
                if not cand: continue
                idx=int(cand[rng.randrange(len(cand))]); cells=set(int(x) for x in E.PL_INT[L][idx])
                ships_by_len[oi]=cells; used|=cells; covered|=cells; used_mask|=E.PL_MASK[L][idx]; remaining[oi]=None; placed=True; break
            if not placed: ok=False; break
        if not ok: continue
        for oi, L in enumerate(remaining_lengths):
            if oi in ships_by_len: continue
            idxs=valid_idx[L]; placed=False
            for _try in range(60):
                idx=int(idxs[rng.randrange(len(idxs))])
                if E.PL_MASK[L][idx]&used_mask: continue
                cells=set(int(x) for x in E.PL_INT[L][idx]); ships_by_len[oi]=cells; used|=cells; used_mask|=E.PL_MASK[L][idx]; placed=True; break
            if not placed: ok=False; break
        if not ok: continue
        ships=[ships_by_len[i] for i in range(len(remaining_lengths))]
        if not hits<=set().union(*ships): continue
        # A remaining ship whose cells are all already shot would already be
        # sunk (and its hits removed) - such a fleet is inconsistent with the
        # sunk announcement, and it makes density_continue loop forever.
        if any(s<=hits for s in ships): continue
        return sunk_ships_cells+ships
    for _ in range(200):
        used_mask=sunk_mask; used=set()
        for sc in sunk_ships_cells: used|=sc
        ships=[]; ok=True
        for L in remaining_lengths:
            idxs=valid_idx[L]
            if len(idxs)==0: ok=False; break
            placed=False
            for _try in range(60):
                idx=int(idxs[rng.randrange(len(idxs))])
                if E.PL_MASK[L][idx]&used_mask: continue
                cells=set(int(x) for x in E.PL_INT[L][idx]); ships.append(cells); used|=cells; used_mask|=E.PL_MASK[L][idx]; placed=True; break
            if not placed: ok=False; break
        if not ok: continue
        covered=set().union(*ships) if ships else set()
        if not hits<=covered: continue
        if any(s<=hits for s in ships): continue
        return sunk_ships_cells+ships
    return None

def play_rollout_first10(ships, rng, k=3, M=6):
    cells=[set(s) for s in ships]
    remaining=list(range(len(FLEET))); shot=set(); misses=set(); sunk=set(); hits=set()
    sunk_ships=[]; n=0
    while remaining and n<100:
        rem=[FLEET[i] for i in remaining]
        scores,_,_=density_detail(rem, misses, sunk, hits)
        # top-k unshot by score, centre tie-break
        unshot=[q for q in range(N*N) if q not in shot]
        unshot.sort(key=lambda q:(-int(scores[q]), float(E.DIST2[q]), q))
        cands=unshot[:k]
        if n<10 and len(cands)>1:
            # sample M consistent fleets
            samples=[]
            tries=0
            while len(samples)<M and tries<40:
                tries+=1
                f=sample_consistent(rng, misses, sunk, hits, rem, sunk_ships, max_tries=200)
                if f is not None: samples.append(f)
            if samples:
                best=None; best_val=None
                for q in cands:
                    tot=0; cnt=0
                    for f in samples:
                        # state after shooting q on fleet f
                        f_cells=[set(s) for s in f]
                        # map remaining in f: sunk ships are first len(sunk_ships) entries; remaining indices follow
                        # For continuation we need remaining list aligned to f: remaining ships are those not sunk.
                        # f = sunk_ships + remaining ships in order of rem lengths? sample_consistent returns sunk + ships in rem order.
                        # So remaining indices in f are len(sunk_ships)..len(f)-1
                        f_rem=list(range(len(sunk_ships), len(f)))
                        f_shot=set(shot); f_miss=set(misses); f_sunk=set(sunk); f_hits=set(hits)
                        f_shot.add(q); hit_idx=None
                        for idx in f_rem:
                            if q in f_cells[idx]: hit_idx=idx; break
                        if hit_idx is None: f_miss.add(q)
                        else:
                            f_hits.add(q)
                            if f_cells[hit_idx]<=f_shot:
                                f_rem.remove(hit_idx); f_sunk|=f_cells[hit_idx]; f_hits-=f_cells[hit_idx]
                        # lengths for f_rem: sampled fleet = sunk_ships + remaining ships in rem order
                        len_by_idx={idx: rem[idx-len(sunk_ships)] for idx in f_rem}
                        tot+=1+density_continue(f, f_shot, f_miss, f_sunk, f_hits, f_rem, len_by_idx=len_by_idx); cnt+=1
                    val=tot/cnt if cnt else 1e9
                    if best_val is None or val<best_val: best_val=val; best=q
                sq=best
            else:
                sq=cands[0]
        else:
            sq=cands[0]
        n+=1; shot.add(sq); hit=None
        for idx in remaining:
            if sq in cells[idx]: hit=idx; break
        if hit is None: misses.add(sq)
        else:
            hits.add(sq)
            if cells[hit]<=shot:
                remaining.remove(hit); sunk|=cells[hit]; hits-=cells[hit]; sunk_ships.append(cells[hit])
    return n

# ---------- profile DP counting ----------
# State at row boundary: (row, used_mask, profile) where profile is tuple len 10:
#  0 = free, else (ship_pos_in_fleet_index, remaining_including_current_row?) 
# We implement DP by iterating rows and generating, for each row, all ways to
# cover the row cells consistent with: profile continuations, hits, misses,
# horizontal ships, vertical starts. This is a recursive row filler with memo.
# Ships are labelled by fleet index 0..4 (two 3s distinct). used_mask tracks ships
# already placed (started) - a vertical ship counts as used at its start row.
# profile entry: 0 free, or (idx, rem) where rem = cells still to place INCLUDING this row.
# At row start, continuations must occupy their column cell this row.
# A horizontal ship occupies a contiguous run in this row, all cells free (profile 0),
# not miss, and if it is length L it must be an unused ship of that length.
# A vertical start occupies one cell this row, profile becomes (idx, L-1) for next row
# if L>1, else completed immediately.
# Cells: hit cells must be occupied (by continuation, horizontal, or vertical start);
# miss cells must be empty; other cells may be occupied or empty.
# At the end (row 10), profile must be all 0 and used_mask all ships.

def dp_count(misses, hits, sunk_lengths, remaining_lengths_with_idx):
    """Count fleets (labelled, remaining ships only) consistent with hits/misses.
    sunk ships are removed entirely (their cells are treated as blocked, passed in misses).
    remaining_lengths_with_idx: list of (fleet_idx, length) for remaining ships.
    Returns (count, n_states_visited, seconds).
    """
    import functools, time
    t0=time.time()
    miss_set=set(misses); hit_set=set(hits)
    # blocked for remaining ships = misses (sunk cells should be included by caller)
    ships=remaining_lengths_with_idx
    # map length -> list of idx
    from functools import lru_cache
    # Encode profile as tuple of ints: 0 free, else idx*10+rem (rem 1..5, idx 0..4 -> code idx*10+rem, but rem can be confused? use idx*8+rem)
    def code(idx, rem): return idx*8+rem+1  # +1 to avoid 0
    def decode(c):
        if c==0: return None
        c2=c-1; return (c2//8, c2%8)
    # Use recursion with lru_cache on (row, used_mask, profile_tuple)
    # Row generation is recursive over columns, returning list of (next_profile, used_mask_delta, hit_ok)
    # Instead implement DP iterative: dict state->count, for each row generate next states via DFS.
    init_profile=tuple([0]*N)
    dp={ (0, init_profile): 1 }
    states_visited=0
    for r in range(N):
        ndp={}
        row_miss=[c in miss_set for c in range(N)]
        row_hit=[c in hit_set for c in range(N)]
        for (used, prof), cnt in dp.items():
            states_visited+=1
            # DFS over columns c=0..9 building next_profile and used2
            # next_profile initially: for each column, if prof has continuation, it occupies cell;
            #   if rem==1, it ends (next 0); if rem>1, next = code(idx, rem-1)
            # We need to decide horizontal/vertical/empty for free columns.
            # Recursive function:
            def dfs(c, next_prof_list, used2, covered_hit_ok):
                if c>=N:
                    # all hit cells in row must be covered: check via occupancy list
                    yield (tuple(next_prof_list), used2)
                    return
                cont=decode(prof[c])
                if cont is not None:
                    idx, rem = cont
                    # continuation occupies cell c: must not be miss
                    if row_miss[c]: return
                    # occupied, so hit satisfied if present
                    nxt = 0 if rem==1 else code(idx, rem-1)
                    next_prof_list.append(nxt)
                    yield from dfs(c+1, next_prof_list, used2, covered_hit_ok)
                    next_prof_list.pop()
                    return
                # free column: options
                # 1) empty, if not hit
                if not row_hit[c]:
                    next_prof_list.append(0)
                    yield from dfs(c+1, next_prof_list, used2, covered_hit_ok)
                    next_prof_list.pop()
                # 2) horizontal ship starting at c: try each unused ship length L, run c..c+L-1 all free, not miss
                #    (hit cells inside run are covered)
                for (idx, L) in ships:
                    if used2>>idx & 1: continue
                    if c+L> N: continue
                    ok=True
                    for cc in range(c, c+L):
                        if decode(prof[cc]) is not None: ok=False; break
                        if row_miss[cc]: ok=False; break
                    if not ok: continue
                    # horizontal occupies c..c+L-1: next profile 0 for those, and we jump
                    # To implement jump in DFS, we append L zeros and recurse at c+L
                    # Use a helper recursion via closure: we can do iterative by calling a sub-DFS with c+L
                    # Simpler: handle by extending next_prof_list and recursing, but need to skip columns.
                    # We implement this branch by a nested function call that appends and recurses:
                    for _ in range(L): next_prof_list.append(0)
                    yield from dfs(c+L, next_prof_list, used2 | (1<<idx), covered_hit_ok)
                    for _ in range(L): next_prof_list.pop()
                # 3) vertical start at c: unused ship length L>=2 (L==1 none), occupies cell, next = code(idx,L-1) if L>1
                #    Note: a vertical start of length L will occupy this cell and L-1 below; it must not be miss (already not hit? hit ok)
                for (idx, L) in ships:
                    if used2>>idx & 1: continue
                    if L<2: continue
                    if r+L> N: continue  # must fit vertically
                    # also cells below that are misses would be caught when those rows are processed as continuations (miss -> prune), but prune early: check below cells not miss? continuation cells cannot be miss, so if any below cell is miss, this start is invalid.
                    bad=False
                    for rr in range(r+1, r+L):
                        if (rr*N+c) in miss_set: bad=True; break
                    if bad: continue
                    next_prof_list.append(code(idx, L-1))
                    yield from dfs(c+1, next_prof_list, used2 | (1<<idx), covered_hit_ok)
                    next_prof_list.pop()
            for (nprof, used2) in dfs(0, [], used, False):
                # hit coverage check: for each hit cell in row, it must be occupied.
                # Occupancy in row: continuation (prof!=0) OR horizontal (we lost track) OR vertical start.
                # Reconstruct occupancy: a cell is occupied if prof!=0, or next_prof from horizontal/vertical?
                # Horizontal cells have next 0, same as empty, so we cannot distinguish after the fact.
                # Fix: track occupancy in DFS. Patch: redo check inside DFS by requiring hit cells to be covered at decision time.
                # Our DFS above already: empty option forbidden on hit cells, so hit cells are never empty.
                # Horizontal/vertical branches are only taken on cells that are not forced empty, and they occupy.
                # Continuation occupies. So any completed row automatically covers hits. Good.
                key=(used2, nprof)
                ndp[key]=ndp.get(key,0)+cnt
        dp=ndp
        if not dp:
            break
    total=0
    full_mask=0
    for (idx,_) in ships: full_mask|=1<<idx
    for (used, prof), cnt in dp.items():
        if used==full_mask and all(p==0 for p in prof): total+=cnt
    return total, states_visited, time.time()-t0

if __name__=='__main__':
    print("DP / rollout module loaded")
