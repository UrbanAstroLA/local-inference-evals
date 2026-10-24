#!/usr/bin/env python3
"""audit-s0 task 1/2/5: recount the Stage-0 G-entry comparison at digest point 0, split by what each compared
block IS at that boundary (assessor 22ax section 3).  CPU only; reads the retained records read-only.

Inputs (read-only, mounted at /p4 by run_cpu.sh):
  /p4/runs-stage0/stage0-attempt-1/{client.jsonl, p4log/p4_r0_p*.jsonl, p4log/sched_r0_p*.jsonl}   (public)
  /p4/PRIVATE-runs-stage0/stage0-attempt-1/p4/entry_<tag>_r<rank>.npz                               (private)
Outputs (public-safe: counts, categories, block ids, equal/differ flags; no digest values, no token ids):
  /out/recount_point0.json, /out/recount_point0.out, /out/recount_point0_blocks.csv

Categories at point 0 (before the forced request's first remainder-prefill forward; the digest runs inside
model_runner.prepare_attn, BEFORE model_state.preprocess_state = the mamba align pre-copy):
  group 0 (MLA KV 'attn', indexer K 'idxk'; block 4352 tokens):
    cached_full          (j+1)*4352 <= n_cached
    cached_partial       j*4352 < n_cached < (j+1)*4352
    new_beyond_cached    j*4352 >= n_cached (allocated for this step, not yet written by this request)
    idxk cached_full blocks are further split by whether the warm prefill that built them left pool records
    unwritten (chunk start not a multiple of kpool=4 inside the block; _kpool_compress_insert write_mask):
    idxk_cached_full_with_unwritten_pools / idxk_cached_full_all_pools_written
  group 1 (kpool tail ring, 1 block/request): tail_ring_new (n_cached % 4 == 0 -> no in-progress pool at point 0)
  groups 2..8 (KDA, block 4352, align mode):
    kda_cached_state     j == n_cached//4352 - 1 : the src column of the align pre-copy (state at n_cached)
                         recurrent (#1): whole view is live state;   conv (#0): rows 0..2 live, rows 3..5 inactive
    kda_running_new      j == (n_cached + n_sched - 1)//4352 : pre-copy destination, still stale at point 0
    kda_spec_scratch     j > running column (num_speculative_blocks = 3)
    kda_older            non-null j < cached-state column (not read by the next step)
  group 9 (DFlash2 draft, sliding window, block 1088): draft_cached_full / draft_new_beyond_cached
  null_placeholder     physical block 0 in the row in both repeats (skipped by g_entry, counted here)
  null_mismatch        physical block 0 in one repeat only
"""
import collections
import csv
import glob
import json
import os

import numpy as np

AO = "/p4/runs-stage0/stage0-attempt-1"
PO = "/p4/PRIVATE-runs-stage0/stage0-attempt-1"
OUT = "/out"
KPOOL = 4
BS = {0: 4352, 1: 8, 9: 1088}
for g in range(2, 9):
    BS[g] = 4352


def jl(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        for line in open(p):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


client = jl(os.path.join(AO, "client.jsonl"))
windows = [d for d in client if d.get("kind") == "start"][-1]["windows"]
logs0 = jl(os.path.join(AO, "p4log", "p4_r0_p*.jsonl"))
blocks = {r["tag"]: r for r in logs0 if r.get("kind") == "blocks"}
reqs = {r["tag"]: r for r in logs0 if r.get("kind") == "req"}
steps = []
for r in jl(os.path.join(AO, "p4log", "sched_r0_p*.jsonl")):
    steps += r.get("steps", [])
steps.sort(key=lambda d: d["step"])

# chunk starts of every keyed request (n_computed at each of its prefilling steps)
chunk_starts = collections.defaultdict(list)
for d in steps:
    for r in d["rows"]:
        if r[3] and r[8]:
            chunk_starts[r[3]].append(int(r[5]))


def family(name, g):
    if g == 0:
        return "idxk" if "indexer.k_cache" in name else "mla"
    if g == 1:
        return "tail"
    if 2 <= g <= 8:
        return "kda_conv" if name.endswith("#0") else "kda_rec"
    return "draft"


def layer(name):
    s = name.split("layers.")[1]
    return int(s.split(".")[0].split("#")[0])


def unwritten_pools(warm_tag, j):
    """Pool indices (block-local) of block j that the warm prefill never wrote: one per chunk start inside the
    block that is not a multiple of KPOOL (the straddling pool is masked by write_mask in both chunks)."""
    lo, hi = j * 4352, (j + 1) * 4352
    out = []
    for s in chunk_starts.get(warm_tag, []):
        if lo < s < hi and s % KPOOL:
            out.append((s - lo) // KPOOL)
    return out


def category(fam, g, j, b1, b2, nc, ns):
    if b1 == 0 and b2 == 0:
        return "null_placeholder"
    if (b1 == 0) != (b2 == 0):
        return "null_mismatch"
    bs = BS[g]
    if g == 0:
        if (j + 1) * bs <= nc:
            return "cached_full"
        if j * bs < nc:
            return "cached_partial"
        return "new_beyond_cached"
    if g == 1:
        return "tail_ring_new"
    if 2 <= g <= 8:
        c = nc // bs - 1
        run = (nc + ns - 1) // bs
        if j == c:
            return "kda_cached_state"
        if j == run:
            return "kda_running_new"
        if j > run:
            return "kda_spec_scratch"
        return "kda_older"
    if (j + 1) * bs <= nc:
        return "draft_cached_full"
    if j * bs < nc:
        return "draft_cached_partial"
    return "draft_new_beyond_cached"


agg = collections.defaultdict(lambda: [0, 0])          # (var, fam, cat) -> [differ, compared]
zero = collections.Counter()                           # (var, fam, cat, 'r1'|'r2') -> blocks whose digest pair is (0, 0)
earliest_layer = {}                                    # (var, window, rank) -> lowest layer with a valid-state difference
focus = []                                             # private: digest pairs of the focus slices
agg_layer = collections.defaultdict(lambda: [0, 0])    # (var, fam, cat, layer) -> [differ, compared]
first = {}                                             # var -> first mismatch in g_entry order, per scope
prov = collections.Counter()
alias = collections.Counter()
nullblk = collections.Counter()
unwritten_summary = collections.Counter()
rows_out = []
SCOPES = {
    # scope name -> set of categories (with fam qualifiers) that count as "valid consumed state"
    "valid_strict": {("mla", "cached_full"), ("kda_rec", "kda_cached_state"), ("draft", "draft_cached_full")},
    "valid_incl_idxk": {("mla", "cached_full"), ("kda_rec", "kda_cached_state"), ("draft", "draft_cached_full"),
                        ("idxk", "cached_full")},
    "as_published": None,
}

for var in ("V1", "V2"):
    for wid in windows:
        reply = wid.split("-")[0]
        for rank in (0, 1):
            t1, t2 = f"{var}-{wid}-r1", f"{var}-{wid}-r2"
            p1 = os.path.join(PO, "p4", f"entry_{t1}_r{rank}.npz")
            p2 = os.path.join(PO, "p4", f"entry_{t2}_r{rank}.npz")
            e1, e2 = np.load(p1), np.load(p2)
            pts = e1["points"]
            assert np.array_equal(pts[:, 1:], e2["points"][:, 1:])
            nc, ns = int(pts[0][2]), int(pts[0][4])
            tens = [str(x) for x in e1["tensors"]]
            grp = [int(x) for x in e1["tensor_group"]]
            d1, d2 = e1["edig"], e2["edig"]
            # aliasing: does any physical block id appear in two different groups' rows at point 0?
            for e, rep in ((e1, 1), (e2, 2)):
                seen = {}
                for g in range(e["eblk"].shape[1]):
                    r = e["eblk"][0, g]
                    for x in r[r > 0].tolist():
                        if x in seen and seen[x] != g:
                            alias[(var, "cross_group_same_block")] += 1
                        seen.setdefault(x, g)
                alias[(var, "rows_checked")] += 1
            for ti, name in enumerate(tens):
                g = grp[ti]
                if g < 0:
                    continue
                fam = family(name, g)
                L = layer(name)
                nlog = int((e1["eblk"][0, g] >= 0).sum())
                assert nlog == int((e2["eblk"][0, g] >= 0).sum())
                for j in range(nlog):
                    b1, b2 = int(e1["eblk"][0, g, j]), int(e2["eblk"][0, g, j])
                    cat = category(fam, g, j, b1, b2, nc, ns)
                    eq = bool(np.array_equal(d1[0, ti, j], d2[0, ti, j]))
                    z1 = bool((d1[0, ti, j] == 0).all())
                    z2 = bool((d2[0, ti, j] == 0).all())
                    extra = ""
                    if cat == "null_placeholder":
                        nullblk[(var, fam)] += 1
                        continue
                    if fam == "idxk" and cat == "cached_full":
                        u1 = unwritten_pools(f"W-{var}-r1-{reply}", j)
                        u2 = unwritten_pools(f"W-{var}-r2-{reply}", j)
                        sub = "with_unwritten_pools" if (u1 or u2) else "all_pools_written"
                        unwritten_summary[(var, sub)] += 1
                        extra = "pools_unwritten_r1=%s;r2=%s" % (u1, u2)
                    key = (var, fam, cat)
                    zero[(var, fam, cat, "r1")] += int(z1)
                    zero[(var, fam, cat, "r2")] += int(z2)
                    if not eq and (fam, cat) in (("mla", "cached_full"), ("kda_rec", "kda_cached_state")):
                        k3 = (var, wid, rank)
                        earliest_layer[k3] = min(earliest_layer.get(k3, 99), L)
                    if var == "V1" and wid == "R01-early" and rank == 0 and (
                            (L == 7 and j == 0 and fam in ("idxk", "mla")) or (L in (2, 4) and cat == "kda_cached_state")):
                        focus.append(dict(tensor=name, logical_block=j, phys=[b1, b2], category=cat,
                                          digest_r1=d1[0, ti, j].tolist(), digest_r2=d2[0, ti, j].tolist()))
                    agg[key][0] += int(not eq)
                    agg_layer[(var, fam, cat, L)][1] += 1
                    agg_layer[(var, fam, cat, L)][0] += int(not eq)
                    if cat in ("cached_full", "kda_cached_state", "draft_cached_full"):
                        wk = blocks.get(f"W-{var}-r1-{reply}"), blocks.get(f"W-{var}-r2-{reply}")
                        ok = all(w is not None and j < len(w["groups"][g]) and w["groups"][g][j] == b
                                 for w, b in zip(wk, (b1, b2)))
                        freed = all(w is not None and j < len(w["groups"][g]) and w["groups"][g][j] == 0 for w in wk)
                        prov[(var, fam, "from_own_warm" if ok else (
                            "warm_row_null_at_removal(unverifiable)" if freed else "NOT_from_own_warm"))] += 1
                    rows_out.append(dict(variant=var, window=wid, rank=rank, tensor=name, family=fam, layer=L,
                                         group=g, logical_block=j, phys_r1=b1, phys_r2=b2, category=cat,
                                         equal=int(eq), note=extra))
                    for scope, allowed in SCOPES.items():
                        if (var, scope) in first:
                            continue
                        if not eq and (allowed is None or (fam, cat) in allowed):
                            first[(var, scope)] = dict(window=wid, rank=rank, tensor=name, family=fam, group=g,
                                                       logical_block=j, phys_r1=b1, phys_r2=b2, category=cat,
                                                       note=extra)

# --- write outputs
doc = dict(
    point=0,
    note=("counts are [differing, compared] over windows x ranks x tensors x logical blocks at digest point 0; "
          "physical null-block placeholders excluded (counted separately)"),
    by_family_category={"|".join(k): v for k, v in sorted(agg.items())},
    by_layer={"|".join(map(str, k)): v for k, v in sorted(agg_layer.items())},
    first_mismatch_in_g_entry_order={f"{k[0]}|{k[1]}": v for k, v in sorted(first.items())},
    provenance_of_valid_blocks={"|".join(k): v for k, v in sorted(prov.items())},
    aliasing_point0={"|".join(k): v for k, v in sorted(alias.items())},
    null_placeholders={"|".join(k): v for k, v in sorted(nullblk.items())},
    idxk_cached_full_blocks={"|".join(k): v for k, v in sorted(unwritten_summary.items())},
    zero_digest_blocks={"|".join(k): v for k, v in sorted(zero.items())},
    earliest_layer_with_valid_difference=collections.Counter(
        f"{k[0]}|L{v}" for k, v in earliest_layer.items()),
)
os.makedirs("/pout", exist_ok=True)
old = os.umask(0o077)
json.dump(dict(note="PRIVATE: digest pairs (two int64 sums) for the audit focus slices, V1 R01-early rank 0, point 0",
               slices=focus), open("/pout/focus_digests_point0.json", "w"), indent=1)
os.umask(old)
json.dump(doc, open(os.path.join(OUT, "recount_point0.json"), "w"), indent=1)
with open(os.path.join(OUT, "recount_point0_blocks.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

L = ["# audit-s0 recount at digest point 0 (scripts/recount_point0.py)", "",
     "differ/compared per variant, family, category (physical null placeholders excluded)", ""]
for k, (a, n) in sorted(agg.items()):
    L.append(f"{k[0]:3s} {k[1]:9s} {k[2]:22s} {a:5d} / {n:5d}")
L += ["", "first mismatch in g_entry enumeration order (window -> rank -> tensor -> logical block), per scope:"]
for k, v in sorted(first.items()):
    L.append(f"{k[0]} {k[1]:16s} {json.dumps(v)}")
for k in SCOPES:
    for var in ("V1", "V2"):
        if (var, k) not in first:
            L.append(f"{var} {k:16s} NO MISMATCH in scope")
L += ["", "provenance of valid-category blocks (same physical id as own replicate's warm at same logical index):"]
L += [f"  {k}: {v}" for k, v in sorted(prov.items())]
L += ["", "aliasing at point 0 (a non-null physical id in two groups' rows of the same request):"]
L += [f"  {k}: {v}" for k, v in sorted(alias.items())]
L += ["", "idxk cached_full blocks by warm pool coverage:"]
L += [f"  {k}: {v}" for k, v in sorted(unwritten_summary.items())]
L += ["", "blocks whose digest pair is (0, 0) (consistent with all-zero bytes, i.e. never written since allocation):"]
L += [f"  {k}: {v}" for k, v in sorted(zero.items()) if v]
L += ["", "earliest (lowest) layer with a difference in MLA cached_full or KDA recurrent cached_state, per window x rank:"]
L += [f"  {k}: {v}" for k, v in sorted(collections.Counter(f"{k[0]} L{v}" for k, v in earliest_layer.items()).items())]
L += ["", "null placeholders skipped (both repeats):"]
L += [f"  {k}: {v}" for k, v in sorted(nullblk.items())]
L += ["", "valid categories by layer (differ/compared):"]
for k, (a, n) in sorted(agg_layer.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
    if k[2] in ("cached_full", "kda_cached_state", "draft_cached_full"):
        L.append(f"  {k[0]} {k[1]:9s} {k[2]:18s} L{k[3]:<3d} {a:4d} / {n:4d}")
open(os.path.join(OUT, "recount_point0.out"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:60]))
