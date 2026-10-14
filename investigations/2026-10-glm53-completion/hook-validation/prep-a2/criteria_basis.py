#!/usr/bin/env python3
"""Measured basis for ATTEMPT2-CRITERIA-DRAFT.md (CPU only; attempt-1 data read-only). POST-INSPECTION, labelled.

Run in a CPU container of the B1 image (torch for the snapshot pages):
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v HV/run-a1:/run:ro -v HV/diagnosis-a1/out:/dout:ro \
      -v HV/prep-a2:/p --entrypoint python3 <B1 image> /p/criteria_basis.py
Writes /p/out/criteria_basis.json.

Every number comes from attempt 1's own records (run-a1) or from the CPU diagnosis outputs derived from them
(diagnosis-a1/out). Nothing here is used by the frozen analysis; it justifies (or fails to justify) each proposed
revision, including evaluating each proposed rule on the attempt-1 data.

Sections:
  A  index-K content: B1 / solo / V3 decode-vs-prefill gross fractions against the measured nulls, with the
     proposed and the literal forms of the content rule evaluated on attempt 1.
  B  index-K structure: page concentration (pages with >= MINP compared pools), layers 3-15, end-of-window pools,
     byte stability over time; and a PLANTED page-misaddressing control (each decode pool compared with the rebuild
     pool in the same slot of the adjacent 64-pool page) showing the structural rules fire on misaddressing.
  C  KDA: slot identification, layers 0-2 tolerance and margin, per-depth table.
  D  H2 argmax shift control per request.  E  H1f kept mass.
"""
import collections
import glob
import json
import os
import re

import numpy as np
import torch

RUN, DOUT, OUT = "/run", "/dout", "/p/out"
os.makedirs(OUT, exist_ok=True)
GROSS = 0.25
MINP = 16                      # pages need at least this many compared pools before a fraction is judged
R = {}


def frac(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x > GROSS)) if x.size else float("nan")


def jl(p):
    return [json.loads(ln) for ln in open(p)]


# ---------------------------------------------------------------- A/B from the per-pool table (V4)
pools = jl(os.path.join(DOUT, "h5_pools.jsonl"))       # arm, tag, tid, rank, layer, pool, P, W, last, rel, ...


def struct(rows, key_page=lambda r: (r["tag"], r["rank"], r["layer"], r["pool"] // 64)):
    pg = collections.defaultdict(list)
    for r in rows:
        pg[key_page(r)].append(r["rel"])
    judged = {k: v for k, v in pg.items() if len(v) >= MINP}
    fr = sorted((frac(v) for v in judged.values()), reverse=True)
    early = [r for r in rows if r["layer"] <= 15]
    return dict(n=len(rows), gross=frac([r["rel"] for r in rows]), pages_judged=len(judged),
                page_max_gross=fr[0] if fr else None, pages_over_25pct=sum(1 for f in fr if f > 0.25),
                page_top5=fr[:5], layers3_15_n=len(early), layers3_15_gross=int(sum(r["rel"] > GROSS for r in early)))


A = {}
for arm in ("b1", "d2solo", "d2"):
    rows = [r for r in pools if r["arm"] == arm]
    endp = [r for r in rows if r["last"] >= r["P"] + r["W"] - 1]
    inner = [r for r in rows if r["last"] <= r["P"] + r["W"] - 2]
    A[arm] = dict(all=struct(rows), end_of_window=dict(n=len(endp), gross=int(sum(r["rel"] > GROSS for r in endp))),
                  window_excluding_end=struct(inner),
                  per_trace={t: frac([r["rel"] for r in inner if r["tid"] == t]) for t in sorted({r["tid"] for r in inner})})
R["V4_decode_vs_rebuild"] = A

# V3 decode vs prefill (H4c) and the prefill-vs-prefill / decode-over-time nulls
h4c = json.load(open(os.path.join(DOUT, "h4c_idx.json")))
R["V3_decode_vs_prefill"] = struct(h4c)
nl = json.load(open(os.path.join(DOUT, "h4c_nulls.json")))
Pn = nl["P"]
R["NULL_prefill_vs_prefill"] = struct(Pn, key_page=lambda r: (r["k1"], r["k2"], r["rank"], r["layer"], r["pool"] // 64))
R["NULL_prefill_vs_prefill"]["per_trace"] = {g: frac([r["rel"] for r in Pn if r["g"] == g])
                                             for g in sorted({r["g"] for r in Pn})}
Dn = nl["D"]
R["NULL_decode_over_time"] = dict(n=len(Dn), byte_identical=float(np.mean([r["same"] for r in Dn])) if Dn else None,
                                  gross=frac([r["rel"] for r in Dn]))
dd = [r for r in json.load(open(os.path.join(DOUT, "h5_decode_null.json"))) if r["cmp"] == "b1_vs_solo"]
R["NULL_decode_b8_vs_b1_cbac17"] = dict(n=len(dd), gross=frac([r["rel"] for r in dd]),
                                         layers3_15_gross=int(sum(r["rel"] > GROSS for r in dd if r["layer"] <= 15)))
# matched-trace view for the decode-path null (only cbac17c70af8 has both arms)
b1c = [r["rel"] for r in pools if r["arm"] == "b1" and r["tid"] == "cbac17c70af8" and r["last"] <= r["P"] + r["W"] - 2]
soc = [r["rel"] for r in pools if r["arm"] == "d2solo" and r["tid"] == "cbac17c70af8" and r["last"] <= r["P"] + r["W"] - 2]
R["matched_cbac17"] = dict(b1_vs_rebuild=frac(b1c), solo_vs_rebuild=frac(soc),
                           decode_b8_vs_b1=R["NULL_decode_b8_vs_b1_cbac17"]["gross"])

# ---------------------------------------------------------------- A: the content rule evaluated on attempt 1
G = R["V4_decode_vs_rebuild"]["b1"]["window_excluding_end"]["gross"]
Gs = R["V4_decode_vs_rebuild"]["d2solo"]["window_excluding_end"]["gross"]
Gd = R["V4_decode_vs_rebuild"]["d2"]["window_excluding_end"]["gross"]
Npp = R["NULL_prefill_vs_prefill"]["gross"]
Ndd = R["NULL_decode_b8_vs_b1_cbac17"]["gross"]
FLOOR_ABS = 0.001
lim = max(2 * max(Ndd, Npp), FLOOR_ABS)
R["content_rule_on_attempt1"] = dict(
    G_b1=G, G_solo=Gs, G_d2=Gd, N_pp=Npp, N_dd_cbac17_only=Ndd,
    literal_b1_le_2Ndd=dict(limit=2 * Ndd, b1_pass=G <= 2 * Ndd, solo_pass=Gs <= 2 * Ndd,
                            note="N_dd measured on one trace (cbac17), the least fragile one; not matched to B1's 8"),
    matched_cbac17_b1_le_2Ndd=R["matched_cbac17"]["b1_vs_rebuild"] <= 2 * Ndd,
    proposed_b1_le_max2null_floor=dict(limit=lim, b1_pass=G <= lim, solo_pass=Gs <= lim, b1_ratio_to_Npp=G / Npp,
                                       d2_over_30pct=Gd > 0.30, d2_over_limit_x=Gd / lim),
    V3_vs_same_pool_null=dict(G_v3=R["V3_decode_vs_prefill"]["gross"]))

# ---------------------------------------------------------------- B: planted page misaddressing (positive control)
SN = {}
for p in glob.glob(os.path.join(RUN, "forcewin", "snap_V4-*win_r*.pt")):
    m = re.search(r"snap_(.*)_F(\d+)win_r(\d+)\.pt$", os.path.basename(p))
    SN.setdefault(m.group(1), {})[int(m.group(3))] = p


def attn_blocks(rec):
    for g in rec["groups"]:
        if g["bpk"] > 1:
            ids, bpk = g["ids"].tolist(), g["bpk"]
            return lambda c, ids=ids, bpk=bpk: ids[c * bpk] // bpk if c * bpk < len(ids) else -1
    return lambda c: -1


def idx_pool(pages, blk_of, pool):
    blk = blk_of((4 * pool + 3) // 4352)
    t = pages.get(blk * 17 + (pool % 1088) // 64)
    if t is None:
        return None
    raw = t.reshape(-1).view(torch.uint8)
    rr = pool % 64
    return raw[rr * 128:(rr + 1) * 128].view(torch.float8_e4m3fn).float() * raw[8192 + rr * 4: 8192 + rr * 4 + 4].view(torch.float32)


planted, aligned = [], []
for tag in sorted(t for t in SN if t.startswith("V4-b1-")):
    tid = tag.rsplit("-", 1)[-1]
    reb = SN.get(f"V4-rebuild-{tid}")
    if not reb:
        continue
    for rank in sorted(SN[tag]):
        a = torch.load(SN[tag][rank], weights_only=False)
        b = torch.load(reb[rank], weights_only=False)
        fa, fb = attn_blocks(a), attn_blocks(b)
        lo, hi = a["range"]
        for name, La in a["layers"].items():
            if La.get("kind") != "paged" or La.get("shape", [0])[-1:] != [132]:
                continue
            Lb = b["layers"][name]
            lay = int(name.split(".")[-4]) if name.split(".")[-4].isdigit() else int(re.search(r"layers\.(\d+)", name).group(1))
            for pool in range((lo + 3) // 4, (hi - 1) // 4):
                x = idx_pool(La["pages"], fa, pool)
                if x is None:
                    continue
                y = idx_pool(Lb["pages"], fb, pool)
                if y is not None:
                    aligned.append(dict(tag=tag, rank=rank, layer=lay, pool=pool,
                                        rel=float((x - y).norm() / max(float(y.norm()), 1e-12))))
                for shift in (64, -64):          # the same slot of the adjacent page (a wrong page id)
                    z = idx_pool(Lb["pages"], fb, pool + shift) if (lo + 3) // 4 <= pool + shift < (hi - 1) // 4 else None
                    if z is not None:
                        planted.append(dict(tag=tag, rank=rank, layer=lay, pool=pool,
                                            rel=float((x - z).norm() / max(float(z.norm()), 1e-12))))
                        break
R["planted_page_misaddress"] = dict(aligned_recomputed=struct(aligned), planted=struct(planted))

# ---------------------------------------------------------------- C: KDA
K = json.load(open(os.path.join(DOUT, "h4b_kda.json")))
ident = sum(1 for k in K if int(np.argmin(k["dist"])) == k["na"] - 1)
early = [k for k in K if k["layer"] <= 2]
by = collections.defaultdict(list)
for k in K:
    by[k["layer"]].append(k)
R["kda"] = dict(n=len(K), identified=ident, early_n=len(early), early_rel_max=max(k["rel"] for k in early),
                early_rel_median=float(np.median([k["rel"] for k in early])),
                early_margin_min=min(k["margin"] for k in early),
                all_margin_min=min(k["margin"] for k in K), all_rel_max=max(k["rel"] for k in K),
                per_layer={L: dict(n=len(v), rel_q50=float(np.median([k["rel"] for k in v])),
                                   rel_max=max(k["rel"] for k in v),
                                   margin_min=min(k["margin"] for k in v)) for L, v in sorted(by.items())})

# ---------------------------------------------------------------- D/E: H2 argmax control, H1f
h2 = json.load(open(os.path.join(DOUT, "h2_corrected.json")))
R["h2"] = dict(aligned_top1=h2["top1"], per_request_aligned_min=h2["per_request_top1_min"],
               per_request_shifted_max=h2["per_request_shifted_top1_max"], shift=h2["shift"],
               aligned_quantiles=h2["aligned"], null_quantiles=h2["null"], null_n=h2["null_n"], n=h2["n"],
               H2a_corrected_pass=h2["H2a_corrected_pass"], H2c_argmax_failing=h2["H2c_argmax_rule_failing"],
               H2c_lp_failing=h2["H2c_lp_rule_failing"])
h1f = json.load(open(os.path.join(DOUT, "h1f_rows.json")))
R["h1f"] = dict(cells=len(h1f), rows=len({(v["tag"], v["o"]) for v in h1f}),
                kept_not_eligible=sum(1 for v in h1f if v["kind"] == "kept_not_eligible"),
                masked_but_eligible=sum(1 for v in h1f if v["kind"] == "masked_but_eligible"),
                max_p_post=max(v["p_post"] for v in h1f), max_p_pre=max(v["p_pre"] for v in h1f),
                min_top1_p=min(v["top1_p"] for v in h1f))
json.dump(R, open(os.path.join(OUT, "criteria_basis.json"), "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in ("content_rule_on_attempt1", "planted_page_misaddress")}, indent=1, default=float)[:6000])
print(json.dumps({k: R[k] for k in ("kda", "h2", "h1f")}, indent=1, default=float)[:3000])
