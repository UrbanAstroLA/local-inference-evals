#!/usr/bin/env python3
"""P4 launch-verification gates G1-G3 (PROPOSAL.md section 3) from the local-p4 bound-launch recorder.  Host
python3, stdlib only.  G4 is the separate single-GPU offline check (g4/).

usage: gates_p4.py --arm A|B|A2 --binds DIR [--ref-a DIR] [--p3b CAPLOG_JSONL] [--out JSON] [--stage startup|final]

Inputs: DIR holds bind_r<rank>_p<pid>.jsonl of one session (both ranks).  --ref-a: session A's bind directory
(required for B: G2, and for A2: the session control).  --p3b: captured-p3b's rank-0 capture log
(cap_p370.jsonl; required for A: G3).

Definitions (fixed before any session ran):
  record key   (rank, layer, plan, m) for TARGET layers (draft = false); plan is decode | prefill (the tiled
               variants decode_tile / prefill_tile are kept apart but judged by the same rules); the LAST record
               of a key is its signature; a key whose signature changed during the session is listed ("changed")
               and fails every gate it belongs to.
  tile         (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n) of the bound launch.
  coverage     every MoE layer 3..44 on both ranks has a decode record at every capture size <= 32
               (1, 2, 4, 8, 16, 24, 32) and a prefill record at 40, 48, 56 and 64 (the warm-up / capture binds).
  G1 (arm B)   every decode record with 10 <= m <= 32 binds K64/N256 for fc1 and fc2, both ranks, all layers;
               and the coverage holds (so "every" is not vacuous).
  G2 (arm B)   every other record equals arm A's record of the same key in every launch field, plan and capacity
               (tile, moe_block_size, fc2_moe_block_size, size_m, max_m_blocks, ...).  Keys absent from A are
               judged against A's rule for their plan: decode m <= 9 and every prefill record bind K64/N256 with A's
               block size for that plan; listed.  weight_tile_config is reported (it differs by design).
  G3 (arm A)   rank-0 records of layers 3, 23, 44 equal captured-p3b's recorded binds field by field (keys present
               in both; a p3b prefill key absent from A is compared with A's prefill signature of that layer when
               all of A's prefill signatures of the layer agree); the capture-size keys must all be present; and
               A follows the stock rule on every layer and rank (decode 10-32 -> K128/N128, else K64/N256).
  A2 control   every key common to A2 and A has an identical signature; A2 follows the stock rule.
"""
import argparse
import glob
import json
import os
import sys

FIELDS = ("size_m", "fc1_tile_k", "fc1_tile_n", "fc2_tile_k", "fc2_tile_n", "moe_block_size", "fc2_moe_block_size",
          "max_m_blocks", "blocks_per_sm", "route_ids_dtype", "direct_topk_routes")
TILE = ("fc1_tile_k", "fc1_tile_n", "fc2_tile_k", "fc2_tile_n")
K64 = (64, 256, 64, 256)
K128 = (128, 128, 128, 128)
LAYERS = tuple(range(3, 45))
DEC_SIZES = (1, 2, 4, 8, 16, 24, 32)
PRE_SIZES = (40, 48, 56, 64)


def load_binds(d):
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "bind_r*_p*.jsonl"))):
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("kind") == "bind":
                    recs.append(r)
    sig, changed = {}, set()
    for r in recs:
        if r.get("draft"):
            continue
        key = (int(r["rank"]), int(r["layer"]), r["plan"], int(r["m"]))
        if r.get("changed"):
            changed.add(key)
        sig[key] = r
    return sig, changed, recs


def tile(r):
    return tuple(r["launch"].get(f) for f in TILE)


def launch(r):
    return {f: r["launch"].get(f) for f in FIELDS}


def kind(plan):
    return "decode" if plan.startswith("decode") else "prefill"


def stock_rule(sig):
    bad = []
    for k, r in sig.items():
        exp = K128 if (kind(k[2]) == "decode" and 10 <= k[3] <= 32) else K64
        if tile(r) != exp:
            bad.append(dict(key=list(k), tile=list(tile(r)), expected=list(exp)))
    return bad


def coverage(sig):
    miss = []
    for rank in (0, 1):
        for L in LAYERS:
            for m in DEC_SIZES:
                if (rank, L, "decode", m) not in sig:
                    miss.append([rank, L, "decode", m])
            for m in PRE_SIZES:
                if (rank, L, "prefill", m) not in sig:
                    miss.append([rank, L, "prefill", m])
    return miss


def prefill_canon(sig, rank, L):
    s = {json.dumps(launch(r), sort_keys=True) for k, r in sig.items() if k[:2] == (rank, L) and kind(k[2]) == "prefill"}
    return json.loads(next(iter(s))) if len(s) == 1 else None


def g1(sig, changed):
    keys = [k for k in sig if kind(k[2]) == "decode" and 10 <= k[3] <= 32]
    bad = [dict(key=list(k), tile=list(tile(sig[k]))) for k in keys if tile(sig[k]) != K64]
    ch = [list(k) for k in keys if k in changed]
    cov = coverage(sig)
    per = {(k[0], k[1]) for k in keys}
    ok = not bad and not ch and not cov and len(per) == 2 * len(LAYERS)
    return dict(ok=ok, n_keys=len(keys), rank_layers=len(per), bad=bad[:20], n_bad=len(bad), changed=ch[:20],
                coverage_missing=cov[:20], n_coverage_missing=len(cov))


def g2(sigB, chB, sigA):
    others = [k for k in sigB if not (kind(k[2]) == "decode" and 10 <= k[3] <= 32)]
    mism, absent, rule_bad = [], [], []
    for k in others:
        if k in sigA:
            if launch(sigB[k]) != launch(sigA[k]) or sigB[k].get("capacity") != sigA[k].get("capacity"):
                mism.append(dict(key=list(k), B=launch(sigB[k]), A=launch(sigA[k])))
        else:
            absent.append(list(k))
            canon = prefill_canon(sigA, k[0], k[1]) if kind(k[2]) == "prefill" else None
            if kind(k[2]) == "prefill":
                ok = canon is not None and {f: v for f, v in launch(sigB[k]).items()} == canon
            else:
                ref = sigA.get((k[0], k[1], "decode", 8))
                ok = (tile(sigB[k]) == K64 and ref is not None
                      and sigB[k]["launch"].get("moe_block_size") == ref["launch"].get("moe_block_size"))
            if not ok:
                rule_bad.append(list(k))
    ch = [list(k) for k in others if k in chB]
    wt = sorted({(sigB[k].get("weight_tile_config"), sigA[k].get("weight_tile_config")) for k in others if k in sigA})
    ok = not mism and not rule_bad and not ch and bool(others)
    return dict(ok=ok, n_keys=len(others), n_common=len(others) - len(absent), mismatches=mism[:20],
                n_mismatches=len(mism), absent_from_A=absent[:20], n_absent_from_A=len(absent),
                absent_rule_failures=rule_bad[:20], changed=ch[:20], weight_tile_config_pairs=[list(x) for x in wt])


def load_p3b(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "site" and isinstance(r.get("key"), list) and r["key"][0] == "moe_bind":
            out[(0, int(r["layer"]), r["plan"], int(r["m"]))] = r
    return out


def g3(sigA, chA, p3b):
    mism, compared, notcomp = [], 0, []
    for k, r in p3b.items():
        a = sigA.get(k)
        if a is None and kind(k[2]) == "prefill":
            canon = prefill_canon(sigA, 0, k[1])
            if canon is not None:
                compared += 1
                if canon != {f: r["launch"].get(f) for f in FIELDS}:
                    mism.append(dict(key=list(k), via="prefill canon"))
                continue
        if a is None:
            notcomp.append(list(k))
            continue
        compared += 1
        if launch(a) != {f: r["launch"].get(f) for f in FIELDS} or str(r.get("tile_config")) != a.get("weight_tile_config"):
            mism.append(dict(key=list(k), A=launch(a), p3b={f: r["launch"].get(f) for f in FIELDS},
                             tile_config=(a.get("weight_tile_config"), r.get("tile_config"))))
    need = [(0, L, "decode", m) for L in (3, 23, 44) for m in DEC_SIZES] + \
           [(0, L, "prefill", m) for L in (3, 23, 44) for m in PRE_SIZES]
    missing = [list(k) for k in need if k not in sigA or k not in p3b]
    rule = stock_rule(sigA)
    cov = coverage(sigA)
    ch = [list(k) for k in sigA if k in chA]
    ok = not mism and not missing and not rule and not cov and not ch and compared > 0
    return dict(ok=ok, p3b_keys=len(p3b), compared=compared, mismatches=mism[:20], n_mismatches=len(mism),
                not_comparable=notcomp[:20], capture_size_keys_missing=missing, stock_rule_violations=rule[:20],
                n_stock_rule_violations=len(rule), coverage_missing=cov[:20], n_coverage_missing=len(cov), changed=ch)


def control(sigX, chX, sigA):
    common = [k for k in sigX if k in sigA]
    mism = [list(k) for k in common if launch(sigX[k]) != launch(sigA[k])
            or sigX[k].get("weight_tile_config") != sigA[k].get("weight_tile_config")]
    rule = stock_rule(sigX)
    cov = coverage(sigX)
    ok = not mism and not rule and not cov and not chX and bool(common)
    return dict(ok=ok, n_common=len(common), mismatches=mism[:20], n_mismatches=len(mism), stock_rule_violations=rule[:20],
                coverage_missing=cov[:20], changed=[list(k) for k in chX][:20])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("A", "B", "A2"))
    ap.add_argument("--binds", required=True)
    ap.add_argument("--ref-a")
    ap.add_argument("--p3b")
    ap.add_argument("--out")
    ap.add_argument("--stage", default="final", choices=("startup", "final"))
    a = ap.parse_args()
    sig, ch, recs = load_binds(a.binds)
    doc = dict(arm=a.arm, stage=a.stage, n_records=len(recs), n_target_keys=len(sig),
               ranks=sorted({k[0] for k in sig}), layers=len({k[1] for k in sig}),
               draft_records=sum(1 for r in recs if r.get("draft")), gates={})
    if a.arm == "A":
        if not a.p3b:
            doc["gates"]["G3"] = dict(ok=False, error="--p3b required")
        else:
            doc["gates"]["G3"] = g3(sig, ch, load_p3b(a.p3b))
    elif a.arm == "B":
        doc["gates"]["G1"] = g1(sig, ch)
        if not a.ref_a:
            doc["gates"]["G2"] = dict(ok=False, error="--ref-a (session A binds) required")
        else:
            sigA, _, _ = load_binds(a.ref_a)
            doc["gates"]["G2"] = g2(sig, ch, sigA)
    else:
        if not a.ref_a:
            doc["gates"]["A2-control"] = dict(ok=False, error="--ref-a required")
        else:
            sigA, _, _ = load_binds(a.ref_a)
            doc["gates"]["A2-control"] = control(sig, ch, sigA)
    doc["ok"] = bool(doc["gates"]) and all(g.get("ok") for g in doc["gates"].values())
    txt = json.dumps(doc, indent=1, default=str)
    if a.out:
        open(a.out, "w").write(txt)
    print(("GATES PASS " if doc["ok"] else "GATES FAIL ") + a.arm + " " + a.stage + " " +
          json.dumps({k: v.get("ok") for k, v in doc["gates"].items()}))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
