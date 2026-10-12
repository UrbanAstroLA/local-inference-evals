#!/usr/bin/env python3
"""D3 activation readout for the bundle gate (runs inside the bundle check image, CPU only, --network none).
usage: d3_logits_check.py RUN_DIR  -> JSON on stdout

Reads the verify13 capture of session s1: gate_info lines (dispatch flags per layer and rank) and the sampled GateLinear
outputs. NOTE on labels: verify13 stores the router output under the key `logits_shipped` with the release label
("shipped: bf16-result GEMM cast to fp32"). In the bundle the same key holds whatever GateLinear served; this script
re-labels it per record as `served (tier from gate_info)`, and never calls it fp32 or shipped without the dispatch evidence.

Per record: a value is bf16-representable when its low 16 bits are zero (fp32 bit pattern). A row is "bf16-valued"
when all 288 values are. Tier 6 (release, D3 absent) gives 100% bf16-valued rows; tier 5 (D3, fp32-output GEMM) gives
essentially none (a random fp32 value has zero low 16 bits with probability 2^-16).
"""
import glob
import json
import os
import sys

import torch

run = sys.argv[1]
ev = []
for f in glob.glob(os.path.join(run, "verify13", "v13_p*.jsonl")):
    for line in open(f):
        try:
            ev.append(json.loads(line))
        except ValueError:
            pass
gi = [d for d in ev if d.get("kind") == "gate_info"]
tiers = {}
for d in gi:
    tiers.setdefault(str(d.get("rank")), {})[d.get("layer")] = d.get("tier_from_flags")
out = {"gate_info_lines": len(gi),
       "tiers_by_rank": {r: sorted(set(v.values())) for r, v in tiers.items()},
       "layers_by_rank": {r: len(v) for r, v in tiers.items()},
       "device_capability": sorted({str(d.get("device_capability")) for d in gi}),
       "weight_dtype": sorted({str(d.get("weight_dtype")) for d in gi}),
       "records": 0, "rows": 0, "rows_bf16_valued": 0, "values": 0, "values_bf16_representable": 0,
       "by_batch_kind": {}, "label_note": "key logits_shipped re-labelled: served GateLinear output (tier per gate_info)"}
for f in sorted(glob.glob(os.path.join(run, "verify13", "rec_r*", "*.pt"))):
    r = torch.load(f, map_location="cpu")
    x = r.get("logits_shipped")
    if x is None:
        continue
    x = x.float().contiguous()
    low = (x.view(torch.int32) & 0xFFFF) == 0
    rows_bf16 = int(low.all(dim=1).sum())
    out["records"] += 1
    out["rows"] += int(x.shape[0])
    out["rows_bf16_valued"] += rows_bf16
    out["values"] += int(x.numel())
    out["values_bf16_representable"] += int(low.sum())
    bk = str(((r.get("meta") or {}).get("rows") or {}).get("batch_kind"))
    b = out["by_batch_kind"].setdefault(bk, {"rows": 0, "rows_bf16_valued": 0})
    b["rows"] += int(x.shape[0])
    b["rows_bf16_valued"] += rows_bf16
out["share_rows_bf16_valued"] = round(out["rows_bf16_valued"] / out["rows"], 6) if out["rows"] else None
out["share_values_bf16_representable"] = round(out["values_bf16_representable"] / out["values"], 6) if out["values"] else None
print(json.dumps(out, indent=1))
