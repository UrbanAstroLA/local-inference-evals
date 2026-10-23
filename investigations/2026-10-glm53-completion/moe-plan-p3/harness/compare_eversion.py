#!/usr/bin/env python3
"""E-version comparison (CPU only): the same slice and the same plan type
through 0.7.0's B12x (fe054789) and 0.9.1's B12x (7fcc094e).

Usage: compare_eversion.py V091.json V070.json OUT.json

Inputs are moe_plan_eq.py result files.  Each image ran its own slice on the
full panel and the other engine's slice on the E-version subset
(m in 8/32/48, x1, uniform, all concentrations, both seeds).  For every
fixture present in both files under the same slice:
  * the input digests must be identical (else the fixture is excluded);
  * decode(v091) is compared with decode(v070), prefill(v091) with
    prefill(v070), row by row, by output-row digest.
If both runs wrote <stem>.eversion.pt next to the JSON, magnitudes
(max abs, relative L2) are added; that part needs torch (run in an image,
CPU only).  Plan effects (decode vs prefill) and partition effects
(tp2r0 vs ep2r0) are never mixed into this comparison.
"""
from __future__ import annotations

import json
import os
import sys


def load(path):
    with open(path) as fh:
        return json.load(fh)


def fixtures_by_id(doc, slice_name):
    out = {}
    sd = doc.get("slices", {}).get(slice_name)
    if not sd:
        return out
    for layer, ld in sd.get("layers", {}).items():
        for f in ld.get("fixtures", []):
            if "error" in f or "digests" not in f:
                continue
            out[f["id"]] = f
    return out


def runtime_caps(doc, slice_name):
    rt = doc.get("slices", {}).get(slice_name, {}).get("runtimes", {}).get("default", {})
    out = {}
    for plan in ("decode_plan", "prefill_plan"):
        caps = (rt.get(plan) or {}).get("caps")
        if caps:
            out[plan] = {k: caps.get(k) for k in ("max_tokens", "num_topk", "route_num_experts",
                                                  "w4a16_block_size_m", "full_rotation_output_dtype",
                                                  "mixed_trellis_route_id_dtypes", "decode_config",
                                                  "swiglu_limit")}
    return out


def magnitudes(pt_a, pt_b, key_a, key_b, fid, plan, n_real):
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    a = pt_a.get(key_a, {}).get(fid, {}).get(plan)
    b = pt_b.get(key_b, {}).get(fid, {}).get(plan)
    if a is None or b is None:
        return None
    a = a[:n_real].float()
    b = b[:n_real].float()
    d = a - b
    ref = float(b.norm())
    return dict(max_abs=float(d.abs().max()),
                rel_l2=(float(d.norm()) / ref) if ref > 0 else (0.0 if float(d.norm()) == 0 else float("inf")))


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    p091, p070, outp = sys.argv[1:]
    d091, d070 = load(p091), load(p070)
    pts = {}
    for tag, path, doc in (("v091", p091, d091), ("v070", p070, d070)):
        ev = doc.get("eversion_file")
        full = os.path.join(os.path.dirname(path), ev) if ev else None
        if full and os.path.exists(full):
            try:
                import torch
                pts[tag] = torch.load(full, map_location="cpu")
            except Exception as exc:  # noqa: BLE001
                pts[tag] = None
                print(f"warning: cannot load {full}: {exc!r}")
    result = dict(schema="p3-eversion/1", effect="E-version (B12x fe054789 vs 7fcc094e; same slice, same plan type)",
                  inputs=dict(v091=os.path.basename(p091), v070=os.path.basename(p070)),
                  engines=dict(v091=dict(status=d091.get("status"), mode=d091.get("mode")),
                               v070=dict(status=d070.get("status"), mode=d070.get("mode"))),
                  slices={})
    for slice_name in ("tp2r0", "ep2r0"):
        a = fixtures_by_id(d091, slice_name)
        b = fixtures_by_id(d070, slice_name)
        common = sorted(set(a) & set(b))
        srec = dict(n_common=len(common), n_input_mismatch=0,
                    plan_args=dict(v091=runtime_caps(d091, slice_name), v070=runtime_caps(d070, slice_name)),
                    plans={})
        for plan in ("decode", "prefill"):
            rows = rows_eq = 0
            fixtures_differ = []
            mags = []
            for fid in common:
                fa, fb = a[fid], b[fid]
                if fa.get("input_digest") != fb.get("input_digest"):
                    if plan == "decode":
                        srec["n_input_mismatch"] += 1
                    continue
                n = int(fa.get("n_real", len(fa["digests"][plan])))
                da, db = fa["digests"][plan][:n], fb["digests"][plan][:n]
                eq = [x == y for x, y in zip(da, db)]
                rows += len(eq)
                rows_eq += sum(eq)
                if not all(eq):
                    entry = dict(id=fid, rows_differ=[i for i, e in enumerate(eq) if not e][:48])
                    if "v091" in pts and "v070" in pts and pts["v091"] and pts["v070"]:
                        layer = fid.split(".")[0][1:]
                        m = magnitudes(pts["v091"], pts["v070"], f"{slice_name}|{layer}",
                                       f"{slice_name}|{layer}", fid, plan, n)
                        if m:
                            entry.update(m)
                            mags.append(m)
                    fixtures_differ.append(entry)
            srec["plans"][plan] = dict(
                rows_compared=rows, row_bitequal_frac=(rows_eq / rows) if rows else None,
                n_fixtures_differ=len(fixtures_differ), fixtures_differ=fixtures_differ[:60],
                max_abs=max((m["max_abs"] for m in mags), default=None),
                max_rel_l2=max((m["rel_l2"] for m in mags), default=None))
        result["slices"][slice_name] = srec
    tmp = outp + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=1)
    os.replace(tmp, outp)
    summary = {s: {p: (r["plans"][p]["rows_compared"], r["plans"][p]["row_bitequal_frac"])
                   for p in r["plans"]} | {"n_common": r["n_common"], "n_input_mismatch": r["n_input_mismatch"]}
               for s, r in result["slices"].items()}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
