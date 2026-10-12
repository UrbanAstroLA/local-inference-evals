#!/usr/bin/env python3
"""KV-writer / query-projection comparison -- CPU ANALYSIS (see KVWRITER-PREP.md). No GPU.

Modes
  selftest INPUTS            emulator self-test on the synthetic set only (no GPU outputs needed)
  analyze  INPUTS RUN.pt...  compare GPU outputs written by kvw_gpu_compare.py (one file per image)

Writer emulators (from source + sm_120 SASS):
  vLLM fp8_ds_mla (0.7.0 writer; vLLM _C at 487ecf187, identical .so in the 0.7.0 and B0 images):
      scale = max(fl(amax / 448), FLT_MIN)        IEEE division (FCHK + div_rn slow path in SASS)
      byte  = e4m3_rn_satfinite(fl(v / scale))    IEEE division, F2FP.SATFINITE.E4M3
    -> fully determined; the GPU bytes must equal the emulation bitwise.
  B12x GLM_NEXT (B0 writer, kv_cache.py ConcatAndCacheGlmNextMlaKernel):
      scale = 1.0 if amax == 0 else fl(amax * fl(1/448))
      inv   = rcp.approx.ftz(scale)               (hardware approximation, <= 1 ulp)
      byte  = e4m3_rn_satfinite(fl(v * inv))
    -> bracketed with inv in {RN(1/scale) - 1 ulp, RN(1/scale), RN(1/scale) + 1 ulp}; the GPU byte must be
       one of the bracket outcomes.
"""
import json
import sys

import numpy as np
import torch

L, G = 512, 128
FLT_MIN = np.float32(1.1754943508222875e-38)
C448 = np.float32(1.0 / 448.0)


def e4m3_rn_sat(x32):
    """fp32 -> e4m3 byte, round-to-nearest-even, saturating to +-448 (torch's cast does not saturate)."""
    x = torch.from_numpy(np.clip(x32, -448.0, 448.0).astype(np.float32))
    return x.to(torch.float8_e4m3fn).view(torch.uint8).numpy()


def e4m3_decode(b):
    return torch.from_numpy(b).view(torch.float8_e4m3fn).float().numpy()


def groups(kv):
    v = kv.float().numpy().reshape(-1, L // G, G).astype(np.float32)
    return v, np.abs(v).max(-1)


def vllm_writer(kv):
    v, amax = groups(kv)
    scale = np.maximum((amax / np.float32(448.0)).astype(np.float32), FLT_MIN)
    b = e4m3_rn_sat((v / scale[..., None]).astype(np.float32))
    return b.reshape(-1, L), scale.astype(np.float32)


def b12x_writer_bracket(kv):
    v, amax = groups(kv)
    scale = np.where(amax == 0, np.float32(1.0), (amax * C448).astype(np.float32)).astype(np.float32)
    inv = (1.0 / scale.astype(np.float64)).astype(np.float32)
    outs = []
    for d in (-1, 0, 1):
        i = (inv.view(np.int32) + d).view(np.float32)
        outs.append(e4m3_rn_sat((v * i[..., None]).astype(np.float32)).reshape(-1, L))
    return outs, scale


def dequant(b, scale):
    return (e4m3_decode(b.reshape(-1, L // G, G)) * scale[..., None]).reshape(-1, L)


def e4m3_step(t):
    """quantisation step of e4m3 at |t| (scaled domain): 2^(e-3) for normals, 2^-9 subnormal."""
    a = np.maximum(np.abs(t), 2.0 ** -6)
    return 2.0 ** (np.floor(np.log2(a)) - 3)


def writer_stats(kv, b, scale):
    v, _ = groups(kv)
    deq = dequant(b, scale).reshape(-1, L // G, G).astype(np.float64)
    t = v.astype(np.float64) / scale[..., None].astype(np.float64)
    err_steps = np.abs(deq - v) / (scale[..., None] * e4m3_step(t))
    return {"max_err_in_e4m3_steps": float(err_steps.max()),
            "frac_err_gt_half_step": float((err_steps > 0.5 + 1e-6).mean()),
            "rms_rel_err_vs_group_amax": float(np.sqrt(((deq - v) ** 2).mean()) / max(np.abs(v).max(), 1e-30))}


def code_distance(a, b):
    """distance in E4M3 codes along the real line; +0 and -0 are the same point."""
    def ordinal(c):
        c = c.astype(np.int64)
        mag = c & 0x7F
        return np.where(c & 0x80, -mag, mag)
    return np.abs(ordinal(a) - ordinal(b))


def cross_writer(bv, sv, bb_set, sb):
    same = bv == bb_set[1]
    ambiguous = ~((bb_set[0] == bb_set[1]) & (bb_set[1] == bb_set[2]))
    dist = code_distance(bv, bb_set[1])
    diff = dist > 0
    nz = sv > FLT_MIN
    return {"elements": int(bv.size), "bytes_differ_raw": int((~same).sum()), "bytes_differ": int(diff.sum()),
            "bytes_differ_frac": float(diff.mean()), "bytes_differ_more_than_one_code": int((dist > 1).sum()),
            "max_code_distance": int(dist.max()) if dist.size else 0,
            "b12x_ambiguous_elements": int(ambiguous.sum()),
            # all-zero groups excluded: vLLM stores FLT_MIN, B12x 1.0 (both decode to exact zeros)
            "scale_ulp_diff_max_nonzero_groups": int(np.abs(sv.view(np.int32).astype(np.int64)
                                                            - sb.view(np.int32).astype(np.int64))[nz].max()) if nz.any() else 0,
            "zero_groups": int((~nz).sum())}


def selftest(inputs):
    d = torch.load(inputs, map_location="cpu")
    kv = d["synth"]["kv_c"]
    bv, sv = vllm_writer(kv)
    bb, sb = b12x_writer_bracket(kv)
    res = {"vllm_writer": writer_stats(kv, bv, sv), "b12x_writer_rn": writer_stats(kv, bb[1], sb),
           "cross_writer_emulated": cross_writer(bv, sv, bb, sb),
           "note": "emulator self-test on synthetic inputs only; not the comparison"}
    print(json.dumps(res, indent=1))
    ok = res["vllm_writer"]["frac_err_gt_half_step"] == 0 and res["b12x_writer_rn"]["max_err_in_e4m3_steps"] < 0.5 + 1e-3
    return 0 if ok else 1


def analyze(inputs, runs):
    """Frozen comparisons (KVWRITER-PREP.md, 'Metrics and decision rule')."""
    d = torch.load(inputs, map_location="cpu")
    report = {}
    for rp in runs:
        r = torch.load(rp, map_location="cpu")
        tag = r["meta"]["image_tag"]
        rep = report.setdefault(tag, {"meta": r["meta"], "R_repeatable_and_slot_independence": r.get("repeatable")})
        for name, w in r["writers"].items():                     # name: "<set>/<layer>"
            kv = d["synth"]["kv_c"] if name.startswith("synth") else d["proxy"][int(name.split("/")[1])]["kv_c"]
            bv, sv = vllm_writer(kv)
            bb, sb = b12x_writer_bracket(kv)
            out = {}
            if "nsa" in w:
                rec = w["nsa"].numpy().reshape(-1, 656)
                gb, gs = rec[:, :L], rec[:, L:528].copy().view(np.float32)
                out["W1_vllm_gpu_eq_emulation"] = bool((gb == bv).all() and (gs.reshape(sv.shape) == sv).all())
                out["W1_vllm_bytes_mismatch"] = int((gb != bv).sum())
                out["nsa_rope_tail_zero"] = bool((rec[:, 528:] == 0).all())
                out["W4_vllm"] = writer_stats(kv, gb, gs.reshape(sv.shape))
            if "next" in w:
                rec = w["next"].numpy().reshape(-1, 528)
                gb, gs = rec[:, :L], rec[:, L:528].copy().view(np.float32).reshape(sb.shape)
                inb = (gb == bb[0]) | (gb == bb[1]) | (gb == bb[2])
                out["W2_b12x_bytes_outside_bracket"] = int((~inb).sum())
                out["W2_b12x_bracket_ambiguous_elements"] = int((~((bb[0] == bb[1]) & (bb[1] == bb[2]))).sum())
                out["W2_b12x_scale_eq_emulation"] = bool((gs == sb).all())
                out["W4_b12x"] = writer_stats(kv, gb, gs)
                if "nsa" in w:
                    nb = w["nsa"].numpy().reshape(-1, 656)[:, :L]
                    ns = w["nsa"].numpy().reshape(-1, 656)[:, L:528].copy().view(np.float32).reshape(sb.shape)
                    out["W3_cross_writer_gpu"] = cross_writer(nb, ns, [gb, gb, gb], gs)
                    # ERRATUM-3 (recorded before GPU data): ties.  An element is a tie when the vLLM pre-rounding
                    # value fl(v/scale) sits exactly on an E4M3 midpoint (frequent: bf16 v and amax have 8-bit
                    # significands); vLLM breaks it by RNE, B12x by the rcp.approx perturbation.
                    v, _ = groups(kv)
                    t = (v / sv[..., None]).astype(np.float32).astype(np.float64).reshape(-1, L)
                    st = e4m3_step(t)
                    tie = (np.abs(t) <= 448.0) & (np.mod(np.abs(t) / st, 1.0) == 0.5)
                    gd = code_distance(nb, gb) > 0
                    possible = (bv != bb[0]) | (bv != bb[1]) | (bv != bb[2])
                    out["W3e"] = {"elements": int(gd.size), "gpu_differ": int(gd.sum()), "tie_elements": int(tie.sum()),
                                  "gpu_differ_at_ties": int((gd & tie).sum()), "gpu_differ_not_tie": int((gd & ~tie).sum()),
                                  "rate_not_tie": float((gd & ~tie).mean()),
                                  "emulated_possible_differ": int(possible.sum()),
                                  "gpu_differ_outside_emulated_possible": int((gd & ~possible).sum())}
            rep[name] = out
        for name, p in r.get("projections", {}).items():            # name: "<layer>/rows<B>"
            layer = int(name.split("/")[0])
            q = d["proxy"][layer]["q_nope"][:, : p["rows"]].double()
            W = d["proxy"][layer]["W_UK_T"].double()
            ref = torch.bmm(q, W).transpose(0, 1)                     # (B, N, L) fp64
            bound = 2.0 ** -9 * ref.abs() + 256 * 2.0 ** -24 * torch.bmm(q.abs(), W.abs()).transpose(0, 1)  # frozen as written
            o = {k: p[k] for k in ("bmm_repeatable", "h64_repeatable", "h64_error") if k in p}
            # ERRATUM-1 (recorded before GPU data): the frozen bound's bf16 term 2^-9*|ref| understates half a bf16
            # ulp (up to 2^-8*|ref|); both are reported, frozen-as-written and corrected (exact half ulp of ref).
            half_ulp = torch.from_numpy(np.spacing(np.abs(ref.numpy()).astype(np.float32)).astype(np.float64)) * 2.0 ** 15
            acc = 256 * 2.0 ** -24 * torch.bmm(q.abs(), W.abs()).transpose(0, 1)
            bound_c = half_ulp + acc
            for k in ("bmm", "h64"):
                if k in p:
                    t = p[k].double()[..., :L]
                    o[f"P2_{k}_outside_bound"] = int(((t - ref).abs() > bound).sum())
                    o[f"P2_{k}_max_err_over_bound"] = float(((t - ref).abs() / bound.clamp_min(1e-30)).max())
                    o[f"P2c_{k}_outside_corrected_bound"] = int(((t - ref).abs() > bound_c).sum())
                    o[f"P2c_{k}_max_err_over_corrected_bound"] = float(((t - ref).abs() / bound_c.clamp_min(1e-30)).max())
            if "bmm" in p and "h64" in p:
                o["P1_h64_eq_bmm_bitwise"] = bool(torch.equal(p["h64"][..., :L], p["bmm"]))
                o["P1_elements_differ"] = int((p["h64"][..., :L] != p["bmm"]).sum())
                o["h64_tail_zero"] = bool((p["h64"][..., L:] == 0).all())
            rep["proj/" + name] = o
    report["verdict"] = verdict(report)
    print(json.dumps(report, indent=1, default=str))
    return report


# frozen thresholds (KVWRITER-PREP.md) and the ERRATUM-2 W4 tolerance derived from the frozen delta (7*2^-24)
W3_RATE_MAX = 1.3e-4
W4_TOL_FROZEN = 0.5 + 1e-6
W4_TOL_ERRATUM = 0.5 + 16 * 7 * 2.0 ** -24


def verdict(report):
    fails = {"frozen_as_written": [], "with_errata": []}
    notes = []
    for tag, rep in report.items():
        if tag == "verdict":
            continue
        rr = rep.get("R_repeatable_and_slot_independence") or {}
        for name, flags in rr.items():
            for k, v in flags.items():
                if v is not True:
                    for f in fails.values():
                        f.append(f"{tag} {name} R {k}")
        for name, o in rep.items():
            if not isinstance(o, dict) or name in ("meta", "R_repeatable_and_slot_independence"):
                continue
            def both(msg):
                for f in fails.values():
                    f.append(f"{tag} {name} {msg}")
            if o.get("W1_vllm_gpu_eq_emulation") is False:
                both("W1")
            if o.get("nsa_rope_tail_zero") is False:
                both("W3 rope tail")
            if o.get("W2_b12x_bytes_outside_bracket", 0) > 0 or o.get("W2_b12x_scale_eq_emulation") is False:
                both("W2")
            w3 = o.get("W3_cross_writer_gpu")
            if w3 and (w3["bytes_differ_more_than_one_code"] > 0 or w3["scale_ulp_diff_max_nonzero_groups"] > 1):
                both("W3")
            if w3 and w3["bytes_differ_frac"] > W3_RATE_MAX:
                fails["frozen_as_written"].append(f"{tag} {name} W3 rate {w3['bytes_differ_frac']:.2e}")
            w3e = o.get("W3e")
            if w3e and (w3e["gpu_differ_outside_emulated_possible"] > 0 or w3e["rate_not_tie"] > W3_RATE_MAX):
                fails["with_errata"].append(f"{tag} {name} W3e {w3e}")
            for k in ("W4_vllm", "W4_b12x"):
                if k in o:
                    m = o[k]["max_err_in_e4m3_steps"]
                    if m > W4_TOL_FROZEN:
                        fails["frozen_as_written"].append(f"{tag} {name} {k} max {m:.7f}")
                    if m > W4_TOL_ERRATUM:
                        fails["with_errata"].append(f"{tag} {name} {k} max {m:.7f}")
            if "W4_vllm" in o and "W4_b12x" in o:
                a, b = o["W4_vllm"]["rms_rel_err_vs_group_amax"], o["W4_b12x"]["rms_rel_err_vs_group_amax"]
                if abs(a - b) > 0.01 * max(a, b):
                    both("W4 rms not within 1%")
            for k in ("bmm", "h64"):
                if o.get(f"P2_{k}_outside_bound", 0) > 0:
                    fails["frozen_as_written"].append(f"{tag} {name} P2 {k} {o[f'P2_{k}_outside_bound']}")
                if o.get(f"P2c_{k}_outside_corrected_bound", 0) > 0:
                    fails["with_errata"].append(f"{tag} {name} P2c {k} {o[f'P2c_{k}_outside_corrected_bound']}")
                if o.get(f"{k}_repeatable") is False:
                    both(f"R {k} not repeatable")
            if "h64_error" in o:
                notes.append(f"{tag} {name} h64 error: {o['h64_error']}")
            if o.get("P1_h64_eq_bmm_bitwise") is False:
                notes.append(f"{tag} {name} P1: H64 != bmm in {o['P1_elements_differ']} elements (rounding-level unless P2 fails)")
            if o.get("h64_tail_zero") is False:
                both("P1 tail lanes not zero")
    return {"beyond_expected_rounding_frozen_as_written": bool(fails["frozen_as_written"]),
            "beyond_expected_rounding_with_errata": bool(fails["with_errata"]), "failures": fails, "notes": notes,
            "thresholds": {"W3_rate_max": W3_RATE_MAX, "W4_tol_frozen": W4_TOL_FROZEN, "W4_tol_erratum": W4_TOL_ERRATUM}}


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        sys.exit(selftest(sys.argv[2]))
    rep = analyze(sys.argv[2], sys.argv[3:])
    json.dump(rep, open("kvw_analysis.json", "w"), indent=1, default=str)
