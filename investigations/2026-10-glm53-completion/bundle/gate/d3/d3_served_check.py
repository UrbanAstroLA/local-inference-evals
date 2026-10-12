#!/usr/bin/env python3
"""D3 numerical verification of the fp32-output router path AS SERVED by the bundle (PLAN.md amendment 14c item 2).

Input: the verify13 capture of gate session s1 (bundle d1234-check image): per sampled (eager forward, MoE layer) record the
gate input rows x [a, a+r) of the M-row batch, the served GateLinear output (stored under the legacy key `logits_shipped`;
here labelled SERVED), and the routing consumed by Exl3MoEMethod.apply. Reuses the stage-A helpers of
verify13/offline/d3_router_compare.py (sha256 0b4b20e4..., imported unchanged): aligned_padded, corrected, route_serving_shape,
classify_routing, tau, gap_k, route_ref64, sets_differ, Ckpt, engine_gate, engine_router.

Labels: SERVED = captured GateLinear output in the bundle server (dispatch per gate_info);
        CORRECTED = torch.mm(x, W.T, out_dtype=fp32) recomputed offline (the #54048 tier-5 expression);
        ENGINE = the bundle's own patched GateLinear recomputed offline on this GPU (secondary);
        REFERENCE = x.double() @ W.double().T.
Controls (fixed by the coordinator before this run; tolerances = stage A's, unchanged):
  A0  dispatch: in-server gate_info on every (rank, layer) = tier 5 (allow_cublas_router_gemm True, the five other
      allow_* flags False); weight / bias sha256 in-server = checkpoint; offline GateLinear on this GPU also tier 5.
  A'  CORRECTED at the serving M (rows in place at [a, a+r), zero padding, x at the in-server byte alignment)
      == SERVED BITWISE on every row of every record. (ENGINE == SERVED reported alongside.)
  D   every SERVED logit within tau_d = 2^-14*S + 2^-24*|ref| of REFERENCE; every top-8 disagreement between
      SERVED-based fp64 selection and REFERENCE selection lies inside the band 0.5*max_e tau_d,e; tau_worst reported.
  C   the engine router invoked at the serving M on SERVED reproduces the consumed ids (sets) and weights BITWISE.
No stop rule: every record is evaluated and reported. usage: d3_served_check.py RUN_DIR CKPT_DIR OUT_DIR
"""
import collections
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d3_router_compare as R  # noqa: E402  (verify13/offline copy, unchanged)

run, ckdir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
dev = os.environ.get("D3_DEVICE", "cuda")   # "cpu" only for the plumbing smoke test
R.stub_tp()
log = R.load_log(run)
rinfo = [d for d in log if d.get("kind") == "router_info" and d.get("cls", "").endswith("TopKRouter")]
rinfo0 = rinfo[0] if rinfo else {}
ginfo = {(d["rank"], d["layer"]): d for d in log if d.get("kind") == "gate_info"}
gsha = {(d["rank"], d["layer"]): d for d in log if d.get("kind") == "gate_weight_sha"}
ck = R.Ckpt(ckdir)
files = sorted(glob.glob(os.path.join(run, "**", "rec_r*", "*.pt"), recursive=True))
if os.environ.get("D3_MAXREC"):
    files = files[:int(os.environ["D3_MAXREC"])]   # plumbing smoke test only
by_layer = collections.defaultdict(list)
for f in files:
    by_layer[int(os.path.basename(f).split("_l")[1][:2])].append(f)
T = collections.Counter()
fails = collections.defaultdict(list)
err_rel, a0 = [], []
t0 = time.time()
for layer in sorted(by_layer):
    pfx = f"model.language_model.layers.{layer}.mlp.gate"
    W = ck.get(pfx + ".weight")
    b = ck.get(pfx + ".e_score_correction_bias")
    gate, flags = R.engine_gate(W, dev)
    router = R.engine_router(b, rinfo0, dev)
    Wd = gate.weight.data
    if not (flags["allow_cublas_router_gemm"] and not any(flags[k] for k in R.FLAG_KEYS if k != "allow_cublas_router_gemm")):
        fails["A0"].append(dict(layer=layer, rank="offline", flags=flags))
    for rank in (0, 1):
        gi, gs = ginfo.get((rank, layer)), gsha.get((rank, layer))
        bad = []
        if gi is None or not str(gi.get("tier_from_flags", "")).startswith("5"):
            bad.append(f"server tier {None if gi is None else gi.get('tier_from_flags')}")
        if gi is not None and (not gi.get("allow_cublas_router_gemm")
                               or any(gi.get(k) for k in R.FLAG_KEYS if k != "allow_cublas_router_gemm")):
            bad.append("server flags")
        if gs is None or gs.get("weight_sha256") != R.sha(W) or gs.get("bias_sha256") != R.sha(b.float()):
            bad.append("weight/bias sha256")
        if bad:
            fails["A0"].append(dict(layer=layer, rank=rank, why=bad))
        T["A0_layer_ranks_checked"] += 1
    for f in by_layer[layer]:
        rec = torch.load(f)
        meta = rec["meta"]
        m, a, r = meta["m"], meta["a"], meta["r"]
        x = rec["x"]
        served = rec["logits_shipped"].float()          # legacy key; SERVED by the patched GateLinear
        T["records"] += 1
        T["rows"] += r
        T["rows_served_bf16_valued"] += int(((served.contiguous().view(torch.int32) & 0xFFFF) == 0).all(dim=1).sum())
        with torch.no_grad():
            xs = R.aligned_padded(x, m, a, meta.get("x_ptr_mod256"), dev)
            corr = R.corrected(xs, Wd)[a:a + r].contiguous().cpu()
            eng = gate(xs)[0][a:a + r].contiguous().cpu()
        okA = bool(R.bitwise_equal_rows(corr, served).all())
        okE = bool(R.bitwise_equal_rows(eng, served).all())
        T["Aprime_records_bitwise"] += okA
        T["engine_records_bitwise"] += okE
        if not okA:
            fails["Aprime"].append(dict(file=os.path.basename(f), m=m, a=a, r=r,
                                        rows_not_exact=int((~R.bitwise_equal_rows(corr, served)).sum()),
                                        max_ulp=R.ulp_diff(corr, served)))
        # D: served vs fp64 reference
        ref = x.double() @ W.double().T
        t = R.tau(x, W, ref)
        tw = R.tau(x, W, ref, which="worst")
        e = (served.double() - ref).abs()
        sxw = x.double().abs() @ W.double().abs().T
        err_rel.append((e / sxw.clamp_min(1e-300)).max(dim=1).values)
        T["D_logits"] += int(e.numel())
        T["D_tau_d_violations"] += int((e > t).sum())
        T["D_tau_worst_violations"] += int((e > tw).sum())
        ids_s64, _ = R.route_ref64(served.double(), b)
        ids_r64, _ = R.route_ref64(ref, b)
        nd = R.sets_differ(ids_s64, ids_r64)
        g64 = R.gap_k(ref, b)
        band = 0.5 * t.max(dim=1).values
        T["D_top8_disagreements"] += int((nd > 0).sum())
        T["D_top8_disagreements_outside_band"] += int(((nd > 0) & (g64 > band)).sum())
        # C: routing of SERVED at the serving M == consumed
        if "topk_ids" in rec:
            w_e, i_e = R.route_serving_shape(router, served.to(dev), m, a, meta.get("logits_ptr_mod256"),
                                             rec["topk_ids"].dtype, dev)
            c = R.classify_routing(rec["topk_ids"], rec["topk_weights"], i_e.cpu(), w_e.cpu())
            okC = c["rows_set_differ"] == 0 and c["rows_same_set_weights_not_bitwise"] == 0
            T["C_records_exact"] += okC
            T["C_rows_exact"] += c["rows_exact"]
            if not okC:
                fails["C"].append(dict(file=os.path.basename(f), m=m, **{k: c[k] for k in ("rows_set_differ", "rows_same_set_weights_not_bitwise")}))
        else:
            fails["C"].append(dict(file=os.path.basename(f), missing_routing=True))
        T["batch_" + str((meta.get("rows") or {}).get("batch_kind"))] += r
er = torch.cat(err_rel) if err_rel else torch.zeros(0)
res = dict(
    labels=dict(served="captured GateLinear output in the bundle server (legacy key logits_shipped)",
                corrected=R.LABEL_CORRECTED, engine="bundle's patched GateLinear recomputed offline",
                reference=R.LABEL_REFERENCE),
    corrected_impl=R.CORRECTED_IMPL["impl"], tolerance_d=R.TAU_NOTE, counts=dict(T),
    max_served_err_over_sum_abs_xw=float(er.max()) if er.numel() else None,
    controls=dict(
        A0_dispatch_and_hashes=dict(ok=not fails["A0"], failures=fails["A0"][:20]),
        Aprime_corrected_reproduces_served_bitwise=dict(ok=T["Aprime_records_bitwise"] == T["records"] and T["records"] > 0,
                                                       records=T["records"], failures=fails["Aprime"][:20]),
        engine_gate_reproduces_served_bitwise=T["engine_records_bitwise"] == T["records"],
        D_served_within_tau_d=dict(ok=T["D_tau_d_violations"] == 0 and T["D_top8_disagreements_outside_band"] == 0,
                                   logits=T["D_logits"], violations=T["D_tau_d_violations"],
                                   tau_worst_violations=T["D_tau_worst_violations"],
                                   top8_disagreements=T["D_top8_disagreements"],
                                   outside_band=T["D_top8_disagreements_outside_band"]),
        C_routing_reproduced_bitwise=dict(ok=not fails["C"] and T["C_records_exact"] == T["records"],
                                          records_exact=T["C_records_exact"], failures=fails["C"][:20])),
    seconds=round(time.time() - t0, 1), router_info=rinfo0)
res["verdict"] = "PASS" if all(v["ok"] if isinstance(v, dict) else v for k, v in res["controls"].items()
                               if k != "engine_gate_reproduces_served_bitwise") else "FAIL"
json.dump(res, open(os.path.join(out, "d3_served_summary.json"), "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ("verdict", "corrected_impl", "counts", "max_served_err_over_sum_abs_xw", "controls", "seconds")},
                 indent=1, default=str))
sys.exit(0 if res["verdict"] == "PASS" else 3)
