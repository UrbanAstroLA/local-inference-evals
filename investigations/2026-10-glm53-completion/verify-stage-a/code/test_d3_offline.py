#!/usr/bin/env python3
"""CPU tests for d3_router_compare.py: helpers, plus a synthetic end-to-end run through the
engine's own GateLinear and GroupedTopKRouter on CPU (fake records, fake checkpoint shard).
  CUDA_VISIBLE_DEVICES= python3 test_d3_offline.py
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d3_router_compare as d3  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append(True)
        print(f"PASS {name}", flush=True)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        RESULTS.append(False)
        print(f"FAIL {name}: {exc!r}", flush=True)


def t_helpers():
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b = torch.tensor([[3, 2, 1], [4, 5, 7]])
    assert d3.sets_differ(a, b).tolist() == [0, 1]
    D = d3.dense(a, torch.tensor([[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]]), 8)
    assert D.shape == (2, 8) and abs(D[0, 3].item() - 0.3) < 1e-6
    lo, hi = d3.wilson(0, 100)
    assert lo == 0.0 and 0 < hi < 0.05
    assert d3.bf16_representable(torch.tensor([1.0, 0.5])) and not d3.bf16_representable(torch.tensor([1.0001]))
    # fp64 reference routing: weights renormalised x2.5, selection uses bias
    lg = torch.zeros(1, 12, dtype=torch.float64)
    bias = torch.zeros(12)
    bias[11] = 1.0
    ids, w = d3.route_ref64(lg, bias, k=2, scale=2.5)
    assert 11 in ids[0].tolist() and abs(w.sum().item() - 2.5) < 1e-12
    g = d3.gap_k(torch.tensor([[3.0, 2.0, 1.0, 0.0]]), torch.zeros(4), k=2)
    assert abs(g.item() - (torch.sigmoid(torch.tensor(2.0)) - torch.sigmoid(torch.tensor(1.0))).item()) < 1e-6
    lo, hi = d3.cluster_boot({1: 1, 2: 0, 3: 2}, {1: 10, 2: 10, 3: 10}, reps=200)
    assert 0 <= lo <= hi <= 1


def write_safetensors(path, tensors):
    hdr, off, blobs = {}, 0, []
    for k, t in tensors.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes()
        hdr[k] = dict(dtype={torch.bfloat16: "BF16", torch.float32: "F32"}[t.dtype],
                      shape=list(t.shape), data_offsets=[off, off + len(b)])
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def build_run(tmp, corrupt=None):
    """Synthetic session: 3 records of layer 3 on both ranks, fake checkpoint shard, v13 log."""
    d3.stub_tp()
    run, ck = os.path.join(tmp, "run"), os.path.join(tmp, "ckpt")
    os.makedirs(os.path.join(run, "rec_r0"))
    os.makedirs(os.path.join(run, "rec_r1"))
    os.makedirs(ck)
    torch.manual_seed(7)
    layer = 3
    W = (torch.randn(d3.E, d3.H) * 0.02).to(torch.bfloat16)
    b = torch.randn(d3.E) * 0.01
    pfx = f"model.language_model.layers.{layer}.mlp.gate"
    write_safetensors(os.path.join(ck, "shard.safetensors"), {pfx + ".weight": W,
                                                             pfx + ".e_score_correction_bias": b})
    json.dump({"weight_map": {pfx + ".weight": "shard.safetensors",
                              pfx + ".e_score_correction_bias": "shard.safetensors"}},
              open(os.path.join(ck, "model.safetensors.index.json"), "w"))
    gate, flags = d3.engine_gate(W, "cpu")
    rinfo = dict(kind="router_info", cls="GroupedTopKRouter", top_k=8, global_num_experts=288,
                 num_expert_group=1, topk_group=1, renormalize=True, scoring_func="sigmoid",
                 routed_scaling_factor=2.5, rank=0)
    router = d3.engine_router(b, rinfo, "cpu")
    nflip = 0
    plens = [4240, 8240, 16478, 32478]
    for fwd, (m, a, r) in enumerate([(40, 0, 40), (300, 17, 64), (64, 0, 64)], start=1):
        xs = (torch.randn(m, d3.H) * 3).to(torch.bfloat16)
        with torch.no_grad():
            logits = gate(xs)[0]
        w_, i_ = router.select_experts(hidden_states=xs[a:a + r], router_logits=logits[a:a + r],
                                       topk_indices_dtype=torch.int64)
        nreq = 4
        qsl = [0, 1, 2, 3, m] if fwd != 1 else [0, 10, 20, 30, m]
        row_req = [max(k for k in range(nreq) if qsl[k] <= a + j) for j in range(r)]
        rows = dict(mismatch=False, batch_kind="mixed" if fwd != 1 else "prefill_only", num_reqs=nreq,
                    num_tokens=m, m=m, row_req=row_req,
                    reqs={k: dict(req_id=f"cmpl-{fwd}-{k}", prefill_len=plens[k], num_computed=0,
                                  num_scheduled=qsl[k + 1] - qsl[k],
                                  is_prefilling=(fwd == 1) or k == 3) for k in set(row_req)})
        cap = logits[a:a + r].clone()
        ids = i_.clone()
        if corrupt == "logits" and fwd == 2:
            cap.view(torch.int32)[5, 7] ^= 1            # one ulp in one captured shipped logit
        if corrupt == "routing" and fwd == 3:
            ids[0, 0] = (ids[0, 0] + 1) % 288 if (ids[0, 0] + 1) % 288 not in ids[0] else (ids[0, 0] + 2) % 288
        rec = dict(meta=dict(fwd=fwd, layer=layer, m=m, a=a, r=r, rows=rows, x_ptr_mod256=0, mode="NONE",
                             labels=dict(logits_shipped="shipped: bf16-result GEMM cast to fp32 (...)")),
                   x=xs[a:a + r].clone(), logits_shipped=cap, topk_ids=ids,
                   topk_weights=w_.clone(), out=xs[a:a + r].clone())
        fn = f"f{fwd:07d}_l{layer:02d}.pt"
        torch.save(rec, os.path.join(run, "rec_r0", fn))
        torch.save(rec, os.path.join(run, "rec_r1", fn))
        corr = xs.float() @ W.float().T
        wc, ic = router.select_experts(hidden_states=xs[a:a + r], router_logits=corr[a:a + r].contiguous(),
                                       topk_indices_dtype=torch.int64)
        nflip += int((d3.sets_differ(i_, ic) > 0).sum())
    sh = lambda t: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()  # noqa: E731
    with open(os.path.join(run, "v13_p1.jsonl"), "w") as fh:
        for rank in (0, 1):
            fh.write(json.dumps(dict(kind="gate_info", rank=rank, layer=layer,
                                     tier_from_flags="6 F.linear(bf16) then .to(out_dtype)",
                                     **{k: False for k in d3.FLAG_KEYS})) + "\n")
            fh.write(json.dumps(dict(kind="gate_weight_sha", rank=rank, layer=layer, weight_sha256=sh(W),
                                     bias_sha256=sh(b))) + "\n")
        fh.write(json.dumps(rinfo) + "\n")
        fh.write(json.dumps(dict(kind="experts_info", rank=0, layer=layer, swiglu_limit_on_layer=10.0,
                                 quant_method="Exl3MoEMethod", projection_mixed=True,
                                 runtime_plans=[dict(plan="trellis_plan", caps_swiglu_limit=None)])) + "\n")
    return run, ck, nflip


def t_end_to_end_cpu():
    tmp = tempfile.mkdtemp(prefix="d3test_")
    run, ck, nflip_expected = build_run(tmp)
    out = os.path.join(tmp, "out")
    s = d3.analyse(run, ck, out, device="cpu")
    assert not s["stopped"], s
    c = s["controls"]
    assert c["a_dispatch_and_hashes"]["ok"] and c["b_shipped_reproduced_bitwise"]["ok"] \
        and c["c_routing_reproduced_exactly"]["ok"], c
    assert c["rows"] == 40 + 64 + 64
    assert s["flips"] == nflip_expected, (s["flips"], nflip_expected)
    assert s["labels"]["shipped"] == "shipped: bf16-result GEMM cast to fp32"
    assert s["labels"]["corrected"] == "corrected: fp32-output GEMM (upstream #54048 form)"
    assert s["labels"]["reference"] == "reference: fp64"
    assert s["control_d"]["violations"] == 0 and s["control_d"]["disagreements_outside_band"] == 0, s["control_d"]
    assert s["control_d"]["state"] == "pass" and s["control_d"]["violations_of_tau_worst"] == 0
    assert s["shipped_vs_reference_contrast"]["max_rel_err_over_sum_abs_xw"] > \
        s["control_d"]["max_rel_err_over_sum_abs_xw"]                    # bf16 rounding dominates
    assert s["full_graph_decode"].startswith("FULL-graph decode steps are covered by source inspection only")
    assert "routed_scaling_factor=2.5" in s["normalisation"] and "grouped_topk_router.py:337-349" in s["normalisation"]
    assert s["finding_E1"]["note"].startswith("Finding E1") and "M<=1024" in s["finding_E1"]
    assert s["controls"]["router_invocation"].startswith("serving-shaped")
    kinds = {(r["batch_kind"], r["row_kind"]) for r in s["by_type"]}
    assert ("prefill_only", "prefill") in kinds and ("mixed", "decode") in kinds, kinds
    assert s["bootstrap_clustered"]["history"]["clusters"] == 4
    for f in ("d3_summary.json", "d3_by_layer.csv", "d3_by_type.csv", "d3_by_layer_type.csv"):
        assert os.path.exists(os.path.join(out, f)), f
    text = open(os.path.join(out, "d3_summary.json")).read()
    assert '"fp32 logits"' not in text
    print(f"    synthetic: rows {s['rows']} flips {s['flips']} (shipped vs fp64 "
          f"{s['shipped_vs_reference_contrast']['selection_disagreements']}, corrected vs fp64 "
          f"{s['control_d']['selection_disagreements']}); max rel err corrected "
          f"{s['control_d']['max_rel_err_over_sum_abs_xw']:.2e} shipped "
          f"{s['shipped_vs_reference_contrast']['max_rel_err_over_sum_abs_xw']:.2e}")
    d3.d4_inventory(run, out)
    inv = json.load(open(os.path.join(out, "d4_inventory.json")))
    assert inv["inventory"]["r0_records"] == 3 and inv["swiglu_limit_on_layer"] == ["10.0"]
    assert inv["sufficient_for_stage_b"] is True


def t_stop_on_logit_mismatch():
    tmp = tempfile.mkdtemp(prefix="d3test_")
    run, ck, _ = build_run(tmp, corrupt="logits")
    s = d3.analyse(run, ck, os.path.join(tmp, "out"), device="cpu")
    assert s["stopped"] and not s["controls"]["b_shipped_reproduced_bitwise"]["ok"]
    dg = s["diagnostics"][0]
    assert dg["rows_not_exact"] == 1 and dg["max_ulp_diff"] == 1 and dg["repeatable_offline"] is True
    assert "row_independent_of_padding" in dg and "alignment_sensitive" in dg
    assert os.path.exists(os.path.join(tmp, "out", "d3_controls_failed.json"))
    assert not os.path.exists(os.path.join(tmp, "out", "d3_summary.json"))


def t_stop_on_routing_mismatch():
    tmp = tempfile.mkdtemp(prefix="d3test_")
    run, ck, _ = build_run(tmp, corrupt="routing")
    s = d3.analyse(run, ck, os.path.join(tmp, "out"), device="cpu")
    assert s["stopped"] and s["controls"]["b_shipped_reproduced_bitwise"]["ok"]
    assert not s["controls"]["c_routing_reproduced_exactly"]["ok"]
    cc = s["controls"]["c_routing_reproduced_exactly"]["classification"]
    assert cc["rows_set_differ"] == 1 and cc["records_failed"] == 1, cc
    dg = [d for d in s["diagnostics"] if d.get("control") == "c"]
    assert len(dg) == 1 and dg[0]["repeatable_window"] is True, dg
    assert dg[0]["window_vs_captured"]["rows_set_differ"] == 1
    assert "serving_M_zero_pad_vs_captured" in dg[0] and "by_batch_rows_window_at_0" in dg[0]
    assert os.path.exists(os.path.join(tmp, "out", "d3_control_c_records.jsonl"))


def t_tau_and_attribution():
    torch.manual_seed(3)
    x = (torch.randn(16, d3.H) * 3).to(torch.bfloat16)
    W = (torch.randn(32, d3.H) * 0.02).to(torch.bfloat16)
    ref = x.double() @ W.double().T
    t = d3.tau(x, W, ref)
    assert ((x.float() @ W.float().T).double() - ref).abs().le(t).all()
    tw = d3.tau(x, W, ref, which="worst")
    assert (tw > t).all() and abs((tw / t).median().item() - 8.0) < 0.5   # 4096 vs 2*256 steps
    meta = dict(rows=dict(batch_kind="mixed", row_req=[0, 1, -1],
                          reqs={0: dict(req_id="a", prefill_len=4240, is_prefilling=False),
                                1: dict(req_id="b", prefill_len=8240, is_prefilling=True)}))
    at = d3.row_attribution(meta, 3)
    assert at[0] == ("decode", "mixed", "prompt_len=4240", "a") and at[1][0] == "prefill" and at[2][0] == "padding"
    assert d3.row_attribution(dict(rows=None), 2) == [("unknown",) * 4] * 2


check("helpers", t_helpers)
check("tau_and_attribution", t_tau_and_attribution)
check("end_to_end_cpu", t_end_to_end_cpu)
check("stop_on_logit_mismatch", t_stop_on_logit_mismatch)
check("stop_on_routing_mismatch", t_stop_on_routing_mismatch)
print(f"{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
sys.exit(0 if all(RESULTS) else 1)
