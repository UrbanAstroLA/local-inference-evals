#!/usr/bin/env python3
"""CPU checks for the B0 all-reduce fixture (run inside the B0 image without GPUs).

T1 generator: deterministic, value families as specified, owner zero layers, no
   NaN/Inf/subnormal, int-family sums exact.
T2 reference: (a.float()+b.float()).to(bf16) equals exact rational round-to-nearest-even
   of a+b (Fraction arithmetic) on generated pairs and on adversarial ties; torch's native
   bf16 add agrees.
T3 distinguishability: for real failure candidates (other steps, layers, ranks, shifted
   indices, zeros) no 16-byte pack of the candidate output equals the correct pack; the
   word-level coincidence rate is reported.
T4 classifier: injected faults of every class are attributed to that class.
T5 source: launch geometry and transport for rows 1..64 from the real B12x
   _plain_launch_config, the env transport policy, vLLM's byte-size parser, and the
   dispatch rule rows <= 48.
T6 plan/schedule determinism.
T7 end-to-end: analyze.py on a synthetic CLEAN run and on a synthetic run with an injected
   stale-same-slot failure.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import analyze  # noqa: E402
import fixture  # noqa: E402
import patterns as pat  # noqa: E402

RESULTS: dict = {}


def check(name, cond, detail=""):
    RESULTS[name] = {"pass": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def bits(t):
    return t.contiguous().view(torch.int16)


def t1_generator():
    L, N = 12, 3 * pat.HIDDEN
    tables = pat.cpu_tables()
    lvec = torch.arange(L, dtype=torch.int64)
    iota = torch.arange(N, dtype=torch.int64)
    check("T1 tables deterministic", torch.equal(bits(pat.make_tables("cpu")), bits(tables)))
    ti = tables[:, :pat.TABLE_LEN].float()
    ta = tables[:, pat.TABLE_LEN:].float()
    check("T1 int table: integers in [-127,127]",
          bool((ti == ti.round()).all() and (ti.abs() <= 127).all()),
          f"distinct {int(torch.unique(ti).numel())}")
    nz = ta[ta != 0].abs()
    check("T1 arith table: finite, normal, |x| in [2^-10, 2^11), -0.0 lanes",
          bool(torch.isfinite(ta).all() and (nz >= 2.0 ** -10).all() and (nz < 2.0 ** 11).all())
          and bool(((ta == 0) & torch.signbit(ta)).any()),
          f"min {float(nz.min()):.3g} max {float(nz.max()):.3g}; "
          f"-0 fraction {float(((ta == 0) & torch.signbit(ta)).float().mean()):.3f}")
    check("T1 rank tables independent",
          float((tables[0].view(torch.int16) == tables[1].view(torch.int16)).float().mean()) < 0.01)
    idx = pat.call_index(17, lvec, iota)
    a = pat.generate(tables, 17, 0, lvec, idx)
    a_t = pat.generate(tables, torch.tensor(17), 0, lvec,
                       pat.call_index(torch.tensor(17), lvec, iota))
    check("T1 tensor step == int step", torch.equal(bits(a), bits(a_t)))
    ok_rows = True
    zero_layers = []
    for s in range(6):
        idx = pat.call_index(s, lvec, iota)
        p0 = pat.generate(tables, s, 0, lvec, idx).float()
        p1 = pat.generate(tables, s, 1, lvec, idx).float()
        for layer in range(L):
            arith = (s + layer) % 2 == 1
            for r, p in ((0, p0), (1, p1)):
                owner = layer % 3 == 2 and ((s + layer) // 3) % 2 == r
                row = p[layer]
                if owner:
                    ok_rows &= bool((row == 0).all()) and not bool(torch.signbit(row).any())
                    zero_layers.append((s, layer, r))
                elif not arith:
                    ok_rows &= bool((row == row.round()).all() and (row.abs() <= 127).all())
                else:
                    ok_rows &= bool((row != row.round()).any())
            if not arith:
                exact = pat.reference_sum(pat.generate_layer(s, layer, 0, N),
                                          pat.generate_layer(s, layer, 1, N)).float()
                ok_rows &= bool(torch.equal(exact, p0[layer] + p1[layer]))
    check("T1 families/owner layers/int sums exact", ok_rows,
          f"owner zero (s,l,r) examples {zero_layers[:3]}")
    one = pat.generate_layer(9, 4, 1, N)
    lv5 = torch.arange(5, dtype=torch.int64)
    check("T1 generate_layer == batched row", torch.equal(
        bits(one), bits(pat.generate(tables, 9, 1, lv5, pat.call_index(9, lv5, iota))[4])))
    lv91 = torch.arange(91, dtype=torch.int64)
    z = torch.zeros(1, dtype=torch.int64)
    rows = [pat.call_index(s, lv91, z)[:, 0] for s in range(400)]
    near = 0
    for s in range(400):
        near += 91 - int(torch.unique(rows[s]).numel())             # same step, any layers
        if s:
            near += int((rows[s] == rows[s - 1]).sum())               # same layer, next step
    check("T1 no table-position collision among nearby calls (400 steps x 91 layers)",
          near == 0, f"{near} collisions")


def t2_reference():
    pairs = []
    for s in range(1, 41, 2):           # arith family at layer 0 for odd steps
        a = pat.generate_layer(s, 0, 0, 2048)
        b = pat.generate_layer(s, 0, 1, 2048)
        pairs.append((a, b))
    a = torch.cat([p[0] for p in pairs])
    b = torch.cat([p[1] for p in pairs])
    ref = pat.reference_sum(a, b)
    native = a + b
    check("T2 torch native bf16 add == reference_sum", torch.equal(bits(ref), bits(native)))
    idx = torch.randperm(a.numel(), generator=torch.Generator().manual_seed(1))[:20000]
    bad = 0
    for i in idx.tolist():
        fa = pat.bf16_bits_to_fraction(int(bits(a[i:i + 1])[0]))
        fb = pat.bf16_bits_to_fraction(int(bits(b[i:i + 1])[0]))
        exact = pat.exact_rn_bf16(fa + fb)
        if exact != float(ref[i]):
            bad += 1
    check("T2 reference == exact RN (20000 generated pairs)", bad == 0, f"{bad} differ")
    # adversarial: exact ties, large exponent gaps, cancellation
    adv_a, adv_b = [], []
    rng = random.Random(7)
    for _ in range(20000):
        e1 = rng.randint(-20, 20)
        m1 = rng.randint(128, 255)
        k = rng.choice([0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 30])
        m2 = rng.randint(128, 255)
        sgn = rng.choice([1, -1])
        adv_a.append(m1 * 2.0 ** (e1 - 7))
        adv_b.append(sgn * m2 * 2.0 ** (e1 - k - 7))
    A = torch.tensor(adv_a).to(torch.bfloat16)
    B = torch.tensor(adv_b).to(torch.bfloat16)
    R = pat.reference_sum(A, B)
    bad = 0
    for i in range(A.numel()):
        fa = pat.bf16_bits_to_fraction(int(bits(A[i:i + 1])[0]))
        fb = pat.bf16_bits_to_fraction(int(bits(B[i:i + 1])[0]))
        if pat.exact_rn_bf16(fa + fb) != float(R[i]):
            bad += 1
    check("T2 reference == exact RN (20000 adversarial gaps/ties)", bad == 0, f"{bad} differ")
    z = torch.tensor([0.0, -0.0, -0.0, 3.0]).to(torch.bfloat16)
    w = torch.tensor([-0.0, 0.0, -0.0, -0.0]).to(torch.bfloat16)
    zs = pat.reference_sum(z, w).float()
    check("T2 signed zeros follow RN (+0, +0, -0, 3)",
          [str(float(v)) for v in zs] == ["0.0", "0.0", "-0.0", "3.0"],
          str([float(v) for v in zs]))


def t3_distinguishable():
    N = 8 * pat.HIDDEN
    worst_pack, word_rates = 0, []
    for s, layer in ((10, 4), (11, 5), (12, 8), (13, 3)):
        for rank in (0, 1):
            peer = 1 - rank
            local = pat.generate_layer(s, layer, rank, N)
            right = pat.reference_sum(local, pat.generate_layer(s, layer, peer, N))
            cands = {
                "own": local,
                "zero": torch.zeros_like(local),
                "prev-layer": pat.generate_layer(s, layer - 1, peer, N),
                "prev2-layer": pat.generate_layer(s, layer - 2, peer, N),
                "prev-step": pat.generate_layer(s - 1, layer, peer, N),
                "prev2-step": pat.generate_layer(s - 2, layer, peer, N),
                "shift8": pat._shift(pat.generate_layer(s, layer, peer, N), 8)[0],
                "shift4096": pat._shift(pat.generate_layer(s, layer, peer, N), 4096)[0],
            }
            truth = pat.generate_layer(s, layer, peer, N)
            for name, cand in cands.items():
                if torch.equal(bits(cand), bits(truth)):
                    continue      # identical source (e.g. two all-zero owner inputs)
                out = pat.reference_sum(local, cand)
                eqw = (out.view(torch.int32) == right.view(torch.int32))
                if name.startswith("shift"):
                    eqw = eqw[4096:]           # shifted-in region only
                packs = eqw.view(-1, 4).all(dim=1)
                worst_pack = max(worst_pack, float(packs.float().mean()))
                word_rates.append(float(eqw.float().mean()))
    # A wrong source is invisible in a pack only if all eight sums coincide with the
    # right ones (equal values, or absorption of a small operand by a large one).
    check("T3 wrong-source pack coincidence < 1e-3 for every candidate",
          worst_pack < 1e-3,
          f"worst pack coincidence {worst_pack:.2e}; word coincidence mean "
          f"{sum(word_rates) / len(word_rates):.2e}, max {max(word_rates):.2e}")


def t4_classifier():
    N = 4 * pat.HIDDEN
    rank = 0
    peer = 1 - rank

    def setup(s, layer):
        local = pat.generate_layer(s, layer, rank, N)
        peer_in = pat.generate_layer(s, layer, peer, N)
        prevs = [pat.generate_layer(s, layer - k, peer, N) for k in range(1, 7)]
        prev_local = [pat.generate_layer(s, layer - k, rank, N) for k in (1, 2)]
        prev_step = [pat.generate_layer(s - 1, layer, peer, N)]
        return local, peer_in, prevs, prev_local, prev_step

    def run(y, ctx):
        local, peer_in, prevs, prev_local, prev_step = ctx
        return pat.classify(y, local, peer_in, prev_calls_peer=prevs,
                            prev_steps_peer=prev_step, prev_calls_local=prev_local)

    def inject(ctx, src_peer=None, src_local=None, packs=range(10, 14)):
        local, peer_in = ctx[0], ctx[1]
        y = pat.reference_sum(local, peer_in)
        bad = pat.reference_sum(local if src_local is None else src_local,
                                peer_in if src_peer is None else src_peer)
        for p in packs:
            y[8 * p:8 * p + 8] = bad[8 * p:8 * p + 8]
        return y

    # s=22, layer 4: int family; layers 3, 2 and step 21 give non-zero peer sources
    c = setup(22, 4)
    local, peer_in, prevs, prev_local, prev_step = c
    cases = {
        "stale-same-slot": inject(c, prevs[1]),
        "stale-other-slot": inject(c, prevs[0]),
        "wrong-rank": inject(c, local),
        "missing-peer": inject(c, torch.zeros_like(peer_in)),
        "cross-step": inject(c, prev_step[0]),
        "addressing": inject(c, pat._shift(peer_in, 8)[0]),
        "local-stale": inject(c, src_local=prev_local[0]),
    }
    ok, detail = True, {}
    for want, y in cases.items():
        got = run(y, c)
        packs = {k: v for k, v in got["pack_classes"].items() if k != "ok"}
        detail[want] = packs
        ok &= packs == {want: 4}
    # arithmetic: s=21, layer 4 is arith family; replace 8 words by the RZ rounding
    ca = setup(21, 4)
    right = pat.reference_sum(ca[0], ca[1])
    rz = pat.rz_sum(ca[0], ca[1])
    y = right.clone()
    words = torch.nonzero((bits(rz) != bits(right)).view(-1, 2).any(dim=1)).view(-1)[:8]
    for w in words.tolist():
        y[2 * w:2 * w + 2] = rz[2 * w:2 * w + 2]
    got = run(y, ca)
    wc = {k: v for k, v in got["word_classes"].items() if k != "ok"}
    detail["arithmetic"] = wc
    ok &= wc == {"arithmetic": len(words)} and len(words) == 8
    check("T4 injected classes recovered at pack level", ok, json.dumps(detail))

    # a stale source that is an all-zero owner input is reported as missing-peer and the
    # alternative is visible in word_matches (inherent ambiguity, not a mis-attribution)
    cz = setup(21, 4)          # layer 2 at s=21: owner rank ((21+2)//3)%2 = 1 = peer
    zero_prev = bool((cz[2][1] == 0).all())
    got = run(inject(cz, cz[2][1]), cz)
    m = got["word_matches"]
    check("T4 zero stale source -> missing-peer, alternative listed",
          zero_prev and got["pack_classes"].get("missing-peer") == 4
          and m.get("stale-same-slot", 0) == m.get("missing-peer", -1) >= 12,
          json.dumps({"packs": got["pack_classes"], "matches": got["word_matches"]}))
    y = pat.reference_sum(local, peer_in)
    stale = pat.reference_sum(local, prevs[1])
    y[8 * 20:8 * 20 + 4] = stale[8 * 20:8 * 20 + 4]       # half a pack stale
    got = run(y, c)
    check("T4 half-stale pack reported torn", "torn-pack (mixed sources)" in got["pack_classes"],
          json.dumps(got["pack_classes"]))
    y = pat.reference_sum(local, peer_in)
    y.view(torch.int16)[8 * 30:8 * 30 + 8] ^= 0x1234
    got = run(y, c)
    check("T4 random bits unexplained", set(got["word_classes"]) - {"ok"} <= {"unexplained"}
          and "unexplained" in got["word_classes"], json.dumps(got["word_classes"]))
    got = run(pat.reference_sum(local, peer_in), c)
    check("T4 clean output all ok", set(got["word_classes"]) == {"ok"})


def t5_source():
    from b12x.comm.pcie.pcie_oneshot import (
        _CuTeOneshotBackend, _CuTeOneshotState, _transport_policy_contract,
        _tp2_plain_remote_push_rows)
    from vllm.distributed.device_communicators.b12x_pcie_all_reduce import _parse_byte_size

    policy = _transport_policy_contract()
    check("T5 B0 env transport policy (push, tp2-fused, plain-push, tp4, tp8)",
          list(policy) == analyze.EXPECTED_POLICY, str(policy))
    max_size = _parse_byte_size("384KB")
    check("T5 VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=384KB -> 393216", max_size == 393216)
    st = _CuTeOneshotState(rank=0, world_size=2, signal_ptrs=(1, 2),
                           rank_data=torch.empty(0, dtype=torch.uint8),
                           signal_table_address=0, next_table_offset=0,
                           registered_tables={}, eager_tables=(0, 0),
                           eager_ptrs=((1, 2), (3, 4)), eager_buffer_bytes=max_size,
                           transport_policy=policy)
    table, ok = {}, True
    for rows in range(1, 65):
        x = torch.empty(rows, pat.HIDDEN, dtype=torch.bfloat16)
        graph = _CuTeOneshotBackend._plain_launch_config(st, x, graph_channel=True)
        eager = _CuTeOneshotBackend._plain_launch_config(st, x, graph_channel=False)
        vllm_b12x = rows * pat.HIDDEN * 2 <= max_size       # should_allreduce size rule
        table[rows] = {"graph": graph, "eager": eager, "vllm_dispatch":
                       "b12x" if vllm_b12x else "pynccl"}
        if rows <= 48:
            ok &= tuple(graph) == analyze.expected_geometry(rows)
        ok &= eager[0] == "pull"
        ok &= _tp2_plain_remote_push_rows(x) == rows
    check("T5 geometry rows 1..48 matches analyze.expected_geometry; eager = pull", ok)
    check("T5 dispatch rule: rows <= 48 -> B12x, 49..64 -> PyNCCL",
          [r for r in table if table[r]["vllm_dispatch"] == "b12x"] == list(range(1, 49)))
    x32 = torch.empty(4, pat.HIDDEN, dtype=torch.float32)
    x6k = torch.empty(4, 6144, dtype=torch.bfloat16)
    check("T5 fp32 / non-4096 hidden never take peer-push",
          _CuTeOneshotBackend._plain_launch_config(st, x32, graph_channel=True)[0] == "pull"
          and _CuTeOneshotBackend._plain_launch_config(st, x6k, graph_channel=True)[0] == "pull")
    RESULTS["T5 table"] = {r: {"graph": list(v["graph"]), "vllm": v["vllm_dispatch"]}
                           for r, v in table.items()}


def t6_plan():
    for name in fixture.PROFILES:
        prof = fixture.PROFILES[name]
        a = fixture.build_cases(prof)
        b = fixture.build_cases(prof)
        sa = [fixture.phase_schedule(a, p, 1) for p in ("main", "skew", "boundary")]
        sb = [fixture.phase_schedule(b, p, 1) for p in ("main", "skew", "boundary")]
        check(f"T6 {name} schedule deterministic", sa == sb)
        disp = {(c.rows, c.expected_dispatch) for c in a if c.ar_kind == "comm"}
        check(f"T6 {name} expected dispatch by rows",
              all((d == "b12x") == (r <= 48) for r, d in disp))


# ------------------------------------------------------------------ T7 synthetic runs
def synth_run(root: str, inject: bool) -> str:
    prof = fixture.PROFILES["smoke"]
    cases = fixture.build_cases(prof)
    seed = 5
    log, s = [], 0
    first = True
    for c in cases:                                   # warm-ups in capture order
        if c.expected_dispatch != "b12x":
            t = "pynccl"
        else:
            t = "pull" if first else "push"
            first = False
        log.append([s, c.gid, "warmup", t])
        s += 1
    warm_push = sum(cases[g].layers for (_, g, _, t) in log if t == "push")
    phases = []
    for name in ("main", "skew", "boundary"):
        items = fixture.phase_schedule(cases, name, seed)
        push = 0
        for g in items:
            c = cases[g]
            t = "push" if c.expected_dispatch == "b12x" else c.expected_dispatch
            log.append([s, g, "replay", t])
            s += 1
            push += c.layers if t == "push" else 0
        phases.append({"phase": name, "items": len(items), "forwards": len(items),
                       "seconds": 0, "epoch_delta": push, "expected_push_calls": push})
    main_b12x = [c for c in cases if c.phase == "main" and c.expected_dispatch == "b12x"
                 and c.kind == "FULL"]
    mixed = [main_b12x[i % len(main_b12x)].gid for i in range(prof["mixed"])]
    push = 0
    for g in mixed:
        log.append([s, g, "replay", "push"]); s += 1
        log.append([s, -1, "eager-prefill", "pynccl"]); s += 1
        push += cases[g].layers
    phases.append({"phase": "mixed", "items": len(mixed), "forwards": 2 * len(mixed),
                   "seconds": 0, "epoch_delta": push, "expected_push_calls": push})
    items = fixture.phase_schedule(cases, "nccl-ref", seed)
    for g in items:
        log.append([s, g, "replay", "pynccl-direct"]); s += 1
    phases.append({"phase": "nccl-ref", "items": len(items), "forwards": len(items),
                   "seconds": 0, "epoch_delta": 0, "expected_push_calls": 0})
    plans = [{"key": ["bfloat16", f"({c.rows}, 4096)", "(4096, 1)", "0"],
              "transport": analyze.expected_geometry(c.rows)[0],
              "threads": analyze.expected_geometry(c.rows)[1],
              "blocks": analyze.expected_geometry(c.rows)[2], "size_packs": c.rows * 512,
              "region_packs": 24576}
             for c in cases if c.expected_dispatch == "b12x"]
    run = os.path.join(root, "inject" if inject else "clean")
    for rank in (0, 1):
        d = os.path.join(run, f"rank{rank}")
        os.makedirs(d)
        crow = []
        for c in cases:
            m = c.meta()
            m["dispatch"] = c.expected_dispatch
            crow.append({**m, "mismatched_elements": 0, "layers_with_mismatch": []})
        if inject and rank == 0:
            # gid of the 48-row main FULL graph; its third replay in the log; layer 6
            tgt = next(c for c in cases if c.phase == "main" and c.kind == "FULL"
                       and c.rows == 48)
            steps = [e[0] for e in log if e[1] == tgt.gid and e[2] == "replay"]
            step, layer = steps[2], 6
            seq, index = analyze.push_sequence([tuple(e) for e in log],
                                               {c.gid: c.meta() for c in cases})
            n = index[(step, layer)]
            s2, g2, l2 = seq[n - 2]
            N = tgt.N
            local = pat.generate_layer(step, layer, 0, N)
            right = pat.reference_sum(local, pat.generate_layer(step, layer, 1, N))
            stale_peer = pat.generate_layer(s2, l2, 1, cases[g2].N)
            stale = pat.reference_sum(local, pat._fit(stale_peer, N)[0])
            y = right.clone()
            y[8 * 100:8 * 103] = stale[8 * 100:8 * 103]
            nbad = int((bits(y) != bits(right)).sum())
            acc = torch.zeros(tgt.layers, dtype=torch.int64)
            acc[layer] = nbad
            fy = torch.zeros(tgt.layers, N, dtype=torch.bfloat16)
            fy[layer] = y
            fstep = torch.full((tgt.layers,), -1, dtype=torch.int64)
            fstep[layer] = step
            torch.save({"meta": tgt.meta(), "acc": acc, "fstep": fstep, "lstep": fstep,
                        "fy": fy, "ly": fy}, os.path.join(d, f"fail_g{tgt.gid}.pt"))
            crow[tgt.gid]["mismatched_elements"] = nbad
            crow[tgt.gid]["layers_with_mismatch"] = [layer]
        res = {"backends": {**analyze.EXPECTED_BACKENDS,
                            "transport_policy": analyze.EXPECTED_POLICY},
               "cases": crow, "plans": plans, "phases": phases,
               "epoch_after_capture": {"value": warm_push, "delta": warm_push,
                                       "expected_push_calls": warm_push},
               "warmup_mismatches": {}, "prefill_mismatched_elements": 0,
               "cross_rank_reference_identical": True,
               "cross_rank_last_output_identical": True,
               "capture": {"seconds": 0}, "plan": {}, "mismatch_steps": []}
        with open(os.path.join(d, "result.json"), "w") as f:
            json.dump(res, f)
        with open(os.path.join(d, "status.json"), "w") as f:
            json.dump({"status": "completed"}, f)
        with open(os.path.join(d, "schedule.jsonl"), "w") as f:
            for e in log:
                f.write(json.dumps(e) + "\n")
    return run


def t7_end_to_end():
    root = tempfile.mkdtemp(prefix="b0ar-")
    try:
        rep = analyze.analyze(synth_run(root, False))
        check("T7 synthetic clean run -> CLEAN", rep["verdict"] == "CLEAN",
              json.dumps(rep.get("invalid", []) + rep.get("fails", []))[:300])
        rep = analyze.analyze(synth_run(root, True))
        cls = rep.get("classifications", [])
        packs = cls[0]["pack_classes"] if cls else {}
        check("T7 injected stale-same-slot run -> FAIL, classified",
              rep["verdict"] == "FAIL" and packs.get("stale-same-slot", 0) >= 2,
              json.dumps({"verdict": rep["verdict"], "packs": packs,
                          "words": cls[0]["word_classes"] if cls else None}))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    torch.manual_seed(0)
    for t in (t1_generator, t2_reference, t3_distinguishable, t4_classifier, t5_source,
              t6_plan, t7_end_to_end):
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            import traceback
            check(f"{t.__name__} raised", False, repr(exc))
            traceback.print_exc()
    out = sys.argv[1] if len(sys.argv) > 1 else None
    if out:
        with open(out, "w") as f:
            json.dump(RESULTS, f, indent=1, default=str)
    failed = [k for k, v in RESULTS.items() if isinstance(v, dict) and v.get("pass") is False]
    print(f"{len(failed)} failed" if failed else "all passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
