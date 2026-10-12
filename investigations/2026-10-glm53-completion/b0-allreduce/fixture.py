#!/usr/bin/env python3
"""B0 TP2 all-reduce correctness fixture: two processes, one per GPU, no model.

Exercises the transport B0 dispatches under shipped graph execution through the real
code path: vLLM's CudaCommunicator (dispatch order B12X_PCIE_ONESHOT then PYNCCL) ->
vLLM's B12xPcieAllReduce adapter (plan / prepare_b12x_plan / capture context) -> B12x
OneshotAllReducePool single channel -> TP2 plain peer-push CuTe kernels.  Nothing in the
transport is transcribed.  PIECEWISE graphs use vLLM's BreakableCUDAGraphCapture with
eager breaks, FULL graphs torch.cuda.graph on a shared pool, captured in vLLM's order
(PIECEWISE then FULL, largest first) inside the communicator's capture context after one
eager warm-up per shape, exactly as v1/worker/gpu/cudagraph_utils.py:336-389 does.

Every replay generates new per-(step, layer, rank, element) inputs in-graph from a device
step counter (patterns.py), runs `layers` all-reduces back to back (alternating a static
input slice and a reused scratch input, a small GEMM every fourth layer, output copied out
immediately), and compares every output element bitwise with the reference on the device.
Mismatch counts are accumulated per graph/layer and per step; the first and last failing
output of each layer is kept for classification (analyze.py).  The B12x plain peer-push
epoch counter is read before and after each phase: its delta must equal the number of
peer-push calls the schedule issued (dispatch evidence).

Usage (inside the B0 image; see run.sh):
  python3 fixture.py --profile smoke|full --out /out        two GPUs
  python3 fixture.py --profile full --dry                    CPU only: print the plan
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import socket
import sys
import time
import traceback
from datetime import timedelta

FX_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FX_DIR)

import torch  # noqa: E402

import patterns as pat  # noqa: E402

HIDDEN = pat.HIDDEN
ONESHOT_MAX_ROWS = 48            # 384 KB / (4096 * 2 B): VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE

# B0 serving environment for the TP all-reduce (recipe env.sh/recipe.json and the
# container inspect of the gate and both screens).  B12X_PCIE_* are unset in B0, which
# selects plain TP2 peer-push (pcie_oneshot.py:137-148).
ENGINE_ENV = {
    "VLLM_ENABLE_PCIE_ALLREDUCE": "1",
    "VLLM_PCIE_ALLREDUCE_BACKEND": "b12x",
    "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE": "384KB",
    "VLLM_B12X_PCIE_EAGER": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
B12X_TRANSPORT_ENV = ("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", "B12X_PCIE_TP2_REMOTE_PUSH",
                      "B12X_PCIE_ONESHOT_PUSH", "B12X_PCIE_TP4_REMOTE_PUSH",
                      "B12X_PCIE_ONESHOT_THREADS", "B12X_PCIE_ONESHOT_BLOCK_LIMIT")

# (rows, replays).  Engine sizes: cudagraph_capture_sizes [1,2,4,8,16,24,32,40,48,56,64]
# for PIECEWISE, uniform-decode multiples of 4 (DFlash2 x3) for FULL; rows > 48 dispatch to
# PyNCCL.  Boundary rows hit each launch-geometry branch of
# _tp2_plain_remote_push_geometry (pcie_oneshot.py:1535-1579) and the 384 KB switch.
PROFILES = {
    "smoke": dict(
        layers=91, boundary_layers=31, chunk=32,
        full=[(64, 6), (48, 40), (4, 40)],
        pw=[(48, 12), (1, 12)],
        skew=[(4, 12)],
        boundary=[(3, 12), (49, 4)],
        nccl=[(48, 12)],
        mixed=8,
    ),
    "full": dict(
        layers=91, boundary_layers=31, chunk=256,
        full=[(64, 200), (56, 200), (48, 2000), (40, 2000), (32, 2000), (24, 2000),
              (16, 2000), (8, 2000), (4, 2000)],
        pw=[(64, 100), (56, 100), (48, 600), (40, 600), (32, 600), (24, 600), (16, 600),
            (8, 600), (4, 600), (2, 600), (1, 600)],
        skew=[(1, 300), (4, 300), (16, 300), (24, 300), (48, 300)],
        boundary=[(3, 300), (5, 300), (6, 300), (7, 300), (9, 300), (14, 300), (15, 300),
                  (17, 300), (22, 300), (23, 300), (25, 300), (47, 300), (49, 50)],
        nccl=[(1, 300), (8, 300), (48, 300)],
        mixed=300,
    ),
}
PREFILL_ROWS = 2048               # max_num_batched_tokens: eager prefill chunk (PyNCCL)
PREFILL_LAYER_ID = 7777


class Case:
    """One captured graph: kind FULL|PIECEWISE, a phase, rows, layers, all-reduce kind."""

    def __init__(self, gid, phase, kind, rows, layers, replays, ar_kind="comm",
                 skew=False):
        self.gid, self.phase, self.kind = gid, phase, kind
        self.rows, self.layers, self.replays = rows, layers, replays
        self.ar_kind, self.skew = ar_kind, skew
        self.N = rows * HIDDEN
        self.expected_dispatch = (
            "pynccl-direct" if ar_kind == "pynccl"
            else ("b12x" if rows <= ONESHOT_MAX_ROWS else "pynccl"))
        self.dispatch = None      # observed at capture
        self.graph = None
        self.cap = None

    def meta(self):
        return {k: getattr(self, k) for k in (
            "gid", "phase", "kind", "rows", "layers", "replays", "ar_kind", "skew",
            "expected_dispatch", "dispatch")}


def build_cases(profile: dict) -> list[Case]:
    cases: list[Case] = []
    L, LB = profile["layers"], profile["boundary_layers"]

    def add(*a, **k):
        cases.append(Case(len(cases), *a, **k))

    # vLLM capture order: PIECEWISE descending, then FULL descending (cudagraph_utils.py
    # 340-347 with descs sorted by num_tokens reverse=True, :313-315).
    for rows, n in sorted(profile["pw"], key=lambda t: -t[0]):
        add("main", "PIECEWISE", rows, L, n)
    for rows, n in sorted(profile["full"], key=lambda t: -t[0]):
        add("main", "FULL", rows, L, n)
    for rows, n in sorted(profile["skew"], key=lambda t: -t[0]):
        add("skew", "FULL", rows, L, n, skew=True)
    for rows, n in sorted(profile["boundary"], key=lambda t: -t[0]):
        add("boundary", "FULL", rows, LB, n)
    for rows, n in sorted(profile["nccl"], key=lambda t: -t[0]):
        add("nccl-ref", "FULL", rows, L, n, ar_kind="pynccl")
    return cases


def phase_schedule(cases: list[Case], phase: str, seed: int) -> list[int]:
    """Rank-identical shuffled multiset of graph ids for one phase."""
    ids = [c.gid for c in cases if c.phase == phase for _ in range(c.replays)]
    random.Random(f"{seed}:{phase}").shuffle(ids)
    return ids


def plan_summary(profile: dict, cases: list[Case], seed: int) -> dict:
    main_b12x = [c for c in cases if c.phase == "main" and c.expected_dispatch == "b12x"
                 and c.kind == "FULL"]
    mixed = [main_b12x[i % len(main_b12x)].gid for i in range(profile["mixed"])]
    random.Random(f"{seed}:mixed").shuffle(mixed)
    forwards = sum(c.replays for c in cases) + len(mixed) * 2 + len(cases)
    push = sum(c.replays * c.layers for c in cases if c.expected_dispatch == "b12x")
    push += sum(next(c for c in cases if c.gid == g).layers for g in mixed)
    elems = sum(c.replays * c.layers * c.N for c in cases)
    return {"graphs": len(cases), "forwards_incl_warmup_and_prefill": forwards,
            "b12x_peer_push_calls_in_replays": push,
            "checked_elements_in_replays": elems,
            "by_phase": {p: sum(c.replays for c in cases if c.phase == p)
                         for p in ("main", "skew", "boundary", "nccl-ref")},
            "mixed_iterations": len(mixed)}


# --------------------------------------------------------------------------- GPU side
class Hang(RuntimeError):
    pass


def worker(rank: int, args, port: int) -> None:
    import torch.distributed as dist

    out = os.path.join(args.out, f"rank{rank}")
    os.makedirs(out, exist_ok=True)
    status = {"rank": rank, "status": "started"}
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                                world_size=2, timeout=timedelta(seconds=600))
        cpu_group = dist.new_group(ranks=[0, 1], backend="gloo",
                                   timeout=timedelta(seconds=600))
        status.update(run(rank, args, device, cpu_group, out))
        status["status"] = "completed"
    except Hang as exc:
        status.update({"status": "hang", "error": str(exc)})
        _write(out, "status.json", status)
        sys.stdout.flush()
        os._exit(4)
    except BaseException as exc:  # noqa: BLE001 - record and re-raise
        status.update({"status": "error", "error": repr(exc),
                       "traceback": traceback.format_exc()})
        _write(out, "status.json", status)
        raise
    _write(out, "status.json", status)


def _write(out, name, obj):
    tmp = os.path.join(out, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, os.path.join(out, name))


def run(rank, args, device, cpu_group, out) -> dict:
    import torch.distributed as dist
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.distributed.device_communicators.b12x_pcie_all_reduce import (
        B12xPcieAllReduce)
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    from b12x.comm.pcie._oneshot_cute import _PLAIN_GRAPH_EPOCH_OFFSET

    profile = dict(PROFILES[args.profile])
    if args.layers:
        profile["layers"] = args.layers
    cases = build_cases(profile)
    peer = 1 - rank
    res: dict = {"profile": args.profile, "seed": args.seed,
                 "env": {k: os.environ.get(k) for k in
                         list(ENGINE_ENV) + list(B12X_TRANSPORT_ENV) +
                         ["NCCL_DEBUG", "NCCL_SOCKET_IFNAME", "CUDA_DEVICE_ORDER"]},
                 "torch": torch.__version__,
                 "device": torch.cuda.get_device_name(device),
                 "plan": plan_summary(profile, cases, args.seed)}

    # -- communicator exactly as vLLM builds it for group 'tp:0' (cuda_communicator.py)
    t0 = time.time()
    comm = CudaCommunicator(cpu_group=cpu_group, device=device,
                            device_group=dist.group.WORLD, unique_name="tp:0")
    ca = comm.ca_comm
    if not isinstance(ca, B12xPcieAllReduce) or ca.disabled:
        raise RuntimeError(f"expected B12xPcieAllReduce, got {type(ca)}")
    if comm.pynccl_comm is None or comm.pynccl_comm.disabled:
        raise RuntimeError("PyNCCL communicator unavailable")
    channel = ca.runtime.for_stream()
    state = channel._ext._state(channel._ptr)
    backends = {"ca": ca.backend_name(), "pynccl": not comm.pynccl_comm.disabled,
                "fi_ar": comm.fi_ar_comm is not None, "qr": comm.qr_comm is not None,
                "aiter": comm.aiter_ar_comm is not None,
                "symm_mem": (comm.symm_mem_comm is not None
                             and not comm.symm_mem_comm.disabled),
                "max_size": ca.max_size, "enable_eager": ca.enable_eager,
                "transport_policy": list(state.transport_policy),
                "eager_buffer_bytes": state.eager_buffer_bytes,
                "plain_remote_push_region_packs": state.plain_remote_push_region_packs,
                "init_s": round(time.time() - t0, 2)}
    res["backends"] = backends
    expect_push = os.environ.get("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH",
                                 os.environ.get("B12X_PCIE_TP2_REMOTE_PUSH", "1")) not in ("", "0")
    if bool(state.transport_policy[2]) != expect_push:
        raise RuntimeError(f"transport policy {state.transport_policy} != env")
    _write(out, "progress.json", {"stage": "init", **res})

    def read_epoch() -> int:
        torch.cuda.synchronize(device)
        v = ctypes.c_uint32()
        channel._ipc.cudaMemcpyAsync(
            ctypes.addressof(v), channel.signal_ptrs[rank] + _PLAIN_GRAPH_EPOCH_OFFSET, 4,
            int(torch.cuda.current_stream(device).cuda_stream))
        torch.cuda.synchronize(device)
        return int(v.value)

    # -- static buffers shared by every graph (graphs replay serially on one stream)
    L = max(c.layers for c in cases)
    Nmax = max(c.N for c in cases)
    tables = pat.make_tables(device)
    iota = torch.arange(max(Nmax, PREFILL_ROWS * HIDDEN), dtype=torch.int64, device=device)
    lvec = torch.arange(L, dtype=torch.int64, device=device)
    P_self = torch.zeros((L, Nmax), dtype=torch.bfloat16, device=device)
    E = torch.zeros_like(P_self)
    Y = torch.zeros_like(P_self)
    scratch = torch.zeros(Nmax, dtype=torch.bfloat16, device=device)
    ga = torch.randn(64, HIDDEN, device=device, dtype=torch.bfloat16)
    gb = torch.randn(HIDDEN, 512, device=device, dtype=torch.bfloat16)
    gc = torch.empty(64, 512, device=device, dtype=torch.bfloat16)
    ea = torch.randn(256, HIDDEN, device=device, dtype=torch.bfloat16)
    eb = torch.randn(HIDDEN, 1024, device=device, dtype=torch.bfloat16)
    ec = torch.empty(256, 1024, device=device, dtype=torch.bfloat16)
    step_t = torch.zeros((), dtype=torch.int64, device=device)
    max_steps = res["plan"]["forwards_incl_warmup_and_prefill"] + 1024
    mism_by_step = torch.zeros(max_steps, dtype=torch.int64, device=device)
    pf_lvec = torch.tensor([PREFILL_LAYER_ID], dtype=torch.int64, device=device)
    pf_acc = torch.zeros((), dtype=torch.int64, device=device)
    for c in cases:
        c.acc = torch.zeros(c.layers, dtype=torch.int64, device=device)
        c.fflag = torch.zeros(c.layers, dtype=torch.bool, device=device)
        c.fstep = torch.full((c.layers,), -1, dtype=torch.int64, device=device)
        c.lstep = torch.full((c.layers,), -1, dtype=torch.int64, device=device)
        c.fy = torch.zeros((c.layers, c.N), dtype=torch.bfloat16, device=device)
        c.ly = torch.zeros_like(c.fy)
    pool = torch.cuda.graph_pool_handle()
    host = {"s": 0, "log": []}

    def eager_break():
        torch.mm(ea, eb, out=ec)   # stands in for an eager attention break

    def all_reduce(case, x):
        if case.ar_kind == "pynccl":
            return comm.pynccl_comm.all_reduce(x)
        return comm.all_reduce(x)          # real vLLM dispatch

    def forward(case: Case, breaker=None, record_dispatch=False):
        Lc, N, R = case.layers, case.N, case.rows
        lv = lvec[:Lc]
        index = pat.call_index(step_t, lv, iota[:N])
        p0 = pat.generate(tables, step_t, 0, lv, index)
        p1 = pat.generate(tables, step_t, 1, lv, index)
        P_self[:Lc, :N].copy_(p0 if rank == 0 else p1)
        E[:Lc, :N].copy_(pat.reference_sum(p0, p1))
        del p0, p1, index
        for l in range(Lc):
            if case.skew and l % 4 == 2 * rank:
                torch.cuda._sleep(args.sleep_cycles)
            if l % 2 == 1:
                scratch[:N].copy_(P_self[l, :N])
                x = scratch[:N].view(R, HIDDEN)
            else:
                x = P_self[l, :N].view(R, HIDDEN)
            if l % 4 == 3:
                torch.mm(ga, gb, out=gc)
            if record_dispatch and l == 0:
                if case.ar_kind == "pynccl":
                    case.dispatch = "pynccl-direct"
                else:
                    case.dispatch = "b12x" if ca.should_custom_ar(x) else "pynccl"
            y = all_reduce(case, x)
            Y[l, :N].copy_(y.view(-1))
            del y
            if breaker is not None and l % 2 == 1 and l != Lc - 1:
                breaker.add_eager(eager_break)
        check(case)
        step_t.add_(1)

    def check(case: Case):
        Lc, N = case.layers, case.N
        Yv, Ev = Y[:Lc, :N], E[:Lc, :N]
        cnt = (Yv.view(torch.int16) != Ev.view(torch.int16)).sum(dim=1)
        case.acc.add_(cnt)
        anyl = cnt > 0
        newf = anyl & ~case.fflag
        sv = step_t.expand(Lc)
        case.fy.copy_(torch.where(newf[:, None], Yv, case.fy))
        case.fstep.copy_(torch.where(newf, sv, case.fstep))
        case.fflag.logical_or_(anyl)
        case.ly.copy_(torch.where(anyl[:, None], Yv, case.ly))
        case.lstep.copy_(torch.where(anyl, sv, case.lstep))
        mism_by_step.index_add_(0, step_t.view(1), cnt.sum().view(1))

    def prefill_eager():
        index = pat.call_index(step_t, pf_lvec, iota[:PREFILL_ROWS * HIDDEN])
        x = pat.generate(tables, step_t, rank, pf_lvec, index).view(PREFILL_ROWS, HIDDEN)
        e = pat.reference_sum(pat.generate(tables, step_t, 0, pf_lvec, index),
                              pat.generate(tables, step_t, 1, pf_lvec, index))
        y = comm.all_reduce(x)            # eager, outside capture context -> PyNCCL
        bad = (y.view(-1).view(torch.int16) != e.view(-1).view(torch.int16)).sum()
        pf_acc.add_(bad)
        mism_by_step.index_add_(0, step_t.view(1), bad.view(1))
        step_t.add_(1)

    def log(gid, mode, transport):
        host["log"].append((host["s"], gid, mode, transport))
        host["s"] += 1

    def wait(deadline_s, where):
        ev = torch.cuda.Event()
        ev.record()
        t = time.time()
        while not ev.query():
            if time.time() - t > deadline_s:
                _write(out, "hang.json", {"where": where, "host_step": host["s"],
                                           "last": host["log"][-8:]})
                raise Hang(f"no progress for {deadline_s}s at {where}")
            time.sleep(0.0005)

    # -- warm-up + capture in vLLM order inside the communicator's capture context
    e_start = read_epoch()
    res["epoch_start"] = e_start
    cap_stream = torch.cuda.Stream(device=device)
    cap_stream.wait_stream(torch.cuda.current_stream(device))
    t0 = time.time()
    warm = []
    with torch.cuda.stream(cap_stream), ca.capture():
        for c in cases:
            transport = ("pynccl" if c.expected_dispatch != "b12x" else
                         ("push" if state.device_slot_selection else "pull"))
            tw = time.time()
            forward(c, record_dispatch=False)       # eager warm-up (prepares the plan)
            cap_stream.synchronize()
            log(c.gid, "warmup", transport)
            tc = time.time()
            if c.kind == "PIECEWISE":
                torch.cuda.synchronize(device)
                cap = BreakableCUDAGraphCapture(pool=pool)
                with cap:
                    forward(c, breaker=cap, record_dispatch=True)
                c.cap = cap
            else:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool):
                    forward(c, record_dispatch=True)
                c.graph = g
            cap_stream.synchronize()
            warm.append({"gid": c.gid, "warmup_transport": transport,
                         "warmup_s": round(tc - tw, 3), "capture_s": round(time.time() - tc, 3)})
            if c.dispatch != c.expected_dispatch:
                raise RuntimeError(f"dispatch {c.dispatch} != {c.expected_dispatch} "
                                   f"for {c.meta()}")
    torch.cuda.synchronize(device)
    res["capture"] = {"seconds": round(time.time() - t0, 1), "per_graph": warm,
                      "device_slot_selection": bool(state.device_slot_selection),
                      "slot_bias": int(state.slot_bias)}
    res["plans"] = [{"key": [str(k) for k in key], "transport": p.transport,
                     "threads": p.threads, "blocks": p.blocks, "size_packs": p.size_packs,
                     "region_packs": p.remote_push_region_packs}
                    for key, p in state.plain_graph_plans.items()]
    warm_push = sum(c.layers for (s, g, m, t) in host["log"] if t == "push"
                    for c in [cases[g]])
    e_cap = read_epoch()
    res["epoch_after_capture"] = {"value": e_cap, "delta": (e_cap - e_start) & 0xFFFFFFFF,
                                  "expected_push_calls": warm_push}
    acc_warm = {c.gid: int(c.acc.sum()) for c in cases}
    res["warmup_mismatches"] = {g: v for g, v in acc_warm.items() if v}
    for c in cases:
        c.acc.zero_()
    _write(out, "progress.json", {"stage": "captured", **res})
    dist.barrier(group=cpu_group)

    def replay(c: Case):
        if c.cap is not None:
            c.cap.replay()
        else:
            c.graph.replay()
        log(c.gid, "replay", "push" if c.dispatch == "b12x" else c.dispatch)

    phases = []

    def run_phase(name, items):
        e0 = read_epoch()
        dist.barrier(group=cpu_group)
        s0 = host["s"]
        t = time.time()
        push = 0
        for i, item in enumerate(items):
            if name == "mixed":
                c = cases[item]
                replay(c)
                push += c.layers
                prefill_eager()
                log(-1, "eager-prefill", "pynccl")
            else:
                c = cases[item]
                replay(c)
                if c.dispatch == "b12x":
                    push += c.layers
            if (i + 1) % profile["chunk"] == 0:
                wait(args.watchdog_s, f"{name}:{i}")
        wait(args.watchdog_s, f"{name}:end")
        secs = time.time() - t
        e1 = read_epoch()
        phases.append({"phase": name, "items": len(items), "forwards": host["s"] - s0,
                       "seconds": round(secs, 2),
                       "epoch_delta": (e1 - e0) & 0xFFFFFFFF, "expected_push_calls": push})
        _write(out, "progress.json", {"stage": f"phase {name} done", "phases": phases})
        dist.barrier(group=cpu_group)

    plan = res["plan"]
    for name in ("main", "skew", "boundary"):
        run_phase(name, phase_schedule(cases, name, args.seed))
    main_b12x = [c for c in cases if c.phase == "main" and c.expected_dispatch == "b12x"
                 and c.kind == "FULL"]
    mixed = [main_b12x[i % len(main_b12x)].gid for i in range(profile["mixed"])]
    random.Random(f"{args.seed}:mixed").shuffle(mixed)
    run_phase("mixed", mixed)
    run_phase("nccl-ref", phase_schedule(cases, "nccl-ref", args.seed))
    res["phases"] = phases

    # -- collect
    torch.cuda.synchronize(device)
    case_rows = []
    for c in cases:
        acc = c.acc.cpu()
        row = {**c.meta(), "mismatched_elements": int(acc.sum()),
               "layers_with_mismatch": [int(i) for i in torch.nonzero(acc).view(-1)]}
        if row["mismatched_elements"] or acc_warm[c.gid]:
            torch.save({"meta": c.meta(), "acc": acc, "fstep": c.fstep.cpu(),
                        "lstep": c.lstep.cpu(), "fy": c.fy.cpu(), "ly": c.ly.cpu()},
                       os.path.join(out, f"fail_g{c.gid}.pt"))
        case_rows.append(row)
    res["cases"] = case_rows
    res["prefill_mismatched_elements"] = int(pf_acc)
    mbs = mism_by_step.cpu()
    steps = torch.nonzero(mbs).view(-1)
    res["mismatch_steps"] = [[int(s), int(mbs[s])] for s in steps[:10000]]
    # cross-rank bitwise identity of the last outputs and references
    w = torch.arange(Nmax, device=device, dtype=torch.int64) % 65521 + 1
    dig = [int((Y[l].view(torch.int16).to(torch.int64) * w).sum()) for l in range(L)]
    dig_e = [int((E[l].view(torch.int16).to(torch.int64) * w).sum()) for l in range(L)]
    both = [None, None]
    dist.all_gather_object(both, {"Y": dig, "E": dig_e}, group=cpu_group)
    res["cross_rank_last_output_identical"] = both[0]["Y"] == both[1]["Y"]
    res["cross_rank_reference_identical"] = both[0]["E"] == both[1]["E"]
    res["epoch_end"] = read_epoch()
    with open(os.path.join(out, "schedule.jsonl"), "w") as f:
        for row in host["log"]:
            f.write(json.dumps(row) + "\n")
    res["peer"] = peer
    _write(out, "result.json", res)
    dist.barrier(group=cpu_group)
    try:
        comm.destroy()   # coordinated B12x close while the CPU group is alive
    except Exception as exc:  # noqa: BLE001 - teardown only; results already written
        return {"result": "result.json", "teardown_error": repr(exc)}
    return {"result": "result.json"}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--layers", type=int, default=0, help="override all-reduces per graph")
    ap.add_argument("--sleep-cycles", type=int, default=50_000)
    ap.add_argument("--watchdog-s", type=float, default=90.0)
    ap.add_argument("--dry", action="store_true", help="CPU only: print the plan")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args()

    profile = dict(PROFILES[args.profile])
    if args.layers:
        profile["layers"] = args.layers
    cases = build_cases(profile)
    summary = {"profile": args.profile, **plan_summary(profile, cases, args.seed),
               "cases": [c.meta() for c in cases]}
    if args.dry:
        print(json.dumps(summary, indent=1))
        return 0
    bad = {k: os.environ.get(k) for k, v in ENGINE_ENV.items() if os.environ.get(k) != v}
    if bad and not args.allow_env_mismatch:
        print(f"environment differs from B0 serving: {bad}", file=sys.stderr)
        return 2
    if torch.cuda.device_count() < 2:
        print("needs two visible GPUs", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    _write(args.out, "plan.json", summary)
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args, _free_port()), nprocs=2, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
