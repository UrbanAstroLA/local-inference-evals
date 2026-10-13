#!/usr/bin/env python3
"""Derived-image smoke (CPU, NO GPU).  Run twice: GLM53_MAPGUARD unset and =device.

  I1  the patched modules import; with the variable unset nothing is patched (inert)
  I2  with GLM53_MAPGUARD=device the worker hooks and the EngineCore scheduler hook install
  I3  scheduler-side truth vs worker ledger: the EngineCore hook (reading the REAL
      KVCacheManager.get_block_ids) and the worker Ledger (rebuilt from the SchedulerOutput
      stream only) produce the same per-step digest and fingerprint over a multi-step scenario
      with prefix hits, completions and row reuse
"""
import glob
import json
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODE = os.environ.get("GLM53_MAPGUARD", "")
OUT = sys.argv[1]
res = []


def result(name, ok, **kw):
    res.append(dict(test=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name + " " + json.dumps(kw, default=str)[:300], flush=True)


import vllm.model_executor.layers.sparse_attn_indexer  # noqa: E402,F401
import vllm.models.glm5next.nvidia.ops.kpool_compress  # noqa: E402,F401
import vllm.model_executor.layers.sparse_attn_indexer_kpool as kp  # noqa: E402
import vllm.v1.attention.backends.mla.b12x_mla_sparse as bm  # noqa: E402
import vllm.v1.core.sched.scheduler as sch  # noqa: E402
from vllm import mapguard_hooks as mg  # noqa: E402
from vllm.v1.worker.gpu import model_runner as mr, states as stm, cudagraph_utils as cg  # noqa: E402

patched = dict(
    runner=getattr(mr.GPUModelRunner, "_mapguard_hooks", False),
    scheduler=getattr(sch.Scheduler, "_mapguard_hooks", False),
    kpool_on=kp._MAPGUARD_ON, b12x_on=bm._MAPGUARD_ON,
    mla_writer_wrapped=bm.B12xMLASparseImpl.do_kv_cache_update.__qualname__.startswith("install."),
    rowstate_wrapped=stm.RequestState.add_request.__qualname__.startswith("install."),
    fullgraph_wrapped=cg.ModelCudaGraphManager.run_fullgraph.__qualname__.startswith("install."),
)
if MODE == "":
    result("I1 inert with GLM53_MAPGUARD unset: nothing patched", not any(patched.values()), **patched)
else:
    result("I2 GLM53_MAPGUARD=device: worker + scheduler hooks installed", all(patched.values()), **patched)
    # I3: scheduler-side digest (real KVCacheManager) == worker ledger digest, step by step
    import b0emu
    from b0emu import Emu
    import random
    emu = Emu(seed=3, concurrency=3)
    A = [700000 + i for i in range(9000)]
    emu.add(A, 5, rid="A")
    emu.add([710000 + i for i in range(5000)], 6, rid="B")
    emu.add(A[:8704] + [720000 + i for i in range(300)], 4, rid="C")
    emu.add([730000 + i for i in range(300)], 6, rid="D")

    class FakeSched:   # the EngineCore Scheduler surface the hook uses
        def __init__(self):
            self.kv_cache_manager = emu.mgr
            self.requests = {}
            self.next = None

        def schedule(self):
            return self.next

        def update_from_output(self, so, mro):
            return None
    os.makedirs(mg.OUT_DIR, exist_ok=True)
    mg.install_scheduler_side(FakeSched)
    fs = FakeSched()
    led = mg.Ledger(mg.Geo(nb=1275, R=16, W0=241))
    rnd = random.Random(1)
    worker = []
    for s in range(40):
        if not (emu.running or emu.waiting):
            break
        so, plan = emu.schedule()
        for st in emu.running:
            fs.requests[st["rid"]] = st["req"]
        fs.next = so
        fs.schedule()                              # EngineCore side logs from the real manager
        led.update(so)                             # worker side from the stream only
        worker.append(dict(fp=mg.fingerprint(so), dig=led.digest(list(so.num_scheduled_tokens))))
        acc = {r: rnd.randint(1, pl["k"] + 1) for r, pl in plan.items()
               if pl["k"] or (pl["st"]["prefill_done"] and not pl["st"].get("first_decode"))}
        emu.complete(plan, acc)
    sched = [json.loads(l) for p in glob.glob(os.path.join(mg.OUT_DIR, "ms_p*.jsonl")) for l in open(p)]
    sched = {r["fp"]: r for r in sched if r.get("kind") == "sched"}
    joined = sum(1 for w in worker if w["fp"] in sched)
    same = sum(1 for w in worker if w["fp"] in sched and sched[w["fp"]]["dig"] == w["dig"])
    result("I3 EngineCore digest (KVCacheManager lists) == worker ledger digest at every step",
           joined == len(worker) and same == joined and joined > 5, steps=len(worker), joined=joined,
           equal=same, excl=[r.get("excl") for r in sched.values() if "excl" in r][:1])
if MODE == "device":
    # I4 (post-a2): execute_model wrapper takes the exact baseline readout once, just before the
    # first real (non-_warmup_) request, and never for runner warm-up outputs
    import torch
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
    geo = mg.Geo(nb=1275, R=16, W0=241)
    mg.S.dev, mg.S.ledger = mg.Dev(geo, torch.device("cpu")), mg.Ledger(geo)

    def so_for(rids):
        return SchedulerOutput(scheduled_new_reqs=[], scheduled_cached_reqs=CachedRequestData.make_empty(),
                               num_scheduled_tokens={r: 1 for r in rids}, total_num_scheduled_tokens=len(rids),
                               scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
                               num_common_prefix_blocks=[], finished_req_ids=set(), free_encoder_mm_hashes=[])
    base_before = sum(1 for p in glob.glob(os.path.join(mg.OUT_DIR, "mg_p*.jsonl")) for l in open(p) if '"baseline"' in l)
    seen = []
    for rids in (["_warmup_0_"], ["_warmup_1_"], ["cmpl-x-0"], ["cmpl-y-0"]):
        try:
            mr.GPUModelRunner.execute_model(object(), so_for(rids))
        except Exception:  # noqa: BLE001  (the real execute_model cannot run on a fake runner)
            pass
        n = sum(1 for p in glob.glob(os.path.join(mg.OUT_DIR, "mg_p*.jsonl")) for l in open(p) if '"baseline"' in l)
        seen.append(n - base_before)
    errs = sum(1 for p in glob.glob(os.path.join(mg.OUT_DIR, "mg_p*.jsonl")) for l in open(p) if '"kind": "guard_error"' in l and "baseline" in l)
    result("I4 exact baseline readout: none during runner warm-up, exactly one at the first real request",
           seen == [0, 0, 1, 1] and errs == 0, baseline_records_after_each_call=seen, baseline_errors=errs)
json.dump(dict(mode=MODE or "unset", results=res, module_sha256=__import__("hashlib").sha256(
    open(mg.__file__, "rb").read()).hexdigest()), open(OUT, "w"), indent=1, default=str)
sys.exit(0 if all(r["ok"] for r in res) else 1)
