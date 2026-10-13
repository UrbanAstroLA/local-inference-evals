#!/usr/bin/env python3
"""End-to-end CPU dry run of the guard's logging path and the frozen analysis (NO GPU).

usage (inside the derived image, once per rank, then the analysis):
  test_pipeline_dryrun.py RUN_DIR RANK      (RANK 0 also runs the EngineCore-side hook)
  analyze_mapguard.py RUN_DIR

Drives the guard's REAL _register_runner / _upload / execute-model epilogue (disarm + step log)
/ poll() code paths with a minimal fake runner (B0 geometry), the emulated scheduler/runner of
b0emu.py and the real hook functions for every owned MLA layer of the rank, writing the same
JSONL files the GPU session writes.  Expected analysis result: no violation, no INVALID item
other than those a CPU run cannot satisfy (no FULL graph capture in server.log), and coverage
INCOMPLETE (the scenario is tiny).
"""
import json
import os
import random
import sys
import time
import types

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
RUN, RANK = sys.argv[1], int(sys.argv[2])
os.makedirs(os.path.join(RUN, "mapguard"), exist_ok=True)
os.environ["GLM53_MAPGUARD"] = "device"
os.environ["GLM53_MAPGUARD_DIR"] = os.path.join(RUN, "mapguard")
os.environ["VLLM_GLM53_MLA_OWNERS"] = "split:25"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import vllm.distributed as vd  # noqa: E402
vd.get_tensor_model_parallel_rank = lambda: RANK
from vllm import mapguard_hooks as mg  # noqa: E402
import b0emu  # noqa: E402
from b0emu import Emu  # noqa: E402

MLA = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
OWNED = [L for L in MLA if int(L >= 25) == RANK]


from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl  # noqa: E402


class Model:
    """named_modules() yields B12x MLA impls exactly where the owned layers live."""

    def __init__(self):
        self.impls = {L: object.__new__(B12xMLASparseImpl) for L in OWNED}

    def named_modules(self):
        return [(f"model.layers.{L}.self_attn.mla_attn", types.SimpleNamespace(impl=i))
                for L, i in self.impls.items()]


def fake_runner():
    bt = types.SimpleNamespace(
        block_sizes=[4352, 8] + [4352] * 7 + [1088], kernel_block_sizes=[64, 8] + [4352] * 7 + [1088],
        blocks_per_kv_block=[68, 1] + [1] * 8,
        block_tables=[types.SimpleNamespace(gpu=torch.zeros(1, 241 * 68, dtype=torch.int32)),
                      types.SimpleNamespace(gpu=torch.zeros(1, 1, dtype=torch.int32))])
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs
    mla = MLAAttentionSpec(block_size=4352, num_kv_heads=1, head_size=512, dtype=torch.uint8,
                           cache_dtype_str="fp8_ds_mla")
    idx = MLAAttentionSpec(block_size=4352, num_kv_heads=1, head_size=132, dtype=torch.uint8,
                           compress_ratio=4)
    groups = [types.SimpleNamespace(kv_cache_spec=types.SimpleNamespace()) for _ in range(10)]
    groups[0] = types.SimpleNamespace(kv_cache_spec=UniformTypeKVCacheSpecs.from_specs(
        {"mla.3": mla, "idx.3": idx}))       # as sim_pool.build_config (B0 group 0)
    hf = types.SimpleNamespace(num_hidden_layers=46, is_kda_layer=lambda i: i not in MLA)
    return types.SimpleNamespace(
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=groups, num_blocks=1275),
        block_tables=bt, max_num_reqs=16, device=torch.device("cpu"),
        input_buffers=types.SimpleNamespace(positions=torch.zeros(4096, dtype=torch.int64),
                                            query_start_loc=torch.zeros(17, dtype=torch.int32)),
        model=Model(), adaptive_verification=None,
        scheduler_config=types.SimpleNamespace(async_scheduling=True),
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(hf_text_config=hf)))


runner = fake_runner()
mg._register_runner(runner)
assert mg.S.owned_layers == OWNED, (mg.S.owned_layers, OWNED)


class FakeSched:
    def __init__(self, emu):
        self.kv_cache_manager = emu.mgr
        self.requests = {}
        self.next = None

    def schedule(self):
        return self.next

    def update_from_output(self, so, mro):
        return None


emu = Emu(seed=11, concurrency=3)
A = [700000 + i for i in range(9000)]
emu.add(A, 6, rid="A")
emu.add([710000 + i for i in range(5000)], 8, rid="B")
emu.add(A[:8704] + [720000 + i for i in range(300)], 5, rid="C")
emu.add([730000 + i for i in range(300)], 8, rid="D")
emu.add([740000 + i for i in range(4400)], 5, rid="E")
fs = None
if RANK == 0:
    mg.install_scheduler_side(FakeSched)
    fs = FakeSched(emu)
rnd = random.Random(2)
client = []
mg.poll("baseline")
time.sleep(0.05)
t_first = None
for s in range(60):
    if not (emu.running or emu.waiting):
        break
    so, plan = emu.schedule()
    if fs is not None:
        for st in emu.running:
            fs.requests[st["rid"]] = st["req"]
        fs.next = so
        fs.schedule()
    if not so.num_scheduled_tokens:
        emu.complete(plan, {})
        continue
    if t_first is None:
        t_first = time.time()
        time.sleep(0.01)
    # execute_model wrapper, part 1 (ledger from the SchedulerOutput)
    mg.S.cur_fp = mg.fingerprint(so)
    mg.S.ledger.update(so)
    mg.S.mode = "none"
    out = emu.runner_step(so)
    runner.input_buffers.positions[:out["pos"].shape[0]] = out["pos"]
    runner.input_buffers.query_start_loc.copy_(out["qsl"][:17])
    ib = types.SimpleNamespace(req_ids=out["req_ids"], num_scheduled_tokens=out["ns"],
                               num_tokens=out["ntok"], num_tokens_after_padding=out["ntok_pad"],
                               num_reqs_after_padding=out["nreq_pad"], idx_mapping_np=out["idx"])
    mg._upload(runner, ib)                       # prepare_attn wrapper
    mg.S.mode = "FULL" if out["nd"] == out["nreq"] else "PIECEWISE"
    ntok, ndt = out["ntok"], out["ndt"]
    for L in OWNED:                               # the forward: every owned layer's hooks
        mg.hook_mla_write(runner.model.impls[L], torch.ones(1), out["slot0"])
        name = f"model.layers.{L}.self_attn.indexer.k_cache"
        if out["nd"] < out["nreq"]:
            mg.hook_idx_prefill_write(name, out["idx_slot"][ndt:ntok], runner.input_buffers.positions, ndt, ntok, 4)
            mg.hook_tail_seed(name, out["tail"][ndt:ntok], runner.input_buffers.positions, ndt, 4)
            mg.hook_idx_prefill_read(name, out["pread"], runner.input_buffers.positions)
        if out["nd"]:
            ds, dp, dt = out["dec"]
            mg.hook_idx_decode_write(name, ds, dp, dt, runner.input_buffers.positions, ndt)
            if "dread" in out:
                rt, rl = out["dread"]
                mg.hook_idx_decode_read(name, rt, rl, runner.input_buffers.positions, int(rt.shape[0]))
        lyr = types.SimpleNamespace(layer_name=f"model.layers.{L}.self_attn.mla_attn")
        mg.hook_mla_read(lyr, out["sel"][0], out["sel"][1], out["ntok_pad"])
    # execute_model wrapper, part 2 (disarm after the forward, step log)
    mg.S.dev.E[mg.H_ARMED].zero_()
    mg.S.armed_host = False
    mg.S.pending["cg"] = mg.S.mode
    mg._log(mg.S.pending)
    mg.S.pending = None
    acc = {r: rnd.randint(1, pl["k"] + 1) for r, pl in plan.items()
           if pl["k"] or (pl["st"]["prefill_done"] and not pl["st"].get("first_decode"))}
    emu.commit(out, acc, plan)
    rej = emu.complete(plan, acc)
    if fs is not None:       # EngineCore update_from_output hook: sampled = 1 + accepted drafts
        rids = list(so.num_scheduled_tokens)
        mro = types.SimpleNamespace(req_id_to_index={r: i for i, r in enumerate(rids)},
                                    sampled_token_ids=[[0] * acc.get(r, 1) for r in rids])
        fs.update_from_output(so, mro)
    if s % 5 == 4:
        mg.poll("periodic")
time.sleep(0.05)
mg.poll("final")
if RANK == 0:
    with open(os.path.join(RUN, "client.jsonl"), "w") as f:
        f.write(json.dumps(dict(kind="request", phase="dry", id="A", t0=t_first, t1=time.time(),
                                finish="length", prompt_tokens=9000, error=None)) + "\n")
        f.write(json.dumps(dict(kind="request", phase="dry", id="D", t0=t_first, t1=time.time(),
                                finish="length", prompt_tokens=300, error=None)) + "\n")
    with open(os.path.join(RUN, "server.log"), "w") as f:
        f.write("(CPU dry run: no server)\n")
print(f"rank {RANK}: steps {mg.S.step} guard_errors {mg.S.errors}")
