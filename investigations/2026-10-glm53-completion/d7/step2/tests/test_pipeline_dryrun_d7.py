#!/usr/bin/env python3
"""End-to-end CPU dry run of the B1 guard's logging path and the amended analysis (NO GPU).

B1 variant of the frozen test_pipeline_dryrun.py: the index-K prefill write of every owned layer runs
B1's REAL writer (_kpool_compress_insert with the ring) and the real tail seed on per-layer caches, with
the amended hooks called in the call site's order (pre-write hook with ring status and content inputs,
writer, post-write hook, tail-seed hook, seed); the scheduler budget is 2,045 tokens so chunk starts
are unaligned (straddling pools, as in serving).  PIPELINE TEST ONLY: the fused compress kernel is
replaced by a fast deterministic torch stand-in (same record addressing) in BOTH the writer and the
oracle, because the Triton interpreter is too slow for this scenario; numerics are tested in
test_d7_guard.py.  Expected analysis: no violation, INVALID only for what a CPU run cannot satisfy,
INCOMPLETE coverage, D7 counters flowing into verdict.json (events, positive control).

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


# ---- B1 writer, stand-in compress kernel, per-layer caches (B1 variant)
import vllm.model_executor.layers.sparse_attn_indexer  # noqa: E402,F401
import vllm.models.glm5next.nvidia.ops.kpool_compress as kc  # noqa: E402
from vllm.model_executor.layers import sparse_attn_indexer_kpool as kp  # noqa: E402
from d7lib import make_caches  # noqa: E402


def fast_compress(kv_cache, slot_k, slot_score, ape, loc, pool_size, head_dim=128, write_mask=None,
                  round_scale=True, return_compressed=False, write_cache=True):
    """Deterministic torch stand-in for the fused compress kernel (pipeline dry run only)."""
    if slot_k.shape[0] == 0:
        return None
    w = torch.softmax(slot_score.float() + ape.float()[None], dim=1)
    x = (w * slot_k.float()).sum(1).to(torch.bfloat16).float()
    scale = (x.abs().amax(1).clamp(min=1e-4) / 448.0)
    q = (x / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    rec = torch.cat([q, scale.contiguous().view(torch.uint8).reshape(-1, 4)], 1)
    m = write_mask if write_mask is not None else torch.ones(loc.shape[0], dtype=torch.bool)
    ps, s0 = int(kv_cache.shape[1]), int(kv_cache.stride(0))
    size = (int(kv_cache.shape[0]) - 1) * s0 + ps * int(kv_cache.shape[2])
    flat = torch.as_strided(kv_cache, (size,), (1,), kv_cache.storage_offset())
    page, r = loc // ps, loc % ps
    ko = (page * s0 + r * 128)[:, None] + torch.arange(128)[None]
    so = (page * s0 + ps * 128 + r * 4)[:, None] + torch.arange(4)[None]
    idx = torch.cat([ko, so], 1)[m]
    flat[idx.reshape(-1)] = rec[m].reshape(-1)


kc.kpool_compress_and_write_cache = fast_compress
kp.kpool_compress_and_write_cache = fast_compress
NB_CACHE = 1275
CACHES = {L: make_caches(NB_CACHE, 8) for L in OWNED}          # raw, tail ring view, index-K view
APE = (torch.randn(4, 128, generator=torch.Generator().manual_seed(23)) * 0.1).float()
D7 = dict(prefill_calls=0)


def kg(req_ids, qsl, pos, L, n):
    """Per-token K and gate as a deterministic function of (request, logical position, layer)."""
    rid_h = torch.zeros(n, dtype=torch.float32)
    for i, r in enumerate(req_ids):
        rid_h[int(qsl[i]):int(qsl[i + 1])] = (int(__import__("hashlib").sha1(r.encode()).hexdigest()[:6], 16) % 997) / 97.0
    d = torch.arange(128, dtype=torch.float32)[None]
    p = pos[:n].float()[:, None]
    k = torch.sin(p * 0.37 + rid_h[:, None] * 1.3 + d * 0.11 + L).to(torch.bfloat16)
    g = torch.cos(p * 0.21 + rid_h[:, None] * 0.7 + d * 0.05 - L).to(torch.bfloat16)
    return k, g


emu = Emu(seed=11, concurrency=3, budget=2045)
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
            raw, tail_c, idx_c = CACHES[L]
            ps_ = slice(ndt, ntok)
            k_, g_ = kg(out["req_ids"], out["qsl"], runner.input_buffers.positions, L, ntok)
            posb = runner.input_buffers.positions
            # B1 call site order (hooked sparse_attn_indexer_kpool.py): pre hook, writer, post hook, seed
            mg.hook_idx_prefill_write(name, out["idx_slot"][ps_], posb, ndt, ntok, 4, k=k_[ps_], gate=g_[ps_],
                                      ape=APE, kv_cache=idx_c, round_scale=True, ring_mode="ring",
                                      ring_tslot=out["tail"][ps_])
            kp._kpool_compress_insert(k_[ps_], g_[ps_], APE, idx_c, out["idx_slot"][ps_], 4, 128, round_scale=True,
                                      positions=posb[ps_], tail_kv_cache=tail_c, tail_slot_mapping=out["tail"][ps_])
            mg.hook_idx_prefill_post(name, idx_c)
            mg.hook_tail_seed(name, out["tail"][ps_], posb, ndt, 4)
            kc.kpool_seed_tail_cache(tail_c, k_[ps_].contiguous(), g_[ps_].contiguous(), out["tail"][ps_], 4, 128)
            D7["prefill_calls"] += 1
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
c = mg.S.dev.cnt[mg.S_IDX_W_PF]
print(f"rank {RANK}: steps {mg.S.step} guard_errors {mg.S.errors} prefill_calls {D7['prefill_calls']} "
      + str({n: int(c[mg.CI[n]]) for n in ("s_events", "s_verified", "s_indet", "s_unavail", "s_missing", "s_mism",
                                         "pc_checked", "pc_mism", "d7_fallback", "missing_pred", "pool_cross")}))
