#!/usr/bin/env python3
"""Draft-slot CPU reproduction (recon 2026-10-22). Runs INSIDE the stock 0.9.1 image (a9e985c0) as a CPU container:
--network none, no GPU, TRITON_INTERPRET=1. Read-only mounts; writes only to /out.

What is real (the image's own code):
  * KVCacheManager / BlockPool / sliding-window manager with the served 10-group geometry, draft group =
    SlidingWindowSpec(block 1088, window 2048, page padded to the MLA page) (b0-mapguard/reach/sim_pool.build_config,
    shown byte-identical between 0.9.1 and B0 by the earlier reachability work);
  * the drafter's slot-mapping kernel _prepare_dflash_inputs_kernel (dflash/speculator.py, identical in every release).
What is emulated (stated): the scheduler loop (FCFS, chunked prefill 2048, K=3 drafts per decode step, lookahead K+1
as config/vllm.py:616-620), acceptance draws, async optimistic num_computed with rollback after the next schedule,
and the runner's persistent block-table rows (append-only per request row, rows reused across requests; the
scheduler's sliding-window frees are NOT propagated to the runner row, as in the runner).

Invariants checked on every kernel output row (all statements are about this harness's exercised states only):
  I1  every non-PAD context/query slot lies in a block the SAME request holds in the draft group now, and not block 0
  I2  that block is not held by any other request (any group) and not by the same request in a non-draft group
      (with draft-slot sharing a block id names the same bytes in an MLA layer tensor)
  I3  every rejected context row's slot is rewritten by a query row of the same step (overwrite coverage)
  I4  every query position is inside the request's allocated draft-group horizon
Minimal-patch control: the same kernel with an upstream-style guard (rejected context rows and block-0 lookups ->
PAD_SLOT_ID; cf. upstream 97388c44f) on the same scenarios.
Positive controls (planted runner/scheduler faults; must be DETECTED): lookahead K instead of K+1; a frontier
block-table entry replaced by another live request's block; a missed block-table append.
"""
import inspect, json, os, random, sys, textwrap, types

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/inv/b0-mapguard/reach")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from sim_pool import build_config  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import get_hash_fn_by_name  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.request import Request  # noqa: E402
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as spec  # noqa: E402

K = 3
QPR = K + 1
BS = 1088
DRAFT = 9          # group index of the draft group in build_config()
PAD = -1
R_ROWS = 16
W = 1024           # runner row width (draft blocks); > 327,680 / 1088
MAXLEN = 1_048_576
PAD_SLOT_ID = getattr(spec, "PAD_SLOT_ID", -1)

# ---------------------------------------------------------------- kernels: stock and minimal-patch control
STOCK = spec._prepare_dflash_inputs_kernel
src = inspect.getsource(STOCK.fn)
src = textwrap.dedent(src[src.index("def "):])
patched = src.replace(
    "ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)",
    "ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)\n"
    "    ctx_ok = (j < (num_ctx - num_rejected)) & (ctx_block_id != 0)\n"
    "    ctx_slot = tl.where(ctx_ok, ctx_slot, PAD_SLOT_ID)").replace(
    "q_slot = q_block_id * block_size + (query_pos % block_size)",
    "q_slot = q_block_id * block_size + (query_pos % block_size)\n"
    "    q_slot = tl.where(q_block_id != 0, q_slot, PAD_SLOT_ID)")
assert patched != src and patched.count("PAD_SLOT_ID, ctx") == 0
ns = {"tl": tl, "triton": triton}
exec(compile(patched.replace("def _prepare_dflash_inputs_kernel", "def _patched_kernel"), "<patched>", "exec"), ns)
PATCHED = triton.jit(ns["_patched_kernel"])
PATCH_DIFF_LINES = [l for l in patched.splitlines() if l not in src.splitlines()]


def run_kernel(kern, batch, table):
    """batch: list of dict(pos=[...], num_sampled, num_rejected, row). Returns per-request ctx and query slots."""
    n = len(batch)
    positions = torch.tensor([p for b in batch for p in b["pos"]], dtype=torch.int64)
    qsl = [0]
    for b in batch:
        qsl.append(qsl[-1] + len(b["pos"]))
    T = qsl[-1]
    maxq = max(len(b["pos"]) for b in batch)
    BLOCK = min(256, triton.next_power_of_2(max(1, maxq + QPR)))
    nblk = triton.cdiv(maxq + QPR, BLOCK)
    max_num_reqs, buf_tokens = R_ROWS, 4096
    max_num_tokens = R_ROWS * QPR   # scalar bound of the query-slot pad loop only (served value is larger; same semantics)
    i64 = lambda m: torch.zeros(m, dtype=torch.int64)
    out = dict(ids=torch.zeros(buf_tokens, dtype=torch.int32), qpos=i64(buf_tokens),
               qsl=torch.zeros(max_num_reqs + 1, dtype=torch.int32), seql=torch.zeros(max_num_reqs, dtype=torch.int32),
               qslot=i64(buf_tokens), cpos=i64(buf_tokens), cslot=i64(buf_tokens),
               sidx=i64(max_num_reqs * K), spos=i64(max_num_reqs * K), smap=torch.zeros(max_num_reqs * K, dtype=torch.int32),
               temp=torch.zeros(max_num_reqs), seeds=i64(max_num_reqs))
    kern[(n, nblk)](
        out["ids"], out["qpos"], out["qsl"], out["seql"], out["qslot"], out["cpos"], out["cslot"],
        out["sidx"], out["spos"], out["smap"], out["temp"], out["seeds"],
        positions, torch.tensor(qsl, dtype=torch.int32), torch.tensor([b["row"] for b in batch], dtype=torch.int32),
        torch.full((max_num_reqs,), 7, dtype=torch.int64), torch.full((max_num_reqs,), 9, dtype=torch.int32),
        torch.tensor([b["num_sampled"] for b in batch], dtype=torch.int32),
        torch.tensor([b["num_rejected"] for b in batch], dtype=torch.int32),
        torch.ones(max_num_reqs), torch.zeros(max_num_reqs, dtype=torch.int64),
        table, table.stride(0),
        5, BS, QPR, K, max_num_reqs, max_num_tokens, MAXLEN,
        SAMPLE_FROM_ANCHOR=False, PAD_SLOT_ID=PAD_SLOT_ID, BLOCK_SIZE=BLOCK)
    res = []
    for i, b in enumerate(batch):
        cs = out["cslot"][qsl[i]:qsl[i + 1]].tolist()
        qs = out["qslot"][i * QPR:(i + 1) * QPR].tolist()
        qp = out["qpos"][i * QPR:(i + 1) * QPR].tolist()
        res.append(dict(ctx=cs, q=qs, qpos=qp))
    return res


# ---------------------------------------------------------------- emulated scheduler + runner
class Sim:
    def __init__(self, seed, conc, lookahead=QPR, async_sched=True, precycle=True):
        self.rng = random.Random(seed)
        cfg, _ = build_config()
        hf = get_hash_fn_by_name("sha256")
        init_none_hash(hf)
        self.mgr = KVCacheManager(cfg, max_model_len=MAXLEN, scheduler_block_size=4352, hash_block_size=1088,
                                  max_in_flight_tokens=2048, enable_caching=True, use_eagle=True)
        self.hasher = get_request_block_hasher(1088, hf)
        self.conc, self.look, self.async_sched = conc, lookahead, async_sched
        self.waiting, self.running = [], []
        self.free_rows = list(range(R_ROWS))[::-1]
        self.rows = torch.zeros(R_ROWS, W, dtype=torch.int32)    # runner rows: never cleared (stale tails)
        self.nrow = [0] * R_ROWS
        self.n = 0
        self.cov = {"prefix_hit_requests": 0, "row_reuse": 0}
        self.used_rows = set()
        if precycle:
            k = 0
            while True:
                ids = [300000 + 7919 * k + i for i in range(4352 * 6)]
                req = Request(f"pre{k}", ids, SamplingParams(max_tokens=1), None, block_hasher=self.hasher)
                self.mgr.new_step_starts()
                m = 0
                while m < len(ids):
                    c = min(2048, len(ids) - m)
                    assert self.mgr.allocate_slots(req, c, num_lookahead_tokens=QPR) is not None
                    req.num_computed_tokens += c
                    m += c
                mx = max(max(g) if g else 0 for g in self.mgr.get_block_ids(req.request_id))
                self.mgr.free(req)
                k += 1
                if mx >= 960:
                    break

    def add(self, prompt, gen, arrive=0):
        rid = f"r{self.n}"
        self.n += 1
        req = Request(rid, list(prompt), SamplingParams(max_tokens=gen), None, block_hasher=self.hasher)
        self.waiting.append(dict(rid=rid, req=req, gen=gen, done=0, pre=False, row=None, committed=0, seen=0,
                                 pending_rej=0, started=False, arrive=arrive))

    def owners(self):
        own = {}
        for st in self.running:
            for g, ids in enumerate(self.mgr.get_block_ids(st["rid"])):
                for b in ids:
                    if b != 0:
                        own.setdefault(b, set()).add((st["rid"], g))
        return own

    def step(self, accept_draw, faults=frozenset()):
        self.mgr.new_step_starts()
        self.tick = getattr(self, "tick", 0) + 1
        while self.waiting and len(self.running) < self.conc and self.waiting[0]["arrive"] <= self.tick:
            st = self.waiting.pop(0)
            st["row"] = self.free_rows.pop()
            self.cov["row_reuse"] += st["row"] in self.used_rows
            self.used_rows.add(st["row"])
            self.nrow[st["row"]] = 0
            self.running.append(st)
        budget = 2048
        batch, meta = [], []
        prev_rej = {st["rid"]: st.get("pending_rej", 0) for st in self.running}
        for st in sorted(self.running, key=lambda s: s["pre"]):
            req = st["req"]
            if not st["pre"]:
                first = not st["started"]
                comp, nhit = self.mgr.get_computed_blocks(req)[:2] if first else (None, 0)
                if first and nhit:
                    self.cov["prefix_hit_requests"] += 1
                rem = req.num_prompt_tokens - req.num_computed_tokens - nhit
                c = min(rem, budget)
                if c <= 0:
                    continue
                look = self.look - (1 if "lookahead_K" in faults else 0)
                b = self.mgr.allocate_slots(req, c, num_new_computed_tokens=nhit, new_computed_blocks=comp,
                                            num_lookahead_tokens=look)
                if b is None:
                    continue
                st["started"] = True
                start = req.num_computed_tokens + nhit
                req.num_computed_tokens = start + c
                st["committed"] = start + c
                pos = list(range(start, start + c))
                last = req.num_computed_tokens >= req.num_prompt_tokens
                ns, nr = (1 if last else 0), 0
                if last:
                    st["pre"] = True
                kind = "prefill"
            else:
                c = 1 + K
                if budget < c:
                    continue
                look = self.look - (1 if "lookahead_K" in faults else 0)
                b = self.mgr.allocate_slots(req, c, num_lookahead_tokens=look)
                if b is None:
                    continue
                base = st["committed"]           # runner-side committed length (true positions)
                req.num_computed_tokens = req.num_computed_tokens + c if self.async_sched else base + c
                pos = list(range(base, base + c))
                a = accept_draw(self.rng)
                ns, nr = a + 1, K - a
                st["committed"] = base + a + 1
                st["new_rej"] = nr
                kind = "decode"
            budget -= c
            # runner row: append-only (sliding-window frees are not propagated to the runner)
            ids = self.mgr.get_block_ids(st["rid"])[DRAFT]
            row = st["row"]
            have = self.nrow[row]
            new = list(ids[have:])
            if "missed_append" in faults and kind == "decode" and new and have not in st.setdefault("skipped", set()):
                st["skipped"].add(have)      # skip each new block's first append exactly once (deterministic plant)
                new = []
            if new:
                self.rows[row, have:have + len(new)] = torch.tensor(new, dtype=torch.int32)
                self.nrow[row] = have + len(new)
            batch.append(dict(pos=pos, num_sampled=ns, num_rejected=nr, row=row))
            meta.append(dict(st=st, kind=kind, pos=pos, nr=nr))
        if not batch:
            return None
        # async: the PREVIOUS step's rejections are rolled back only after this schedule (allocation above used the
        # optimistic count); sync: the scheduler sees the committed length
        for st in self.running:
            if self.async_sched:
                st["req"].num_computed_tokens -= prev_rej.get(st["rid"], 0)
                st["pending_rej"] = st.pop("new_rej", 0)
            else:
                st["req"].num_computed_tokens = st["committed"]
                st.pop("new_rej", None)
        table = torch.stack([self.rows[b["row"]] for b in batch])
        if "foreign_entry" in faults:
            own = self.owners()
            for i, m in enumerate(meta):
                if m["kind"] == "decode" and self.rng.random() < 0.3:
                    others = [blk for blk, s in own.items() if all(r != m["st"]["rid"] for r, g in s)]
                    if others:
                        table[i, m["pos"][-1] // BS] = self.rng.choice(others)
        return batch, meta, table

    def finish_step(self, meta):
        for m in meta:
            st = m["st"]
            if m["kind"] == "decode":
                st["done"] = st["committed"] - st["req"].num_prompt_tokens
            if st["pre"] and st["done"] >= st["gen"]:
                self.running.remove(st)
                self.mgr.free(st["req"])
                self.free_rows.append(st["row"])


def check(sim, meta, res, table):
    own = sim.owners()
    v = dict(I1_not_own_draft=0, I1_block0=0, I2_shared_other=0, I3_rejected_not_rewritten=0, I4_beyond_horizon=0)
    n = dict(rows=0, ctx_slots=0, q_slots=0, rejected_rows=0, pad=0, rows_with_window_nulls=0, query_rows_crossing_block=0,
             slots_in_high_blocks=0)
    ex = []
    for m, r in zip(meta, res):
        rid = m["st"]["rid"]
        mine = set(b for b in sim.mgr.get_block_ids(rid)[DRAFT] if b != 0)
        dl = sim.mgr.get_block_ids(rid)[DRAFT]
        horizon = len(dl) * BS
        n["rows"] += 1
        n["rows_with_window_nulls"] += any(b == 0 for b in dl)
        n["query_rows_crossing_block"] += (r["qpos"][0] // BS) != (r["qpos"][-1] // BS)
        slots = [("ctx", s, p) for s, p in zip(r["ctx"], m["pos"])] + [("q", s, p) for s, p in zip(r["q"], r["qpos"])]
        for kind, s, p in slots:
            n["ctx_slots" if kind == "ctx" else "q_slots"] += 1
            if s == PAD_SLOT_ID:
                n["pad"] += 1
                continue
            blk = s // BS
            n["slots_in_high_blocks"] += blk >= 935
            if blk == 0:
                v["I1_block0"] += 1
            elif blk not in mine:
                v["I1_not_own_draft"] += 1
                if len(ex) < 5:
                    ex.append(dict(rid=rid, kind=kind, pos=p, blk=blk, owners=sorted(map(str, own.get(blk, set())))))
            holders = own.get(blk, set())
            if any(r_ != rid or g != DRAFT for r_, g in holders):
                v["I2_shared_other"] += 1
        for p in r["qpos"]:
            if p >= horizon:
                v["I4_beyond_horizon"] += 1
        if m["nr"]:
            n["rejected_rows"] += m["nr"]
            rej = set(r["ctx"][len(m["pos"]) - m["nr"]:])
            if any(s != PAD_SLOT_ID and s not in set(r["q"]) for s in rej):
                v["I3_rejected_not_rewritten"] += 1
    return v, n, ex


def scenario(name, seed, conc, prompts, gens, accept, faults=frozenset(), async_sched=True, kernels=("stock", "patched"),
             arrivals=None):
    sims = {k: Sim(seed, conc, async_sched=async_sched) for k in kernels}
    out = {}
    for k, sim in sims.items():
        for i, (p, g) in enumerate(zip(prompts, gens)):
            sim.add(p, g, arrive=(arrivals[i] if arrivals else 0))
        tot_v, tot_n, exs, steps, slots_by_step = {}, {}, [], 0, []
        while sim.running or sim.waiting:
            st = sim.step(accept, faults)
            if st is None:
                break
            batch, meta, table = st
            res = run_kernel(STOCK if k == "stock" else PATCHED, batch, table)
            if "drop_bonus_query" in faults:
                for m_, r_ in zip(meta, res):
                    if m_["nr"]:
                        r_["q"][0] = PAD_SLOT_ID
            v, n, ex = check(sim, meta, res, table)
            for d, s in ((tot_v, v), (tot_n, n)):
                for a, b in s.items():
                    d[a] = d.get(a, 0) + b
            exs += ex[: max(0, 5 - len(exs))]
            slots_by_step.append([(r["ctx"][:len(m["pos"]) - m["nr"]], r["q"]) for m, r in zip(meta, res)])
            sim.finish_step(meta)
            steps += 1
            if steps % 250 == 0:
                print(f'  {name} [{k}] step {steps}', flush=True)
        out[k] = dict(steps=steps, counts={**tot_n, **sim.cov}, violations=tot_v, examples=exs, _slots=slots_by_step)
    if "stock" in out and "patched" in out:
        same = sum(1 for a, b in zip(out["stock"]["_slots"], out["patched"]["_slots"]) if a == b)
        out["valid_rows_identical_steps"] = f"{same}/{len(out['stock']['_slots'])}"
    for k in kernels:
        out[k].pop("_slots")
    return {"scenario": name, "seed": seed, "concurrency": conc, "faults": sorted(faults), "async": async_sched, **out}


if __name__ == "__main__":
    OUT = sys.argv[1] if len(sys.argv) > 1 else "/out/result.json"
    quick = os.environ.get("QUICK") == "1"
    rng = random.Random(7)
    gen_long = 900 if quick else 5000
    tmpl = [11 + (i % 97) for i in range(150)]
    accept_mix = lambda r: r.choices(range(K + 1), weights=[0.25, 0.2, 0.2, 0.35])[0]
    accept_all = lambda r: K
    accept_none = lambda r: 0
    results = []
    gp = lambda n, lo, hi: [tmpl + [rng.randrange(1000, 150000) for _ in range(rng.randrange(lo, hi))] for _ in range(n)]
    long_prefix = [rng.randrange(1000, 150000) for _ in range(9000 if not quick else 4400)]
    S = [
        ("gpqa-like c8 mixed acceptance", 1, 8, gp(12, 30, 400), [rng.randrange(200, gen_long) for _ in range(12)], accept_mix),
        ("all drafts accepted (query reaches lv+1+K)", 2, 4, gp(4, 30, 400), [gen_long] * 4, accept_all),
        ("all drafts rejected", 3, 4, gp(4, 30, 400), [gen_long // 2] * 4, accept_none),
        ("multi-chunk prefill then decode", 4, 2, [[rng.randrange(1000, 150000) for _ in range(7000)] for _ in range(2)], [gen_long // 2] * 2, accept_mix),
        ("shared long prefix x4, staggered (prefix-cache hits, as B+/E)", 5, 4,
         [list(long_prefix) + [rng.randrange(1000, 150000) for _ in range(50)] for _ in range(4)], [gen_long // 2] * 4, accept_mix,
         [0, 12, 24, 36]),
        ("concurrency 1, block-boundary crossings", 6, 1, gp(2, 30, 400), [gen_long] * 2, accept_mix),
    ]
    for row in S:
        name, seed, conc, prompts, gens, acc = row[:6]
        arr = row[6] if len(row) > 6 else None
        results.append(scenario(name, seed, conc, prompts, gens, acc, arrivals=arr))
        results.append(scenario(name + " [sync scheduler]", seed, conc, prompts, gens, acc, async_sched=False, kernels=("stock",), arrivals=arr))
        print(json.dumps({k: results[-2][k] for k in ("scenario",)}), results[-2]["stock"]["violations"], flush=True)
    # positive controls: planted faults must be detected by the invariant checker on the stock kernel
    P = [("PLANT lookahead K instead of K+1", frozenset({"lookahead_K"}), accept_all),
         ("PLANT foreign block-table entry at the frontier", frozenset({"foreign_entry"}), accept_mix),
         ("PLANT missed block-table append", frozenset({"missed_append"}), accept_mix)]
    P += [("PLANT drop the bonus query slot when drafts were rejected (checker control for I3)", frozenset({"drop_bonus_query"}), accept_mix)]
    for name, f, acc in P:
        results.append(scenario(name, 11, 4, gp(4, 30, 400), [gen_long] * 4, acc, faults=f, kernels=("stock",)))
        print(name, results[-1]["stock"]["violations"], flush=True)
        if "lookahead_K" in f:
            results.append(scenario(name + " [sync scheduler]", 11, 4, gp(4, 30, 400), [gen_long] * 4, acc, faults=f,
                                    async_sched=False, kernels=("stock",)))
            print(name + " [sync scheduler]", results[-1]["stock"]["violations"], flush=True)
    json.dump({"image_note": "run inside the stock 0.9.1 image; see run.sh", "K": K, "block_size": BS,
               "patch_lines": PATCH_DIFF_LINES, "results": results}, open(OUT, "w"), indent=1)
    print("written", OUT)
