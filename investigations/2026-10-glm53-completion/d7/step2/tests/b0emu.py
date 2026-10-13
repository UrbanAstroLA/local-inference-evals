# CPU emulation of B0's scheduler -> runner -> mapping-builder path for the mapguard tests.
#
# * Block ids come from the REAL B0 KVCacheManager (10-group B0 scheduler config,
#   b0-cache/reachability/sim_pool.py:build_config), so prefix hits, LIFO/FIFO reuse and
#   high block ids are the real allocator's.
# * SchedulerOutputs are the REAL vllm dataclasses; async-scheduling semantics are emulated
#   (num_computed is optimistic by the previous step's drafts; rejections roll back after
#   the next schedule, scheduler.py:1836-1856).
# * The runner side uses B0's REAL kernels/functions under TRITON_INTERPRET=1:
#   _prepare_pos_seq_lens_kernel, _gather_block_tables_kernel, _compute_slot_mappings_kernel,
#   get_compressed_slot_mapping, compute_kpool_tail_slot_mapping, _prepare_uniform_decode_kernel,
#   _scatter_decode_tokens_by_request, _logical_topk_to_physical_slots,
#   _compact_dropped_selection, _mask_page_table_after_nsa_len, sort_batch_req_ids.
#   Persistent tables follow BlockTables.append_block_ids (kernel-block expansion x68 for the
#   attention group, overwrite for new requests, append otherwise); rows follow RequestState
#   (free_indices.pop() / append).
# * Faults are injected only into the runner emulation (never into the guard's inputs from
#   the scheduler), mirroring where the real defect classes would live.
import os
import random
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reach"))

from sim_pool import build_config  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import get_hash_fn_by_name  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402
from vllm.v1.worker.gpu.block_table import (  # noqa: E402
    _compute_slot_mappings_kernel, _gather_block_tables_kernel)
from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens  # noqa: E402
from vllm.v1.worker.gpu.model_runner import sort_batch_req_ids  # noqa: E402
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping  # noqa: E402
from vllm.v1.attention.backends.mla.indexer import (  # noqa: E402
    _prepare_uniform_decode_kernel, compute_kpool_tail_slot_mapping)
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (  # noqa: E402
    _compact_dropped_selection, _logical_topk_to_physical_slots, _mask_page_table_after_nsa_len)
import vllm.model_executor.layers.sparse_attn_indexer  # noqa: E402,F401  (import order)
import vllm.models.glm5next.nvidia.ops.kpool_compress  # noqa: E402,F401  (import order)
from vllm.model_executor.layers.sparse_attn_indexer_kpool import (  # noqa: E402
    _build_decode_scatter_indices, _scatter_decode_tokens_by_request)

BS, KBS, BPK, KPOOL, PPB, RING = 4352, 64, 68, 4, 1088, 8


def _indexer_builder_geometry():
    """The indexer builder's table factor and storage block, derived with B0's REAL code
    (post-inspection fix after run-a2; the a2 emulation hard-coded factor 68 / storage 1088,
    copying the guard's own assumption):  AttentionGroup.create_metadata_builders
    (worker/utils.py:302-311) gives the builder spec block = compressed_kernel_block_size(spec)
    * compress_ratio; DeepseekV32IndexerMetadataBuilder.build translates the shared 64-token
    table with factor = spec.block_size // kernel_block_size (indexer.py:1089-1096)."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.utils import compressed_kernel_block_size
    idx = MLAAttentionSpec(block_size=BS, num_kv_heads=1, head_size=132, dtype=torch.uint8,
                           compress_ratio=KPOOL)
    cr = idx.block_size // idx.storage_block_size
    spec_b = idx.copy_with_new_block_size(compressed_kernel_block_size(idx) * cr)
    return spec_b.block_size // KBS, spec_b.storage_block_size


IDX_FACTOR, IDX_STORAGE = _indexer_builder_geometry()   # 4, 64 on B0
NB = 1275
R = 16
W0 = 241
PAD = -1
NSPEC = 3
TOPK = 128   # MLA read width in the emulation (the guard is width-agnostic)


class Fault:
    def __init__(self, kind, step, **kw):
        self.kind, self.step, self.kw = kind, step, kw


class Emu:
    """One scenario: mini scheduler (real allocator) + runner emulation (real kernels)."""

    def __init__(self, seed=0, budget=2048, concurrency=4, precycle=True):
        self.rng = random.Random(seed)
        cfg, self.names = build_config()
        hash_fn = get_hash_fn_by_name("sha256")
        init_none_hash(hash_fn)
        self.mgr = KVCacheManager(cfg, max_model_len=1_048_576, scheduler_block_size=BS,
                                  hash_block_size=1088, max_in_flight_tokens=2048,
                                  enable_caching=True, use_eagle=True)
        self.hasher = get_request_block_hasher(1088, hash_fn)
        self.budget = budget
        self.concurrency = concurrency
        self.waiting, self.running = [], []
        self.finished_pending = set()
        self.sched_step = 0
        self.prev_drafts = {}           # rid -> drafts scheduled in the previous step
        self.pending_rej = {}           # rid -> rejected count to apply after next schedule
        # runner state (B0 semantics)
        self.free_rows = list(range(R))
        self.row_of = {}
        self.tab = [torch.zeros(R, W0 * BPK, dtype=torch.int32), torch.zeros(R, 4, dtype=torch.int32)]
        self.nblk = torch.zeros(2, R, dtype=torch.int32)
        self.committed = torch.zeros(R, dtype=torch.int32)     # GPU num_computed_tokens
        self.pos_buf = torch.zeros(4096, dtype=torch.int64)    # persistent positions (stale beyond ntok)
        self.slot_buf = torch.full((2, 4096), -1, dtype=torch.int64)
        self.prev_slot0 = None
        self.in_tabs = [torch.zeros(R, W0 * BPK, dtype=torch.int32), torch.zeros(R, 4, dtype=torch.int32)]
        self.prompt_len = {}
        self.ids = 0
        if precycle:
            self._precycle()

    # -------------------------------------------------------------- scheduler side
    def _precycle(self):
        """Advance the free-queue head past block 935 with requests the scenario never sees
        (hashed full blocks are appended back FIFO, block_pool.py:719-743)."""
        k = 0
        while True:
            ids = [200000 + 7919 * k + i for i in range(BS * 6)]
            req = Request(f"pre{k}", ids, SamplingParams(max_tokens=1), None, block_hasher=self.hasher)
            self.mgr.new_step_starts()
            n = 0
            while n < len(ids):
                c = min(2048, len(ids) - n)
                b = self.mgr.allocate_slots(req, c, num_lookahead_tokens=4)
                assert b is not None
                req.num_computed_tokens += c
                n += c
            mx = max(max(g) if g else 0 for g in self.mgr.get_block_ids(req.request_id))
            self.mgr.free(req)
            k += 1
            if mx >= 960:
                break

    def add(self, prompt_ids, max_tokens, rid=None):
        rid = rid or f"r{self.ids}"
        self.ids += 1
        req = Request(rid, list(prompt_ids), SamplingParams(max_tokens=max_tokens), None,
                      block_hasher=self.hasher)
        self.waiting.append(dict(req=req, rid=rid, gen=0, max=max_tokens, prefill_done=False,
                                 started=False, sent=False, hit=0))
        return rid

    def schedule(self, chunk_override=None):
        """One schedule() call -> real SchedulerOutput + per-request plan."""
        self.mgr.new_step_starts()
        self.sched_step += 1
        budget = self.budget
        new_reqs, cached = [], dict(req_ids=[], new_block_ids=[], num_computed=[], resumed=set())
        nsched, spec = {}, {}
        while self.waiting and len(self.running) < self.concurrency:
            self.running.append(self.waiting.pop(0))
        plan = {}
        for st in sorted(self.running, key=lambda s: not s["prefill_done"]):
            req = st["req"]
            before = [list(g) for g in self.mgr.get_block_ids(st["rid"])] if st["started"] else None
            if not st["prefill_done"]:
                first = not st["started"]
                comp, nhit = (self.mgr.get_computed_blocks(req)[:2] if first else (None, 0))
                remaining = req.num_prompt_tokens - req.num_computed_tokens - nhit
                n = min(remaining, budget)
                if chunk_override and st["rid"] in chunk_override:
                    n = min(n, chunk_override[st["rid"]])
                if n <= 0:
                    continue
                b = self.mgr.allocate_slots(req, n, num_new_computed_tokens=nhit,
                                            new_computed_blocks=comp, num_lookahead_tokens=4)
                if b is None:
                    continue
                nc = req.num_computed_tokens + nhit
                req.num_computed_tokens = nc + n
                k = 0
                st["started"] = True
                st["hit"] = st["hit"] or nhit
                if req.num_computed_tokens >= req.num_prompt_tokens:
                    st["prefill_done"] = True
                    st["first_decode"] = True
            else:
                # B0 schedules full draft sets (scheduler.py:528-541: no partial drafts)
                k = NSPEC
                n = 1 + k
                if budget < n:
                    continue
                b = self.mgr.allocate_slots(req, n, num_lookahead_tokens=4)
                if b is None:
                    continue
                nc = req.num_computed_tokens
                req.num_computed_tokens = nc + n
            budget -= n
            nsched[st["rid"]] = n
            if k:
                spec[st["rid"]] = [-1] * k
            if not st["sent"]:
                st["sent"] = True
                new_reqs.append(NewRequestData(
                    req_id=st["rid"], prompt_token_ids=list(req.prompt_token_ids), mm_features=[],
                    sampling_params=None, pooling_params=None,
                    block_ids=tuple(list(g) for g in self.mgr.get_block_ids(st["rid"])),
                    num_computed_tokens=nc, lora_request=None,
                    prefill_token_ids=list(req.prompt_token_ids)))
            else:
                after = [list(g) for g in self.mgr.get_block_ids(st["rid"])]
                delta = tuple(a[len(bb):] for a, bb in zip(after, before))
                cached["req_ids"].append(st["rid"])
                cached["new_block_ids"].append(delta if any(delta) else None)
                cached["num_computed"].append(nc)
            plan[st["rid"]] = dict(n=n, k=k, nc=nc, st=st)
        so = SchedulerOutput(
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=CachedRequestData(
                req_ids=cached["req_ids"], resumed_req_ids=set(), new_token_ids=[],
                all_token_ids={}, new_block_ids=cached["new_block_ids"],
                num_computed_tokens=cached["num_computed"],
                num_output_tokens=[0] * len(cached["req_ids"])),
            num_scheduled_tokens=nsched, total_num_scheduled_tokens=sum(nsched.values()),
            scheduled_spec_decode_tokens=spec, scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * 10, finished_req_ids=set(self.finished_pending),
            free_encoder_mm_hashes=[])
        self.finished_pending = set()
        # async: rejections of the previous step are applied after this schedule
        for rid, rej in self.pending_rej.items():
            st = next((s for s in self.running if s["rid"] == rid), None)
            if st is not None:
                st["req"].num_computed_tokens -= rej
        self.pending_rej = {}
        return so, plan

    def complete(self, plan, accepted):
        """Scheduler update after the runner: accepted[rid] tokens for decode rows."""
        rej = {}
        for rid, pl in plan.items():
            st = pl["st"]
            if pl["k"] or (st["prefill_done"] and not st.get("first_decode")):
                a = accepted[rid]
                rej[rid] = pl["k"] + 1 - a
                st["gen"] += a
                for _ in range(a):
                    st["req"].append_output_token_ids(self.rng.randrange(1000, 150000))
            elif st["prefill_done"] and st.get("first_decode"):
                st["first_decode"] = False
                st["gen"] += 1
                st["req"].append_output_token_ids(self.rng.randrange(1000, 150000))
            if st["gen"] >= st["max"]:
                self.running.remove(st)
                self.mgr.free(st["req"])
                self.finished_pending.add(rid)
        self.pending_rej = {r: v for r, v in rej.items() if v > 0}
        return rej

    # -------------------------------------------------------------- runner side
    def runner_step(self, so, faults=(), pad_to=None):
        """Apply the SchedulerOutput as B0's runner v2 does and build every consumed mapping."""
        F = {f.kind: f for f in faults}
        for rid in so.finished_req_ids:
            row = self.row_of.pop(rid, None)
            if row is not None:
                self.free_rows.append(row)
        for nr in so.scheduled_new_reqs:
            row = self.free_rows.pop()
            self.row_of[nr.req_id] = row
            self.prompt_len[nr.req_id] = len(nr.prompt_token_ids)
            self.committed[row] = nr.num_computed_tokens
            self._append(row, nr.block_ids, overwrite=True)
        cr = so.scheduled_cached_reqs
        for i, rid in enumerate(cr.req_ids):
            if cr.new_block_ids[i] is not None:
                self._append(self.row_of[rid], cr.new_block_ids[i], overwrite=False,
                             drop=("drop_append" in F and F["drop_append"].kw["rid"] == rid))
        if "wrong_req" in F:
            kw = F["wrong_req"].kw
            ra, rb = self.row_of[kw["victim"]], self.row_of[kw["donor"]]
            c = int(self.committed[ra]) // BS if kw.get("col") is None else kw["col"]
            dc = kw.get("donor_col", 0)
            self.tab[0][ra, c * BPK:(c + 1) * BPK] = self.tab[0][rb, dc * BPK:(dc + 1) * BPK]
        if "wrong_group" in F:
            kw = F["wrong_group"].kw
            ra = self.row_of[kw["victim"]]
            c = kw.get("col", 0)
            blk = kw["block"]
            self.tab[0][ra, c * BPK:(c + 1) * BPK] = torch.arange(BPK, dtype=torch.int32) + blk * BPK
        req_ids = sort_batch_req_ids(so.num_scheduled_tokens, so.scheduled_spec_decode_tokens, 1 + NSPEC)
        nreq = len(req_ids)
        ns = np.array([so.num_scheduled_tokens[r] for r in req_ids], np.int32)
        ntok = int(ns.sum())
        idx = [self.row_of[r] for r in req_ids]
        if "row_reuse" in F:   # the batch row reads the previous occupant's runner row
            kw = F["row_reuse"].kw
            if kw["rid"] in req_ids:
                idx[req_ids.index(kw["rid"])] = kw["stale_row"]
        idx_mapping = torch.tensor(idx, dtype=torch.int32)
        qsl = torch.zeros(R + 1, dtype=torch.int32)
        qsl[1:nreq + 1] = torch.from_numpy(np.cumsum(ns))
        qsl[nreq + 1:] = ntok
        ntok_pad = pad_to or ntok
        nreq_pad = nreq if pad_to is None else max(nreq, pad_to // (1 + NSPEC))
        seq_lens = torch.zeros(R, dtype=torch.int32)
        prepare_pos_seq_lens(idx_mapping, qsl, self.committed, self.pos_buf, seq_lens)
        if "wrong_pos" in F:
            kw = F["wrong_pos"].kw
            r = req_ids.index(kw["rid"])
            self.pos_buf[int(qsl[r]):int(qsl[r + 1])] += kw.get("delta", 1)
        # gathered (input) tables, real kernel
        ptr_src = torch.tensor([t.data_ptr() for t in self.tab], dtype=torch.uint64)
        ptr_dst = torch.tensor([t.data_ptr() for t in self.in_tabs], dtype=torch.uint64)
        strides = torch.tensor([t.stride(0) for t in self.tab], dtype=torch.int64)
        _gather_block_tables_kernel[(2, nreq_pad)](idx_mapping, ptr_src, ptr_dst, strides,
                                                   self.nblk, self.nblk.stride(0), nreq, BLOCK_SIZE=1024)
        gathered = [t[:nreq_pad] for t in self.in_tabs]
        # slot mappings (persistent buffer), real kernel
        bsz = torch.tensor([KBS, RING], dtype=torch.int32)
        _compute_slot_mappings_kernel[(2, nreq + 1)](self.slot_buf.shape[1], idx_mapping, qsl,
                                                    self.pos_buf, ptr_src, strides, bsz,
                                                    self.slot_buf, self.slot_buf.stride(0), 0,
                                                    CP_SIZE=1, CP_INTERLEAVE=1, PAD_ID=PAD,
                                                    TRITON_BLOCK_SIZE=1024)
        slot0 = self.slot_buf[0, :ntok_pad].clone()
        if "offset_bug" in F:
            r = req_ids.index(F["offset_bug"].kw["rid"])
            a_, b_ = int(qsl[r]), int(qsl[r + 1])
            slot0[a_:b_] = torch.div(slot0[a_:b_], BS, rounding_mode="floor") * BS + \
                torch.remainder(slot0[a_:b_] + 1, BS)
        if "stale_replay" in F and self.prev_slot0 is not None:
            m = min(ntok_pad, self.prev_slot0.shape[0])
            slot0[:m] = self.prev_slot0[:m]
        if "inact_write" in F and ntok_pad > ntok:
            slot0[ntok] = F["inact_write"].kw["block"] * BS + 5
        self.prev_slot0 = self.slot_buf[0, :ntok_pad].clone()
        # tail mapping: B0 (D2) circular mapping, or the generic one with a stale column (T1)
        tail_generic = self.slot_buf[1, :ntok_pad].clone()
        if "tail_t1" in F:
            kw = F["tail_t1"].kw
            row = self.row_of[kw["rid"]]
            self.tab[1][row, 1] = kw["block"]
            _compute_slot_mappings_kernel[(2, nreq + 1)](self.slot_buf.shape[1], idx_mapping, qsl,
                                                        self.pos_buf, ptr_src, strides, bsz,
                                                        self.slot_buf, self.slot_buf.stride(0), 0,
                                                        CP_SIZE=1, CP_INTERLEAVE=1, PAD_ID=PAD,
                                                        TRITON_BLOCK_SIZE=1024)
            tail = self.slot_buf[1, :ntok_pad].clone()
            self.tab[1][row, 1] = 0
        else:
            tail = compute_kpool_tail_slot_mapping(tail_generic, gathered[1], qsl[:nreq + 1],
                                                   self.pos_buf, ntok, nreq, RING)
        # index-K pool-granular slot mapping on the translated table (indexer.py:1081-1112)
        idx_bt = (gathered[0][:, ::IDX_FACTOR] // IDX_FACTOR).contiguous()
        idx_buf = torch.full((4096,), -1, dtype=torch.int64)
        idx_slot = get_compressed_slot_mapping(ntok, qsl[:nreq + 1], seq_lens[:nreq], idx_bt[:nreq],
                                               IDX_STORAGE, KPOOL, out=idx_buf)
        idx_slot = idx_buf[:ntok_pad]
        # decode / prefill split (decode-like rows first, sort_batch_req_ids)
        dec_rows = [i for i, r in enumerate(req_ids) if so.scheduled_spec_decode_tokens.get(r)
                    or (ns[i] == 1 and self.committed[idx[i]] >= self.prompt_len[r])]
        nd = len(dec_rows)
        ndt = int(ns[:nd].sum())
        out = dict(req_ids=req_ids, nreq=nreq, ntok=ntok, ntok_pad=ntok_pad, nreq_pad=nreq_pad,
                   qsl=qsl, pos=self.pos_buf[:ntok_pad].clone(), slot0=slot0, tail=tail,
                   idx_slot=idx_slot, gathered=gathered, idx_bt=idx_bt, seq_lens=seq_lens,
                   nd=nd, ndt=ndt, ns=ns, idx=idx)
        # grouped decode write inputs (kpool decode path)
        if nd:
            pos_d = self.pos_buf[:ndt]
            if (ns[:nd] == ns[0]).all() and ndt == nd * int(ns[0]):
                sh = (nd, int(ns[0]))
                out["dec"] = (idx_slot[:ndt].view(sh), pos_d.to(torch.int32).view(sh),
                              tail[:ndt].view(sh))
            else:
                lmax = int(ns[:nd].max())
                lens = torch.from_numpy(ns[:nd].astype(np.int64))
                sc = _build_decode_scatter_indices(lens, nd, ndt)
                out["dec"] = (_scatter_decode_tokens_by_request(idx_slot[:ndt], -1, nd, lmax, sc),
                              _scatter_decode_tokens_by_request(pos_d.to(torch.int32), -1, nd, lmax, sc),
                              _scatter_decode_tokens_by_request(tail[:ndt], -1, nd, lmax, sc))
            # decode read: per-token expanded table + lengths (indexer.py:885-923), uniform case
            if (ns[:nd] == ns[0]).all():
                ml = int(ns[0])
                dsl = torch.zeros(4096, dtype=torch.int32)
                ebt = torch.zeros(max(ndt, 1), gathered[0].shape[1], dtype=torch.int32)
                dl = torch.zeros(4096, dtype=torch.int32)
                _prepare_uniform_decode_kernel[(ndt,)](seq_lens, dsl, gathered[0], gathered[0].stride(0),
                                                       ebt, ebt.stride(0), dl, ml, BLOCK_SIZE=1024)
                rt = (ebt[:ndt, ::IDX_FACTOR] // IDX_FACTOR).contiguous()
                rl = dsl[:ndt] // KPOOL
                if "idx_read_len" in F:
                    rl = rl.clone()
                    rl[0] += F["idx_read_len"].kw.get("delta", 1)
                if "idx_read_foreign" in F:
                    rt = rt.clone()
                    rt[0, 0] = F["idx_read_foreign"].kw["block"] * (PPB // IDX_STORAGE)
                out["dread"] = (rt, rl)
        # prefill read (one chunk over all prefill rows; indexer.py:1130-1170)
        if nd < nreq:
            cl = (seq_lens[nd:nreq] // KPOOL).to(torch.int32)
            cu = torch.zeros(nreq - nd + 1, dtype=torch.int32)
            cu[1:] = torch.cumsum(cl, 0)

            class Chunk:
                pass
            ch = Chunk()
            ch.block_table = idx_bt[nd:nreq]
            ch.local_cu_seq_lens = cu
            ch.cu_seq_lens = cu
            ch.token_start = int(qsl[nd])
            out["pread"] = ch
        # MLA sparse read: logical top-k -> physical (b12x_mla_sparse.py:2996-3046)
        n = ntok_pad
        logical = torch.full((n, TOPK), -1, dtype=torch.int32)
        rid_tok = torch.zeros(n, dtype=torch.int32)
        for i in range(nreq):
            rid_tok[int(qsl[i]):int(qsl[i + 1])] = i
        for t in range(ntok):
            p = int(self.pos_buf[t])
            m = min(p + 1, TOPK)
            lo_n = max(p - 15, 0)
            far = sorted(self.rng.sample(range(0, lo_n), k=min(max(m - 16, 0), lo_n)))
            near = list(range(max(p - 15, 0), p + 1))
            vals = (far + near)[-m:]
            logical[t, :len(vals)] = torch.tensor(vals, dtype=torch.int32)
        if "sel_causal" in F:
            logical[0, 0] = int(self.pos_buf[0]) + 5
        sel = torch.full((n, TOPK), -1, dtype=torch.int32)
        _logical_topk_to_physical_slots(rid_tok, gathered[0], logical, sel, KBS)
        nsa = torch.zeros(n, dtype=torch.int32)
        nsa[:ntok] = torch.clamp(self.pos_buf[:ntok] + 1, max=TOPK).to(torch.int32)
        _compact_dropped_selection(sel, nsa)
        _mask_page_table_after_nsa_len(sel, nsa)
        if "sel_foreign" in F and ntok > 1:     # token 1 reads another request's block
            sel[1, 0] = F["sel_foreign"].kw["block"] * BS + 3
        if "masked_live" in F and ntok > 2:     # length shortened after masking: entry stays live
            nsa[2] -= 1
        out["sel"] = (sel, nsa)
        if "qsl_bad" in F and nreq >= 2:
            q2 = qsl.clone()
            q2[1] += 1
            out["qsl_consumed"] = q2
        else:
            out["qsl_consumed"] = qsl
        return out

    def _append(self, row, block_ids, overwrite, drop=False):
        for g in (0, 1):
            ids = list(block_ids[g])
            start = 0 if overwrite else int(self.nblk[g, row])
            if g == 0:
                kid = [b * BPK + k for b in ids for k in range(BPK)]
                if kid and not drop:
                    self.tab[0][row, start:start + len(kid)] = torch.tensor(kid, dtype=torch.int32)
                # drop=True: the runner misses this append (the column keeps the row's old,
                # previous occupant's block) but still counts the block
                self.nblk[g, row] = start + len(kid)
            else:
                if ids:
                    self.tab[1][row, start:start + len(ids)] = torch.tensor(ids, dtype=torch.int32)
                self.nblk[g, row] = start + len(ids)

    def commit(self, out, accepted_by_rid, plan):
        """GPU num_computed_tokens after the step (accepted tokens for decode rows)."""
        for i, rid in enumerate(out["req_ids"]):
            row = self.row_of.get(rid)
            if row is None:
                continue
            pl = plan[rid]
            if pl["k"] or pl["st"]["prefill_done"] and int(self.committed[row]) >= self.prompt_len[rid]:
                self.committed[row] += accepted_by_rid[rid]
            else:
                self.committed[row] += pl["n"]
