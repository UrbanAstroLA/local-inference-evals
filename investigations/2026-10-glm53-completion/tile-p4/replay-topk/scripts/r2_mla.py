"""R2: order sensitivity of the served b12x sparse-MLA prefill, per image; R3: R1's top-k outputs composed into it.

ALL Q AND KV ARE SYNTHETIC (labelled).  The kernel, its plan, its cache writer and every index-construction step are
the image's own served code, called with the served arguments of layer 3 at warm chunk 1 (PROTOCOL.md section 4):

v0.9.1 (DCP=1, MLA owner map split:25 -> layer 3 owner-local on GPU 0, all 64 heads; VLLM_GLM53_NOPE_RECORD=1):
  plan     b12x_mla_sparse._prepare_b12x_sparse_mla_plan(sparse_mla, Caps(... mode="extend", max_q_rows=2048,
           num_q_heads=64, max_batch=16, head_dim=512, v_head_dim=512, model_type=GLM_NEXT, scale_format=None,
           cache_record_bytes=528, fp8_rope=None, max_chunks_per_row=32, page_size=64, head_major_output=False,
           return_lse=True, lse_scale="natural", softmax_scale=256**-0.5), forced_num_splits=32)   (091:1839-1890)
  writer   sparse_mla.concat_and_cache_glm_next_mla(kv_c, kv_cache, slots, plan=_prepare_glm_next_writer_plan)
           (do_kv_cache_update 091:2101-2112)
  index    topk_indices_buffer[:n] = -1; [:, :2047] = expand_pools_and_append_tail(pool_ids[:, :511], p + 1, 4)
           (kpool:461, 633-653); _logical_topk_to_physical_slots; nsa = min(p + 1, 2048);
           _compact_dropped_selection; _mask_page_table_after_nsa_len   (forward_mqa 091:3024-3048)
  run      sparse_mla.bind(plan, scratch, q, kv_cache, selected_indices, cache_lengths=[seq_len], selected_lengths)
           -> sparse_mla.run -> (out, lse_base2)   (091:3219-3248)
v0.7.0 (DCP=2, TP=2: layer 3 is DCP-sharded; the kernel call of DCP rank 0 is replayed; 32 local heads x 2 gathered
        = 64 query heads; fp8_ds_mla 656-B record with a zero 64-lane RoPE tail, q 576 wide):
  plan     sparse_mla.plan(Caps(device, num_q_heads=64, max_q_rows=2048, max_width=2048, dtype=bf16, kv_dtype=uint8,
           head_dim=576, v_head_dim=512, mode="extend", max_batch=16, max_chunks_per_row=32, page_size=64,
           head_major_output=True))   (070:1621-1645)
  writer   ops.concat_and_cache_mla(kv_c, zeros(n, 64) bf16, kv_cache, slots, "fp8_ds_mla", scale)  (backend.py:1118)
  index    the same expand, then triton_filter_and_convert_dcp_index(req_id, block_table, topk, dcp_size=2,
           dcp_rank=0, interleave=1, BLOCK_SIZE=64, NUM_TOPK_TOKENS=2048, out=, valid_counts=);
           nsa = min(valid, local p + 1); _mask_page_table_after_nsa_len   (070:2765-2785)
  run      plan.bind(scratch, q, selected_indices, cache_seqlens_int32=[local seq_len], nsa_cache_seqlens_int32)
           -> run_extend(binding=, kv_cache, sm_scale, v_head_dim=512, return_lse=True, lse_scale="natural")
           (070:3007-3030)
"""
from __future__ import annotations

import time
import traceback
import zlib

import rt_common as rc

N_PAGES = 80               # physical pages in the synthetic cache (64 hold the 4,090 tokens)
N_PAGES_V070 = 48          # rank-local: 2,045 tokens -> 32 pages
VARIANTS = ("asc", "desc", "perm1", "perm2", "perm3", "atomic2")


def _decode_records(torch, recs):
    """[n, >=528] uint8 records -> fp32 latent [n, 512]: 512 E4M3 in 4 groups of 128, then 4 fp32 scales."""
    lat = recs[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    sc = recs[:, 512:528].contiguous().view(torch.float32)
    return (lat.view(-1, 4, 128) * sc[:, :, None]).reshape(-1, 512)


class Synth:
    """Synthetic layer-3 inputs for one image, built once and reused by every variant."""

    def __init__(self, torch, dev, version, seed, q_std):
        self.torch, self.dev, self.version = torch, dev, version
        g = rc.gen(torch, seed, dev)
        n = rc.SEQ_LEN
        self.pos = torch.arange(rc.CHUNK_START, rc.SEQ_LEN, dtype=torch.int32, device=dev)
        self.kv_c = (torch.randn((n, rc.KV_LORA), generator=g) * 1.0).to(torch.bfloat16).to(dev)
        self.q_lat = (torch.randn((rc.CHUNK_ROWS, rc.MLA_HEADS, rc.KV_LORA), generator=g) * q_std).to(torch.bfloat16).to(dev)
        # pool selections: rows with <= 512 pools take the shortcut (index order, -1 pad); selecting rows get a fixed
        # random 512-pool SET; its lowest synthetic score is the fixed last column (the threshold-bin minimum)
        plen = ((self.pos + 1) // rc.KPOOL).cpu()
        self.selm = (plen > rc.SELECT_K)
        self.plen = plen
        K = rc.SELECT_K
        ar = torch.arange(K)[None, :]
        short = torch.where(ar < plen[:, None], ar.expand(rc.CHUNK_ROWS, -1), torch.full((1, K), -1))
        self.sets = []
        self.scores = []
        base = short.clone()
        for i in range(rc.CHUNK_ROWS):
            if not bool(self.selm[i]):
                self.sets.append(None)
                self.scores.append(None)
                continue
            s = torch.randperm(int(plen[i]), generator=g)[:K]
            sc = torch.randn(K, generator=g)
            self.sets.append(s)
            self.scores.append(sc)
        self.short = short
        self.g = g

    def pool_topk(self, variant, rng_seed=0):
        """[rows, 512] int32 pool ids; selecting rows hold the same SET in every variant and the same last column
        (lowest score); columns 0..510 are ordered per variant:
          base     above-threshold block (all but the 16 lowest) in a random ('atomic-like') order, then the 16 lowest
                   sorted by score descending (the upstream kernel's threshold-bin layout; block size ASSUMED)
          atomic2  the same with another random above-threshold order (what two runs of the served kernel differ by)
          asc      ascending pool id;  desc  score descending;  permN  uniform random permutations
        """
        torch = self.torch
        K = rc.SELECT_K
        out = self.short.clone()
        g = torch.Generator().manual_seed(1000 + rng_seed + zlib.crc32(variant.encode()) % 997)
        for i in range(rc.CHUNK_ROWS):
            s = self.sets[i]
            if s is None:
                continue
            sc = self.scores[i]
            order = torch.argsort(sc, descending=True)            # score-descending
            last = order[-1]
            head = order[:-1]                                     # 511 columns to arrange
            if variant in ("base", "atomic2"):
                tb = 15                                           # threshold block size among the 511 (+ the last)
                above, thr = head[: K - 1 - tb], head[K - 1 - tb:]
                gg = torch.Generator().manual_seed(7 + i * 31 + (0 if variant == "base" else 99991) + rng_seed)
                above = above[torch.randperm(len(above), generator=gg)]
                head = torch.cat([above, thr])
            elif variant == "asc":
                head = head[torch.argsort(s[head])]
            elif variant == "desc":
                pass
            elif variant.startswith("perm"):
                head = head[torch.randperm(len(head), generator=g)]
            else:
                raise ValueError(variant)
            out[i] = torch.cat([s[head], s[last].reshape(1)]).to(torch.int32)
        return out.to(self.dev)


class V091:
    name = "v091"

    def __init__(self, torch, dev, kc, bm, log):
        from b12x.attention import sparse_mla
        from b12x.attention._shared.mla.traits import ModelType
        self.torch, self.dev, self.kc, self.bm, self.sm, self.log = torch, dev, kc, bm, sparse_mla, log
        self.info = dict(path="sparse_mla.bind/run with the plan from b12x_mla_sparse._prepare_b12x_sparse_mla_plan",
                         heads=64, record_bytes=528, head_dim=512)
        self.caps = sparse_mla.Caps(
            device=dev, num_q_heads=64, max_q_rows=2048, max_width=rc.W, softmax_scale=256 ** -0.5,
            dtype=torch.bfloat16, kv_dtype=torch.uint8, head_dim=512, v_head_dim=512,
            model_type=int(ModelType.GLM_NEXT), scale_format=None, cache_record_bytes=528, fp8_rope=None,
            latent_scale_per_token=False, mode="extend", max_batch=16, max_chunks_per_row=32, page_size=rc.PAGE,
            head_major_output=False, return_lse=True, lse_scale="natural")
        t = time.time()
        self.plan, _ = bm._prepare_b12x_sparse_mla_plan(sparse_mla, self.caps, forced_num_splits=32)
        self.info["plan_prepare_s"] = round(time.time() - t, 2)
        st = getattr(self.plan, "prepared", None)
        q = getattr(getattr(st, "state", None), "query", None)
        self.info["prefill_mg_enabled"] = getattr(q, "prefill_mg_enabled", None)
        t = time.time()
        self.writer = bm._prepare_glm_next_writer_plan(dev, rc.PAGE)
        self.info["writer_prepare_s"] = round(time.time() - t, 2)
        spec = self.plan.scratch_specs()[0]
        self.scratch_spec = spec
        self.rb = 528

    def build_cache(self, syn, poison, perm_seed):
        torch = self.torch
        kv = torch.full((N_PAGES, rc.PAGE, self.rb), poison, dtype=torch.uint8, device=self.dev)
        g = torch.Generator().manual_seed(perm_seed)
        phys = torch.randperm(N_PAGES, generator=g)[:68].to(torch.int32)
        bt = phys[None, :].to(self.dev).contiguous()                      # [1, 68] (one 4,352-token MLA block)
        t = torch.arange(rc.SEQ_LEN, device=self.dev)
        slots = (bt[0, (t // rc.PAGE)].long() * rc.PAGE + t % rc.PAGE).contiguous()
        self.sm.concat_and_cache_glm_next_mla(syn.kv_c, kv.view(torch.uint8), slots, plan=self.writer)
        torch.cuda.synchronize(self.dev)
        return dict(kv=kv, bt=bt, slots=slots)

    def build_index(self, syn, pool_topk, cache):
        torch, bm = self.torch, self.bm
        n = rc.CHUNK_ROWS
        q_seq = syn.pos + 1
        expanded = self.kc.expand_pools_and_append_tail(pool_topk.to(torch.int64)[:, : rc.SELECT_K - 1], q_seq, rc.KPOOL)
        topk_buf = torch.full((2048, rc.W), -1, dtype=torch.int32, device=self.dev)   # persistent [max_batched, 2048]
        topk_buf[:n] = -1
        topk_buf[:n, : expanded.shape[-1]] = expanded
        topk = topk_buf[:n]
        req = torch.zeros(n, dtype=torch.int32, device=self.dev)
        pt1 = torch.full((2048, rc.W), -1, dtype=torch.int32, device=self.dev)
        sel = pt1[:n, : topk.shape[1]]
        bm._logical_topk_to_physical_slots(req, cache["bt"], topk, sel, rc.PAGE)
        nsa_buf = torch.zeros(2048, dtype=torch.int32, device=self.dev)
        nsa = nsa_buf[:n]
        nsa.copy_(q_seq)
        nsa.clamp_max_(topk.shape[1])
        bm._compact_dropped_selection(sel, nsa)
        bm._mask_page_table_after_nsa_len(sel, nsa)
        return dict(sel=sel, nsa=nsa, topk=topk, expanded=expanded, keep=(topk_buf, pt1, nsa_buf))

    def make_q(self, syn):
        ws = self.torch.empty((2048, 64, 512), dtype=self.torch.bfloat16, device=self.dev)  # q workspace [max_batched]
        q = ws[: rc.CHUNK_ROWS]
        q.copy_(syn.q_lat)
        return q, ws

    def run(self, q, cache, idx, scratch=None):
        torch = self.torch
        if scratch is None:
            scratch = torch.empty(self.scratch_spec.shape, dtype=self.scratch_spec.dtype, device=self.dev)
        clen = torch.tensor([rc.SEQ_LEN], dtype=torch.int32, device=self.dev)
        b = self.sm.bind(self.plan, scratch=scratch, q=q, kv_cache=cache["kv"], selected_indices=idx["sel"],
                         cache_lengths=clen, selected_lengths=idx["nsa"])
        out, lse = self.sm.run(b)
        torch.cuda.synchronize(self.dev)
        o, l = _token_major(torch, out, lse, q.shape[0], self.info)
        return o, l, scratch

    def ref_rows(self, syn, cache, idx, rows):
        """fp32 reference attention for the given rows from the decoded cache records (q is the bf16 query)."""
        torch = self.torch
        kv = cache["kv"].view(-1, self.rb)
        outs = []
        for r in rows:
            nsa = int(idx["nsa"][r])
            s = idx["sel"][r, :nsa]
            s = s[s >= 0].long()
            lat = _decode_records(torch, kv[s])
            qq = syn.q_lat[r].float()
            att = torch.softmax((qq @ lat.T) * (256 ** -0.5), dim=-1)
            outs.append(att @ lat)
        return torch.stack(outs)

    def decode_check(self, syn, cache):
        torch = self.torch
        kv = cache["kv"].view(-1, self.rb)
        lat = _decode_records(torch, kv[cache["slots"]])
        ref = syn.kv_c.float()
        return float((lat - ref).norm() / ref.norm())


class V070:
    name = "v070"

    def __init__(self, torch, dev, kc, bm, log):
        from b12x.attention import sparse_mla
        import vllm.v1.attention.backends.mla.sparse_utils as su
        from vllm import _custom_ops as ops
        from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
        self.torch, self.dev, self.kc, self.bm, self.sm, self.log = torch, dev, kc, bm, sparse_mla, log
        self.su, self.ops, self.local_lens = su, ops, get_dcp_local_seq_lens
        self.info = dict(path="sparse_mla.plan(Caps) -> plan.bind -> sparse_mla.run_extend (DCP rank 0 of 2)",
                         heads=64, record_bytes=656, head_dim=576, dcp_rank=0, dcp_size=2)
        self.caps = sparse_mla.Caps(device=dev, num_q_heads=64, max_q_rows=2048, max_width=rc.W,
                                    dtype=torch.bfloat16, kv_dtype=torch.uint8, head_dim=576, v_head_dim=512,
                                    mode="extend", max_batch=16, max_chunks_per_row=32, page_size=rc.PAGE,
                                    head_major_output=True)
        t = time.time()
        self.plan = sparse_mla.plan(self.caps)
        self.info["plan_s"] = round(time.time() - t, 2)
        self.nbytes = int(self.plan.layout.nbytes)
        self.rb = 656
        g = torch.arange(rc.CHUNK_START + 1, rc.SEQ_LEN + 1, dtype=torch.int32)
        self.per_token_local = self.local_lens(g, 2, 0, 1).to(torch.int32).to(dev)
        self.seq_local = int(self.local_lens(torch.tensor([rc.SEQ_LEN], dtype=torch.int32), 2, 0, 1)[0])

    def build_cache(self, syn, poison, perm_seed):
        torch = self.torch
        kv = torch.full((N_PAGES_V070, rc.PAGE, self.rb), poison, dtype=torch.uint8, device=self.dev)
        g = torch.Generator().manual_seed(perm_seed)
        phys = torch.randperm(N_PAGES_V070, generator=g)[:34].to(torch.int32)
        bt = phys[None, :].to(self.dev).contiguous()
        own = torch.arange(0, rc.SEQ_LEN, 2, device=self.dev)                 # rank 0 owns even positions
        loc = own // 2
        slots = (bt[0, (loc // rc.PAGE)].long() * rc.PAGE + loc % rc.PAGE).contiguous()
        kpe = torch.zeros((own.numel(), 64), dtype=torch.bfloat16, device=self.dev)
        self.ops.concat_and_cache_mla(syn.kv_c[own].contiguous(), kpe, kv, slots, kv_cache_dtype="fp8_ds_mla",
                                      scale=torch.ones(1, dtype=torch.float32, device=self.dev))
        torch.cuda.synchronize(self.dev)
        return dict(kv=kv, bt=bt, slots=slots, own=own)

    def build_index(self, syn, pool_topk, cache):
        torch, bm = self.torch, self.bm
        n = rc.CHUNK_ROWS
        q_seq = syn.pos + 1
        expanded = self.kc.expand_pools_and_append_tail(pool_topk.to(torch.int64)[:, : rc.SELECT_K - 1], q_seq, rc.KPOOL)
        topk_buf = torch.full((2048, rc.W), -1, dtype=torch.int32, device=self.dev)
        topk_buf[:n, : expanded.shape[-1]] = expanded
        topk = topk_buf[:n]
        req = torch.zeros(n, dtype=torch.int32, device=self.dev)
        pt1 = torch.full((2048, rc.W), -1, dtype=torch.int32, device=self.dev)
        sel = pt1[:n, : topk.shape[1]]
        nsa_buf = torch.zeros(2048, dtype=torch.int32, device=self.dev)
        nsa = nsa_buf[:n]
        self.su.triton_filter_and_convert_dcp_index(req, cache["bt"], topk, dcp_size=2, dcp_rank=0,
                                                    cp_kv_cache_interleave_size=1, BLOCK_SIZE=rc.PAGE,
                                                    NUM_TOPK_TOKENS=topk.shape[1], out=sel, valid_counts=nsa)
        torch.minimum(nsa, self.per_token_local, out=nsa)
        bm._mask_page_table_after_nsa_len(sel, nsa)
        return dict(sel=sel, nsa=nsa, topk=topk, expanded=expanded, keep=(topk_buf, pt1, nsa_buf))

    def make_q(self, syn):
        ws = self.torch.empty((2048, 64, 576), dtype=self.torch.bfloat16, device=self.dev)
        q = ws[: rc.CHUNK_ROWS]
        q.zero_()
        q[..., :512].copy_(syn.q_lat)
        return q, ws

    def run(self, q, cache, idx, scratch=None):
        torch = self.torch
        if scratch is None:
            scratch = torch.empty((self.nbytes,), dtype=torch.uint8, device=self.dev)
        clen = torch.tensor([self.seq_local], dtype=torch.int32, device=self.dev)
        b = self.plan.bind(scratch=scratch, q=q, selected_indices=idx["sel"], cache_seqlens_int32=clen,
                           nsa_cache_seqlens_int32=idx["nsa"])
        out, lse = self.sm.run_extend(binding=b, kv_cache=cache["kv"], sm_scale=256 ** -0.5, v_head_dim=512,
                                      return_lse=True, lse_scale="natural")
        torch.cuda.synchronize(self.dev)
        o, l = _token_major(torch, out, lse, q.shape[0], self.info)
        return o, l, scratch

    ref_rows = V091.ref_rows

    def decode_check(self, syn, cache):
        torch = self.torch
        kv = cache["kv"].view(-1, self.rb)
        lat = _decode_records(torch, kv[cache["slots"]])
        ref = syn.kv_c[cache["own"]].float()
        return float((lat - ref).norm() / ref.norm())


def _token_major(torch, out, lse, n, info):
    """Return (out [n, heads, dv] token-major copy, lse full copy).  Records the kernel's returned shapes / strides once.
    A head-major [heads, n, dv] return is transposed to token-major (comparisons are per token row)."""
    if "out_shape" not in info:
        info["out_shape"], info["out_stride"] = list(out.shape), list(out.stride())
        info["lse_shape"] = list(lse.shape) if lse is not None else None
    if out.dim() == 3 and out.shape[0] >= n and out.shape[1] != n:
        o = out[:n]
    elif out.dim() == 3 and out.shape[1] == n:
        o = out.permute(1, 0, 2)
        info["out_transposed"] = True
    else:
        o = out[:n]
    return o.contiguous().clone(), (lse.contiguous().clone() if lse is not None else None)


def permute_valid_prefix(torch, sel, nsa, rows_mask, seed, mode):
    """v0.7.0 kernel-order variants applied after the filter: permute the valid (>= 0) prefix of selecting rows."""
    out = sel.clone()
    g = torch.Generator().manual_seed(seed)
    cpu = sel.cpu()
    nn = nsa.cpu()
    for i in torch.nonzero(rows_mask).flatten().tolist():
        k = int(nn[i])
        row = cpu[i, :k]
        v = row[row >= 0]
        if mode == "asc":
            v = torch.sort(v).values
        elif mode == "rev":
            v = torch.flip(v, [0])
        else:
            v = v[torch.randperm(len(v), generator=g)]
        cpu_row = cpu[i].clone()
        cpu_row[: len(v)] = v
        out[i] = cpu_row.to(out.device)
    return out


def compare(torch, a, b, rows_sel):
    eq = rc.bits_equal_rows(torch, a, b)
    mx, rel = rc.row_magnitudes(torch, a, b)
    sel = rows_sel.to(a.device)
    return dict(rows_differ=int((~eq).sum()), rows_differ_selecting=int((~eq & sel).sum()),
                rows_differ_shortcut=int((~eq & ~sel).sum()), selecting_rows=int(sel.sum()),
                mag_selecting=rc.summarize_mag(torch, ~eq & sel, mx, rel)), eq


def run_r2(torch, dev, version, cfg, dl, log, r1_kept=None):
    doc = dict(part="r2", version=version, status="running", label="SYNTHETIC q and KV (no model data)",
               regimes={}, errors=[])
    try:
        kp, kc, ix, bm = rc.import_vllm_modules()
        K = V091 if version == "v091" else V070
        t = time.time()
        kern = K(torch, dev, kc, bm, log)
        doc["kernel"] = kern.info
        doc["setup_s"] = round(time.time() - t, 2)
        for qname, qstd in cfg["q_regimes"]:
            if dl.passed():
                doc["stopped_reason"] = "soft deadline"
                break
            try:
                doc["regimes"][qname] = run_q_regime(torch, dev, kern, version, qname, qstd, cfg, dl, log)
            except Exception as exc:  # noqa: BLE001
                doc["errors"].append(dict(regime=qname, error=repr(exc)[:400], tb=traceback.format_exc()[-1500:]))
        if version == "v070" and not dl.passed():
            try:
                doc["filter"] = run_filter_check(torch, dev, kern, cfg)
            except Exception as exc:  # noqa: BLE001
                doc["errors"].append(dict(part="filter", error=repr(exc)[:400], tb=traceback.format_exc()[-1500:]))
        if r1_kept is not None and version == "v091" and not dl.passed():
            try:
                doc["r3"] = run_r3(torch, dev, kern, cfg, r1_kept, log)
            except Exception as exc:  # noqa: BLE001
                doc["errors"].append(dict(part="r3", error=repr(exc)[:400], tb=traceback.format_exc()[-1500:]))
        doc["status"] = "partial" if doc.get("stopped_reason") else ("error" if doc["errors"] else "complete")
    except Exception as exc:  # noqa: BLE001
        doc["status"] = "error"
        doc["errors"].append(dict(error=repr(exc)[:400], tb=traceback.format_exc()[-1500:]))
    return doc


def run_q_regime(torch, dev, kern, version, qname, qstd, cfg, dl, log):
    t0 = time.time()
    syn = Synth(torch, dev, version, cfg["seed"], qstd)
    selm = syn.selm
    r = dict(q_std=qstd, rows=rc.CHUNK_ROWS, rows_selecting=int(selm.sum()))
    cache = kern.build_cache(syn, poison=0x5A, perm_seed=cfg["seed"])
    r["record_decode_rel_err"] = kern.decode_check(syn, cache)
    q, qws = kern.make_q(syn)
    idx_base = kern.build_index(syn, syn.pool_topk("base"), cache)
    r["nsa_selecting_median"] = float(idx_base["nsa"][selm.to(dev)].float().median())
    for _ in range(cfg["r2_warm"]):
        kern.run(q, cache, idx_base)
    r["warmup_calls"] = cfg["r2_warm"]
    base, lse0, scratch = kern.run(q, cache, idx_base)
    r["nan_rows_base"] = int(torch.isnan(base.float()).reshape(base.shape[0], -1).any(dim=1).sum())
    # (a) same order, same allocation, repeated
    rep = []
    for _ in range(cfg["r2_repeats"]):
        o, l, _ = kern.run(q, cache, idx_base, scratch=scratch)
        rep.append(dict(out_equal=bool(rc.bits_equal_rows(torch, o, base).all()),
                        lse_equal=bool(rc.bits_equal_rows(torch, l, lse0).all())))
    r["repeat_same_alloc"] = dict(calls=len(rep), all_out_equal=all(x["out_equal"] for x in rep),
                                  all_lse_equal=all(x["lse_equal"] for x in rep))
    # (a') scratch poisoned before the call (scratch is output-only for prefill)
    pz = []
    for v in (0xFF, 0x00):
        scratch.fill_(v)
        o, _, _ = kern.run(q, cache, idx_base, scratch=scratch)
        pz.append(bool(rc.bits_equal_rows(torch, o, base).all()))
    r["scratch_poison_equal"] = pz
    # (b) relocation: fresh q workspace, fresh cache (same records at other physical pages AND same pages), fresh
    # index buffers and scratch
    reloc = []
    for k in range(cfg["r2_reloc"]):
        pad = torch.empty((k + 1) * 3 * 1048576, dtype=torch.uint8, device=dev)
        q2, qws2 = kern.make_q(syn)
        cache2 = kern.build_cache(syn, poison=0x5A, perm_seed=cfg["seed"] + (0 if k == 0 else 17 * k))
        idx2 = kern.build_index(syn, syn.pool_topk("base"), cache2)
        o, _, _ = kern.run(q2, cache2, idx2)
        reloc.append(dict(same_pages=(k == 0), out_equal=bool(rc.bits_equal_rows(torch, o, base).all()),
                          q_ptr_differs=int(q2.data_ptr()) != int(q.data_ptr()),
                          kv_ptr_differs=int(cache2["kv"].data_ptr()) != int(cache["kv"].data_ptr())))
        del pad, qws2
    r["relocation"] = reloc
    # (c) hidden reads: a different poison in every record the writer does not write
    cache3 = kern.build_cache(syn, poison=0xA5, perm_seed=cfg["seed"])
    o, _, _ = kern.run(q, cache3, idx_base)
    r["cache_poison_equal"] = bool(rc.bits_equal_rows(torch, o, base).all())
    # (d) order variants: same set per row, same last column; the index rebuilt through the served path
    var = {}
    fp32_rows = [int(i) for i in torch.nonzero(selm).flatten()[:: max(1, int(selm.sum()) // cfg["ref_rows"])][: cfg["ref_rows"]]]
    ref = kern.ref_rows(syn, cache, idx_base, fp32_rows)
    def vs_ref(o):
        a = o[fp32_rows].float()
        return float(((a - ref).reshape(len(fp32_rows), -1).norm(dim=1) / ref.reshape(len(fp32_rows), -1).norm(dim=1)).median())
    r["base_vs_fp32_ref_rel_l2_median"] = vs_ref(base)
    r["fp32_ref_rows"] = len(fp32_rows)
    for name in VARIANTS:
        if dl.passed():
            r["stopped"] = "soft deadline"
            break
        if version == "v091":
            idx_v = kern.build_index(syn, syn.pool_topk(name), cache)
            same_set = bool((torch.sort(idx_v["sel"], dim=1).values == torch.sort(idx_base["sel"], dim=1).values).all())
            same_nsa = bool(torch.equal(idx_v["nsa"], idx_base["nsa"]))
        else:
            if name in ("asc", "perm1", "perm2", "perm3"):
                mode = "asc" if name == "asc" else "perm"
            elif name == "desc":
                mode = "rev"
            else:
                mode = "perm"
            sel_v = permute_valid_prefix(torch, idx_base["sel"], idx_base["nsa"], selm,
                                         cfg["seed"] + zlib.crc32(name.encode()) % 1000, mode)
            idx_v = dict(sel=sel_v, nsa=idx_base["nsa"].clone())
            same_set = bool((torch.sort(sel_v, dim=1).values == torch.sort(idx_base["sel"], dim=1).values).all())
            same_nsa = True
        order_changed = int((~(idx_v["sel"] == idx_base["sel"]).all(dim=1)).sum())
        o1, _, _ = kern.run(q, cache, idx_v)
        o2, _, _ = kern.run(q, cache, idx_v)
        c, eq = compare(torch, o1, base, selm)
        c.update(same_set=same_set, same_nsa=same_nsa, rows_index_order_changed=order_changed,
                 repeat_equal=bool(rc.bits_equal_rows(torch, o1, o2).all()), vs_fp32_ref_rel_l2_median=vs_ref(o1))
        var[name] = c
        log(f"R2 {version} {qname} {name}: rows differ {c['rows_differ_selecting']}/{c['selecting_rows']} "
            f"(shortcut {c['rows_differ_shortcut']}), max_abs {c['mag_selecting']['max_abs']:.3g}")
    r["variants"] = var
    # (e) positive control: change the SET grossly (every selected pool p -> (p + 1) mod pool_len, a bijection, so
    # the row stays a set of distinct pools; most columns change), through the same served index path
    pt = syn.pool_topk("base").cpu()
    for i in torch.nonzero(selm).flatten().tolist():
        n_p = int(syn.plen[i])
        pt[i] = (pt[i].long() + 1) % n_p
    idx_p = kern.build_index(syn, pt.to(dev), cache)
    o, _, _ = kern.run(q, cache, idx_p)
    c, _ = compare(torch, o, base, selm)
    r["pos_ctrl_set_change"] = dict(rows_differ_selecting=c["rows_differ_selecting"], selecting_rows=c["selecting_rows"],
                                    rows_differ_shortcut=c["rows_differ_shortcut"])
    r["seconds"] = round(time.time() - t0, 2)
    return r


def run_filter_check(torch, dev, kern, cfg):
    """v0.7.0 only: does triton_filter_and_convert_dcp_index give the same output order for the same input, and does it
    keep the input's relative order?"""
    syn = Synth(torch, dev, "v070", cfg["seed"], 1.0)
    cache = kern.build_cache(syn, poison=0x5A, perm_seed=cfg["seed"])
    selm = syn.selm.to(dev)
    i0 = kern.build_index(syn, syn.pool_topk("base"), cache)
    same = 0
    for _ in range(cfg["r2_repeats"]):
        i1 = kern.build_index(syn, syn.pool_topk("base"), cache)
        same += int(torch.equal(i1["sel"], i0["sel"]) and torch.equal(i1["nsa"], i0["nsa"]))
    # order preservation: map each output slot back to its position in the expanded input row
    sel, topk = i0["sel"].cpu(), i0["topk"].cpu()
    bt = cache["bt"].cpu()[0]
    preserved = 0
    checked = 0
    for i in torch.nonzero(syn.selm).flatten().tolist()[:: 16]:
        row_in = topk[i]
        own = row_in[(row_in >= 0) & (row_in % 2 == 0)]
        loc = own // 2
        slot = bt[(loc // rc.PAGE).long()].long() * rc.PAGE + loc % rc.PAGE
        k = int(i0["nsa"][i])
        checked += 1
        preserved += int(torch.equal(sel[i, :k].long(), slot[:k]))
    return dict(repeats=cfg["r2_repeats"], identical_repeats=same, rows_checked=checked,
                rows_input_order_preserved=preserved)


def run_r3(torch, dev, kern, cfg, kept, log):
    """R1's same-allocation outputs (same synthetic logits, observed orderings) through the served expand / slot
    conversion into the v0.9.1 kernel; outputs compared with the call-0 output."""
    syn = Synth(torch, dev, "v091", cfg["seed"], cfg["q_regimes"][0][1])
    cache = kern.build_cache(syn, poison=0x5A, perm_seed=cfg["seed"])
    q, _ = kern.make_q(syn)
    outs = kept["outputs"]
    ref_top = outs[0]
    idx0 = kern.build_index(syn, ref_top, cache)
    base, _, _ = kern.run(q, cache, idx0)
    rows_sel = ((syn.pos + 1) // rc.KPOOL > rc.SELECT_K).cpu()
    calls = []
    for j, t in enumerate(outs[1: cfg["r3_calls"] + 1], 1):
        order_diff = ~(t == ref_top).all(dim=1)
        set_diff = ~(torch.sort(t, dim=1).values == torch.sort(ref_top, dim=1).values).all(dim=1)
        idx = kern.build_index(syn, t, cache)
        o, _, _ = kern.run(q, cache, idx)
        eq = rc.bits_equal_rows(torch, o, base)
        mx, rel = rc.row_magnitudes(torch, o, base)
        diff = ~eq
        calls.append(dict(call=j, rows_topk_order_diff=int(order_diff.sum()), rows_topk_set_diff=int(set_diff.sum()),
                          rows_out_diff=int(diff.sum()),
                          rows_out_diff_without_topk_diff=int((diff & ~order_diff).sum()),
                          rows_topk_diff_out_equal=int((order_diff & ~diff).sum()),
                          mag=rc.summarize_mag(torch, diff, mx, rel)))
    log(f"R3: {sum(1 for c in calls if c['rows_out_diff'])}/{len(calls)} calls with output differences")
    return dict(label="SYNTHETIC logits/q/KV; R1 regime " + kept["regime"], calls=calls,
                rows_selecting=int(rows_sel.sum()))


def dry_r2(torch, version, log):
    """CPU dry run: imports, Caps construction as served, index construction through the served Triton helpers under
    TRITON_INTERPRET=1 (CPU tensors), shapes.  Stops before plan preparation (CuTe compile needs the GPU)."""
    out = dict(part="r2-dry", version=version)
    kp, kc, ix, bm = rc.import_vllm_modules()
    from b12x.attention import sparse_mla
    dev = torch.device("cpu")
    if version == "v091":
        from b12x.attention._shared.mla.traits import ModelType
        caps = sparse_mla.Caps(
            device=torch.device("cuda", 0), num_q_heads=64, max_q_rows=2048, max_width=rc.W, softmax_scale=256 ** -0.5,
            dtype=torch.bfloat16, kv_dtype=torch.uint8, head_dim=512, v_head_dim=512,
            model_type=int(ModelType.GLM_NEXT), scale_format=None, cache_record_bytes=528, fp8_rope=None,
            latent_scale_per_token=False, mode="extend", max_batch=16, max_chunks_per_row=32, page_size=rc.PAGE,
            head_major_output=False, return_lse=True, lse_scale="natural")
    else:
        caps = sparse_mla.Caps(device=torch.device("cuda", 0), num_q_heads=64, max_q_rows=2048, max_width=rc.W,
                               dtype=torch.bfloat16, kv_dtype=torch.uint8, head_dim=576, v_head_dim=512, mode="extend",
                               max_batch=16, max_chunks_per_row=32, page_size=rc.PAGE, head_major_output=True)
    out["caps"] = {k: str(getattr(caps, k)) for k in caps.__dataclass_fields__}
    syn = Synth(torch, dev, version, 1234, 1.0)
    pt = syn.pool_topk("base")
    pa = syn.pool_topk("asc")
    out["variant_same_set"] = bool((torch.sort(pt, 1).values == torch.sort(pa, 1).values).all())
    out["variant_same_last"] = bool((pt[:, -1] == pa[:, -1]).all())
    out["variant_order_differs_rows"] = int((~(pt == pa).all(dim=1)).sum())
    q_seq = syn.pos + 1
    exp = kc.expand_pools_and_append_tail(pt.to(torch.int64)[:, : rc.SELECT_K - 1], q_seq, rc.KPOOL)
    out["expanded_shape"] = list(exp.shape)
    topk = torch.full((rc.CHUNK_ROWS, rc.W), -1, dtype=torch.int32)
    topk[:, : exp.shape[1]] = exp
    bt = torch.randperm(N_PAGES, generator=torch.Generator().manual_seed(5))[:68].to(torch.int32)[None, :]
    req = torch.zeros(rc.CHUNK_ROWS, dtype=torch.int32)
    sel = torch.full_like(topk, -1)
    if version == "v091":
        bm._logical_topk_to_physical_slots(req, bt, topk, sel, rc.PAGE)
        nsa = q_seq.clone().clamp_max_(rc.W)
        bm._compact_dropped_selection(sel, nsa)
        bm._mask_page_table_after_nsa_len(sel, nsa)
    else:
        import vllm.v1.attention.backends.mla.sparse_utils as su
        from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
        nsa = torch.zeros(rc.CHUNK_ROWS, dtype=torch.int32)
        su.triton_filter_and_convert_dcp_index(req, bt, topk, dcp_size=2, dcp_rank=0, cp_kv_cache_interleave_size=1,
                                               BLOCK_SIZE=rc.PAGE, NUM_TOPK_TOKENS=rc.W, out=sel, valid_counts=nsa)
        loc = get_dcp_local_seq_lens(q_seq, 2, 0, 1).to(torch.int32)
        torch.minimum(nsa, loc, out=nsa)
        bm._mask_page_table_after_nsa_len(sel, nsa)
    out["nsa_selecting_min_max"] = [int(nsa[syn.selm].min()), int(nsa[syn.selm].max())]
    out["sel_valid_selecting_median"] = float((sel[syn.selm] >= 0).sum(dim=1).float().median())
    out["sel_shape"] = list(sel.shape)
    log(f"R2 dry {version}: expanded {out['expanded_shape']}, nsa selecting {out['nsa_selecting_min_max']}")
    return out
