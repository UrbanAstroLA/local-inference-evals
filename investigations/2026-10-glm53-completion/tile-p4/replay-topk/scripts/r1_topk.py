"""R1: repeatability of the served prefill top-k, torch.ops._C.top_k_per_row_prefill (PROTOCOL.md section 3).

The call is the served one (sparse_attn_indexer_kpool.py:600-609, identical in both images):
    torch.ops._C.top_k_per_row_prefill(logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, topk_dst, num_rows,
                                       logits.stride(0), logits.stride(1), select_k)
with topk_dst = torch.full((num_rows, 512), -1, int32) (kpool:576-581), select_k = 2048 // 4 (kpool:574), and the
window from the image's own build_prefill_chunk_metadata (indexer.py).  ALL LOGITS ARE SYNTHETIC (labelled per regime):
  K  served producer: the image's DeepGEMM fp8_fp4_mqa_logits (deep_gemm.py wrapper, called as kpool:561-568 with
     clean_logits=False) on synthetic FP8 q / k, power-of-two k scales (ue8m0, attention.py scale_fmt) and fp32
     weights.  Gives the served logits tensor's real shape, strides and value granularity.
  C  continuous: fp32 standard normal (no exact ties with probability ~1), in a tensor with K's strides.
  T  tied: C rounded to bf16 precision (8 significant bits), stored as fp32: many exact ties near the threshold.
Per regime: warm-up calls, then N same-allocation calls (output buffer re-filled with -1 before each, as production
allocates it fresh with -1), then N relocated calls (fresh input copy with the same strides + fresh output buffer, the
allocator shifted by a dummy allocation).  Every call is compared exactly with the first same-allocation call.
"""
from __future__ import annotations

import time
import traceback

import rt_common as rc

MASK61 = (1 << 61) - 1


def _row_sig(torch, out):
    """Order-sensitive per-row signature (int64); informational only (distinct-ordering counts)."""
    w = torch.arange(1, out.shape[1] + 1, device=out.device, dtype=torch.int64) * 1000003
    return ((out.to(torch.int64) + 7) * w).sum(dim=1)


def _compare(torch, ref, out, ref_sorted):
    order_eq = (out == ref).all(dim=1)
    set_eq = (torch.sort(out, dim=1).values == ref_sorted).all(dim=1)
    last_eq = out[:, -1] == ref[:, -1]
    return order_eq, set_eq, last_eq


def make_logits_K(torch, dev, dg, ks, ke, seed):
    """Served producer on synthetic FP8 inputs.  Shapes as kpool:545-568 with use_fp4_cache=False:
    q (M, 32, 128) e4m3 with q_scale None, k (N, 128) e4m3, k_scale (N,) fp32 (a view of the (N, 4) uint8 workspace
    in production; ue8m0 => powers of two), weights (M, 32) fp32."""
    g = rc.gen(torch, seed, dev)
    M, N = rc.CHUNK_ROWS, rc.N_POOLS
    f8 = torch.float8_e4m3fn
    q = (torch.randn((M, rc.INDEX_HEADS, rc.INDEX_DIM), generator=g) * 2.0).to(f8).to(dev)
    k = (torch.randn((N, rc.INDEX_DIM), generator=g) * 2.0).to(f8).to(dev)
    ksc = torch.pow(2.0, torch.randint(-8, -3, (N,), generator=g).float()).to(dev)
    w = (torch.randn((M, rc.INDEX_HEADS), generator=g) * 0.1).float().to(dev)
    logits = dg.fp8_fp4_mqa_logits((q, None), (k, ksc), w, ks, ke, clean_logits=False)
    torch.cuda.synchronize(dev)
    return logits


def strided_copy(torch, src, values=None):
    """A fresh tensor with src's size and strides holding src's (or values') logical contents."""
    dst = torch.empty_strided(tuple(src.shape), tuple(src.stride()), dtype=src.dtype, device=src.device)
    dst.copy_(src if values is None else values)
    return dst


def window_ok_rows(torch, ks, ke):
    return (ke - ks).cpu()


def ref_sets(torch, logits, ks, ke, rows, k):
    """CPU fp64 reference top-k sets for the given rows; returns {row: (set, tie_at_boundary)}."""
    out = {}
    L = logits.detach().cpu().double()
    for r in rows:
        a, b = int(ks[r]), int(ke[r])
        v = L[r, a:b]
        if b - a <= k:
            out[r] = (set(range(b - a)), False)
            continue
        s = torch.sort(v, descending=True)
        tie = bool(s.values[k - 1] == s.values[k])
        out[r] = (set(s.indices[:k].tolist()), tie)
    return out


def _sync(torch, dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def run_regime(torch, dev, name, logits, ks, ke, n_calls, n_warm, dl, keep_outputs=False, op=None):
    """op: the served torch.ops._C.top_k_per_row_prefill; tests inject a CPU stand-in with the same signature."""
    if op is None:
        op = torch.ops._C.top_k_per_row_prefill
    M = logits.shape[0]
    K = rc.SELECT_K
    res = dict(regime=name, label="SYNTHETIC", shape=list(logits.shape), stride=list(logits.stride()),
               dtype=str(logits.dtype), storage_offset=int(logits.storage_offset()), n_calls_requested=n_calls)
    plen = (ke - ks).cpu()
    sel_rows = (plen > K)
    res["rows"] = M
    res["rows_selecting"] = int(sel_rows.sum())
    res["rows_shortcut"] = int((~sel_rows).sum())
    res["first_selecting_position"] = int(rc.CHUNK_START + int(torch.nonzero(sel_rows)[0])) if bool(sel_rows.any()) else None
    lbits = logits.view(torch.int32).clone()          # input-unchanged check (bitwise, NaN-safe)

    def call(lg, out):
        op(lg, ks, ke, out, M, lg.stride(0), lg.stride(1), K)
        _sync(torch, dev)

    out_a = torch.full((M, K), -1, dtype=torch.int32, device=dev)
    t0 = time.time()
    for _ in range(n_warm):
        out_a.fill_(-1)
        call(logits, out_a)
    res["warmup_calls"] = n_warm
    out_a.fill_(-1)
    call(logits, out_a)
    ref = out_a.clone()
    ref_sorted = torch.sort(ref, dim=1).values
    res["ptr_ref"] = dict(logits=int(logits.data_ptr()), out=int(out_a.data_ptr()))
    ref_ptrs = (int(logits.data_ptr()), int(out_a.data_ptr()))
    # shortcut-row negative control (rows with <= 512 pools: index order then -1)
    ar = torch.arange(K, device=dev, dtype=torch.int32)[None, :]
    expect_short = torch.where(ar < (ke - ks)[:, None], ar, torch.full_like(ar, -1))
    short_ok = (ref == expect_short).all(dim=1)
    res["neg_shortcut_rows_index_ordered"] = int(short_ok[~sel_rows.to(dev)].sum())
    # reference sets (CPU fp64) on a row sample: selecting rows only
    sample = [int(i) for i in torch.nonzero(sel_rows).flatten()[:: max(1, int(sel_rows.sum()) // 128)][:128]]
    refs = ref_sets(torch, logits, ks.cpu(), ke.cpu(), sample, K)
    agree = sum(1 for r in sample if set(ref[r].cpu().tolist()) == refs[r][0])
    ties = sum(1 for r in sample if refs[r][1])
    agree_untied = sum(1 for r in sample if not refs[r][1] and set(ref[r].cpu().tolist()) == refs[r][0])
    res["refset_sample_rows"] = len(sample)
    res["refset_agree"] = agree
    res["refset_rows_tied_at_boundary"] = ties
    res["refset_agree_untied"] = agree_untied
    # order pattern of the reference call (informational): is the last column the minimum logit of the set; is the
    # row sorted by logit descending; is it index-ascending
    Lr = logits.float()
    idx = ref.clamp_min(0).long()
    vals = torch.gather(Lr[:, :], 1, idx + ks[:, None].long())
    selm = sel_rows.to(dev)
    last_is_min = (vals[:, -1] == vals.min(dim=1).values)
    desc = (vals[:, :-1] >= vals[:, 1:]).all(dim=1)
    asc_idx = (ref[:, :-1] < ref[:, 1:]).all(dim=1)
    res["pattern_last_is_min"] = int((last_is_min & selm).sum())
    res["pattern_fully_desc"] = int((desc & selm).sum())
    res["pattern_index_ascending"] = int((asc_idx & selm).sum())
    # sorted suffix length (the threshold-bin block is sorted by logit, placed last): median over selecting rows
    nondesc = ~(vals[:, :-1] >= vals[:, 1:])
    pos = torch.arange(K - 1, device=dev)[None, :].expand(M, -1)
    last_break = torch.where(nondesc, pos, torch.full_like(pos, -1)).max(dim=1).values
    suffix = (K - 1 - last_break).float()
    res["pattern_sorted_suffix_len_median"] = float(suffix[selm].median()) if bool(selm.any()) else None
    # ties at the threshold, all selecting rows (exact fp32 equality of the 512th and 513th largest in-window values)
    neg_inf = torch.tensor(float("-inf"), device=dev)
    cols = torch.arange(logits.shape[1], device=dev)[None, :]
    inwin = (cols >= ks[:, None]) & (cols < ke[:, None])
    srt = torch.sort(torch.where(inwin, Lr, neg_inf), dim=1, descending=True).values
    if srt.shape[1] > K:
        tie_rows = (srt[:, K - 1] == srt[:, K]) & selm
        res["rows_tied_at_threshold"] = int(tie_rows.sum())
    res["nonfinite_in_window"] = int((~torch.isfinite(Lr) & inwin).sum())

    def tally(prefix, outs_iter):
        n = 0
        order_rows = torch.zeros(M, dtype=torch.int32, device=dev)
        set_rows = torch.zeros(M, dtype=torch.int32, device=dev)
        last_rows = torch.zeros(M, dtype=torch.int32, device=dev)
        calls_order_diff = calls_set_diff = 0
        input_changed = 0
        short_bad = 0
        sigs = [_row_sig(torch, ref)]
        kept = [ref.clone()] if keep_outputs else None
        ptrs = []
        for out, lg in outs_iter:
            if dl.passed():
                res[prefix + "_stopped"] = "soft deadline"
                break
            oe, se, le = _compare(torch, ref, out, ref_sorted)
            order_rows += (~oe).int()
            set_rows += (~se).int()
            last_rows += (~le).int()
            calls_order_diff += int(bool((~oe).any()))
            calls_set_diff += int(bool((~se).any()))
            short_bad += int((~(out == expect_short).all(dim=1) & ~selm).sum())
            if not torch.equal(lg.view(torch.int32), lbits):
                input_changed += 1
            sigs.append(_row_sig(torch, out))
            if keep_outputs:
                kept.append(out.clone())
            ptrs.append((int(lg.data_ptr()), int(out.data_ptr())))
            n += 1
        S = torch.stack(sigs, dim=1)
        distinct = torch.tensor([len(torch.unique(S[i])) for i in range(M)]) if n else torch.ones(M)
        d = dict(calls=n, calls_with_order_diff=calls_order_diff, calls_with_set_diff=calls_set_diff,
                 rows_order_varied=int((order_rows > 0).sum()),
                 rows_order_varied_selecting=int(((order_rows > 0) & selm).sum()),
                 rows_order_varied_shortcut=int(((order_rows > 0) & ~selm).sum()),
                 rows_set_varied=int((set_rows > 0).sum()),
                 rows_last_col_varied=int((last_rows > 0).sum()),
                 rows_last_col_varied_selecting=int(((last_rows > 0) & selm).sum()),
                 shortcut_row_failures=short_bad, input_changed_calls=input_changed,
                 distinct_orderings_max=int(distinct.max()),
                 distinct_orderings_median_selecting=float(distinct[sel_rows].float().median()) if bool(sel_rows.any()) else None,
                 per_row_order_diff_counts=[int(v) for v in order_rows.cpu().tolist()],
                 per_row_set_diff_counts=[int(v) for v in set_rows.cpu().tolist()],
                 distinct_ptrs=len(set(ptrs)),
                 calls_relocated_from_ref=sum(1 for a, b in ptrs if a != ref_ptrs[0] and b != ref_ptrs[1]))
        res[prefix] = d
        return kept

    def same_alloc():
        for _ in range(n_calls - 1):
            out_a.fill_(-1)
            call(logits, out_a)
            yield out_a, logits

    kept = tally("same_alloc", same_alloc())
    res["same_alloc"]["includes_reference_call"] = True

    def relocated():
        hold = []
        for i in range(n_calls):
            pad = torch.empty(((i % 7) + 1) * 1048576 + 4096 * i, dtype=torch.uint8, device=dev)
            lg = strided_copy(torch, logits)
            ks2, ke2 = ks.clone(), ke.clone()
            out = torch.full((M, K), -1, dtype=torch.int32, device=dev)
            op(lg, ks2, ke2, out, M, lg.stride(0), lg.stride(1), K)
            _sync(torch, dev)
            hold.append(pad)
            if len(hold) > 3:
                hold.pop(0)
            yield out, lg

    tally("relocated", relocated())
    # comparator positive control: swap two columns of the first selecting row; replace one element with an unselected
    # pool in the second selecting row.  The comparator must flag exactly those rows (order; set).
    pc = ref.clone()
    srows = torch.nonzero(sel_rows).flatten().tolist()
    if len(srows) >= 2:
        r0, r1 = srows[0], srows[1]
        a, b = pc[r0, 0].item(), pc[r0, 1].item()
        pc[r0, 0], pc[r0, 1] = b, a
        chosen = set(pc[r1].tolist())
        repl = next(v for v in range(int(plen[r1])) if v not in chosen)
        pc[r1, 0] = repl
        oe, se, _ = _compare(torch, ref, pc, ref_sorted)
        res["pos_ctrl"] = dict(order_flagged=sorted(torch.nonzero(~oe).flatten().tolist()) == sorted({r0, r1}),
                               set_flagged=torch.nonzero(~se).flatten().tolist() == [r1])
    res["seconds"] = round(time.time() - t0, 3)
    return res, kept


def run_r1(torch, dev, cfg, dl, log):
    """cfg: dict(regimes=[...], n_calls=int, n_warm=int, keep_for_r3=str|None, seed=int).  Returns (doc, kept)."""
    doc = dict(part="r1", status="running", label="SYNTHETIC logits (no model data)", regimes={}, errors=[])
    kept_out = None
    try:
        import vllm._custom_ops  # noqa: F401  (registers torch.ops._C, as the served process does)
        kp, kc, ix, bm = rc.import_vllm_modules()
        if not hasattr(torch.ops._C, "top_k_per_row_prefill"):
            raise RuntimeError("torch.ops._C.top_k_per_row_prefill is not registered in this image")
        doc["op_schema"] = str(torch.ops._C.top_k_per_row_prefill.default._schema)
        try:
            m = rc.served_window(torch, ix, dev)
            ks, ke = m.cu_seqlen_ks, m.cu_seqlen_ke
            fks, fke = rc.formula_window(torch, dev)
            doc["window"] = dict(source="indexer.build_prefill_chunk_metadata (served)",
                                 equals_formula=bool(torch.equal(ks, fks) and torch.equal(ke, fke)),
                                 max_local_total_seq_lens=int(m.max_local_total_seq_lens),
                                 token_start=int(m.token_start), token_end=int(m.token_end))
        except Exception as exc:  # noqa: BLE001
            ks, ke = rc.formula_window(torch, dev)
            doc["window"] = dict(source="formula (served builder failed)", error=repr(exc)[:300], equals_formula=True)
        stride_src = None
        K_logits = None
        if "K" in cfg["regimes"]:
            t = time.time()
            try:
                import vllm.utils.deep_gemm as dg
                K_logits = make_logits_K(torch, dev, dg, ks, ke, cfg["seed"])
                doc["K_producer"] = dict(ok=True, seconds=round(time.time() - t, 2), shape=list(K_logits.shape),
                                         stride=list(K_logits.stride()), dtype=str(K_logits.dtype),
                                         contiguous=bool(K_logits.is_contiguous()))
                # informational: is the producer itself repeatable on identical inputs (in-window values, bitwise)?
                K2 = make_logits_K(torch, dev, dg, ks, ke, cfg["seed"])
                cols = torch.arange(K_logits.shape[1], device=dev)[None, :]
                inwin = (cols >= ks[:, None]) & (cols < ke[:, None])
                same = (K2.view(torch.int32) == K_logits.view(torch.int32)) | ~inwin
                doc["K_producer"]["repeat_rows_bitwise_equal_in_window"] = int(same.all(dim=1).sum())
                del K2
                stride_src = K_logits
            except Exception as exc:  # noqa: BLE001
                doc["K_producer"] = dict(ok=False, seconds=round(time.time() - t, 2), error=repr(exc)[:400],
                                         tb=traceback.format_exc()[-800:])
        if stride_src is None:
            stride_src = torch.empty((rc.CHUNK_ROWS, rc.N_POOLS), dtype=torch.float32, device=dev)
            doc["stride_source"] = "contiguous (served producer unavailable: stride ASSUMED, see K_producer)"
        else:
            doc["stride_source"] = "served producer output (K)"
        g = rc.gen(torch, cfg["seed"] + 1, dev)
        base = torch.randn((rc.CHUNK_ROWS, rc.N_POOLS), generator=g).float()
        for name in cfg["regimes"]:
            if dl.passed():
                doc["stopped_reason"] = "soft deadline"
                break
            try:
                if name == "K":
                    if K_logits is None:
                        doc["regimes"]["K"] = dict(regime="K", skipped="served producer unavailable")
                        continue
                    lg = K_logits
                elif name == "C":
                    lg = strided_copy(torch, stride_src, base.to(dev))
                elif name == "T":
                    lg = strided_copy(torch, stride_src, base.to(torch.bfloat16).float().to(dev))
                else:
                    raise ValueError(name)
                keep = cfg.get("keep_for_r3") == name
                r, kept = run_regime(torch, dev, name, lg, ks, ke, cfg["n_calls"], cfg["n_warm"], dl, keep_outputs=keep)
                doc["regimes"][name] = r
                if keep:
                    kept_out = dict(regime=name, logits=lg, ks=ks, ke=ke, outputs=kept)
                log(f"R1 {name}: same-alloc order-varied rows {r['same_alloc']['rows_order_varied']}, set-varied "
                    f"{r['same_alloc']['rows_set_varied']}; relocated order-varied {r['relocated']['rows_order_varied']}"
                    f" ({r['seconds']}s)")
            except Exception as exc:  # noqa: BLE001
                doc["errors"].append(dict(regime=name, error=repr(exc)[:400], tb=traceback.format_exc()[-1200:]))
        doc["status"] = "partial" if doc.get("stopped_reason") else ("error" if doc["errors"] else "complete")
    except Exception as exc:  # noqa: BLE001
        doc["status"] = "error"
        doc["errors"].append(dict(error=repr(exc)[:400], tb=traceback.format_exc()[-1500:]))
    return doc, kept_out


def dry_r1(torch, log):
    """CPU dry run (CUDA hidden): imports, the served window builder under TRITON_INTERPRET=1, argument construction
    and shapes; stops before any CUDA op.  Returns a dict of checks."""
    out = dict(part="r1-dry")
    import vllm._custom_ops  # noqa: F401
    kp, kc, ix, bm = rc.import_vllm_modules()
    dev = torch.device("cpu")
    m = rc.served_window(torch, ix, dev)
    fks, fke = rc.formula_window(torch, dev)
    out["window_equals_formula"] = bool(torch.equal(m.cu_seqlen_ks, fks) and torch.equal(m.cu_seqlen_ke, fke))
    out["max_local_total_seq_lens"] = int(m.max_local_total_seq_lens)
    plen = (m.cu_seqlen_ke - m.cu_seqlen_ks)
    out["first_selecting_position"] = rc.CHUNK_START + int(torch.nonzero(plen > rc.SELECT_K)[0])
    out["rows_shortcut"] = int((plen <= rc.SELECT_K).sum())
    out["select_k"] = rc.SELECT_K
    lg = torch.empty((rc.CHUNK_ROWS, rc.N_POOLS), dtype=torch.float32)
    topk_dst = torch.full((lg.shape[0], rc.SELECT_K), -1, dtype=torch.int32)
    out["call_args"] = ["logits%s" % (tuple(lg.shape),), "cu_seqlen_ks int32", "cu_seqlen_ke int32",
                        "topk_dst%s int32 filled -1" % (tuple(topk_dst.shape),), lg.shape[0], lg.stride(0),
                        lg.stride(1), rc.SELECT_K]
    out["op_registered_on_cpu"] = hasattr(torch.ops._C, "top_k_per_row_prefill")
    log(f"R1 dry: window==formula {out['window_equals_formula']}, first selecting p {out['first_selecting_position']}")
    return out
