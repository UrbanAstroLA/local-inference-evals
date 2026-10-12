"""B0 cache isolation + high-offset addressing fixture (model-free, ONE GPU).

NOT RUN YET. Launched only by run_b0_cache_fixture.sh after an operator budget decision.

Allocates one B0 MLA slot tensor (2,929,766,400 B: MLA + KDA + DFlash draft co-owners) and
one idx tensor (183,110,400 B: index-K + kpool tail ring) with B0's real view code
(b0views.py), fills both with a sentinel pattern and keeps a shadow image of the bytes they
must contain. Each part drives REAL B0 kernels on chosen block ids (low twins and high
blocks >= 935, the record straddling 2^31, the last block 1274), updates the shadow with the
operation's reference effect, and requires raw == shadow after every operation.

Primary criterion per writer: exact sentinel integrity outside the declared footprint, and
inside it either exact expected bytes (copies, zeroing, tagged payloads) or byte equality
with a low-block twin computed by the same kernel from identical inputs. Readers: bit
equality between low- and high-address twins, plus a reference check with tolerances
frozen here (before data):
  FA2 draft read vs fp32 torch:      cos >= 0.999, max|err| <= 3e-2
  B12x GLM_NEXT read vs fp32 ref:    cos > 0.995,  max|err| < 3e-2  (B12x test_glm_next_mla.py:108-119)
  exact (gather/scatter/copies/tags): bit-exact

Results: JSON at argv[1], rewritten after every part (partial results survive a wall cap).
"""
from __future__ import annotations

import json, os, sys, time, traceback

assert os.environ.get("VLLM_GLM53_NOPE_RECORD") == "1"
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b0geom as G
import b0shadow as SH
import b0views as V

OUT = sys.argv[1]
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else None
DEV = torch.device(os.environ.get("B0_FIXTURE_DEVICE", "cuda:0"))  # "cpu" only for the CPU import smoke test
torch.manual_seed(20261012)
R: dict = {"meta": {}, "parts": {}, "checks": []}
T0 = time.time()

LOW = [1, 2, 3, 4, 5, 6]
HIGH = [934, 935, 1100, 1273, 1274]
EDGE = [1, 934, 935, 1274]


def dump():
    R["meta"]["elapsed_s"] = round(time.time() - T0, 1)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=1, default=str)
    os.replace(tmp, OUT)


def part(name):
    def deco(fn):
        def run():
            if ONLY and name.split("_")[0] not in ONLY:
                return
            t0 = time.time()
            rec = {"ok": None}
            R["parts"][name] = rec
            try:
                out = fn() or {}
                torch.cuda.synchronize()
                rec.update(out)
                rec["ok"] = all(c["ok"] for c in R["checks"] if c.get("part") == name)
            except Exception as e:  # noqa: BLE001
                rec["ok"] = False
                rec["error"] = repr(e)
                rec["traceback"] = traceback.format_exc()[-4000:]
            rec["secs"] = round(time.time() - t0, 2)
            dump()
        run.__name__ = name
        return run
    return deco


CUR = {"part": None}


def check(sh: SH.Shadow, label: str, part_name: str) -> dict:
    torch.cuda.synchronize()
    r = sh.check(label)
    r["part"] = part_name
    R["checks"].append({k: v for k, v in r.items() if k != "runs"})
    return r


def record(part_name: str, label: str, ok: bool, **kw):
    R["checks"].append({"part": part_name, "label": label, "ok": bool(ok), **kw})


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),
                                              b.contiguous().view(torch.uint8))


def cos_maxabs(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return (torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
            (a - b).abs().max().item())


# ------------------------------------------------------------------ setup
if DEV.type == "cuda":
    free0, total = torch.cuda.mem_get_info(DEV)
    R["meta"].update({"device": torch.cuda.get_device_name(DEV),
                      "cc": list(torch.cuda.get_device_capability(DEV))})
else:
    free0 = total = None
    torch.cuda.synchronize = lambda *a, **k: None  # CPU smoke test only
R["meta"].update({
    "free_bytes_at_start": free0, "total_bytes": total, "torch": torch.__version__,
    "image_id": os.environ.get("B0_IMAGE_ID"), "geometry": G.check_against_log(None),
    "env_nope_record": os.environ.get("VLLM_GLM53_NOPE_RECORD"),
})
raw = torch.empty(G.MLA_TENSOR, dtype=torch.int8, device=DEV)
rawi = torch.empty(G.IDX_TENSOR, dtype=torch.int8, device=DEV)
S = SH.Shadow(raw, salt=0xB0C1, kind="mla")
SI = SH.Shadow(rawi, salt=0xB0C2, kind="idx")
MLA = V.mla_view(raw)
KV = V.draft_view(raw)
KC, VC = V.draft_kv_split(KV)
CONV, CONV_K, SSM, _ = V.kda_views(raw)
IDX = V.idx_view(rawi)
TAIL = V.tail_view(rawi)
R["meta"]["views"] = {n: [list(t.shape), list(t.stride())] for n, t in
                      [("mla", MLA), ("draft_k", KC), ("conv_k", CONV_K), ("ssm", SSM),
                       ("idx", IDX), ("tail", TAIL)]}
R["meta"]["setup_check"] = [check(S, "fresh mla", "setup")["ok"], check(SI, "fresh idx", "setup")["ok"]]
dump()


# ------------------------------------------------------------------ A: DFlash draft writer
def draft_write(slots_bt, key, value):
    """slots: list of (block, tok) or None for PAD; FlashAttentionImpl.do_kv_cache_update path."""
    from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
    sm = torch.tensor([(-1 if s is None else s[0] * G.DRAFT_BLOCK_TOKENS + s[1]) for s in slots_bt],
                      dtype=torch.int64, device=DEV)
    one = torch.ones((), dtype=torch.float32, device=DEV)
    reshape_and_cache_flash(key, value, KC, VC, sm, "auto", one, one)
    for i, s in enumerate(slots_bt):
        if s is None:
            continue
        b, t = s
        for h in range(G.DRAFT_HEADS):
            S.expect_bytes(G.draft_k_range(b, t, h)[0], key[i, h])
            S.expect_bytes(G.draft_v_range(b, t, h)[0], value[i, h])


def draft_payload(slots_bt, epoch):
    n = len(slots_bt)
    k = torch.empty(n, G.DRAFT_HEADS, G.DRAFT_HEAD, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    for i, s in enumerate(slots_bt):
        b, t = s if s is not None else (0, 0)
        for h in range(G.DRAFT_HEADS):
            k[i, h] = SH.bf16_tag_values("draft", b, t, epoch, G.DRAFT_HEAD)
            v[i, h] = -SH.bf16_tag_values("draft", b, t, epoch, G.DRAFT_HEAD)
    return k.to(DEV), v.to(DEV)


@part("A_draft_writer")
def part_a():
    slots = []
    for b in EDGE:
        for t in (0, 1, 543, 1087):
            slots += [(b, t), None]
    k, v = draft_payload(slots, 1)
    draft_write(slots, k, v)
    r = check(S, "draft reshape_and_cache_flash edge blocks + PAD", "A_draft_writer")
    return {"rows": len(slots), "clean": r["ok"]}


# ------------------------------------------------------------------ B: DFlash FA2 reader
@part("B_draft_fa2_reader")
def part_b():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
    L = 2100
    sets = {"low": [2, 3], "high": [1273, 1274], "edge": [934, 935]}
    g = torch.Generator(device="cpu").manual_seed(7)
    kctx = (torch.randn(L, G.DRAFT_HEADS, G.DRAFT_HEAD, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    vctx = (torch.randn(L, G.DRAFT_HEADS, G.DRAFT_HEAD, generator=g)).to(torch.bfloat16).to(DEV)
    # different stale contents beyond seq_len in each set's last block (reads must not see them)
    for name, bl in sets.items():
        slots = [(bl[p // G.DRAFT_BLOCK_TOKENS], p % G.DRAFT_BLOCK_TOKENS) for p in range(L)]
        draft_write(slots, kctx, vctx)
        stale = [(bl[-1], t) for t in range(L - G.DRAFT_BLOCK_TOKENS, G.DRAFT_BLOCK_TOKENS)]
        sk, sv = draft_payload(stale, 3 + len(name))
        draft_write(stale, sk * 7, sv * 7)
    check(S, "context writes for 3 block sets", "B_draft_fa2_reader")
    nq, hq = 4, 16  # bonus + 3 masks; 32 heads / TP2
    q = (torch.randn(nq, hq, G.DRAFT_HEAD, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    outs = {}
    win = (2047, 2047)  # non-causal DFlash layer: _maybe_symmetrize_window((2047, 0), False)
    for name, bl in sets.items():
        o = torch.empty(nq, hq, G.DRAFT_HEAD, dtype=torch.bfloat16, device=DEV)
        flash_attn_varlen_func(
            q=q, k=KC, v=VC, out=o,
            cu_seqlens_q=torch.tensor([0, nq], dtype=torch.int32, device=DEV), max_seqlen_q=nq,
            seqused_k=torch.tensor([L], dtype=torch.int32, device=DEV), max_seqlen_k=L,
            softmax_scale=G.DRAFT_HEAD ** -0.5, causal=False, window_size=list(win),
            block_table=torch.tensor([bl], dtype=torch.int32, device=DEV), fa_version=2)
        outs[name] = o
    # fp32 reference over the logical K/V (window covers all L keys for the last 4 queries)
    rep = hq // G.DRAFT_HEADS
    kf = kctx.float().repeat_interleave(rep, dim=1)
    vf = vctx.float().repeat_interleave(rep, dim=1)
    qpos = torch.arange(L - nq, L, device=DEV)
    kpos = torch.arange(L, device=DEV)
    mask = (kpos[None, :] >= qpos[:, None] - win[0]) & (kpos[None, :] <= qpos[:, None] + win[1])
    s = torch.einsum("qhd,khd->hqk", q.float(), kf) * G.DRAFT_HEAD ** -0.5
    s = s.masked_fill(~mask[None], float("-inf"))
    ref = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vf)
    res = {}
    for name, o in outs.items():
        c, m = cos_maxabs(o, ref)
        eq = bit_equal(o, outs["low"])
        record("B_draft_fa2_reader", f"FA2 {name} vs low bit-equal", eq)
        record("B_draft_fa2_reader", f"FA2 {name} vs fp32 ref", c >= 0.999 and m <= 3e-2, cos=c, max_abs=m)
        res[name] = {"bit_equal_low": eq, "cos": c, "max_abs": m}
    check(S, "after FA2 reads (reads must not write)", "B_draft_fa2_reader")
    return res


# ------------------------------------------------------------------ C: KDA state writers/readers
def kda_set_ssm(blocks, vals):
    from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
    idx = torch.tensor(blocks, dtype=torch.int32, device=DEV)
    scatter_states(SSM, vals, idx)
    for i, b in enumerate(blocks):
        S.expect_bytes(G.kda_ssm_range(b)[0], vals[i])


def kda_ssm_payload(blocks, epoch):
    out = torch.empty(len(blocks), 32, 128, 128, dtype=torch.float32)
    for i, b in enumerate(blocks):
        out[i] = SH.fp32_tag_values("kda", b, 0, epoch, 32 * 128 * 128).view(32, 128, 128)
    return out.to(DEV)


def kda_set_conv(blocks, vals):
    """Test setup (torch indexing, not a kernel under test): SD rows (6, 12288) bf16."""
    for i, b in enumerate(blocks):
        CONV[b].copy_(vals[i])
        S.expect_bytes(G.kda_conv_range(b)[0], vals[i])


@part("C1_kda_scatter_gather")
def part_c1():
    from vllm.model_executor.layers.mamba.ops.gather_initial_states import gather_initial_states
    vals = kda_ssm_payload(EDGE, 1)
    kda_set_ssm(EDGE, vals)
    check(S, "scatter_states edge blocks", "C1_kda_scatter_gather")
    idx = torch.tensor(EDGE, dtype=torch.int32, device=DEV)
    got = gather_initial_states(SSM, idx, torch.ones(len(EDGE), dtype=torch.bool, device=DEV))
    record("C1_kda_scatter_gather", "gather_initial_states exact", bit_equal(got, vals))
    check(S, "after gather", "C1_kda_scatter_gather")


@part("C2_kda_conv_update")
def part_c2():
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    g = torch.Generator(device="cpu").manual_seed(11)
    pairs = [(2, 1274), (3, 935), (4, 934)]
    st = (torch.randn(6, G.KDA_CONV_DIM, generator=g) * 0.3).to(torch.bfloat16)
    kda_set_conv([b for p in pairs for b in p], [st] * 6)
    check(S, "conv setup", "C2_kda_conv_update")
    w = (torch.randn(G.KDA_CONV_DIM, 4, generator=g) * 0.3).to(torch.bfloat16).to(DEV)
    res = {}
    # plain decode: one token per row
    x = (torch.randn(1, G.KDA_CONV_DIM, generator=g)).to(torch.bfloat16).to(DEV).repeat(6, 1)
    idx = torch.tensor([b for p in pairs for b in p], dtype=torch.int32, device=DEV)
    y = causal_conv1d_update(x.clone(), CONV_K, w, None, "silu", conv_state_indices=idx)
    torch.cuda.synchronize()
    for i, (lo, hi) in enumerate(pairs):
        record("C2_kda_conv_update", f"decode out {hi} == {lo}", bit_equal(y[2 * i + 1], y[2 * i]))
        record("C2_kda_conv_update", f"decode state {hi} == {lo}", bit_equal(CONV[hi], CONV[lo]))
        for b in (lo, hi):
            S.accept_actual(*G.kda_conv_range(b))
    check(S, "conv update decode: only conv regions changed", "C2_kda_conv_update")
    # spec decode: 4 tokens per row, num_accepted 2 (window shift), state column 0 only
    xs = (torch.randn(4, G.KDA_CONV_DIM, generator=g)).to(torch.bfloat16).to(DEV).repeat(6, 1)
    qsl = torch.arange(0, 25, 4, dtype=torch.int32, device=DEV)
    acc = torch.full((6,), 2, dtype=torch.int32, device=DEV)
    ys = causal_conv1d_update(xs.clone(), CONV_K, w, None, "silu", conv_state_indices=idx,
                              num_accepted_tokens=acc, query_start_loc=qsl, max_query_len=4)
    torch.cuda.synchronize()
    for i, (lo, hi) in enumerate(pairs):
        record("C2_kda_conv_update", f"spec out {hi} == {lo}",
               bit_equal(ys[8 * i + 4: 8 * i + 8], ys[8 * i: 8 * i + 4]))
        record("C2_kda_conv_update", f"spec state {hi} == {lo}", bit_equal(CONV[hi], CONV[lo]))
        for b in (lo, hi):
            S.accept_actual(*G.kda_conv_range(b))
    check(S, "conv update spec: only conv regions changed", "C2_kda_conv_update")
    return res


@part("C2b_kda_conv_prefill")
def part_c2b():
    """causal_conv1d_fn (prefill / chunked-prefill path, kda.py:529-540); metadata=None
    (grid computed on the host instead of from the attention metadata)."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    g = torch.Generator(device="cpu").manual_seed(37)
    pairs = [(2, 1274), (3, 935)]
    blocks = [b for p in pairs for b in p]
    st = (torch.randn(6, G.KDA_CONV_DIM, generator=g) * 0.3).to(torch.bfloat16)
    kda_set_conv(blocks, [st] * len(blocks))
    check(S, "conv prefill setup", "C2b_kda_conv_prefill")
    T = 37
    xs = (torch.randn(T, G.KDA_CONV_DIM, generator=g)).to(torch.bfloat16).to(DEV)
    x = xs.repeat(len(blocks), 1).contiguous()
    w = (torch.randn(G.KDA_CONV_DIM, 4, generator=g) * 0.3).to(torch.bfloat16).to(DEV)
    qsl = torch.arange(0, T * len(blocks) + 1, T, dtype=torch.int32, device=DEV)
    y = causal_conv1d_fn(x.transpose(0, 1), w, None, conv_states=CONV_K, query_start_loc=qsl,
                         cache_indices=torch.tensor(blocks, dtype=torch.int32, device=DEV),
                         has_initial_state=torch.ones(len(blocks), dtype=torch.bool, device=DEV),
                         activation="silu", metadata=None).transpose(0, 1)
    torch.cuda.synchronize()
    for i, (lo, hi) in enumerate(pairs):
        record("C2b_kda_conv_prefill", f"prefill out {hi} == {lo}",
               bit_equal(y[(2 * i + 1) * T:(2 * i + 2) * T], y[2 * i * T:(2 * i + 1) * T]))
        record("C2b_kda_conv_prefill", f"prefill state {hi} == {lo}", bit_equal(CONV[hi], CONV[lo]))
        for b in (lo, hi):
            S.accept_actual(*G.kda_conv_range(b))
    check(S, "conv prefill: only conv regions changed", "C2b_kda_conv_prefill")


@part("C3_kda_fused_recurrent_spec")
def part_c3():
    from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda
    g = torch.Generator(device="cpu").manual_seed(13)
    rows = [[1, 2, 3, 4], [935, 1100, 1273, 1274], [5, 6, 7, 8], [930, 933, 934, 1271]]
    init = (torch.randn(32, 128, 128, generator=g) * 0.05).to(DEV)
    # h0 is read from column (num_accepted - 1) = 1
    kda_set_ssm([r[1] for r in rows], torch.stack([init] * len(rows)))
    check(S, "recurrent setup", "C3_kda_fused_recurrent_spec")
    T = 4

    def rnd(*s, sc=1.0):
        return (torch.randn(*s, generator=g) * sc).to(torch.bfloat16).to(DEV)
    q1, k1, v1 = rnd(1, T, 32, 128), rnd(1, T, 32, 128), rnd(1, T, 32, 128)
    g1, b1 = rnd(1, T, 32, 128, sc=0.5), rnd(1, T, 32)
    rep = lambda t: t.repeat(1, len(rows), *([1] * (t.ndim - 2)))
    a_log = (torch.randn(32, generator=g) * 0.1).float().to(DEV)
    dt_bias = (torch.randn(32 * 128, generator=g) * 0.1).float().to(DEV)
    idx = torch.tensor(rows, dtype=torch.int32, device=DEV)
    cu = torch.arange(0, T * len(rows) + 1, T, dtype=torch.int32, device=DEV)
    acc = torch.full((len(rows),), 2, dtype=torch.int32, device=DEV)
    o, _ = fused_recurrent_kda(q=rep(q1), k=rep(k1), v=rep(v1), g=rep(g1), beta=rep(b1),
                               initial_state=SSM, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
                               ssm_state_indices=idx, num_accepted_tokens=acc, sigmoid_beta=True,
                               a_log=a_log, g_bias=dt_bias, compute_gate=True, lower_bound=-5.0)
    torch.cuda.synchronize()
    for ri in (1, 3):
        lo = 0 if ri == 1 else 2
        record("C3_kda_fused_recurrent_spec", f"out row {ri} == row {lo}",
               bit_equal(o[:, ri * T:(ri + 1) * T], o[:, lo * T:(lo + 1) * T]))
        for c in range(4):
            record("C3_kda_fused_recurrent_spec", f"state {rows[ri][c]} == {rows[lo][c]}",
                   bit_equal(SSM[rows[ri][c]], SSM[rows[lo][c]]))
    for r in rows:
        for b in r:
            S.accept_actual(*G.kda_ssm_range(b))
    check(S, "fused_recurrent spec: only the 4 ssm columns per row changed", "C3_kda_fused_recurrent_spec")


@part("C4_kda_align_precopy")
def part_c4():
    """precopy_mamba_align_fused_kernel launched directly (moderate glue; the metadata is
    built as MambaSpecDecodeGPUContext would: mamba_utils.py:792-840)."""
    from vllm.v1.worker import mamba_utils as MU
    rows = [[10, 11, 12, 13], [1271, 1272, 1273, 1274]]
    sv = kda_ssm_payload([11, 1272], 4)
    kda_set_ssm([11, 1272], sv)
    g = torch.Generator(device="cpu").manual_seed(17)
    cv = (torch.randn(6, G.KDA_CONV_DIM, generator=g)).to(torch.bfloat16)
    kda_set_conv([10, 1271], [cv, cv])
    check(S, "precopy setup", "C4_kda_align_precopy")
    bt = torch.tensor(rows, dtype=torch.int32, device=DEV)
    src_col = torch.tensor([0, 0], dtype=torch.int32, device=DEV)
    dst_col = torch.tensor([2, 2], dtype=torch.int32, device=DEV)
    bias = torch.tensor([1, 1], dtype=torch.int32, device=DEV)
    i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=DEV)
    MU.precopy_mamba_align_fused_kernel[(2, 2, 16)](
        dst_col, src_col, bias,
        i64([bt.data_ptr()]), bt.stride(0),
        i64([CONV.data_ptr(), SSM.data_ptr()]), i64([G.MLA_PAGE, G.MLA_PAGE]),
        i64([2, 4]), i64([G.KDA_CONV_DIM, 32 * 128 * 128]), i64([6, 0]), i64([0, 0]),
        i64([0, 0]), i64([0, 0]),
        torch.tensor([0, 1], dtype=torch.int32, device=DEV), 2,
        COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=False, TEMPORAL_ACCEPT_BIAS=True,
        HAS_IDX_MAPPING=True, TEMPORAL_TILES=16)
    torch.cuda.synchronize()
    row = G.KDA_CONV_DIM * 2
    for r in rows:
        # conv: state[bt[0], bias:] -> state[bt[2], :6-bias];  ssm: state[bt[0+bias]] -> state[bt[2]]
        S.expect_copy(G.kda_conv_range(r[0])[0] + row, G.kda_conv_range(r[2])[0], 5 * row)
        S.expect_copy(G.kda_ssm_range(r[1])[0], G.kda_ssm_range(r[2])[0], G.KDA_SSM_BYTES)
    check(S, "precopy: exact copies only", "C4_kda_align_precopy")


@part("C5_page_copy_on_write")
def part_c5():
    from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace
    copy_kv_cache_blocks_inplace([MLA], G.NUM_BLOCKS, [(2, 1270), (1274, 7)])
    torch.cuda.synchronize()
    S.expect_copy(2 * G.MLA_PAGE, 1270 * G.MLA_PAGE, G.MLA_PAGE)
    S.expect_copy(1274 * G.MLA_PAGE, 7 * G.MLA_PAGE, G.MLA_PAGE)
    check(S, "copy_kv_cache_blocks_inplace whole pages", "C5_page_copy_on_write")


# ------------------------------------------------------------------ D: MLA GLM_NEXT writer/readers
MLA_SETS = {
    "low": [G.mla_slot(2, t) for t in range(64)],
    "straddle": list(range(G.first_high_mla_slot() - 32, G.first_high_mla_slot() + 32)),
    "b935": [G.mla_slot(935, t) for t in range(64)],
    "last": [G.mla_slot(1274, t) for t in range(4288, 4352)],
    "kpage_cross": [G.mla_slot(1100, t) for t in range(32, 96)],
}
LAT_POW2 = -6


def mla_latents(epoch):
    return torch.stack([SH.e4m3_exact_latent(SH.tag_code("mla", 0, i, epoch), scale_pow2=LAT_POW2)
                        for i in range(64)]).to(DEV)


def mla_writer_plan():
    try:
        from vllm.v1.attention.backends.mla.b12x_mla_sparse import _prepare_glm_next_writer_plan
        return _prepare_glm_next_writer_plan(DEV, 64), "vllm._prepare_glm_next_writer_plan"
    except Exception as e:  # noqa: BLE001
        from b12x.attention import sparse_mla
        kv_c = torch.zeros((1, 512), dtype=torch.bfloat16, device=DEV)
        cache = torch.zeros((1, 64, 528), dtype=torch.uint8, device=DEV)
        slots = torch.full((1,), -1, dtype=torch.int64, device=DEV)
        return sparse_mla.plan_cache_writer(kv_c, cache, slots), f"b12x lazy ({e!r})"


def mla_write(plan, lat, slots):
    from b12x.attention import sparse_mla
    sm = torch.tensor(slots, dtype=torch.int64, device=DEV)
    sparse_mla.concat_and_cache_glm_next_mla(lat, MLA.view(torch.uint8), sm, plan=plan)


def mla_records(slots):
    idx = torch.tensor(slots, dtype=torch.int64, device=DEV)
    return MLA.view(torch.uint8).view(-1, 528)[idx]


@part("D1_mla_writer")
def part_d1():
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference
    plan, how = mla_writer_plan()
    lat = mla_latents(1)
    res = {"plan": how}
    for name, slots in MLA_SETS.items():
        padded = []
        for s in slots:
            padded += [s, -1]
        mla_write(plan, torch.repeat_interleave(lat, 2, dim=0), padded)
    torch.cuda.synchronize()
    low = mla_records(MLA_SETS["low"])
    ref = pack_mla_kv_cache_reference(lat.cpu()).view(64, 528).to(DEV)
    res["low_vs_reference_bytes_differing"] = int((low != ref).sum())
    deq = unpack_mla_kv_cache_reference(low.view(64, 1, 528)).view(64, 512)
    c, m = cos_maxabs(deq, lat)
    record("D1_mla_writer", "low records decode to the latents", c > 0.9999 and m <= 2 ** LAT_POW2 * 0.01,
           cos=c, max_abs=m)
    for name, slots in MLA_SETS.items():
        got = mla_records(slots)
        record("D1_mla_writer", f"{name} records == low twin bytes", bit_equal(got, low))
        for s in slots:
            S.accept_actual(*G.mla_record_range(s))
    check(S, "GLM_NEXT writer: only the declared records changed", "D1_mla_writer")
    R["meta"]["mla_plan"] = how
    return res


def mla_read(selected_slots, mode, q):
    from b12x.attention import sparse_mla
    from b12x.attention._shared.mla.traits import ModelType, ScaleFormat
    rows, width = len(selected_slots), len(selected_slots[0])
    sel = torch.tensor(selected_slots, dtype=torch.int32, device=DEV)
    active = torch.full((rows,), width, dtype=torch.int32, device=DEV)
    seql = torch.full((rows,), int(sel.max()) + 1, dtype=torch.int32, device=DEV)
    plan = sparse_mla.plan(sparse_mla.Caps(
        device=DEV, num_q_heads=int(q.shape[1]), max_q_rows=rows, max_batch=rows, max_width=width,
        softmax_scale=256 ** -0.5, max_kv_rows=width, kv_dtype=torch.uint8, head_dim=512,
        v_head_dim=512, page_size=64, model_type=ModelType.GLM_NEXT,
        scale_format=ScaleFormat.ARBITRARY_FP32, cache_record_bytes=528, fp8_rope=False,
        latent_scale_per_token=False, mode=mode, return_lse=False, lse_scale="base2"))
    spec = plan.scratch_specs()[0]
    b = sparse_mla.bind(plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=DEV), q=q,
                        selected_indices=sel, kv_cache=MLA.view(torch.uint8), cache_lengths=seql,
                        selected_lengths=active)
    out = sparse_mla.run(b)
    # Post-run-3 change: run_unified_prefill returns (O, base-2 LSE) for extend
    # (b12x/attention/_shared/mla/prefill.py:158, 373); keep O.
    if isinstance(out, tuple):
        out = out[0]
    return out, (plan, b, sel)


def mla_reference(q, slots_rows):
    from b12x.attention._shared.mla.reference import _sparse_attention_reference, unpack_mla_kv_cache_reference
    flat = sorted({s for r in slots_rows for s in r})
    pos = {s: i for i, s in enumerate(flat)}
    kv = unpack_mla_kv_cache_reference(mla_records(flat).view(-1, 1, 528)).view(len(flat), 512).float()
    pt = torch.tensor([[pos[s] for s in r] for r in slots_rows], dtype=torch.int32, device=DEV)
    return _sparse_attention_reference(q_all=q, k_all=kv, v_all=kv, page_table_1=pt,
                                       active_token_counts=None, sm_scale=256 ** -0.5)


# Post-run-2 change (2026-10-12, made after inspecting run-2 data): run 2 used width 64 for
# both modes; GLM_NEXT extend rejects topk 64 (b12x/attention/_shared/mla/prefill.py:430-432,
# 547-560: topk in {512,1024,2048,2051,2112}) and B0 serves decode at w2048 (server-s1.log:74
# "decode plan ...h64.rows64.w2048.rec528"). Decode at width 64 is kept unchanged; decode and
# extend at the serving width 2048 are added over 2048 distinct written records per set.
MLA_SETS_2048 = {
    "low": [G.mla_slot(2, t) for t in range(2048)],
    "straddle": list(range(G.first_high_mla_slot() - 1024, G.first_high_mla_slot() + 1024)),
    "b935": [G.mla_slot(935, t) for t in range(2048)],
    "last": [G.mla_slot(1274, t) for t in range(4352 - 2048, 4352)],
    "kpage_cross": [G.mla_slot(1100, t) for t in range(32, 32 + 2048)],
}


def _d2_reads(sets, width, modes, q, g, res):
    perm = [torch.randperm(width, generator=g).tolist() for _ in range(q.shape[0])]
    for mode in modes:
        outs = {}
        for name, slots in sets.items():
            sel = [[slots[j] for j in p] for p in perm]
            o, _ = mla_read(sel, mode, q)
            outs[name] = o.clone()  # outputs live in the binding's scratch (kernel.py:3183)
            if name in ("low", "straddle", "last"):
                ref = mla_reference(q, sel)
                c, m = cos_maxabs(outs[name], ref)
                record("D2_mla_readers", f"{mode} w{width} {name} vs fp32 ref",
                       c > 0.995 and m < 3e-2, cos=c, max_abs=m)
                res[f"{mode}_w{width}_{name}_ref"] = [c, m]
        for name in sets:
            record("D2_mla_readers", f"{mode} w{width} {name} == low bit-equal",
                   bit_equal(outs[name], outs["low"]))


@part("D2_mla_readers")
def part_d2():
    g = torch.Generator(device="cpu").manual_seed(19)
    q = (torch.randn(4, 64, 512, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    res = {}
    _d2_reads(MLA_SETS, 64, ("decode",), q, g, res)  # unchanged from run 2 (passed there)
    # write the 2048-record sets (same tagged latents, tok = i) and twin-check them
    plan, _ = mla_writer_plan()
    lat = torch.stack([SH.e4m3_exact_latent(SH.tag_code("mla", 0, i, 1), scale_pow2=LAT_POW2)
                       for i in range(2048)]).to(DEV)
    for name, slots in MLA_SETS_2048.items():
        mla_write(plan, lat, slots)
    torch.cuda.synchronize()
    low = mla_records(MLA_SETS_2048["low"])
    for name, slots in MLA_SETS_2048.items():
        record("D2_mla_readers", f"w2048 set {name} records == low twin bytes",
               bit_equal(mla_records(slots), low))
        for s_ in slots:
            S.accept_actual(*G.mla_record_range(s_))
    check(S, "w2048 set writes: only declared records changed", "D2_mla_readers")
    _d2_reads(MLA_SETS_2048, 2048, ("decode", "extend"), q, g, res)
    check(S, "after GLM_NEXT reads", "D2_mla_readers")
    return res


@part("D3_mla_graph_replay")
def part_d3():
    """Writer and decode reader captured once in CUDA graphs (as under FULL graph
    execution), replayed with new slot / selection contents including high blocks."""
    plan, _ = mla_writer_plan()
    n = 16
    lat_buf = torch.zeros(n, 512, dtype=torch.bfloat16, device=DEV)
    slot_buf = torch.full((n,), -1, dtype=torch.int64, device=DEV)
    from b12x.attention import sparse_mla
    mla_write(plan, lat_buf, [-1] * n)  # warm
    gw = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gw):
        sparse_mla.concat_and_cache_glm_next_mla(lat_buf, MLA.view(torch.uint8), slot_buf, plan=plan)
    replays = [
        [G.mla_slot(3, t) for t in range(n)],
        [G.mla_slot(1274, 4352 - n + t) for t in range(n)],
        list(range(G.first_high_mla_slot() - 8, G.first_high_mla_slot() + 8)),
        [G.mla_slot(3, t) for t in range(n)],  # reuse same slots, new epoch
    ]
    for ep, slots in enumerate(replays):
        lat = torch.stack([SH.e4m3_exact_latent(SH.tag_code("mla", 0, i, ep + 2), scale_pow2=LAT_POW2)
                           for i in range(n)]).to(DEV)
        lat_buf.copy_(lat)
        slot_buf.copy_(torch.tensor(slots, dtype=torch.int64, device=DEV))
        gw.replay()
        torch.cuda.synchronize()
        # eager twin on block 4 for the same latents
        twin = [G.mla_slot(4, t) for t in range(n)]
        mla_write(plan, lat, twin)
        torch.cuda.synchronize()
        record("D3_mla_graph_replay", f"replay {ep} records == eager twin",
               bit_equal(mla_records(slots), mla_records(twin)))
        for s in slots + twin:
            S.accept_actual(*G.mla_record_range(s))
        check(S, f"graph writer replay {ep}", "D3_mla_graph_replay")
    # reader graph: capture on low selection, replay with high selections written into the bound buffer
    g = torch.Generator(device="cpu").manual_seed(23)
    q = (torch.randn(4, 64, 512, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    sel_low = [MLA_SETS["low"]] * 4
    # Post-run-2 change: run_unified_decode writes into the binding scratch's output_buffer
    # (b12x/attention/_shared/mla/kernel.py:3183), so the run-2 `eager_low` aliased `out_g` and
    # the replay comparisons compared the buffer with itself. Eager references are now cloned
    # and computed per selection with their own plan/scratch; the control's two conditions are
    # recorded separately.
    # Post-run-3 change: writer replays 1 and 2 above overwrote 16 records each of the
    # "last" (block 1274 t4336-4351) and "straddle" (slot 4,067,203 +-8) reader sets, so those
    # sets were no longer twins of "low". Restore the D1 latents on all reader sets first.
    for name, slots in MLA_SETS.items():
        mla_write(plan, mla_latents(1), slots)
    torch.cuda.synchronize()
    for name, slots in MLA_SETS.items():
        record("D3_mla_graph_replay", f"restored {name} == low twin bytes",
               bit_equal(mla_records(slots), mla_records(MLA_SETS["low"])))
        for s_ in slots:
            S.accept_actual(*G.mla_record_range(s_))
    check(S, "restore reader sets", "D3_mla_graph_replay")
    eager_low_raw, (_, binding, sel_t) = mla_read(sel_low, "decode", q)
    eager_low = eager_low_raw.clone()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out_g = sparse_mla.run(binding)
    for name in ("low", "straddle", "last", "b935"):
        sel = [MLA_SETS[name]] * 4
        eager_name = mla_read(sel, "decode", q)[0].clone()
        sel_t.copy_(torch.tensor(sel, dtype=torch.int32, device=DEV))
        gr.replay()
        torch.cuda.synchronize()
        record("D3_mla_graph_replay", f"reader replay {name} == eager {name}", bit_equal(out_g, eager_name))
        record("D3_mla_graph_replay", f"eager {name} == eager low (twin records)", bit_equal(eager_name, eager_low))
    ctrl = [[G.mla_slot(3, t) for t in range(16)] * 4] * 4
    eager_ctrl = mla_read(ctrl, "decode", q)[0].clone()
    record("D3_mla_graph_replay", "control selection output differs from low", not bit_equal(eager_ctrl, eager_low))
    sel_t.copy_(torch.tensor(ctrl, dtype=torch.int32, device=DEV))
    gr.replay()
    torch.cuda.synchronize()
    record("D3_mla_graph_replay", "control replay follows new selection", bit_equal(out_g, eager_ctrl))
    check(S, "after graph reads", "D3_mla_graph_replay")


# ------------------------------------------------------------------ E: kpool index-K + tail ring
def tail_expect(b, t, key, gate):
    o = b * G.IDX_PAGE + (t % G.TAIL_SLOTS) * G.TAIL_HEAD * 2
    SI.expect_bytes(o, key)
    SI.expect_bytes(o + G.TAIL_SLOTS * G.TAIL_HEAD * 2, gate)


@part("E1_kpool_tail_seed")
def part_e1():
    from vllm.models.glm5next.nvidia.ops.kpool_compress import kpool_seed_tail_cache
    blocks = [1, 2, 934, 935, 1274]
    toks = []
    for b in blocks:
        for pos in (8, 9, 10, 11):  # the request's last kpool tokens -> all seeded
            toks.append((b, pos))
    key = torch.stack([SH.bf16_tag_values("tail", b, p, 1, 128) for b, p in toks]).to(DEV)
    gate = -key
    ts = torch.tensor([b * G.TAIL_SLOTS + p % G.TAIL_SLOTS for b, p in toks], dtype=torch.int64, device=DEV)
    kpool_seed_tail_cache(TAIL, key, gate, ts, kpool=4)
    for i, (b, p) in enumerate(toks):
        tail_expect(b, p, key[i], gate[i])
    check(SI, "tail seed exact", "E1_kpool_tail_seed")
    # PAD rows (-1) write nothing
    kpool_seed_tail_cache(TAIL, key[:4], gate[:4], torch.full((4,), -1, dtype=torch.int64, device=DEV), kpool=4)
    check(SI, "tail seed PAD rows", "E1_kpool_tail_seed")


@part("E2_kpool_compress")
def part_e2():
    from vllm.models.glm5next.nvidia.ops.kpool_compress import kpool_compress_and_write_cache
    g = torch.Generator(device="cpu").manual_seed(29)
    pairs = [((3, 0), (1274, 1087)), ((3, 63), (935, 64)), ((3, 64), (934, 1023))]
    sk = (torch.randn(1, 4, 128, generator=g)).to(torch.bfloat16).to(DEV).repeat(6, 1, 1)
    ss = (torch.randn(1, 4, 128, generator=g)).to(torch.bfloat16).to(DEV).repeat(6, 1, 1)
    ape = (torch.randn(4, 128, generator=g) * 0.1).float().to(DEV)
    locs = [b * G.IDX_POOLS + p for pr in pairs for (b, p) in pr]
    kpool_compress_and_write_cache(IDX.view(torch.uint8), sk, ss, ape,
                                   torch.tensor(locs, dtype=torch.int64, device=DEV), pool_size=4)
    torch.cuda.synchronize()
    u8 = rawi.view(torch.uint8)
    for (lb, lp), (hb, hp) in pairs:
        for rng in (G.idx_k_range, G.idx_scale_range):
            a, b_ = rng(lb, lp), rng(hb, hp)
            record("E2_kpool_compress", f"{rng.__name__} {hb}/{hp} == {lb}/{lp}",
                   bit_equal(u8[b_[0]:b_[1]], u8[a[0]:a[1]]))
            SI.accept_actual(*a)
            SI.accept_actual(*b_)
    check(SI, "kpool compress: only K + scale of target pools", "E2_kpool_compress")


@part("E3_kpool_decode_update")
def part_e3():
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        kpool_decode_update_and_maybe_write_cache_batched, kpool_seed_tail_cache)
    g = torch.Generator(device="cpu").manual_seed(31)
    reqs = [(5, 6), (1273, 1274)]  # (tail block, idx block) per twin request
    # positions 12..15 (spec 1+3); pool 3 completes at pos 15; prior tail tokens none
    pos = torch.tensor([[12, 13, 14, 15]] * 2, dtype=torch.int32, device=DEV)
    ts = torch.tensor([[tb * 8 + p % 8 for p in range(12, 16)] for tb, _ in reqs], dtype=torch.int32, device=DEV)
    slot = torch.tensor([[-1, -1, -1, ib * G.IDX_POOLS + 3] for _, ib in reqs], dtype=torch.int32, device=DEV)
    key = (torch.randn(1, 4, 128, generator=g)).to(torch.bfloat16).to(DEV).repeat(2, 1, 1)
    sc = (torch.randn(1, 4, 128, generator=g)).to(torch.bfloat16).to(DEV).repeat(2, 1, 1)
    ape = (torch.randn(4, 128, generator=g) * 0.1).float().to(DEV)
    kpool_decode_update_and_maybe_write_cache_batched(IDX.view(torch.uint8), TAIL, ts, key, sc, ape, slot, pos, 4)
    torch.cuda.synchronize()
    u8 = rawi.view(torch.uint8)
    (lt, li), (ht, hi) = reqs
    ta, tb_ = G.tail_range(lt), G.tail_range(ht)
    # Post-run-2 change: run 2 compared the whole 4,096-B tail block, including ring slots
    # 0-3 that this step never writes (positions 12-15 -> slots 4-7, kpool_compress.py:593-594,
    # 702-712); those hold the offset-dependent sentinel and necessarily differ between blocks.
    # Now: twin compare over the written slots only, plus exact expected bytes (the stash stores
    # the bf16 key and gate unchanged, kpool_compress.py:612-622, 702-712).
    for half in (0, 2048):
        a0, b0_ = ta[0] + half + 4 * 256, tb_[0] + half + 4 * 256
        record("E3_kpool_decode_update", f"tail ring twin slots 4-7 half {half // 2048}",
               bit_equal(u8[b0_:b0_ + 1024], u8[a0:a0 + 1024]))
    for r, (tb, _) in enumerate(reqs):
        o = tb * G.IDX_PAGE
        for j, p in enumerate(range(12, 16)):
            kk = u8[o + (p % 8) * 256: o + (p % 8) * 256 + 256]
            gg = u8[o + 2048 + (p % 8) * 256: o + 2048 + (p % 8) * 256 + 256]
            record("E3_kpool_decode_update", f"stash exact b{tb} pos{p}",
                   bit_equal(kk, key[r, j].contiguous().view(torch.uint8))
                   and bit_equal(gg, sc[r, j].contiguous().view(torch.uint8)))
    for rng in (G.idx_k_range, G.idx_scale_range):
        a, b_ = rng(li, 3), rng(hi, 3)
        record("E3_kpool_decode_update", f"{rng.__name__} twin", bit_equal(u8[b_[0]:b_[1]], u8[a[0]:a[1]]))
        SI.accept_actual(*a)
        SI.accept_actual(*b_)
    for b in (lt, ht):
        o = b * G.IDX_PAGE
        for p in range(12, 16):  # stashed raw K + gate, ring slot p % 8
            SI.accept_actual(o + (p % 8) * 256, o + (p % 8) * 256 + 256)
            SI.accept_actual(o + 2048 + (p % 8) * 256, o + 2048 + (p % 8) * 256 + 256)
    check(SI, "decode update: tail ring slots + one pool only", "E3_kpool_decode_update")


# ------------------------------------------------------------------ F: zeroer + cross-group reuse
def zeroer():
    """Real KVBlockZeroer (v1/worker/utils.py) over fake attention groups in serving order."""
    from types import SimpleNamespace as NS
    from vllm.v1.worker.utils import KVBlockZeroer
    B12x, FA, IDXB, TAILB = V._backends()
    mla_s, idx_s, tail_s, draft_s = V.specs()
    ctx = {"mla.3": NS(kv_cache=MLA), "idx.3": NS(kv_cache=IDX), "tail.3": NS(kv_cache=TAIL),
           "draft.0": NS(kv_cache=KV)}
    groups = [NS(kv_cache_spec=mla_s, kv_cache_group_id=0, backend=B12x, layer_names=["mla.3"]),
              NS(kv_cache_spec=idx_s, kv_cache_group_id=0, backend=IDXB, layer_names=["idx.3"]),
              NS(kv_cache_spec=tail_s, kv_cache_group_id=1, backend=TAILB, layer_names=["tail.3"]),
              NS(kv_cache_spec=draft_s, kv_cache_group_id=9, backend=FA, layer_names=["draft.0"])]
    kbs = [64, 8] + [G.MLA_BLOCK_TOKENS] * 7 + [G.DRAFT_BLOCK_TOKENS]
    return KVBlockZeroer(DEV, groups, kbs, "fp8_ds_mla", ctx, G.NUM_BLOCKS)


@part("F1_zeroer")
def part_f1():
    z = zeroer()
    blocks = [5, 935, 1274]
    z.zero_block_ids(blocks)
    torch.cuda.synchronize()
    for b in blocks:
        S.expect_zero(*G.page_range(b))
        SI.expect_zero(*G.page_range(b, G.IDX_PAGE))
    check(S, "zeroer MLA slot tensor pages", "F1_zeroer")
    check(SI, "zeroer idx pages (incl. tail ring)", "F1_zeroer")
    return {"segments": z._meta[5] if z._meta else 0}


@part("F2_cross_group_reuse")
def part_f2():
    """Most frequent cross-group transitions in the reachability simulation (draft->MLA,
    KDA->KDA, MLA->draft, draft->KDA, KDA->MLA) on blocks 935 and 1274: zero as the runner
    does for attention groups, write as the new owner, read back through the new owner's
    reader, check isolation after each step."""
    from vllm.model_executor.layers.mamba.ops.gather_initial_states import gather_initial_states
    z = zeroer()
    plan, _ = mla_writer_plan()
    seq = ["draft", "mla", "kda", "kda", "draft", "kda", "mla"]
    for b in (935, 1274):
        for ep, owner in enumerate(seq):
            if owner in ("mla", "draft"):  # attention groups: new block ids are zeroed
                z.zero_block_ids([b])
                S.expect_zero(*G.page_range(b))
                SI.expect_zero(*G.page_range(b, G.IDX_PAGE))
            if owner == "draft":
                sl = [(b, t) for t in range(0, G.DRAFT_BLOCK_TOKENS, 17)]
                k, v = draft_payload(sl, ep)
                draft_write(sl, k, v)
                got = KC[b, [t for _, t in sl]]
                record("F2_cross_group_reuse", f"b{b} ep{ep} draft readback", bit_equal(got, k))
            elif owner == "mla":
                slots = [G.mla_slot(b, t) for t in range(0, G.MLA_BLOCK_TOKENS, 68)]
                lat = torch.stack([SH.e4m3_exact_latent(SH.tag_code("mla", b, i, ep), scale_pow2=LAT_POW2)
                                   for i in range(len(slots))]).to(DEV)
                mla_write(plan, lat, slots)
                torch.cuda.synchronize()
                for s in slots:
                    S.accept_actual(*G.mla_record_range(s))
                from b12x.attention._shared.mla.reference import unpack_mla_kv_cache_reference
                deq = unpack_mla_kv_cache_reference(mla_records(slots).view(-1, 1, 528)).view(-1, 512)
                ok = all(SH.e4m3_untag(deq[i], LAT_POW2) == SH.untag_code(SH.tag_code("mla", b, i, ep))
                         for i in range(len(slots)))
                record("F2_cross_group_reuse", f"b{b} ep{ep} mla tags", ok)
            else:
                vals = kda_ssm_payload([b], ep)
                kda_set_ssm([b], vals)
                got = gather_initial_states(SSM, torch.tensor([b], dtype=torch.int32, device=DEV),
                                            torch.ones(1, dtype=torch.bool, device=DEV))
                record("F2_cross_group_reuse", f"b{b} ep{ep} kda readback", bit_equal(got, vals))
            check(S, f"b{b} ep{ep} {owner}", "F2_cross_group_reuse")
    check(SI, "idx tensor after reuse cycle", "F2_cross_group_reuse")


@part("G_draft_graph_replay")
def part_g():
    """Draft writer captured in a CUDA graph with a persistent slot buffer (DFlash2 FULL
    graphs), replayed with new contents incl. high blocks, PAD and reused slots."""
    from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
    n = 16
    kb = torch.zeros(n, G.DRAFT_HEADS, G.DRAFT_HEAD, dtype=torch.bfloat16, device=DEV)
    vb = torch.zeros_like(kb)
    sb = torch.full((n,), -1, dtype=torch.int64, device=DEV)
    one = torch.ones((), dtype=torch.float32, device=DEV)
    reshape_and_cache_flash(kb, vb, KC, VC, sb, "auto", one, one)
    gg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gg):
        reshape_and_cache_flash(kb, vb, KC, VC, sb, "auto", one, one)
    plans = [[(6, t) for t in range(n)],
             [(1274, 1087 - t) for t in range(n)],
             [(935, t) if t % 2 else None for t in range(n)],
             [(6, t) for t in range(n)]]
    for ep, sl in enumerate(plans):
        k, v = draft_payload(sl, ep + 4)
        kb.copy_(k)
        vb.copy_(v)
        sb.copy_(torch.tensor([(-1 if s is None else s[0] * G.DRAFT_BLOCK_TOKENS + s[1]) for s in sl],
                              dtype=torch.int64, device=DEV))
        gg.replay()
        torch.cuda.synchronize()
        for i, s in enumerate(sl):
            if s is None:
                continue
            for h in range(G.DRAFT_HEADS):
                S.expect_bytes(G.draft_k_range(*s, h)[0], k[i, h])
                S.expect_bytes(G.draft_v_range(*s, h)[0], v[i, h])
        check(S, f"draft graph replay {ep}", "G_draft_graph_replay")


@part("Z_final")
def part_z():
    check(S, "final mla slot tensor", "Z_final")
    check(SI, "final idx tensor", "Z_final")


if __name__ == "__main__":
    for fn in (part_a, part_b, part_c1, part_c2, part_c2b, part_c3, part_c4, part_c5, part_d1, part_d2,
               part_d3, part_e1, part_e2, part_e3, part_f1, part_f2, part_g, part_z):
        fn()
    R["meta"]["all_parts_ok"] = all(p.get("ok") for p in R["parts"].values())
    R["meta"]["failed_checks"] = [c for c in R["checks"] if not c["ok"]]
    dump()
    print(json.dumps({k: {"ok": v.get("ok"), "secs": v.get("secs"), "error": v.get("error")}
                      for k, v in R["parts"].items()}, indent=1))
    sys.exit(0 if R["meta"]["all_parts_ok"] else 3)
