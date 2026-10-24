#!/usr/bin/env python3
"""CPU test, run INSIDE each image (CUDA hidden, TRITON_INTERPRET=1): the replay's calls against the served code.

usage: test_served_args.py v091|v070
Checks (each prints PASS/FAIL; exit 1 on any FAIL):
  S1  the cited in-image files have the sha256 recorded by version-cmp (so every file:line in PROTOCOL.md is the code
      that runs), and the cited lines hold the cited code (kpool 574, 576-581, 600-609, 644; indexer window kernel;
      backend prefill call sites)
  S2  the compiled op schema (read from the _C binary's registration string) has the argument order the replay uses
  S3  the served window builder gives ks = 0, ke = (p + 1) // 4 for the chunk, and 6 rows with <= 512 pools
  S4  the Caps keyword set the replay passes equals the keyword set of the served _make_plan Caps(...) call (AST of
      the image's b12x_mla_sparse.py), and the values follow the served configuration
  S5  index construction through the served Triton helpers on a row subset (positions incl. 2,050-2,052 and 4,089):
      expand keeps pool order and drops column 511; tail columns 2,044..2,046; v0.9.1 physical slots follow the block
      table in column order and nsa = min(p + 1, 2048); v0.7.0 filter keeps only rank-0 (even) tokens, valid count
      = nsa
  S6  softmax scale: attention.py computes scaling = qk_head_dim ** -0.5 with qk_head_dim = 256 (GLM-5.3 config)
"""
from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/s")

VER = sys.argv[1]
V = "/usr/local/lib/python3.12/dist-packages/vllm/"
B = "/opt/b12x/b12x/"
SHA = {  # version-cmp/sources-sha256.txt (070, 091); None = not cited for that image
    V + "model_executor/layers/sparse_attn_indexer_kpool.py": ("e7935f2a9b8ecbc75410defff571d9940ad2abf511286bf0936d0c324936df39",) * 2,
    V + "v1/attention/backends/mla/indexer.py": ("67449677b787f37e31f69de97269097907ebda395879eb67d0b2ebb991a2f0dc",) * 2,
    V + "v1/attention/backends/mla/sparse_utils.py": ("a3cf9998ba60d51bbc50dfc093162adc01482f37cacbc6b3bee909cfca4cb228",) * 2,
    V + "_custom_ops.py": ("5ca3b8955ff2dca446813057e6678065a44ec283d73db4a9d8fd24328cb4ed07",) * 2,
    V + "_C_stable_libtorch.abi3.so": ("2b096e4bf49725e0498536c60ee0377e366d188ca5dae94d89da2c5d1e87b612",) * 2,
    V + "v1/attention/backends/mla/b12x_mla_sparse.py": (
        "4b160792231daed43816676586fbf42e8269a60ebb1f1e92215880b48d19c757",
        "6e74c42cf4385bb0451243098ec8b4a84005675fa776169fddb2fb230bab0157"),
    V + "models/glm5next/nvidia/ops/kpool_compress.py": (
        "52959ea33234db6e7cfa94289dd263a15add3be4d2107285418c830b5c19e472",
        "7fb8eab0d71c2c576f995cc91ffc79c5055269b9bed3b094247f36bc503b241d"),
    V + "models/glm5next/nvidia/attention.py": (
        "7d42294c63ea52a7845a6ad0289ae3596098f5aee857718cfed34d6dcf0418f2",
        "c3b368160f00a60c192873ba127bb1c5ec8a98bbc7e1ee7a887152f51179db53"),
    B + "attention/_shared/mla/prefill.py": (
        "a48ba27a04629b5cb7eba1e24fc07e6af8942d6d43c312e065438040f5e1e050",
        "1197f081741a2b1977586ccdf519df0855ee710d31d3efc56b2b1f4028d1cb4c"),
    B + "attention/_shared/mla/prefill_mg.py": (
        "70e450ff645684ae3a97aff3004988a6b596dbe6ba0bcfe2e340f94b9eb7fb60",
        "73fdfab3d46fdbdf303ad98665d74ce76f42c13c99287338a1ae607d414b93eb"),
    B + "attention/sparse_mla/_preparation.py": (None, "86ae220dbe5f680f22c9e7726d0c3a8845b57ad492c744ef1cac9d4baa885ced"),
}
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {VER} {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def lines(p, a, b):
    return "".join(open(p).readlines()[a - 1:b])


# S1
k = 0 if VER == "v070" else 1
for p, pair in SHA.items():
    if pair[k] is None:
        continue
    got = sha(p) if os.path.exists(p) else None
    check(f"S1 sha {p.replace(V, 'vllm/').replace(B, 'b12x/')}", got == pair[k], "" if got == pair[k] else f"got {got}")
KP = V + "model_executor/layers/sparse_attn_indexer_kpool.py"
check("S1 kpool:574 select_k", "select_k = topk_tokens // index_kpool" in lines(KP, 574, 574))
check("S1 kpool:576-581 pool_topk full -1 int32", all(s in lines(KP, 576, 581) for s in ("torch.full(", "-1,", "dtype=torch.int32")))
check("S1 kpool:600-609 served call", "torch.ops._C.top_k_per_row_prefill(" in lines(KP, 600, 600)
      and all(s in lines(KP, 601, 609) for s in ("logits,", "chunk.cu_seqlen_ks,", "chunk.cu_seqlen_ke,", "topk_dst,",
                                                 "num_rows,", "logits.stride(0),", "logits.stride(1),", "select_k,")))
check("S1 kpool:461 buffer reset", "topk_indices_buffer[: hidden_states.shape[0]] = -1" in lines(KP, 461, 461))
check("S1 kpool:644 expand drops col 511", "pool_ids[:, : select_k - 1], q_seq, index_kpool" in lines(KP, 644, 644))
check("S1 kpool:561-568 logits clean_logits=False", "fp8_fp4_mqa_logits(" in lines(KP, 561, 561)
      and "clean_logits=False" in lines(KP, 561, 568))
IX = V + "v1/attention/backends/mla/indexer.py"
src_ix = open(IX).read()
check("S1 indexer window rule", "global_ctx = start_pos + 1 + offset" in src_ix
      and "len_per_token = global_ctx // COMPRESS_RATIO" in src_ix and "row_start + len_per_token" in src_ix)
BM = V + "v1/attention/backends/mla/b12x_mla_sparse.py"
src_bm = open(BM).read()
if VER == "v091":
    check("S1 091 prefill bind/run call", "binding = self._b12x_sparse_mla.bind(\n                extend_plan," in src_bm
          and "self._b12x_sparse_mla.run(binding)" in src_bm)
    check("S1 091 index path", all(s in src_bm for s in ("_logical_topk_to_physical_slots(", "nsa_cache_seqlens.copy_(per_token_cache)",
                                                       "nsa_cache_seqlens.clamp_max_(topk_indices.shape[1])",
                                                       "_compact_dropped_selection(", "_mask_page_table_after_nsa_len(")))
    check("S1 091 GLM_NEXT writer", "concat_and_cache_glm_next_mla(" in src_bm)
else:
    check("S1 070 prefill bind/run_extend call", "binding = extend_plan.bind(" in src_bm
          and "self._sparse_mla_extend_forward(" in src_bm and 'lse_scale="natural"' in src_bm)
    check("S1 070 DCP index path", "triton_filter_and_convert_dcp_index(" in src_bm
          and "torch.minimum(\n                nsa_cache_seqlens,\n                per_token_cache," in src_bm)

# S2
so = V + "_C_stable_libtorch.abi3.so"
strs = subprocess.run(["strings", "-n", "8", so], capture_output=True, text=True).stdout
want = ("top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, Tensor! indices, int numRows, "
        "int stride0, int stride1, int topK) -> ()")
check("S2 op schema in binary", want in strs)

import torch  # noqa: E402
import rt_common as rc  # noqa: E402

kp, kc, ix, bm = rc.import_vllm_modules()
# S3
m = rc.served_window(torch, ix, torch.device("cpu"))
fks, fke = rc.formula_window(torch, torch.device("cpu"))
check("S3 window == formula", torch.equal(m.cu_seqlen_ks, fks) and torch.equal(m.cu_seqlen_ke, fke))
check("S3 logits width 1,022", int(m.max_local_total_seq_lens) == rc.N_POOLS)
plen = m.cu_seqlen_ke - m.cu_seqlen_ks
check("S3 rows with <= 512 pools = positions 2045-2050", int((plen <= 512).sum()) == 6 and int(plen[6]) == 513)

# S4: Caps keyword set vs the served _make_plan call
tree = ast.parse(src_bm)
served_kw = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_make_plan":
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and getattr(sub.func, "attr", getattr(sub.func, "id", "")) in (
                    "Caps", "B12XSparseMLAScratchCaps"):
                served_kw = {kw.arg: ast.unparse(kw.value) for kw in sub.keywords}
check("S4 served Caps(...) found", served_kw is not None)
if VER == "v091":
    ours = dict(device="dev", num_q_heads=64, max_q_rows=2048, max_width=2048, softmax_scale=0.0625, dtype="bf16",
                kv_dtype="uint8", head_dim=512, v_head_dim=512, model_type="GLM_NEXT", scale_format=None,
                cache_record_bytes=528, fp8_rope=None, latent_scale_per_token=False, mode="extend", max_batch=16,
                max_chunks_per_row=32, page_size=64, head_major_output=False, return_lse=True, lse_scale="natural")
else:
    ours = dict(device="dev", num_q_heads=64, max_q_rows=2048, max_width=2048, dtype="bf16", kv_dtype="uint8",
                head_dim=576, v_head_dim=512, mode="extend", max_batch=16, max_chunks_per_row=32, page_size=64,
                head_major_output=True)
if served_kw is not None:
    check("S4 keyword set equal", set(served_kw) == set(ours),
          f"served-only {sorted(set(served_kw) - set(ours))}, ours-only {sorted(set(ours) - set(served_kw))}")
    for kname in sorted(served_kw):
        print(f"  INFO {VER} Caps.{kname}: served `{served_kw[kname]}` -> replay {ours.get(kname)!r}")
check("S4 split cap = ceil(2048 / 64)", "self._num_splits_cap = max(1, _cdiv(self.topk_tokens, _DECODE_SPLIT_TILE))" in src_bm
      and "_DECODE_SPLIT_TILE = 64" in src_bm)
check("S4 kernel heads = ceil(heads x dcp / 8) x 8", "self._input_num_heads = self.num_heads * self.dcp_world_size" in src_bm
      and "_HEAD_ALIGNMENT = 8" in src_bm)
ATT = V + "models/glm5next/nvidia/attention.py"
src_att = open(ATT).read()
if VER == "v091":
    check("S4 091 owner-local runs all heads", "tp_size = 1 if self.owner_local else get_tensor_model_parallel_world_size()" in src_att)
    check("S4 091 NoPE record needs DCP=1 (528 B)", "GLM53_NOPE_RECORD_BYTES" in src_bm and "requires DCP=1" in src_bm)
else:
    check("S4 070 rope_pad 64 (q 576, record 656)", "self.rope_pad = 64" in src_bm and "else 656" in src_bm)

# S5 on a row subset
rows = [0, 5, 6, 7, 1000, 2044]                      # positions 2045, 2050, 2051, 2052, 3045, 4089
pos = torch.tensor([rc.CHUNK_START + r for r in rows], dtype=torch.int32)
g = torch.Generator().manual_seed(3)
pool = torch.full((len(rows), 512), -1, dtype=torch.int32)
for i, p in enumerate(pos.tolist()):
    n = (p + 1) // 4
    pool[i] = torch.arange(512, dtype=torch.int32) if n <= 512 else torch.randperm(n, generator=g)[:512].to(torch.int32)
    if n <= 512:
        pool[i, n:] = -1
exp = kc.expand_pools_and_append_tail(pool.to(torch.int64)[:, :511], pos + 1, 4)
ok = exp.shape[1] == 2047
for i, p in enumerate(pos.tolist()):
    n = (p + 1) // 4
    want_hist = torch.tensor([(int(pid) * 4 + o) if 0 <= int(pid) < n else -1 for pid in pool[i, :511] for o in range(4)])
    tail = (p + 1) % 4
    want_tail = torch.tensor([n * 4 + t if t < tail else -1 for t in range(3)])
    ok &= torch.equal(exp[i, :2044].long(), want_hist) and torch.equal(exp[i, 2044:].long(), want_tail)
check("S5 expand keeps pool order, drops col 511, tail at 2044..2046", bool(ok))
topk = torch.full((len(rows), 2048), -1, dtype=torch.int32)
topk[:, :2047] = exp
req = torch.zeros(len(rows), dtype=torch.int32)
if VER == "v091":
    bt = torch.randperm(80, generator=g)[:68].to(torch.int32)[None, :]
    sel = torch.full_like(topk, -1)
    bm._logical_topk_to_physical_slots(req, bt, topk, sel, 64)
    nsa = (pos + 1).clamp_max(2048).to(torch.int32)
    bm._compact_dropped_selection(sel, nsa)
    bm._mask_page_table_after_nsa_len(sel, nsa)
    lg = topk.long()
    want = torch.where(lg >= 0, bt[0][(lg.clamp_min(0) // 64)].long() * 64 + lg.clamp_min(0) % 64, torch.full_like(lg, -1))
    sel_rows = [i for i, p in enumerate(pos.tolist()) if p + 1 >= 2048]
    check("S5 091 physical slots in column order (rows p + 1 >= 2048)",
          all(torch.equal(sel[i].long(), want[i]) for i in sel_rows))
    check("S5 091 nsa = min(p + 1, 2048)", torch.equal(nsa, (pos + 1).clamp_max(2048).to(torch.int32)))
else:
    import vllm.v1.attention.backends.mla.sparse_utils as su
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
    bt = torch.randperm(48, generator=g)[:34].to(torch.int32)[None, :]
    sel = torch.full_like(topk, -1)
    nsa = torch.zeros(len(rows), dtype=torch.int32)
    su.triton_filter_and_convert_dcp_index(req, bt, topk, dcp_size=2, dcp_rank=0, cp_kv_cache_interleave_size=1,
                                           BLOCK_SIZE=64, NUM_TOPK_TOKENS=2048, out=sel, valid_counts=nsa)
    loc = get_dcp_local_seq_lens(pos + 1, 2, 0, 1).to(torch.int32)
    torch.minimum(nsa, loc, out=nsa)
    bm._mask_page_table_after_nsa_len(sel, nsa)
    ok = True
    order_same = 0
    for i in range(len(rows)):
        tok = topk[i][topk[i] >= 0].long()
        own = tok[tok % 2 == 0] // 2
        want_set = set((bt[0][own // 64].long() * 64 + own % 64).tolist())
        k_ = int(nsa[i])
        got = sel[i, :k_].long()
        ok &= set(got.tolist()) == want_set and int((sel[i] >= 0).sum()) == k_ == len(want_set)
        order_same += int(torch.equal(got, (bt[0][own // 64].long() * 64 + own % 64)))
    check("S5 070 filter keeps rank-0 tokens, valid count = nsa", bool(ok))
    print(f"  INFO {VER} filter output in input order on {order_same}/{len(rows)} rows (CPU interpreter; GPU order measured in R2)")
# S6
check("S6 scaling = qk_head_dim ** -0.5", "self.scaling = self.qk_head_dim**-0.5" in src_att
      and "self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim" in src_att)
print(f"{VER}: {len(FAILS)} failure(s)")
sys.exit(1 if FAILS else 0)
