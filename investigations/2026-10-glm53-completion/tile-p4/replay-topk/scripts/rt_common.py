"""replay-topk: shared helpers, run INSIDE the stock images (torch + the image's vLLM / b12x).

Everything here is synthetic-input plumbing: geometry of the served chunk, deterministic synthetic tensors, exact
comparisons, JSON output.  No model weights, no prompt data, no captured tensors.  Outputs carry only synthetic data,
counts and magnitudes, so they are public-safe (nothing is written under a PRIVATE path).

Served geometry (stock v0.9.1, warm chunk 1 of the capture-p0 prompt; PROTOCOL.md section 2):
  query rows        token positions 2,045 .. 4,089 (2,045 rows), request seq_len 4,090
  indexer window    row p sees pools [0, (p + 1) // 4)  (indexer.py BuildPrefillChunkMetadataKernel, DCP_WORLD=1:
                    cu_seqlen_ks = row_start = 0, cu_seqlen_ke = row_start + (start_pos + 1 + offset) // COMPRESS_RATIO)
  logits            [2,045, 1,022] fp32 from DeepGEMM fp8_fp4_mqa_logits (N = compressed seq len 4,090 // 4)
  top-k             select_k = topk_tokens // index_kpool = 2,048 // 4 = 512 pools
  expand            pool_ids[:, :511] -> 2,044 token ids + tail (seq_len % 4) at columns 2,044..2,046; buffer width 2,048
  sparse MLA        64 heads (v0.9.1 owner-local layer 3; v0.7.0 32 local x DCP 2 gathered), latent 512, page 64
"""
from __future__ import annotations

import hashlib
import json
import os
import time

CHUNK_START = 2045
CHUNK_ROWS = 2045
SEQ_LEN = CHUNK_START + CHUNK_ROWS          # 4,090
KPOOL = 4
TOPK_TOKENS = 2048
SELECT_K = TOPK_TOKENS // KPOOL             # 512
N_POOLS = SEQ_LEN // KPOOL                  # 1,022 (compressed seq len = logits width)
INDEX_HEADS = 32
INDEX_DIM = 128
MLA_HEADS = 64
KV_LORA = 512
PAGE = 64
W = TOPK_TOKENS                             # topk_indices_buffer width
EXPANDED_COLS = (SELECT_K - 1) * KPOOL + KPOOL - 1   # 2,047
FIRST_SELECT_POS = 4 * (SELECT_K + 1) - 1   # 2,051: first p with (p + 1) // 4 > 512


def positions():
    return list(range(CHUNK_START, CHUNK_START + CHUNK_ROWS))


def pools_seen(p: int) -> int:
    return (p + 1) // KPOOL


class Deadline:
    def __init__(self, seconds):
        self.t0 = time.time()
        self.s = seconds

    def left(self) -> float:
        return float("inf") if self.s is None else self.s - (time.time() - self.t0)

    def passed(self) -> bool:
        return self.left() <= 0


def write_json(path: str, doc: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, default=str)
    os.replace(tmp, path)


def tensor_sha(torch, t) -> str:
    """sha256 of a tensor's logical contents (host copy, contiguous, raw bytes)."""
    a = t.detach().contiguous().cpu()
    if a.dtype in (torch.bfloat16, torch.float16):
        a = a.view(torch.int16)
    elif a.dtype in (getattr(torch, "float8_e4m3fn", None),):
        a = a.view(torch.uint8)
    return hashlib.sha256(a.numpy().tobytes()).hexdigest()


def gen(torch, seed: int, dev):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def env_record(torch, dev) -> dict:
    out = dict(torch=torch.__version__, cuda=getattr(torch.version, "cuda", None), device=str(dev),
               env={k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("B12X_", "VLLM_", "DG_", "CUDA_",
                                                                                        "TRITON_", "PYTORCH_"))})
    if dev.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(dev)
        out["capability"] = list(torch.cuda.get_device_capability(dev))
        out["stream"] = str(torch.cuda.current_stream(dev))
    return out


def import_vllm_modules():
    """The image's served modules.  The model package is imported first: importing the kpool indexer module on its
    own hits a circular import (glm5next attention imports it back)."""
    import vllm.models.glm5next.nvidia.attention  # noqa: F401
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as kp
    import vllm.models.glm5next.nvidia.ops.kpool_compress as kc
    import vllm.v1.attention.backends.mla.indexer as ix
    import vllm.v1.attention.backends.mla.b12x_mla_sparse as bm
    return kp, kc, ix, bm


def served_window(torch, ix, dev):
    """cu_seqlen_ks / cu_seqlen_ke for the chunk, from the image's own metadata builder (indexer.py
    build_prefill_chunk_metadata), with the arguments DeepseekV32IndexerMetadataBuilder.build passes for one
    single-request prefill chunk at DCP=1 (split_indexer_prefill_chunks -> one (req_slice, query_slice(0, 2045)))."""
    i32 = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    qsl_cpu = i32([0, CHUNK_ROWS])
    m = ix.build_prefill_chunk_metadata(
        0, 1, qsl_cpu.to(dev), qsl_cpu, i32([SEQ_LEN]).to(dev), i32([N_POOLS]).to(dev), i32([N_POOLS]),
        torch.zeros((1, (SEQ_LEN + 4351) // 4352 * 68), dtype=torch.int32, device=dev), KPOOL,
        query_slice=slice(0, CHUNK_ROWS), skip_kv_gather=False, dcp_rank=0, dcp_world_size=1,
        cp_kv_cache_interleave_size=1)
    return m


def formula_window(torch, dev):
    p = torch.arange(CHUNK_START, CHUNK_START + CHUNK_ROWS, dtype=torch.int32)
    ks = torch.zeros(CHUNK_ROWS, dtype=torch.int32)
    ke = ((p + 1) // KPOOL).to(torch.int32)
    return ks.to(dev), ke.to(dev)


def bits_equal_rows(torch, a, b):
    """Per-row exact bit equality of two same-shaped tensors (any dtype): [rows] bool."""
    if a.dtype in (torch.bfloat16, torch.float16):
        a, b = a.view(torch.int16), b.view(torch.int16)
    elif a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)
    return (a == b).reshape(a.shape[0], -1).all(dim=1)


def row_magnitudes(torch, a, b):
    """max |a - b| and ||a - b|| / ||b|| per row, in fp32 of the output values."""
    d = a.float() - b.float()
    absd = torch.nan_to_num(d.abs(), nan=float("inf")).reshape(a.shape[0], -1)
    ref = b.float().reshape(b.shape[0], -1).norm(dim=1)
    rel = d.reshape(d.shape[0], -1).norm(dim=1) / ref.clamp_min(1e-30)
    return absd.max(dim=1).values, rel


def summarize_mag(torch, mask, absmax, rel) -> dict:
    """Aggregate magnitudes over the rows in mask (bool tensor)."""
    n = int(mask.sum())
    if n == 0:
        return dict(rows=0, max_abs=0.0, rel_l2_median=None, rel_l2_max=None)
    a = absmax[mask].float()
    r = rel[mask].float()
    return dict(rows=n, max_abs=float(a.max()), max_abs_median=float(a.median()),
                rel_l2_median=float(r.median()), rel_l2_max=float(r.max()))
