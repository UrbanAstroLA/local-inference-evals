"""Shared helpers for the captured-p3b offline replays (run inside the stock images; numpy + torch).

Capture layout (written by p3b_capture_hooks's poller; see build/files/vllm/p3b_capture_hooks.py):
  <cap>/moe/L<layer>/s<seq>.npz   x (MMAX,H) bf16-as-uint16, ids (MMAX,8) int64, w (MMAX,8) f32, out (MMAX,H) uint16,
                                  hdr (8,) int64 [seq_begin, step, m_static, ntok, ntok_pad, kind, aux, seq_end]
  <cap>/att/L<layer>/s<seq>.npz   q (NA,Hc,D) uint16, pre/post/logical (NA,W) int32, rec (NA,W,RB) uint8,
                                  slot0 (1,RB) uint8, out (NA,Hc,Dv) uint16, scal (NA,6) int64
                                  [row, valid, pos, L, nsa_post, cache_len], [lse], hdr (aux = cache slots)
  <caplog>/cap_p*.jsonl           init (rank 0: att_info incl. the served decode Caps), site and moe_bind records,
                                  entry records (sha256 of each .npz)
Nothing here writes captured values to public outputs: callers put only counts, flags, digests and comparison
magnitudes into their JSON.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time

import numpy as np

H_BEGIN, H_STEP, H_M, H_NTOK, H_NPAD, H_KIND, H_AUX, H_END = range(8)
S_ROW, S_VALID, S_POS, S_L, S_NSA, S_CLEN = range(6)
W = 2048


def load_caplog(caplog: str) -> dict:
    recs = []
    for p in sorted(glob.glob(os.path.join(caplog, "cap_*.jsonl"))):
        for line in open(p):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    init0 = [r for r in recs if r.get("kind") == "init" and r.get("rank") == 0 and "att_info" in r]
    binds, sites = {}, {}
    for r in recs:
        if r.get("kind") != "site":
            continue
        key = r.get("key") or []
        if key and key[0] == "moe_bind":
            binds[(int(r["layer"]), int(r["m"]))] = dict(plan=r["plan"], launch=r.get("launch"),
                                                         tile_config=r.get("tile_config"))
        elif key:
            sites[tuple(key)] = r
    entries = {(r["ring"], int(r["layer"]), int(r["seq"])): r for r in recs if r.get("kind") == "entry"}
    return dict(init=init0[0] if init0 else None, binds=binds, sites=sites, entries=entries, n_records=len(recs))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_entries(cap: str, ring: str, layer: int) -> list[tuple[int, str]]:
    out = []
    for p in sorted(glob.glob(os.path.join(cap, ring, f"L{layer}", "s*.npz"))):
        out.append((int(os.path.basename(p)[1:-4]), p))
    return out


def load_npz(path: str, caplog: dict | None = None, ring: str = "", layer: int = -1, seq: int = -1) -> dict:
    """Load one entry; verify its sha256 against the capture log's entry record when available."""
    d = dict(np.load(path))
    d["_sha_ok"] = None
    if caplog is not None:
        rec = caplog["entries"].get((ring, layer, seq))
        if rec is not None:
            d["_sha_ok"] = file_sha256(path) == rec.get("sha256")
            d["_phase"] = rec.get("phase")
            d["_cats"] = rec.get("cats")
    return d


def bf16(torch, a: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16)


def deserialize_caps(sparse_mla, torch, d: dict, device):
    """Captured Caps dict -> Caps(**constructor fields).  Derived fields are recomputed by Caps.__post_init__."""
    ctor = ("device", "num_q_heads", "max_q_rows", "max_width", "softmax_scale", "dtype", "kv_dtype", "head_dim",
            "v_head_dim", "model_type", "scale_format", "cache_record_bytes", "fp8_rope", "latent_scale_per_token",
            "mode", "max_batch", "max_chunks_per_row", "page_size", "head_major_output", "return_lse", "lse_scale")
    kw, skipped = {}, []
    for k in ctor:
        if k not in d:
            skipped.append(k)
            continue
        v = d[k]
        if isinstance(v, dict):
            if "float_hex" in v:
                v = float.fromhex(v["float_hex"])
            elif "dtype" in v:
                v = getattr(torch, v["dtype"].split(".")[-1])
            elif "device" in v:
                v = device
            elif "enum" in v:
                v = v["value"]
            else:
                skipped.append(k)
                continue
        if k == "device":
            v = device
        kw[k] = v
    return sparse_mla.Caps(**kw), skipped


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


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
