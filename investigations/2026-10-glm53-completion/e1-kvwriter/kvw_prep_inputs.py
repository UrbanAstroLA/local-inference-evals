#!/usr/bin/env python3
"""KV-writer / query-projection comparison -- INPUT PREPARATION (CPU only). See KVWRITER-PREP.md.

Builds two input sets and writes them to OUT/kvw_inputs.pt (+ .sha256):

  real-weight proxy ("proxy"): Stage A captured the MoE gate input x (bf16, 4096) of real serving rows
    (long-context prefill and decode, col511-gate fixture).  MLA latents and queries were NOT captured, so
    the attention input is approximated per row as h = x / w_post * w_in (both RMSNorm weights of the
    same layer; x is unit-RMS before w_post), then the checkpoint's own BF16 weights give
      kv_c   = kv_a_layernorm(kv_a_proj_with_mqa(h))            (T, 512)  -- what the cache writer stores
      q_nope = q_b_proj(q_a_layernorm(q_a_proj(h)))[..., :256]  (64 heads, B, 256)
      W_UK_T = kv_b_proj.weight.T.view(512, 64, 512)[..., :256].permute(1, 2, 0)   (64, 256, 512)
    bf16 rounding after each linear and norm (distributionally realistic, not bitwise to serving).
  synthetic ("synth"): Gaussian latents with per-channel scales taken from the proxy latents, plus edge
    cases: all-zero groups (unit-scale path), single large outliers, groups entirely in the E4M3
    subnormal range after scaling, ties (values exactly on E4M3 midpoints after scaling).

Usage: kvw_prep_inputs.py CAPTURE_DIR CKPT_SNAPSHOT OUT_DIR [--layers 3,23,43] [--q-rows 64]
"""
import argparse
import glob
import hashlib
import json
import os
import struct

import torch

H, P, L, NH = 4096, 256, 512, 64
EPS = 1e-5


def ckpt_getter(d):
    mp = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    def get(key):
        f = os.path.join(d, mp[key])
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
            m = h[key]
            s, e = m["data_offsets"]
            fh.seek(8 + n + s)
            buf = bytearray(fh.read(e - s))
        dt = {"F32": torch.float32, "BF16": torch.bfloat16}[m["dtype"]]
        return torch.frombuffer(buf, dtype=dt).reshape(m["shape"]).clone()
    return get


def lin(x, w):
    return (x.float() @ w.float().T).to(torch.bfloat16)


def rms(x, w):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (y * w.float()).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--layers", default="3,23,43")
    ap.add_argument("--q-rows", type=int, default=64)
    a = ap.parse_args()
    get = ckpt_getter(a.ckpt)
    pfx = "model.language_model.layers.{}."
    out = {"meta": {"note": "see kvw_prep_inputs.py docstring", "layers": {}}, "proxy": {}, "synth": {}}
    g = torch.Generator().manual_seed(20261012)
    for layer in [int(v) for v in a.layers.split(",")]:
        files = sorted(glob.glob(os.path.join(a.capture, "rec_r0", f"*_l{layer:02d}.pt")))
        xs, kinds = [], []
        for f in files:
            rec = torch.load(f, map_location="cpu")
            xs.append(rec["x"])
            kinds.append((rec["meta"].get("rows") or {}).get("batch_kind") or "unknown")
        x = torch.cat(xs)
        p = pfx.format(layer)
        w_in, w_post = get(p + "input_layernorm.weight"), get(p + "post_attention_layernorm.weight")
        h = (x.float() / w_post.float() * w_in.float()).to(torch.bfloat16)
        kv = lin(h, get(p + "self_attn.kv_a_proj_with_mqa.weight"))
        kv_c = rms(kv, get(p + "self_attn.kv_a_layernorm.weight"))
        sel = torch.randperm(h.shape[0], generator=g)[: a.q_rows]
        qa = rms(lin(h[sel], get(p + "self_attn.q_a_proj.weight")), get(p + "self_attn.q_a_layernorm.weight"))
        q = lin(qa, get(p + "self_attn.q_b_proj.weight")).view(-1, NH, P)      # qk_rope_head_dim = 0
        q_nope = q.permute(1, 0, 2).contiguous()                                 # (N, B, P)
        wkv = get(p + "self_attn.kv_b_proj.weight")                              # (N*(P+V), L)
        W_UK_T = wkv.T.contiguous().view(L, NH, P + 256)[..., :P].permute(1, 2, 0).contiguous()  # (N, P, L)
        out["proxy"][layer] = {"kv_c": kv_c, "q_nope": q_nope, "W_UK_T": W_UK_T}
        out["meta"]["layers"][layer] = {"tokens": int(kv_c.shape[0]), "q_rows": int(q_nope.shape[1]),
                                        "records": len(files), "batch_kinds": sorted(set(kinds)),
                                        "kv_c_absmax": float(kv_c.float().abs().max()),
                                        "kv_c_rms": float(kv_c.float().pow(2).mean().sqrt())}
    # synthetic set, per-channel scale from the first proxy layer
    ref = next(iter(out["proxy"].values()))["kv_c"].float()
    ch = ref.std(0).clamp_min(1e-3)
    T = 4096
    s = (torch.randn(T, L, generator=g) * ch).to(torch.bfloat16)
    s[::97, :128] = 0                                     # all-zero group -> unit scale (B12x) / FLT_MIN (vLLM)
    s[::13, 5] = 40 * ch[5]                               # single large outlier: rest of group near subnormal
    s[::31, 128:256] = (torch.randn(128, generator=g) * 1e-6).to(torch.bfloat16)  # tiny group
    # values landing near E4M3 rounding midpoints after amax/448 scaling: amax fixed at 448*2^k, then
    # value = (m + 0.5) * step with bf16-exact midpoints
    tie = torch.tensor([448.0] + [float((8 + j % 8) + 0.5) * 2.0 ** ((j // 8) % 14 - 9) for j in range(127)])  # E4M3 midpoints in [2^-6, 248]
    s[7::101, 256:384] = tie.to(torch.bfloat16)
    out["synth"]["kv_c"] = s
    out["meta"]["synth"] = {"tokens": T, "edge_cases": ["zero group every 97th", "outlier every 13th",
                                                       "tiny group every 31st", "midpoint group every 101st (from 7)"]}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "kvw_inputs.pt")
    torch.save(out, path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    open(path + ".sha256", "w").write(f"{digest}  kvw_inputs.pt\n")
    print(json.dumps(out["meta"], indent=1), digest)


if __name__ == "__main__":
    main()
