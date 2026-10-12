#!/usr/bin/env python3
"""KV-writer / query-projection comparison -- GPU PART. PREPARED, NOT RUN (PLAN.md amendment 16).

Runs inside one serving image on ONE idle GPU, model-free (no server, no checkpoint load):
  writers      vLLM ops.concat_and_cache_mla(kv_cache_dtype="fp8_ds_mla")  -> 656 B records (0.7.0 writer)
               B12x sparse_mla.concat_and_cache_glm_next_mla                -> 528 B records (B0 writer)
               on the identical bf16 latents of kvw_inputs.pt (identity slots, then permuted slots)
  projections  torch.bmm(q_nope, W_UK_T)                     (B0 path, 512 wide)
               b12x mla_query_projection.run_glm_h64_bf16    (0.7.0 path for 2 <= rows <= 16, 576 wide)
               on production-style views (token-major q exposed head-major), rows in {1, 2, 4, 8, 16}
Each op runs twice (repeatability).  Outputs are saved raw for kvw_analyze.py; nothing is judged here.

  docker run --rm --gpus '"device=0"' --network none -e IMAGE_TAG=b0 -e IMAGE_ID=<sha256> \\
      -v <e1-kvwriter>:/w --entrypoint python3 <image> /w/kvw_gpu_compare.py \\
      /w/kvw-inputs/kvw_inputs.pt /w/kvw-runs/b0.pt
"""
import hashlib
import json
import os
import sys

import torch

PAGE, L, NSA, NEXT = 64, 512, 656, 528


def main():
    inputs, out_path = sys.argv[1:3]
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    d = torch.load(inputs, map_location="cpu")
    from vllm import _custom_ops as ops
    try:
        from b12x.attention import sparse_mla
        from vllm.v1.attention.backends.mla import b12x_mla_sparse as backend
        have_next = hasattr(sparse_mla, "concat_and_cache_glm_next_mla") and hasattr(backend, "_prepare_glm_next_writer_plan")
    except Exception:  # noqa: BLE001
        have_next = False
    from b12x.gemm import mla_query_projection as mqp

    meta = {"image_tag": os.environ.get("IMAGE_TAG"), "image_id": os.environ.get("IMAGE_ID"),
            "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__,
            "inputs_sha256": hashlib.sha256(open(inputs, "rb").read()).hexdigest(), "have_glm_next_writer": have_next}
    try:
        import vllm
        meta["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001
        pass
    res = {"meta": meta, "writers": {}, "projections": {}, "repeatable": {}}

    def write_all(kv_c, slots):
        T = kv_c.shape[0]
        pages = (T + PAGE - 1) // PAGE
        nsa = torch.zeros((pages, PAGE, NSA), dtype=torch.uint8, device=dev)
        ops.concat_and_cache_mla(kv_c, torch.zeros((T, 64), dtype=torch.bfloat16, device=dev), nsa, slots,
                                 kv_cache_dtype="fp8_ds_mla", scale=torch.ones((1,), dtype=torch.float32, device=dev))
        out = {"nsa": nsa.view(-1, NSA)}
        if have_next:
            nxt = torch.zeros((pages, PAGE, NEXT), dtype=torch.uint8, device=dev)
            plan = backend._prepare_glm_next_writer_plan(dev, PAGE)
            sparse_mla.concat_and_cache_glm_next_mla(kv_c, nxt, slots, plan=plan)
            out["next"] = nxt.view(-1, NEXT)
        torch.cuda.synchronize()
        return out

    sets = [("synth/0", d["synth"]["kv_c"])] + [(f"proxy/{k}", v["kv_c"]) for k, v in d["proxy"].items()]
    for name, kv in sets:
        kv_c = kv.to(dev).contiguous()
        T = kv_c.shape[0]
        ident = torch.arange(T, dtype=torch.int64, device=dev)
        a = write_all(kv_c, ident)
        b = write_all(kv_c, ident)
        perm = torch.randperm(T, generator=torch.Generator().manual_seed(7)).to(dev)
        c = write_all(kv_c, perm)
        rep = {}
        for k in a:
            rep[k] = bool(torch.equal(a[k], b[k]))
            rep[k + "_slot_independent"] = bool(torch.equal(c[k][perm], a[k][:T]))
        res["repeatable"][name] = rep
        res["writers"][name] = {k: v[:T].cpu() for k, v in a.items()}

    for layer, v in d["proxy"].items():
        W = v["W_UK_T"].to(dev).contiguous()                                  # (N, P, L)
        for rows in (1, 2, 4, 8, 16):
            q_tm = v["q_nope"][:, :rows].permute(1, 0, 2).contiguous().to(dev)  # token-major (B, N, P)
            q = q_tm.permute(1, 0, 2)                                         # head-major view (N, B, P)
            p = {"rows": rows}
            o1 = torch.bmm(q, W).transpose(0, 1).contiguous()
            o2 = torch.bmm(q, W).transpose(0, 1).contiguous()
            p["bmm"] = o1.cpu()
            p["bmm_repeatable"] = bool(torch.equal(o1, o2))
            try:
                q_pe = torch.empty((rows, q.shape[0], 0), dtype=torch.bfloat16, device=dev)
                h1 = torch.full((rows, q.shape[0], L + 64), 17.0, dtype=torch.bfloat16, device=dev)
                h2 = torch.full_like(h1, 17.0)
                mqp.run_glm_h64_bf16(q, W, q_pe, h1)
                mqp.run_glm_h64_bf16(q, W, q_pe, h2)
                torch.cuda.synchronize()
                p["h64"] = h1.cpu()
                p["h64_repeatable"] = bool(torch.equal(h1, h2))
            except Exception as exc:  # noqa: BLE001
                p["h64_error"] = repr(exc)[:300]
            res["projections"][f"{layer}/rows{rows}"] = p

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(res, out_path)
    print(json.dumps({"meta": meta, "repeatable": res["repeatable"],
                      "proj": {k: {kk: vv for kk, vv in p.items() if not torch.is_tensor(vv)}
                               for k, p in res["projections"].items()}}, indent=1))


if __name__ == "__main__":
    main()
