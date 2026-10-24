#!/usr/bin/env python3
"""Stage-0 single-GPU pre-check (STAGE0-PROTOCOL.md section 6): CUDA-graph capture with the stage-0 recording.  Runs in
the stage-0 image on GPU0, no model, synthetic tensors at the served sizes (hidden 4096, vocab 154,880, 16 request
slots), env GLM53_P4=on GLM53_P4_S0=on.

  C1  capture (m = 32) of every in-graph hook WITH the per-layer state digests (8 layers: hidden bf16, residual fp32,
      mHC post / comb fp32; two layers also feed the hidden stages), module registered as in serving (all buffers
      allocated before capture), a flush attempted inside the capture: the capture succeeds, the flush is refused and
      writes nothing, and on_batch inside the capture issues no entry digest.
  C2  5 replays (8 forced requests x 4 rows, fixed accepted counts) + the eager bridge after each: stores == the
      step's inputs bit for bit (MoE every position, hidden at o % 8, logits at o % 64); per-layer digests == a host
      (numpy) recomputation of both digests from the inputs at every recorded position; every step-ring row carries
      the uid and tag hash assigned at add time; no p4_error.
  C3  an eager mixed step with 300 tokens (> MMAX, never captured): stages, stores and layer digests take the step's
      8 logits rows (gathered), equal to the inputs at those rows.
  C4  window-entry digests on a synthetic KV cache (4 tensors in 2 groups: fp8-like MLA pages with 4 pages per
      logical block, bf16 indexer pages with 2, KDA conv bf16 / recurrent fp32 with 1): device digests == a host
      recomputation per logical block (and the null block); logical block ids recorded; the removal flush writes
      entry_<tag>_r0.npz equal to the device buffers.
  N1  negative control, last: a device-to-host copy inside a capture fails it.

usage: precheck_s0.py OUT_JSON
"""
import glob
import json
import os
import shutil
import sys
import time
import traceback
import types

OUT_JSON = sys.argv[1]
os.environ.setdefault("GLM53_P4", "on")
os.environ.setdefault("GLM53_P4_S0", "on")
os.environ.setdefault("GLM53_P4_DIR", "/tmp/p4pre0")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from vllm import p4_hooks as p4  # noqa: E402

DEV = torch.device(os.environ.get("P4_PRECHECK_DEV", "cuda:0"))
G = {}
H, V, R = 4096, 154880, 16
M = 32
NL = 8
NB = 40
SCHED = [3, 1, 2, 0, 3]
RESULTS = []
N_EXPECTED = 5


def result(name, ok, **kw):
    RESULTS.append(dict(test=name, ok=bool(ok), **kw))
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:400], flush=True)


class Mod:
    def __init__(self, mhc=False):
        self.mhc, self.n = mhc, 4

    def hc_post(self, h, r, post, comb):
        return (h.unsqueeze(1) * post.unsqueeze(-1)).to(h.dtype)    # [m, 4, H]; post == 1 -> contract == h


def fake_runner():
    g = torch.Generator().manual_seed(11)
    mla = torch.randint(0, 255, (NB * 4, 16, 72), generator=g, dtype=torch.uint8).to(DEV)
    idx = torch.randn(NB * 2, 8, 64, generator=g).to(torch.bfloat16).to(DEV)
    conv = torch.randn(NB, 8, 16, generator=g).to(torch.bfloat16).to(DEV)
    ssm = torch.randn(NB, 4, 16, 16, generator=g).to(DEV)
    bt = types.SimpleNamespace(num_kv_cache_groups=2, blocks_per_kv_block=[4, 1], block_sizes=[4352, 4352],
                               num_blocks=types.SimpleNamespace(np=np.zeros((2, R), dtype=np.int32)),
                               block_tables=[types.SimpleNamespace(gpu=torch.zeros(R, 4 * 12, dtype=torch.int32, device=DEV)),
                                             types.SimpleNamespace(gpu=torch.zeros(R, 12, dtype=torch.int32, device=DEV))])
    perm = torch.randperm(NB - 1, generator=g) + 1
    lids = {}
    for i in range(R):
        n = 5 + i % 3
        ids = perm[(3 * i) % (NB - 8):(3 * i) % (NB - 8) + n].tolist()
        lids[i] = ids
        bt.num_blocks.np[0, i] = 4 * n
        bt.num_blocks.np[1, i] = n
        bt.block_tables[0].gpu[i, :4 * n] = torch.tensor([b * 4 + k for b in ids for k in range(4)], dtype=torch.int32)
        bt.block_tables[1].gpu[i, :n] = torch.tensor(ids, dtype=torch.int32)
    kvc = types.SimpleNamespace(num_blocks=NB, kv_cache_groups=[types.SimpleNamespace(layer_names=["mla.0", "idx.0"]),
                                                                 types.SimpleNamespace(layer_names=["kda.0"])])
    ctx = {"mla.0": types.SimpleNamespace(kv_cache=[mla]), "idx.0": types.SimpleNamespace(kv_cache=idx),
           "kda.0": types.SimpleNamespace(kv_cache=(conv, ssm))}
    r = types.SimpleNamespace(kv_cache_config=kvc, compilation_config=types.SimpleNamespace(static_forward_context=ctx),
                              device=DEV, block_tables=bt)
    return r, lids, dict(mla=(mla, 4, 0), idx=(idx, 2, 0), conv=(conv, 1, 1), ssm=(ssm, 1, 1))


def setup():
    shutil.rmtree(p4.OUT_DIR, ignore_errors=True)
    p4.S.__init__()
    p4.S.rank = 0
    p4._alloc(DEV, H, V, R)
    mods = [Mod(), Mod(), Mod(), Mod(True), Mod(True), Mod()]
    for k, m in enumerate(mods):
        p4.S.ids[id(m)] = k
    layers = [Mod() for _ in range(NL)]
    layers[2], layers[5] = mods[3], mods[4]            # two digest layers are also the hidden stages
    p4.S.lid = {id(m): i for i, m in enumerate(layers)}
    p4.S.lid_layers = list(range(NL))
    p4._alloc_ldig(DEV, NL, H)
    runner, lids, kv = fake_runner()
    p4.S.runner = runner
    cov = p4._entry_setup(runner)
    p4.S.registered = True
    p4._mkdir_private(p4.PRIV_DIR)
    return mods, layers, lids, kv, cov


def make_inputs(m, gen):
    st = dict(x=[torch.randn(m, H, generator=gen, device=DEV).to(torch.bfloat16) for _ in range(6)],
              h=[torch.randn(m, H, generator=gen, device=DEV).to(torch.bfloat16) for _ in range(NL)],
              res=[torch.randn(m, 4, H, generator=gen, device=DEV) for _ in range(NL)],
              post=[torch.randn(m, 4, generator=gen, device=DEV) for _ in range(NL)],
              comb=[torch.randn(m, 4, 4, generator=gen, device=DEV) for _ in range(NL)],
              one=torch.ones(m, 4, device=DEV))
    return st


def forward(mods, layers, st):
    for k in range(3):
        p4.hook_moe_out(mods[k], st["x"][k])
    for i, lay in enumerate(layers):
        if i in (2, 5):           # stage layers: hidden = x[3] / x[4], post = 1, no residual / comb (contract == h)
            p4.hook_hidden(lay, st["x"][3 if i == 2 else 4], None, st["one"], None)
        else:
            p4.hook_hidden(lay, st["h"][i], st["res"][i], st["post"][i], st["comb"][i])
    p4.hook_final(mods[5], st["x"][5])


def ref_ldig(st, row):
    """Host recomputation of the two layer digests of token row `row` for every digest layer."""
    w = ((np.arange(H, dtype=np.int64) * 2654435761) % (2 ** 31)) * 2 + 1
    out = np.zeros((NL, 2), dtype=np.int64)
    for i in range(NL):
        if i in (2, 5):
            h = st["x"][3 if i == 2 else 4][row]
            parts = [h, st["one"][row]]
        else:
            h = st["h"][i][row]
            parts = [h, st["res"][i][row], st["post"][i][row], st["comb"][i][row]]
        d0 = 0
        for t in parts:
            a = t.detach().cpu().contiguous()
            iv = a.view(torch.int16 if a.element_size() == 2 else torch.int32).numpy().astype(np.int64)
            d0 += int(iv.sum())
        hv = h.detach().cpu().contiguous().view(torch.int16).numpy().astype(np.int64).reshape(-1)
        out[i] = [d0, int((hv * w[:hv.size]).sum())]
    return out


def ib(n, ntp, nr=8, lrows=None, prefilling=False):
    li = torch.arange(n, device=DEV) if lrows is None else lrows
    nl = int(li.shape[0])
    return types.SimpleNamespace(num_reqs=nr, num_tokens=ntp, num_tokens_after_padding=ntp, num_reqs_after_padding=nr,
                                 idx_mapping_np=np.arange(nr), req_ids=[f"cmpl-x-{i}" for i in range(nr)],
                                 num_scheduled_tokens=np.full(nr, ntp // nr), num_computed_tokens_np=np.full(nr, 1000),
                                 num_computed_prefill_tokens_np=np.full(nr, 1000), prefill_len_np=np.full(nr, 1000),
                                 is_prefilling_np=np.full(nr, prefilling), num_draft_tokens_per_req=np.full(nr, 3),
                                 num_draft_tokens=3 * nr, cu_num_logits_np=np.array([0] * nr + [nl]),
                                 logits_indices=li)


def ref_block(t, ppb, b):
    pages = t[b * ppb:(b + 1) * ppb].detach().cpu().contiguous()
    iv = pages.view({1: torch.int8, 2: torch.int16, 4: torch.int32}[pages.element_size()]).reshape(-1).numpy().astype(np.int64)
    return np.array([iv.sum(), iv[::3].sum()], dtype=np.int64)


def check_entry(lids, kv, cov, tags, eslots):
    names = [t[0] for t in p4.S.e_tens]
    want = {"mla.0": kv["mla"], "idx.0": kv["idx"], "kda.0#0": kv["conv"], "kda.0#1": kv["ssm"]}
    bad, n = [], 0
    for i, tag in enumerate(tags):
        es = eslots[i]
        ed = p4.S.edig[es, 0].cpu().numpy()
        eb = p4.S.eblk[es, 0].cpu().numpy()
        for g in (0, 1):
            if list(eb[g][:len(lids[i])]) != lids[i] or (eb[g][len(lids[i]):] != -1).any():
                bad.append((tag, "eblk", g))
        for ti, nm in enumerate(names):
            t, ppb, g = want[nm]
            for j, b in enumerate(lids[i] + [0]):
                col = j if j < len(lids[i]) else p4.NE_BLK
                n += 1
                if not np.array_equal(ed[ti, col], ref_block(t, ppb, b)):
                    bad.append((tag, nm, j))
    return bad, n, names


def main():
    t0 = time.time()
    mods, layers, lids, kv, cov = setup()
    gen = torch.Generator(device=DEV).manual_seed(7)
    st = make_inputs(M, gen)
    G.update(mods=mods, layers=layers, st=st)
    for _ in range(2):            # warm-up outside capture
        forward(mods, layers, st)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    p4.on_add(15, "inside-capture", 512)
    err = None
    n_files0 = len(glob.glob(os.path.join(p4.PRIV_DIR, "*")))
    edig0 = p4.S.edig.clone()
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            forward(mods, layers, st)
            p4.S.serving = True
            p4.on_batch(ib(M, M, nr=8), 0)              # inside capture: no stage-0 bookkeeping, no digest
            p4.on_remove(15, "inside-capture", 512)
            p4.S.serving = False
        if DEV.type == "cuda":
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)[:300] + traceback.format_exc()[-600:]
    G["g"] = g if err is None else None
    files1 = len(glob.glob(os.path.join(p4.PRIV_DIR, "*")))
    logs = "".join(open(p).read() for p in glob.glob(os.path.join(p4.LOG_DIR, "p4_r0_*.jsonl")))
    result("C1 capture with every in-graph hook incl. per-layer digests succeeds; flush and digests refused inside",
           err is None and files1 == n_files0 and "flush_refused" in logs and torch.equal(edig0, p4.S.edig),
           error=err, entry_cov=[c.get("covered") for c in cov])
    if err:
        return
    tags = [f"pc-{i}" for i in range(8)]
    for i, t in enumerate(tags):
        p4.on_add(i, t, 512)
    uids = {i: p4.S.uid_host[i] for i in range(8)}
    eslots = {i: p4.S.eslot_host[i] for i in range(8)}
    nout = [0] * 8
    expect, lexpect = {}, {}
    for step in range(len(SCHED)):
        a = SCHED[step]
        p4.on_batch(ib(M, M), step + 1)                  # real batch: serving; step 1 = entry point (not prefilling)
        for k in range(6):
            st["x"][k].copy_(torch.randn(M, H, generator=gen, device=DEV).to(torch.bfloat16))
        for i in range(NL):
            st["h"][i].copy_(torch.randn(M, H, generator=gen, device=DEV).to(torch.bfloat16))
            st["res"][i].copy_(torch.randn(M, 4, H, generator=gen, device=DEV))
            st["post"][i].copy_(torch.randn(M, 4, generator=gen, device=DEV))
            st["comb"][i].copy_(torch.randn(M, 4, 4, generator=gen, device=DEV))
        logits = torch.randn(M, V, generator=gen, device=DEV)
        g.replay()
        ridx = torch.arange(8, device=DEV).repeat_interleave(4)
        j = torch.arange(4, device=DEV).repeat(8)
        o = torch.tensor([nout[i] for i in range(8)], device=DEV).repeat_interleave(4) + j
        sc = dict(L=M, ridx=ridx, o=o, fwin=torch.ones(M, dtype=torch.bool, device=DEV))
        p4.on_logits(sc, logits, types.SimpleNamespace(logits_indices=torch.arange(M, device=DEV)))
        p4.on_commit(sc, j <= a)
        p4.on_step(step + 1, torch.arange(8, device=DEV), o.view(8, 4)[:, 0], torch.full((8,), 4, device=DEV),
                   torch.full((8,), a, device=DEV))
        for i in range(8):
            for jj in range(a + 1):
                oo, row = nout[i] + jj, 4 * i + jj
                for k in range(3):
                    expect[(tags[i], "moe", k, oo)] = st["x"][k][row].clone()
                if oo % 8 == 0:
                    for kk, k in enumerate((3, 4, 5)):
                        expect[(tags[i], "hid", kk, oo)] = st["x"][k][row].clone()
                if oo % 64 == 0:
                    expect[(tags[i], "logits", 0, oo)] = logits[row].clone()
                lexpect[(tags[i], oo)] = ref_ldig(st, row)
            nout[i] += a + 1
        p4.S.last_step = step + 1
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    ring = p4.S.steprec[1:len(SCHED) + 1].cpu().numpy()
    ring_ok = all(int(ring[s, i, 5]) == uids[i] and int(ring[s, i, 6]) == p4.taghash(tags[i])
                  for s in range(len(SCHED)) for i in range(8))
    bad_e, n_e, names = check_entry(lids, kv, cov, tags, eslots)
    for i, t in enumerate(tags):
        p4.on_remove(i, t, 512)
    bad, checked = [], 0
    for i, t in enumerate(tags):
        z = np.load(os.path.join(p4.PRIV_DIR, f"p4_{t}_r0.npz"))
        n = nout[i]
        if not ((z["moe_have"][:, :n] == 1).all() and (z["moe_have"][:, n:] == 0).all() and (z["ldig_have"][:n] == 1).all()):
            bad.append((t, "have"))
        for (tt, kind, k, oo), ref in expect.items():
            if tt != t:
                continue
            checked += 1
            u16 = lambda x: x.view(torch.int16).cpu().numpy().view(np.uint16)  # noqa: E731
            if kind == "moe":
                ok = np.array_equal(z["moe"][k, oo], u16(ref))
            elif kind == "hid":
                ok = np.array_equal(z["hid"][k, oo // 8], u16(ref)) and z["hid_have"][k, oo // 8] == 1
            else:
                ok = np.array_equal(z["logits"][oo // 64], ref.cpu().numpy()) and z["logits_have"][oo // 64] == 1
            if not ok:
                bad.append((t, kind, k, oo))
        for (tt, oo), ref in lexpect.items():
            if tt == t:
                checked += 1
                if not np.array_equal(z["ldig"][oo], ref):
                    bad.append((t, "ldig", oo))
    errs = [l for p in glob.glob(os.path.join(p4.LOG_DIR, "p4_r0_*.jsonl")) for l in open(p) if '"p4_error"' in l]
    result("C2 replays + eager bridge: stores and per-layer digests == inputs; step ring carries uid / tag hash; "
           "no p4_error", not bad and checked > 0 and ring_ok and not errs, checked=checked, bad=bad[:5],
           ring_ok=ring_ok, errors=errs[:3])
    # C4 (entry digests were taken at the step-1 on_batch, before that step's forward)
    ent_ok = []
    for i, t in enumerate(tags):
        p = os.path.join(p4.PRIV_DIR, f"entry_{t}_r0.npz")
        ent_ok.append(os.path.exists(p) and int(np.load(p)["points"][0][1]) == 1)
    result("C4 entry digests == host recomputation per logical block and null block; block ids recorded; flushed",
           not bad_e and n_e > 0 and all(ent_ok) and sorted(names) == ["idx.0", "kda.0#0", "kda.0#1", "mla.0"],
           compared=n_e, bad=bad_e[:5], tensors=names, flushed=sum(ent_ok))
    # C3 eager big step
    btags = [f"big-{i}" for i in range(8)]
    for i, t in enumerate(btags):
        p4.on_add(i, t, 512)
    mb = 300
    stb = make_inputs(mb, gen)
    lrows = torch.tensor([36 * i + 35 for i in range(8)], device=DEV)
    p4.on_batch(ib(8, mb, lrows=lrows), 100)
    forward(mods, layers, stb)
    lg = torch.randn(8, V, generator=gen, device=DEV)
    sc = dict(L=8, ridx=torch.arange(8, device=DEV), o=torch.zeros(8, dtype=torch.int64, device=DEV),
              fwin=torch.ones(8, dtype=torch.bool, device=DEV))
    p4.on_logits(sc, lg, types.SimpleNamespace(logits_indices=lrows))
    p4.on_commit(sc, torch.ones(8, dtype=torch.bool, device=DEV))
    p4.S.last_step = 100
    for i, t in enumerate(btags):
        p4.on_remove(i, t, 512)
    bad3 = []
    for i, t in enumerate(btags):
        z = np.load(os.path.join(p4.PRIV_DIR, f"p4_{t}_r0.npz"))
        row = int(lrows[i])
        for k in range(3):
            if z["moe_have"][k, 0] != 1 or not np.array_equal(
                    z["moe"][k, 0], stb["x"][k][row].view(torch.int16).cpu().numpy().view(np.uint16)):
                bad3.append((t, "moe", k))
        if z["ldig_have"][0] != 1 or not np.array_equal(z["ldig"][0], ref_ldig(stb, row)):
            bad3.append((t, "ldig"))
    result("C3 eager 300-token step: stages / stores / layer digests take the 8 logits rows", not bad3, bad=bad3[:5])
    err = None
    try:
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            forward(mods, layers, st)
            _ = p4.S.ldig[0, 0].cpu()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)[:200]
    result("N1 negative control: a device-to-host copy inside a capture fails it", err is not None, error=err)
    print("secs", round(time.time() - t0, 1))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        ok = bool(RESULTS) and all(r["ok"] for r in RESULTS) and len(RESULTS) == N_EXPECTED
        json.dump(dict(results=RESULTS, verdict="PASS" if ok else "FAIL"), open(OUT_JSON, "w"), indent=1)
        print("VERDICT", "PASS" if ok else "FAIL", flush=True)
        os._exit(0 if ok else 1)
