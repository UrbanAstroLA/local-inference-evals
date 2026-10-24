#!/usr/bin/env python3
"""CPU tests of the stage-0 recording and of the request-identity fix (p4_hooks schema 2 + the forcewin port's
multi-prompt selection), against the installed stock 0.9.1 engine code with the P4 files installed over it
(tests/inner.sh hooked) or in the stage-0 image.  CPU container, Triton interpreter, hook-validation's toy engine
(real Sampler, RejectionSampler, RequestState, post_update).

Env: FORCEWIN=on GLM53_P4=on GLM53_P4_S0=on GLM53_P4_DIR=/tmp/p4s0test P4_PRECHECK_DEV=cpu TRITON_INTERPRET=1

  U1  graph-static: hook_hidden with per-layer digests (mHC and plain path) issues the same aten ops for different
      values, no host-sync op, buffers keep their addresses
  U2  lint: the new in-graph helpers (_ldig_put, _rowsum, _big_rows) use no .item / .cpu / .tolist / nonzero /
      .numpy / bool()
  U3  identity: uids are monotonic and never reused; a re-used req_idx gets a new uid; step-ring rows carry the
      uid / tag hash assigned at add time; removal resets them to -1, so an unkeyed later occupant reads -1
  U4  entry digests (fake KV cache): one point per prefill step and one entry point, then none; == host
      recomputation; flushed per request; a W = 0 request gets no entry slot
  U5  multi-prompt request: two prompts in ONE request (one SamplingParams), each engine request forced with its own
      list and tag (committed ids == its own history, stores flushed under its tag); a prompt whose key is not in
      the spec, or a duplicated key, stays unforced (fw_error logged, no store)
  U6  keyed warm request (forcewin_hooks []): no store slot or store file, uid and "req" record logged, its steps
      appear in the sched log with its tag, and its block ids are logged at removal
  U7  reused request index end to end: request A finishes and is removed, request B is added on the same req_idx;
      the step ring's uid column labels A's and B's steps correctly while the flush-time req_idx map names only B
  U8  sched log: every recorded step has its graph mode, eager binds field, batch rows with tags and counts
  U9  big prefill step (92 rows > 64) with GLM53_P4_S0=on: o = 0 IS staged (logits rows gathered); all stores
      bit-exact; per-layer digests == host recomputation at every position
"""
import ast
import glob
import inspect
import json
import os
import shutil
import sys
import types

sys.path.insert(0, "/hv/tests")
sys.path.insert(0, "/t")
sys.path.insert(0, "/p4pc")
os.environ.setdefault("FORCEWIN", "on")
os.environ.setdefault("GLM53_P4", "on")
os.environ.setdefault("GLM53_P4_S0", "on")
os.environ.setdefault("GLM53_P4_DIR", "/tmp/p4s0test")
os.environ.setdefault("FORCEWIN_DIR", "/tmp/p4s0test/fw")
os.environ.setdefault("FORCEWIN_WMAX", "512")
os.environ.setdefault("P4_PRECHECK_DEV", "cpu")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import test_p4_cpu as T  # noqa: E402  (toy staging wrapper, feat(), run(), check_stores())
import test_forcing as TF  # noqa: E402
from fwtest_common import V, check, summary  # noqa: E402
from vllm import forcewin_hooks as fw  # noqa: E402
from vllm import p4_hooks as p4  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
ARGV = list(sys.argv)
sys.argv = ["precheck_s0.py", "/tmp/p4s0-pc.json"]
import precheck_s0 as PS  # noqa: E402  (fake KV cache / block tables; main() is not run on import)

assert p4.S0 and p4.ENABLED and fw._p4b is p4, "GLM53_P4=on GLM53_P4_S0=on and the bridge are required"
CPU = torch.device("cpu")
H = T.H
_orig_reset = T.reset_p4
FAKE = {}


def reset_s0():
    _orig_reset()
    p4.S.lid = {id(T.MODS[3]): 0, id(T.MODS[4]): 1}
    p4.S.lid_layers = [12, 34]
    p4._alloc_ldig(CPU, 2, H)
    PS.R = TF.MAXR
    runner, lids, kv = PS.fake_runner()
    p4.S.runner = runner
    p4._entry_setup(runner)
    FAKE.update(runner=runner, lids=lids, kv=kv)


T.reset_p4 = reset_s0


def logs(prefix="p4"):
    return [json.loads(l) for p in glob.glob(os.path.join(p4.LOG_DIR, f"{prefix}_r0_*.jsonl")) for l in open(p)]


def fwlogs():
    return [json.loads(l) for p in glob.glob(os.path.join(fw.OUT_DIR, "fw_r0_*.jsonl")) for l in open(p)]


def add_req(e, rid, prompt, sp, max_tokens):
    """TF.Eng.add with prompt_token_ids in new_req_data (the engine passes them; the toy did not)."""
    e.rs.add_request(rid, len(prompt), prompt, 0, max_tokens)
    idx = e.rs.req_id_to_index[rid]
    e.sampler.add_request(idx, len(prompt), sp)
    fw.add_request(e.runner, rid, idx, len(prompt), types.SimpleNamespace(sampling_params=sp, prompt_token_ids=list(prompt)))
    e.rs.apply_staged_writes()
    e.sampler.apply_staged_writes()
    e.reqs[rid] = dict(idx=idx, prompt=list(prompt), out=[], max=max_tokens, prefilled=False, key=len(e.reqs))
    e.full[e.reqs[rid]["key"]] = list(prompt)
    e.commits[rid] = []
    return idx


def finish(e, rid):
    fw.remove_request(e.runner, rid)
    e.rs.remove_request(rid)
    e.reqs.pop(rid, None)


def drive(e, rids, until):
    e.prefill(rids)
    while any(len(e.reqs[r]["out"]) < until[r] for r in rids):
        ib, lg = e.decode(rids)
        e.finish_decode(rids, ib, lg)


def ref_digest(rows):
    """Host digests of a list of (bf16 [H]) rows: d0 = sum of int16 words, d1 = weighted sum (post / residual None)."""
    w = ((np.arange(H, dtype=np.int64) * 2654435761) % (2 ** 31)) * 2 + 1
    a = torch.stack(rows).view(torch.int16).numpy().astype(np.int64)
    return np.stack([a.sum(1), (a * w[None, :H]).sum(1)], axis=1)


# ---------------------------------------------------------------------------------------------------- U1, U2
def test_static():
    from torch.utils._python_dispatch import TorchDispatchMode

    class Rec(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            self.ops.append(str(func))
            return func(*args, **(kwargs or {}))

    def trace(seed, m):
        reset_s0()
        p4.S.big = False
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(m, H, generator=g).to(torch.bfloat16)
        res = torch.randn(m, 4, H, generator=g)
        post = torch.randn(m, 4, generator=g)
        comb = torch.randn(m, 4, 4, generator=g)
        mh = T.Mod(3, mhc=True)
        p4.S.ids[id(mh)] = 3
        p4.S.lid[id(mh)] = 0
        plain = T.Mod(9)
        p4.S.lid[id(plain)] = 1
        ptr = (p4.S.ldig.data_ptr(), p4.S.stage.data_ptr())
        with Rec() as r:
            p4.hook_hidden(mh, x, None, torch.ones(m, 4), None)
            p4.hook_hidden(plain, x, res, post, comb)
        return r.ops, ptr == (p4.S.ldig.data_ptr(), p4.S.stage.data_ptr())
    a, pa = trace(1, 32)
    b, pb = trace(2, 32)
    sync = [o for o in a if any(s in o for s in ("_local_scalar_dense", "item", "nonzero", "is_nonzero", "equal",
                                                 "_to_copy", "masked_select"))]
    check("U1 hook_hidden + layer digests: identical aten ops for different values; buffers keep addresses",
          a == b and pa and pb and len(a) > 0, f"{len(a)} ops")
    check("U1 hook_hidden + layer digests: no host-sync op", not sync, str(sync[:5]))
    src = open(p4.__file__).read()
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    bad = []
    for name in ("_ldig_put", "_rowsum", "_big_rows", "hook_hidden", "_put"):
        seg = ast.get_source_segment(src, fns[name])
        for wd in (".item(", ".cpu(", ".tolist(", "nonzero", ".numpy(", "bool("):
            if wd in seg:
                bad.append((name, wd))
    check("U2 new in-graph helpers: no .item / .cpu / .tolist / nonzero / .numpy / bool()", not bad, str(bad))


# ---------------------------------------------------------------------------------------------------- U3
def test_identity():
    reset_s0()
    p4.S.serving = True
    p4.on_add(1, "id-a", 10)
    ua = p4.S.uid_host[1]
    p4.on_step(5, torch.tensor([1]), torch.tensor([0]), torch.tensor([4]), torch.tensor([2]))
    p4.on_remove(1, "id-a", 10)
    p4.on_step(6, torch.tensor([1]), torch.tensor([0]), torch.tensor([4]), torch.tensor([2]))   # unkeyed occupant
    p4.on_add(1, "id-b", 10)
    ub = p4.S.uid_host[1]
    p4.on_step(7, torch.tensor([1]), torch.tensor([0]), torch.tensor([4]), torch.tensor([2]))
    r5, r6, r7 = (p4.S.steprec[s, 0].tolist() for s in (5, 6, 7))
    reqs = {r["uid"]: r["tag"] for r in logs() if r.get("kind") == "req"}
    check("U3 uids monotonic and never reused; a re-used req_idx gets a new uid; public req records map uid -> tag",
          ub > ua and reqs.get(ua) == "id-a" and reqs.get(ub) == "id-b")
    check("U3 step rows carry the uid / tag hash at record time; removal resets them (unkeyed occupant reads -1)",
          r5[5] == ua and r5[6] == p4.taghash("id-a") and r6[5] == -1 and r6[6] == -1 and r7[5] == ub
          and r7[6] == p4.taghash("id-b"), f"{r5} {r6} {r7}")


# ---------------------------------------------------------------------------------------------------- U4
def test_entry():
    reset_s0()
    p4.S.serving = True
    p4.on_add(0, "en-a", 64)
    p4.on_add(2, "en-w", 0)
    nr = 2

    def ib(prefilling, step):
        return types.SimpleNamespace(num_reqs=nr, num_tokens=40, idx_mapping_np=np.array([0, 2]), req_ids=["a", "w"],
                                     num_scheduled_tokens=np.array([20, 20]), num_computed_tokens_np=np.array([step * 20, 0]),
                                     num_computed_prefill_tokens_np=np.array([step * 20, 0]), prefill_len_np=np.array([60, 60]),
                                     is_prefilling_np=np.array([prefilling, True]), num_draft_tokens_per_req=None,
                                     num_draft_tokens=0, cu_num_logits_np=np.array([0, 1, 2]),
                                     logits_indices=torch.tensor([19, 39]))
    for s, pre in enumerate((True, True, False, False)):
        p4.on_batch(ib(pre, s), s + 1)
    pts = list(p4.S.epoints.get(0, []))
    es = p4.S.eslot_host[0]
    bad = []
    names = [t[0] for t in p4.S.e_tens]
    want = {"mla.0": FAKE["kv"]["mla"], "idx.0": FAKE["kv"]["idx"], "kda.0#0": FAKE["kv"]["conv"], "kda.0#1": FAKE["kv"]["ssm"]}
    for pt in range(len(pts)):
        for ti, nm in enumerate(names):
            t, ppb, g = want[nm]
            for j, b in enumerate(FAKE["lids"][0] + [0]):
                col = j if j < len(FAKE["lids"][0]) else p4.NE_BLK
                if not np.array_equal(p4.S.edig[es, pt, ti, col].numpy(), PS.ref_block(t, ppb, b)):
                    bad.append((pt, nm, j))
    no_w = 2 not in p4.S.eslot_host
    p4.on_remove(0, "en-a", 64)
    z = np.load(os.path.join(p4.PRIV_DIR, "entry_en-a_r0.npz"))
    check("U4 entry points: one per prefill step + one entry point, then none",
          [p[1] for p in pts] == [0, 0, 1] and [p[0] for p in pts] == [1, 2, 3], str(pts))
    check("U4 entry digests == host recomputation per logical block and null block", not bad, str(bad[:4]))
    check("U4 flushed entry file has the points, digests and tensor list; a W = 0 request got no entry slot",
          z["edig"].shape[0] == 3 and list(z["points"][:, 1]) == [0, 0, 1] and list(z["tensors"]) == names and no_w)


# ---------------------------------------------------------------------------------------------------- U5, U6, U7, U8
def test_engine():
    reset_s0()
    p4.S.serving = True
    e = TF.Eng()
    T.CUR["eng"] = e
    sched = [3, 1, 2, 0]
    W = 48
    pa = [5000 + (7 * i) % 900 for i in range(37)]
    pb = [6000 + (11 * i) % 700 for i in range(29)]
    pc = [7000 + (3 * i) % 500 for i in range(31)]
    Fa, Fb = TF.make_history(pa, W, 3), TF.make_history(pb, W, 5)
    spec = json.dumps(dict(entries=[dict(key=fw.multi_key(pa), F=Fa, tag="V2-RA-close-r1"),
                                    dict(key=fw.multi_key(pb), F=Fb, tag="V2-RB-deep-r1")]))
    sp = SamplingParams(temperature=1.0, top_p=0.95, seed=73000, max_tokens=W,
                        extra_args={"local_fw_multi": spec, "local_fw_draft": "list", "local_fw_sched": sched})
    # U6: a keyed warm request first (empty forced list)
    spw = SamplingParams(temperature=1.0, top_p=0.95, seed=72999, max_tokens=1,
                         extra_args={"forcewin_hooks": [], "local_fw_tag": "W-V2-r1-RA"})
    iw = add_req(e, "warm", pa[:20], spw, 1)
    no_slot = p4.S.slot_host.get(iw) == p4.NSLOT and iw not in p4.S.eslot_host
    e.prefill(["warm"])
    finish(e, "warm")
    ia = add_req(e, "m-0", pa, sp, W)
    ib_ = add_req(e, "m-1", pb, sp, W)
    ic = add_req(e, "m-2", pc, sp, W)           # key not in the spec -> unforced
    errs_c = [r for r in fwlogs() if r.get("kind") == "fw_error" and "local_fw_multi" in r.get("error", "")]
    drive(e, ["m-0", "m-1"], {"m-0": W, "m-1": W})
    outa, outb = list(e.reqs["m-0"]["out"]), list(e.reqs["m-1"]["out"])
    finish(e, "m-2")                            # freed first: the toy's free list hands out the last freed index,
    finish(e, "m-0")                            # so the next request re-uses m-1's index (a driven request)
    finish(e, "m-1")
    za = os.path.exists(os.path.join(p4.PRIV_DIR, "p4_V2-RA-close-r1_r0.npz"))
    zb = os.path.exists(os.path.join(p4.PRIV_DIR, "p4_V2-RB-deep-r1_r0.npz"))
    check("U5 multi-prompt: each engine request forced with its own list (committed ids == its own history)",
          outa == Fa and outb == Fb)
    check("U5 multi-prompt: stores flushed under each request's own tag", za and zb)
    check("U5 a prompt whose key is not in the spec stays unforced (fw_error logged, no store slot)",
          bool(errs_c) and ic not in fw.S.reqs and not glob.glob(os.path.join(p4.PRIV_DIR, "p4__r0.npz")))
    dup = json.dumps(dict(entries=[dict(key=fw.multi_key(pa), F=Fa, tag="d1"), dict(key=fw.multi_key(pa), F=Fa, tag="d2")]))
    try:
        fw.multi_pick(dup, pa)
        dup_ok = False
    except ValueError:
        dup_ok = True
    check("U5 a duplicated prompt key is refused (ValueError -> unforced)", dup_ok)
    L = logs()
    reqs = {r["tag"]: r for r in L if r.get("kind") == "req"}
    blocks = {r["tag"]: r for r in L if r.get("kind") == "blocks"}
    sched_steps = [s for r in logs("sched") for s in r.get("steps", [])]
    warm_rows = [row for s in sched_steps for row in s["rows"] if row[3] == "W-V2-r1-RA"]
    check("U6 keyed warm request: no store slot / file, uid + req record, steps in the sched log, block ids at removal",
          no_slot and "W-V2-r1-RA" in reqs and bool(warm_rows) and "W-V2-r1-RA" in blocks
          and not os.path.exists(os.path.join(p4.PRIV_DIR, "p4_W-V2-r1-RA_r0.npz"))
          and blocks["W-V2-r1-RA"]["groups"][1] == FAKE["lids"][iw], str(blocks.get("W-V2-r1-RA"))[:200])
    check("U8 sched log: every recorded step has a graph mode, a binds list and rows with tags and counts",
          bool(sched_steps) and all(s.get("mode") is not None and isinstance(s.get("binds"), list) and s["rows"]
                                    for s in sched_steps)
          and any(row[3] == "V2-RA-close-r1" and row[8] == 0 for s in sched_steps for row in s["rows"]))
    # U7: request index re-use: a new forced request on the slot m-0 had
    Fd = TF.make_history(pa, 24, 9)
    spd = SamplingParams(temperature=1.0, top_p=0.95, seed=73000, max_tokens=24,
                         extra_args={"forcewin_hooks": Fd, "local_fw_tag": "V1-RD-early-r2", "local_fw_draft": "list",
                                     "local_fw_sched": sched})
    idd = add_req(e, "d", pa, spd, 24)
    drive(e, ["d"], {"d": 24})
    finish(e, "d")
    p4.flush_steps()
    uid_tag = {r["uid"]: r["tag"] for r in logs() if r.get("kind") == "req"}
    rows = []
    flush_tags = []
    for f in sorted(glob.glob(os.path.join(p4.PRIV_DIR, "steprec_*_r0.npz"))):
        z = np.load(f)
        flush_tags.append([str(t) for t in z["tags"]])
        rows += [r for blk in z["rows"] for r in blk if int(r[0]) >= 0 and int(r[1]) == idd]
    by = {}
    for r in rows:
        by.setdefault(uid_tag.get(int(r[5]), "?"), 0)
        by[uid_tag.get(int(r[5]), "?")] += 1
    check("U7 re-used req_idx: the ring's uid column labels every occupant's steps correctly (no unknown uid)",
          idd in (ia, ib_) and len(by) >= 2 and by.get("V1-RD-early-r2", 0) > 0 and "?" not in by
          and not any("V1-RD-early-r2" in t for t in flush_tags[0]), f"idx {idd} occupants {by}")


# ---------------------------------------------------------------------------------------------------- U9
def test_big_step():
    r = T.run("BIG", "list", W=130, two=True, sched=[3, 0, 2, 1], co=True)
    T.check_stores("s0-big", r, rows_in_prefill=1)      # 92-row prefill: o = 0 staged under stage 0
    z = r["z"]
    prompt, F = r["prompt"], r["F"]
    bad = []
    for o in range(len(F)):
        pre = prompt + F[:o]
        ref = ref_digest([T.feat(3, pre), T.feat(4, pre)])
        if z["ldig_have"][o] != 1 or not np.array_equal(z["ldig"][o], ref):
            bad.append(o)
    check("U9 big prefill step: o = 0 staged (have == 1) under GLM53_P4_S0=on", int(z["moe_have"][0, 0]) == 1)
    check("U9 per-layer digests == host recomputation at every position (incl. o = 0)", not bad, str(bad[:5]))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("FW_THREADS", "4")))
    shutil.rmtree(p4.OUT_DIR, ignore_errors=True)
    shutil.rmtree(fw.OUT_DIR, ignore_errors=True)
    part = ARGV[1] if len(ARGV) > 1 else "all"
    if part in ("all", "units"):
        test_static()
        test_identity()
        test_entry()
    if part in ("all", "engine"):
        test_engine()
    if part in ("all", "big"):
        test_big_step()
    ok = summary("test_stage0_cpu[%s]" % part)
    sys.exit(0 if ok else 1)
