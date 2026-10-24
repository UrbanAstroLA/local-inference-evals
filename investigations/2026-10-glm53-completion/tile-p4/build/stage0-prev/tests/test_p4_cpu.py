#!/usr/bin/env python3
"""CPU tests of the P4 instrumentation (p4_hooks + the forcewin stock port), run against the installed stock 0.9.1
engine code with the P4 files installed over it (tests/inner.sh hooked) or in the built image (image).  CPU
container, Triton interpreter, the hook-validation scaffolding (fwtest_common: UVA -> CPU tensors, log1p shim, the
engine's PyTorch top-p) and its toy engine (test_forcing.Eng: real Sampler, RejectionSampler, RequestState,
post_update; the model forward is a deterministic toy whose outputs depend on the token prefix only).

Env: FORCEWIN=on GLM53_P4=on GLM53_P4_DIR=/tmp/p4test TRITON_INTERPRET=1

  P1  list draft mode: committed ids == history; every position measured once; metrics == single-token reference
  P2  accepted count of the k-th verify step == list[k] (clamped at the window end); two forced requests with the same
      list advance in lock step (same accepted count every step)
  P3  stores: MoE (3 stages) == stage feature of prompt + F[:o] bit for bit at every staged position, have == 1;
      hidden (3 stages) only at o % 8 == 0; fp32 logits only at o % 64 == 0 and == the toy logits of prompt + F[:o];
      a step with more than 64 tokens is not staged (have 0 there; logits still stored)
  P4  flush file == device stores; the slot is freed and re-used with its have flags reset (no stale rows)
  P5  steprec ring: per step (o0, rows, accepted) of the forced request == the commits
  P6  other schedules through the same stores (oracle, poison): P1 and P3 hold
  S1  stage hooks: copies exact for m <= 64 (mhc and non-mhc hidden path), stage_n = m; m > 64 -> stage_n 0
  S2  graph-static op sequence of every in-graph hook: identical aten ops for different values; no host-sync op
  S3  flush rule (no poller): before the first real batch and during a (simulated) capture the flush is refused and
      writes nothing; afterwards it writes; the module never starts a thread
  S4  static lint: no thread, pinned memory, stream, event or synchronize in p4_hooks.py; no .item / .cpu /
      .tolist / nonzero in the in-graph hooks
  R1  register() on a fake runner: the six stages found by class and layer index; draft modules ignored;
      a duplicate stage is an issue; stores allocated on rank 0 only; installs the graph-mode label once
  B1  bound-launch recorder: one record per (layer, plan, m); repeat with the same signature not logged; a changed
      signature logged with changed=true; draft flag; signature fields as the gates read them
"""
import ast
import glob
import inspect
import json
import os
import shutil
import sys
import threading
import types

sys.path.insert(0, "/hv/tests")
os.environ.setdefault("FORCEWIN", "on")
os.environ.setdefault("GLM53_P4", "on")
os.environ.setdefault("GLM53_P4_DIR", "/tmp/p4test")
os.environ.setdefault("FORCEWIN_DIR", "/tmp/p4test/fw")
os.environ.setdefault("FORCEWIN_WMAX", "512")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import test_forcing as TF  # noqa: E402  (hook-validation toy engine; asserts fw.ON)
from fwtest_common import V, check, summary, toy_logits  # noqa: E402
from vllm import forcewin_hooks as fw  # noqa: E402
from vllm import p4_hooks as p4  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
import vllm.model_executor.layers.mhc  # noqa: E402,F401  (imported before any op trace)

assert p4.ENABLED and fw._p4b is p4, "GLM53_P4=on and the bridge are required"
CPU = torch.device("cpu")
H = 32
K = TF.K


def feat(k, prefix):
    key = (hash(tuple(prefix[-6:])) ^ (len(prefix) * 7919) ^ (k * 104729)) & 0x7FFFFFFF
    g = torch.Generator().manual_seed(key)
    return torch.randn(H, generator=g).to(torch.bfloat16)


class Mod:
    def __init__(self, k, mhc=False):
        self.k, self.mhc, self.n = k, mhc, 4

    def hc_post(self, h, r, post, comb):       # fake mHC post: 4 identical streams (contract == h exactly)
        return h.unsqueeze(1).expand(h.shape[0], 4, h.shape[1]).clone()


MODS = [Mod(k) for k in range(6)]
CUR = {}


def reset_p4():
    shutil.rmtree(p4.OUT_DIR, ignore_errors=True)
    p4.S.__init__()
    p4._alloc(CPU, H, V, TF.MAXR)
    for k, m in enumerate(MODS):
        p4.S.ids[id(m)] = k
    p4.S.registered = True


_orig_on_logits = fw.on_logits


def staged_on_logits(runner, logits, ib):
    """The toy 'forward': before on_logits, write the six stages for every token row (as the in-graph hooks do)."""
    e = CUR.get("eng")
    if e is not None and p4.S.stage is not None:
        qsl = ib.query_start_loc.tolist()
        ids = ib.input_ids.tolist()
        pos = ib.positions.tolist()
        rows = {k: [] for k in range(6)}
        for b, rid in enumerate(ib.req_ids):
            seq = e.full[e.reqs[rid]["key"]]
            lo, hi = qsl[b], qsl[b + 1]
            for t in range(lo, hi):
                prefix = list(seq[:pos[lo]]) + ids[lo:t + 1]
                for k in range(6):
                    rows[k].append(feat(k, prefix))
        X = {k: torch.stack(v) for k, v in rows.items()}
        for k in range(3):
            p4.hook_moe_out(MODS[k], X[k])
        for k in (3, 4):
            p4.hook_hidden(MODS[k], X[k], None, None, None)
        p4.hook_final(MODS[5], X[5])
    return _orig_on_logits(runner, logits, ib)


TF.fw.on_logits = staged_on_logits


def run(name, mode, W=200, prompt_len=37, two=False, sched=None, co=True):
    reset_p4()
    p4.S.serving = True
    e = TF.Eng()
    CUR["eng"] = e
    prompt = [5000 + (7 * i) % 900 for i in range(prompt_len)]
    F = TF.make_history(prompt, W, 3)
    ex = {"forcewin_hooks": F, "local_fw_tag": name, "local_fw_draft": mode}
    if sched is not None:
        ex["local_fw_sched"] = sched
    e.add("forced", prompt, SamplingParams(temperature=1.0, top_p=0.95, seed=11, max_tokens=W, extra_args=ex), W)
    rids = ["forced"]
    F2 = prompt2 = None
    if two:
        prompt2 = [6000 + (11 * i) % 700 for i in range(prompt_len - 5)]
        F2 = TF.make_history(prompt2, W, 5)
        ex2 = dict(ex, forcewin_hooks=F2, local_fw_tag=name + "-2")
        e.add("forced2", prompt2, SamplingParams(temperature=1.0, top_p=0.95, seed=12, max_tokens=W, extra_args=ex2), W)
        rids.append("forced2")
    if co:
        e.add("free", [4000 + i for i in range(23)], SamplingParams(temperature=1.0, top_p=0.95, seed=13, max_tokens=10_000),
              10_000)
        rids.append("free")
    e.prefill(rids)
    while len(e.reqs["forced"]["out"]) < W:
        ib, logits = e.decode(rids)
        e.finish_decode(rids, ib, logits)
    idx = e.reqs["forced"]["idx"]
    slot = p4.S.slot_host.get(idx)
    dev = dict(moe=p4.S.store[:, slot, :W].clone(), have=p4.S.have[:, slot, :W].clone(),
               hid=p4.S.hstore[:, slot].clone(), hhave=p4.S.hhave[:, slot].clone(),
               lst=p4.S.lstore[slot].clone(), lhave=p4.S.lhave[slot].clone())
    buf = fw.S.buf[idx, :W].clone()
    nmeas = fw.S.nmeas[idx, :W].clone()
    commits = list(e.commits["forced"])
    commits2 = list(e.commits.get("forced2", []))
    fw.remove_request(e.runner, "forced")
    path = os.path.join(p4.PRIV_DIR, f"p4_{name}_r0.npz")
    z = dict(np.load(path)) if os.path.exists(path) else None
    return dict(e=e, prompt=prompt, F=F, idx=idx, slot=slot, dev=dev, buf=buf, nmeas=nmeas, z=z, commits=commits,
                commits2=commits2, out=e.reqs["forced"]["out"], out2=(e.reqs.get("forced2") or {}).get("out"), F2=F2)


def stage_staged(prompt, F, o, rows_in_prefill):
    """Was the row predicting output index o staged? o == 0 comes from the prefill step (staged iff <= 64 rows)."""
    return not (o == 0 and rows_in_prefill > p4.MMAX)


def check_stores(tag, r, rows_in_prefill):
    prompt, F, W = r["prompt"], r["F"], len(r["F"])
    z = r["z"]
    ok_moe = ok_have = ok_hid = ok_log = True
    bad = []
    for o in range(W):
        pre = prompt + F[:o]
        staged = stage_staged(prompt, F, o, rows_in_prefill)
        for k in range(3):
            hv = int(z["moe_have"][k, o])
            if staged:
                ref = feat(k, pre).view(torch.int16).numpy().view(np.uint16)
                if hv != 1 or not np.array_equal(z["moe"][k, o], ref):
                    ok_moe = False
                    bad.append(("moe", k, o, hv))
            elif hv != 0:
                ok_have = False
        if o % p4.HID_EVERY == 0:
            for j, k in enumerate((3, 4, 5)):
                hv = int(z["hid_have"][j, o // p4.HID_EVERY])
                if staged:
                    ref = feat(k, pre).view(torch.int16).numpy().view(np.uint16)
                    if hv != 1 or not np.array_equal(z["hid"][j, o // p4.HID_EVERY], ref):
                        ok_hid = False
                        bad.append(("hid", k, o, hv))
                elif hv != 0:
                    ok_have = False
        if o % p4.LOGIT_EVERY == 0:
            hv = int(z["logits_have"][o // p4.LOGIT_EVERY])
            ref = toy_logits(pre).float().numpy()
            if hv != 1 or not np.array_equal(z["logits"][o // p4.LOGIT_EVERY], ref):
                ok_log = False
                bad.append(("logits", o, hv))
    nh = (W + p4.HID_EVERY - 1) // p4.HID_EVERY
    extra = int((z["hid_have"][:, nh:] != 0).sum()) if z["hid_have"].shape[1] > nh else 0
    check(f"P3[{tag}] MoE stores == stage feature of prompt + F[:o], bit for bit, have == 1 (staged positions)",
          ok_moe, f"bad {bad[:3]}")
    check(f"P3[{tag}] hidden stores only at o % 8 == 0, == reference", ok_hid and extra == 0)
    check(f"P3[{tag}] fp32 logits at o % 64 == 0 == toy logits of prompt + F[:o]", ok_log)
    check(f"P3[{tag}] unstaged steps leave have == 0", ok_have)


def test_list():
    sched = [3, 0, 2, 1, 3, 3, 0, 1, 2, 3, 1, 0, 3, 2]
    r = run("L1", "list", two=True, sched=sched)
    W = len(r["F"])
    check("P1[list] committed ids == history", r["out"] == r["F"] and r["out2"] == r["F2"])
    check("P1[list] every window position measured exactly once", bool((r["nmeas"] == 1).all()))
    m, pp = TF.reference(r["prompt"], r["F"])
    d_lp, d_pp, am, ent, ml, el = TF.compare(r["buf"], m, pp, W)
    check("P1[list] metrics == single-token reference", d_lp <= 1e-5 and d_pp <= 1e-5 and am and ent <= 1e-4 and ml <= 1e-5 and el,
          f"dlp {d_lp:.1e} dpp {d_pp:.1e}")
    acc = [a for (_, a) in r["commits"][1:]]
    acc2 = [a for (_, a) in r["commits2"][1:]]
    want = [sched[k % len(sched)] for k in range(len(acc))]
    clamp_ok = all(a == w for a, w in zip(acc[:-1], want[:-1])) and acc[-1] <= want[-1]
    check("P2[list] accepted count of verify step k == list[k] (last step clamped)", clamp_ok,
          f"first {acc[:8]} want {want[:8]}")
    check("P2[list] two forced requests with the same list advance in lock step", acc == acc2[:len(acc)])
    check("P4 flush file written", r["z"] is not None)
    if r["z"] is None:
        return r
    same = (np.array_equal(r["z"]["moe"], r["dev"]["moe"].view(torch.int16).numpy().view(np.uint16))
            and np.array_equal(r["z"]["moe_have"], r["dev"]["have"].numpy())
            and np.array_equal(r["z"]["logits"], r["dev"]["lst"][: r["z"]["logits"].shape[0]].numpy()))
    check("P4 flush file == device stores", same)
    check("P4 slot freed after removal", r["slot"] in p4.S.free and r["idx"] not in p4.S.slot_host)
    check_stores("list", r, rows_in_prefill=37 + 32 + 23)   # prefill step: 3 prompts = 92 rows > 64 -> o = 0 unstaged
    # P5 steprec
    files = sorted(glob.glob(os.path.join(p4.PRIV_DIR, "steprec_*_r0.npz")))
    rows = np.concatenate([np.load(f)["rows"] for f in files]) if files else np.zeros((0, 1, 5))
    per = []
    for blk in rows:
        for row in blk:
            if int(row[1]) == r["idx"] and int(row[0]) >= 0:
                per.append((int(row[0]), int(row[2]), int(row[3]), int(row[4])))
    per.sort()
    exp, o0 = [], 0
    for n, a in r["commits"]:
        exp.append((o0, a if a is not None else 0))
        o0 += n
    got = [(o, a) for (_, o, _, a) in per]
    check("P5 steprec (row-0 output index, accepted) per step == commits", got == exp[:len(got)] and len(got) == len(exp),
          f"{got[:4]} vs {exp[:4]}")
    return r


def test_slot_reuse():
    r1 = run("S-a", "list", W=70, sched=[1, 2], co=False)
    # second request on the same engine state: same slot comes back reset
    e = r1["e"]
    CUR["eng"] = e
    prompt = [7000 + i for i in range(20)]
    F = TF.make_history(prompt, 70, 9)
    e.add("forced_b", prompt, SamplingParams(temperature=1.0, top_p=0.95, seed=21, max_tokens=70,
                                             extra_args={"forcewin_hooks": F, "local_fw_tag": "S-b",
                                                         "local_fw_draft": "list", "local_fw_sched": [2]}), 70)
    idx = e.reqs["forced_b"]["idx"]
    slot = p4.S.slot_host.get(idx)
    fresh = slot is not None and int(p4.S.have[:, slot].sum()) == 0 and int(p4.S.lhave[slot].sum()) == 0
    check("P4 a re-used slot starts with have == 0 (no stale rows)", fresh, f"slot {slot}")
    e.prefill(["forced_b"])
    while len(e.reqs["forced_b"]["out"]) < 70:
        ib, lg = e.decode(["forced_b"])
        e.finish_decode(["forced_b"], ib, lg)
    fw.remove_request(e.runner, "forced_b")
    z = np.load(os.path.join(p4.PRIV_DIR, "p4_S-b_r0.npz"))
    check("P4 second request's stores complete (have == 1 at every staged position)",
          bool((z["moe_have"] == 1).all()), f"have counts {np.bincount(z['moe_have'].ravel())}")


def test_other_modes():
    for mode in ("oracle", "poison"):
        r = run("M-" + mode.replace(":", ""), mode, W=130)
        m, pp = TF.reference(r["prompt"], r["F"])
        d_lp, d_pp, am, ent, ml, el = TF.compare(r["buf"], m, pp, len(r["F"]))
        check(f"P6[{mode}] ids == history, measured once, metrics == reference",
              r["out"] == r["F"] and bool((r["nmeas"] == 1).all()) and d_lp <= 1e-5 and am)
        if r["z"] is not None:
            check_stores(mode, r, rows_in_prefill=37 + 23)       # prefill 60 rows <= 64: o = 0 staged


def test_stage_hooks():
    reset_p4()
    x = torch.randn(48, H).to(torch.bfloat16)
    p4.hook_moe_out(MODS[1], x)
    ok1 = torch.equal(p4.S.stage[1, :48], x) and int(p4.S.stage_n[1]) == 48
    mh = Mod(3, mhc=True)
    p4.S.ids[id(mh)] = 3
    p4.hook_hidden(mh, x, torch.zeros(48, 4, H), torch.zeros(48, 4), torch.zeros(48, 4, 4))
    ok2 = torch.equal(p4.S.stage[3, :48], x) and int(p4.S.stage_n[3]) == 48
    big = torch.randn(100, H).to(torch.bfloat16)
    p4.hook_final(MODS[5], big)
    ok3 = int(p4.S.stage_n[5]) == 0
    p4.hook_moe_out(Mod(9), x)            # an unregistered module: nothing
    other = Mod(7)
    p4.S.ids[id(other)] = 4
    p4.hook_moe_out(other, x)             # a hidden-stage module passed to the MoE hook: ignored
    ok4 = int(p4.S.stage_n[4]) == 0
    check("S1 stage hooks copy exactly (MoE and mHC hidden path), stage_n = m", ok1 and ok2)
    check("S1 m > 64 -> not staged (stage_n 0); unregistered / wrong-kind modules ignored", ok3 and ok4)


def test_graph_static():
    from torch.utils._python_dispatch import TorchDispatchMode

    class Rec(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            self.ops.append(str(func))
            return func(*args, **(kwargs or {}))

    def trace(seed, m):
        reset_p4()
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(m, H, generator=g).to(torch.bfloat16)
        mh = Mod(3, mhc=True)
        p4.S.ids[id(mh)] = 3
        ptr = p4.S.stage.data_ptr()
        with Rec() as r:
            p4.hook_moe_out(MODS[0], x)
            p4.hook_hidden(mh, x, None, torch.zeros(m, 4), None)
            p4.hook_final(MODS[5], x)
        return r.ops, ptr == p4.S.stage.data_ptr()
    a, pa = trace(1, 32)
    b, pb = trace(2, 32)
    sync = [o for o in a if any(s in o for s in ("_local_scalar_dense", "item", "nonzero", "is_nonzero", "equal",
                                                 "_to_copy", "masked_select"))]
    check("S2 in-graph hooks: identical aten op sequence for different values; buffers keep their addresses",
          a == b and pa and pb and len(a) > 0, f"{len(a)} ops")
    check("S2 in-graph hooks: no host-sync op", not sync, str(sync[:5]))


def test_flush_rule():
    n0 = threading.active_count()
    reset_p4()
    p4.S.serving = False
    p4.on_add(0, "rule-a", 10)
    p4.on_remove(0, "rule-a", 10)
    f1 = glob.glob(os.path.join(p4.PRIV_DIR, "p4_rule-a_r0.npz"))
    p4.S.serving = True
    real = p4._capturing
    p4._capturing = lambda: True
    p4.on_add(1, "rule-b", 10)
    p4.on_remove(1, "rule-b", 10)
    p4.flush_steps()
    f2 = glob.glob(os.path.join(p4.PRIV_DIR, "p4_rule-b_r0.npz")) + glob.glob(os.path.join(p4.PRIV_DIR, "steprec_*"))
    p4._capturing = real
    p4.on_add(2, "rule-c", 10)
    p4.on_remove(2, "rule-c", 10)
    f3 = glob.glob(os.path.join(p4.PRIV_DIR, "p4_rule-c_r0.npz"))
    logs = "".join(open(p).read() for p in glob.glob(os.path.join(p4.LOG_DIR, "p4_r*.jsonl")))
    check("S3 flush refused before the first real batch (S.serving False): no file", not f1 and "flush_refused" in logs)
    check("S3 flush refused while a stream is capturing: no file", not f2)
    check("S3 flush after serving starts writes the file; the slot is freed every time", bool(f3) and len(p4.S.free) == p4.NSLOT)
    # on_batch never sets serving while capturing
    p4.S.serving = False
    p4._capturing = lambda: True
    p4.on_batch(types.SimpleNamespace(num_reqs=1, is_prefilling_np=np.array([0]), num_tokens=4), 1)
    s1 = p4.S.serving
    p4._capturing = real
    p4.on_batch(types.SimpleNamespace(num_reqs=1, is_prefilling_np=np.array([0]), num_tokens=4), 2)
    check("S3 serving starts at the first real batch outside capture, never inside", (not s1) and p4.S.serving)
    check("S3 the module starts no thread", threading.active_count() == n0)


def test_lint():
    src = open(p4.__file__).read()
    code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
    banned = [w for w in ("Thread(", "pin_memory", "cuda.Stream", "cuda.Event", ".synchronize(", "record_stream")
              if w in code]
    check("S4 p4_hooks.py: no thread, pinned memory, stream, event or synchronize", not banned, str(banned))
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    bad = []
    for name in ("_put", "hook_moe_out", "hook_hidden", "hook_final"):
        seg = ast.get_source_segment(src, fns[name])
        for w in (".item(", ".cpu(", ".tolist(", "nonzero", ".numpy(", "bool("):
            if w in seg:
                bad.append((name, w))
    check("S4 in-graph hooks: no .item / .cpu / .tolist / nonzero / .numpy / bool()", not bad, str(bad))
    fsrc = open(fw.__file__).read()
    fcode = "\n".join(ln.split("#")[0] for ln in fsrc.splitlines())
    check("S4 forcewin port: no thread or stream added", "Thread(" not in fcode and "cuda.Stream" not in fcode)


def test_register():
    class Glm5NextDecoderLayer:
        def __init__(self, i, moe):
            self.layer_idx, self._mlp_is_moe, self.mhc, self.is_sequence_parallel, self.n = i, moe, True, False, 4

    class Glm5NextModel:
        pass

    class Experts:
        def __init__(self):
            self.exl3_trellis_weights, self.exl3_trellis_tile_config = object(), None

    mods = [("language_model.model", Glm5NextModel())]
    for i in range(45):
        mods.append((f"language_model.model.layers.{i}", Glm5NextDecoderLayer(i, i >= 3)))
        if i >= 3:
            mods.append((f"language_model.model.layers.{i}.mlp.experts.routed_experts", Experts()))
    mods.append(("drafter.model.layers.3", Glm5NextDecoderLayer(3, True)))      # ignored (draft)
    runner = types.SimpleNamespace(model=types.SimpleNamespace(named_modules=lambda: iter(mods)), device=CPU,
                                   max_num_reqs=4, model_config=types.SimpleNamespace(get_hidden_size=lambda: H,
                                                                                      get_vocab_size=lambda: V))
    shutil.rmtree(p4.OUT_DIR, ignore_errors=True)
    p4.S.__init__()
    p4.S.rank = 0
    p4.register(runner)
    init = [json.loads(l) for p in glob.glob(os.path.join(p4.LOG_DIR, "p4_r0_*.jsonl")) for l in open(p)
            if '"init"' in l]
    ok = (len(p4.S.ids) == 6 and p4.S.stage is not None and init and not init[-1]["issues"]
          and init[-1]["n_moe_layers_with_weights"] == 42 and sorted(init[-1]["stage_modules"].values()) == list(range(6)))
    check("R1 register: six stages found by class/layer, draft ignored, stores allocated (rank 0), no issues", bool(ok),
          str(init[-1]["issues"] if init else None))
    import vllm.v1.worker.gpu.cudagraph_utils as cg
    check("R1 register installs the graph-mode label on the CUDA-graph managers",
          getattr(cg.CudaGraphManager, "_p4_hooks", False))
    p4.S.__init__()
    p4.S.rank = 1
    mods.append(("language_model.model.layers.12.dup", Glm5NextDecoderLayer(12, True)))
    p4.register(runner)
    init1 = [json.loads(l) for p in glob.glob(os.path.join(p4.LOG_DIR, "p4_r1_*.jsonl")) for l in open(p)
             if '"init"' in l]
    check("R1 rank 1: no stores; a duplicate stage module is an issue", p4.S.stage is None and init1 and
          any(i[0] == "duplicate" for i in init1[-1]["issues"]))


def test_binds():
    shutil.rmtree(p4.OUT_DIR, ignore_errors=True)
    p4.S.__init__()
    p4.S.rank = 0

    def binding(k, n):
        return types.SimpleNamespace(mixed_trellis_binding=types.SimpleNamespace(launch=types.SimpleNamespace(
            size_m=32, fc1_tile_k=k, fc1_tile_n=n, fc2_tile_k=k, fc2_tile_n=n, moe_block_size=8, fc2_moe_block_size=8,
            max_m_blocks=256, blocks_per_sm=1, route_ids_dtype=torch.int64, direct_topk_routes=False)))
    rt = dict(min_trellis_m=1, max_trellis_m=32, max_batched_tokens=1024)
    lay = types.SimpleNamespace(layer_name="language_model.model.layers.7.mlp.experts.routed_experts",
                                exl3_trellis_tile_config=None)
    p4.hook_bind(lay, "decode", 32, binding(128, 128), rt)
    p4.hook_bind(lay, "decode", 32, binding(128, 128), rt)
    p4.hook_bind(lay, "decode", 32, binding(64, 256), rt)
    p4.hook_bind(lay, "prefill", 900, binding(64, 256), rt)
    recs = [json.loads(l) for p in glob.glob(os.path.join(p4.LOG_DIR, "bind_r0_*.jsonl")) for l in open(p)]
    ok = (len(recs) == 3 and recs[0]["layer"] == 7 and recs[0]["launch"]["fc1_tile_k"] == 128 and not recs[0]["changed"]
          and recs[1]["changed"] and recs[1]["launch"]["fc1_tile_n"] == 256 and recs[2]["capacity"] == 1024
          and recs[0]["capacity"] == 32 and recs[0]["weight_tile_config"] == "None" and recs[0]["draft"] is False)
    check("B1 recorder: one record per key, repeats not logged, changes logged, capacity per plan, fields for the gates",
          ok, str([(r["plan"], r["m"], r["launch"]["fc1_tile_k"], r["changed"]) for r in recs]))
    import vllm.model_executor.layers.quantization.exl3 as e
    check("B1 exl3 hooks present and on (_P4_ON)", e._P4_ON and "hook_bind" in inspect.getsource(e))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("FW_THREADS", "4")))
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("all", "units"):
        test_stage_hooks()
        test_graph_static()
        test_flush_rule()
        test_lint()
        test_register()
        test_binds()
    if part in ("all", "list"):
        test_list()
        test_slot_reuse()
    if part in ("all", "modes"):
        test_other_modes()
    ok = summary("test_p4_cpu[%s]" % part)
    sys.exit(0 if ok else 1)
