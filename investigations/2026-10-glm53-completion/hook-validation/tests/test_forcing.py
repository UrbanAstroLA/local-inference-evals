#!/usr/bin/env python3
"""Forcing, measurement ordering and committed-state bookkeeping of the local-forcewin hook, run against the
installed B1 sampler code (hooked copies of rejection_sampler.py / sampler.py), the real RequestState, the real
post_update kernel and the hook module. CPU container, Triton interpreter, FORCEWIN=on.

Emulated: the model forward (a deterministic toy: next-token logits are a function of the token prefix only, so
every position has an exact single-token reference) and the drafter (greedy on the toy).

Checks
  F1-F6  per draft schedule (natural, oracle, poison, reject:1..3, sched): returned/committed ids == history;
         each window position measured exactly once; measured metrics == the single-token reference on the exact
         prefix; accepted-count histogram as the schedule prescribes; bonus accounting (sum of commits == W);
         integer committed state (all_token_ids, total_len, num_computed, last_sampled) after every step.
  F7     an unforced co-runner in the same batch gets the real sampler's output untouched.
  O1     rejection_sample does not modify the processed logits it reads (post-mask values are pre-decision).
  O2     static order in the hooked files: on_logits < sampler call; on_processed < rejection_sample / gumbel;
         override < get_num_sampled_and_rejected; sample() < postprocess_sampled; after_propose after drafting.
  O3     the measured values do not depend on the sampler's random draws (two request seeds, identical buffers).
  N1-N2  negative controls (power): an off-by-one output index and an "every row valid" mutant are detected.
  R1     prefill reference (local_fw_ref, chunked prefill) equals the forced-decode measurement on the toy.
"""
import ast
import inspect
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("FORCEWIN", "on")
os.environ.setdefault("FORCEWIN_DIR", "/tmp/fwtest")
os.environ.setdefault("FORCEWIN_WMAX", "512")

from fwtest_common import TERM, V, check, summary, toy_logits  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vllm import forcewin_hooks as fw  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from fwtest_common import _topkp_cpu as apply_top_k_top_p  # noqa: E402
from vllm.v1.worker.gpu.input_batch import post_update  # noqa: E402
from vllm.v1.worker.gpu.sample.sampler import Sampler  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample  # noqa: E402
from vllm.v1.worker.gpu.states import RequestState  # noqa: E402

assert fw.ON, "FORCEWIN must be on for this test"
CPU = torch.device("cpu")
K = 3
MAXR = 4


class Eng:
    def __init__(self):
        fw.S.__init__()
        self.rs = RequestState(MAXR, 2048, 1024, K, V, CPU)
        self.sampler = Sampler(MAXR, V, CPU, self.rs, num_speculative_tokens=K)
        spec = types.SimpleNamespace(num_speculative_tokens=K, rejection_sample_method="standard",
                                     synthetic_acceptance_rates=None)
        self.rej = RejectionSampler(self.sampler, spec, CPU)
        self.full = {}
        self.runner = types.SimpleNamespace(
            device=CPU, num_speculative_steps=K, max_num_reqs=MAXR, req_states=self.rs,
            model_config=types.SimpleNamespace(get_vocab_size=lambda: V),
            model=types.SimpleNamespace(compute_logits=self._compute_logits))
        self.reqs = {}
        self.commits = {}

    # prefill-reference plumbing: hidden row = (request key, position) packed as a float pair
    def _compute_logits(self, h):
        out = []
        for row in h.tolist():
            key, pos = int(row[0]), int(row[1])
            out.append(toy_logits(self.full[key][: pos + 1]))
        return torch.stack(out)

    def add(self, rid, prompt, sp, max_tokens):
        self.rs.add_request(rid, len(prompt), prompt, 0, max_tokens)
        idx = self.rs.req_id_to_index[rid]
        self.sampler.add_request(idx, len(prompt), sp)
        fw.add_request(self.runner, rid, idx, len(prompt), types.SimpleNamespace(sampling_params=sp))
        self.rs.apply_staged_writes()
        self.sampler.apply_staged_writes()
        self.reqs[rid] = dict(idx=idx, prompt=list(prompt), out=[], max=max_tokens, prefilled=False, key=len(self.reqs))
        self.full[self.reqs[rid]["key"]] = list(prompt)
        self.commits[rid] = []
        return idx

    def _ib(self, rids, tok, pos, lidx, cu, rowreq, rowj, seq, qsl, ndraft, prefilling, ncp):
        idx = [self.reqs[r]["idx"] for r in rids]
        return types.SimpleNamespace(
            num_reqs=len(rids), req_ids=list(rids),
            idx_mapping=torch.tensor(idx, dtype=torch.int32), idx_mapping_np=np.array(idx),
            input_ids=torch.tensor(tok, dtype=torch.int32), positions=torch.tensor(pos, dtype=torch.int64),
            logits_indices=torch.tensor(lidx, dtype=torch.int64),
            cu_num_logits=torch.tensor(cu, dtype=torch.int32), cu_num_logits_np=np.array(cu, dtype=np.int32),
            expanded_idx_mapping=torch.tensor([idx[b] for b in rowreq], dtype=torch.int32),
            expanded_local_pos=torch.tensor(rowj, dtype=torch.int32),
            seq_lens=torch.tensor(seq, dtype=torch.int32),
            query_start_loc=torch.tensor(qsl, dtype=torch.int32),
            query_start_loc_np=np.array(qsl, dtype=np.int32),
            num_draft_tokens=ndraft, num_draft_tokens_per_req=np.array([K if ndraft else 0] * len(rids)),
            is_prefilling_np=np.array(prefilling), num_tokens=len(tok),
            num_computed_prefill_tokens_np=np.array(ncp, dtype=np.int32))

    def prefill(self, rids, chunk=None, ref_hidden=False):
        """One prefill step (whole prompts, or chunks of `chunk` tokens, Sampler path)."""
        tok, pos, lidx, cu, rowreq, rowj, seq, qsl, ncp, pf = [], [], [], [0], [], [], [], [0], [], []
        hid = []
        for b, r in enumerate(rids):
            q = self.reqs[r]
            done = q.get("pdone", 0)
            n = len(q["prompt"]) - done if chunk is None else min(chunk, len(q["prompt"]) - done)
            tok += q["prompt"][done:done + n]
            pos += list(range(done, done + n))
            hid += [[q["key"], p] for p in range(done, done + n)]
            lidx.append(len(tok) - 1)
            cu.append(cu[-1] + 1)
            rowreq.append(b)
            rowj.append(0)
            seq.append(done + n)
            qsl.append(len(tok))
            ncp.append(done)
            pf.append(done + n < len(q["prompt"]))
            q["pdone"] = done + n
        ib = self._ib(rids, tok, pos, lidx, cu, rowreq, rowj, seq, qsl, 0, pf, ncp)
        fw.on_batch(self.runner, ib)
        logits = torch.stack([toy_logits(self.full[self.reqs[r]["key"]][: seq[b]]) for b, r in enumerate(rids)])
        fw.on_logits(self.runner, logits, ib)
        out = self.sampler(logits, ib)
        if ref_hidden:
            fw.prefill_ref(self.runner, torch.tensor(hid, dtype=torch.float32), ib)
        post_update(ib.idx_mapping, self.rs.num_computed_tokens.gpu, self.rs.last_sampled_tokens, None,
                    out.sampled_token_ids, out.num_sampled, out.num_rejected, ib.query_start_loc,
                    self.rs.all_token_ids.gpu, self.rs.total_len.gpu)
        fw.after_commit(self.runner, ib, out, out.num_sampled, out.num_rejected, None)
        for b, r in enumerate(rids):
            n = int(out.num_sampled[b])
            if n:
                self.reqs[r]["prefilled"] = True
                self._commit(r, out.sampled_token_ids[b, :n].tolist(), 0)
        self._propose(rids, ib)
        return out

    def _commit(self, r, toks, a):
        q = self.reqs[r]
        q["out"] += toks
        self.full[q["key"]] = q["prompt"] + q["out"]
        self.commits[r].append((len(toks), a))

    def _propose(self, rids, ib):
        for r in rids:
            q = self.reqs[r]
            pre = q["prompt"] + q["out"]
            d = []
            for _ in range(K):
                d.append(int(toy_logits(pre + d).argmax()))
            self.rs.draft_tokens[q["idx"]] = torch.tensor(d, dtype=self.rs.draft_tokens.dtype)
        fw.after_propose(self.runner, ib)

    def decode(self, rids, hook=True):
        tok, pos, cu, rowreq, rowj, seq, qsl = [], [], [0], [], [], [], [0]
        logits = []
        for b, r in enumerate(rids):
            q = self.reqs[r]
            d = self.rs.draft_tokens[q["idx"]].tolist()
            inp = [q["out"][-1]] + d
            p0 = len(q["prompt"]) + len(q["out"]) - 1
            tok += inp
            pos += list(range(p0, p0 + K + 1))
            pre = q["prompt"] + q["out"]
            for j in range(K + 1):
                logits.append(toy_logits(pre + d[:j]))
            cu.append(cu[-1] + K + 1)
            rowreq += [b] * (K + 1)
            rowj += list(range(K + 1))
            seq.append(p0 + K + 1)
            qsl.append(len(tok))
        ib = self._ib(rids, tok, pos, list(range(len(tok))), cu, rowreq, rowj, seq, qsl, K * len(rids),
                      [False] * len(rids), [len(self.reqs[r]["prompt"]) for r in rids])
        logits = torch.stack(logits)
        if hook:
            fw.on_batch(self.runner, ib)
            fw.on_logits(self.runner, logits, ib)
        else:
            fw.S.sc = None
        return ib, logits

    def finish_decode(self, rids, ib, logits):
        out = self.rej(logits, ib, None)
        post_update(ib.idx_mapping, self.rs.num_computed_tokens.gpu, self.rs.last_sampled_tokens, None,
                    out.sampled_token_ids, out.num_sampled, out.num_rejected, ib.query_start_loc,
                    self.rs.all_token_ids.gpu, self.rs.total_len.gpu)
        fw.after_commit(self.runner, ib, out, out.num_sampled, out.num_rejected, None)
        acc = fw.S.sc["acc"].tolist() if fw.S.sc is not None and fw.S.sc.get("acc") is not None else None
        for b, r in enumerate(rids):
            n = int(out.num_sampled[b])
            self._commit(r, out.sampled_token_ids[b, :n].tolist(), None if acc is None else acc[b])
        self._propose(rids, ib)
        return out


def make_history(prompt, n, seed):
    """A history sampled from the toy at top_p 0.95 (natural acceptance), with two out-of-nucleus tokens."""
    g = torch.Generator().manual_seed(seed)
    h = []
    for i in range(n):
        x = toy_logits(prompt + h)
        if i in (17, 40):
            h.append(int(torch.argmin(x)))                   # the least likely token: outside the top_p nucleus
            continue
        pm = torch.softmax(apply_top_k_top_p(x[None].clone(), None, torch.tensor([0.95]))[0], -1)
        h.append(int(torch.multinomial(pm, 1, generator=g)))
    return h


def reference(prompt, F):
    """Single-token history-advance reference: metrics of P(. | prompt + F[:o]) for every o."""
    raw = torch.stack([toy_logits(prompt + F[:o]) for o in range(len(F))])
    ft = torch.tensor(F, dtype=torch.int64)
    m = fw.row_metrics(raw, ft)
    pp = fw.post_probs(apply_top_k_top_p(raw.clone(), None, torch.full((len(F),), 0.95)), ft)
    return m, pp


def compare(buf, m, pp, n):
    C = fw.C
    got_lp = buf[:n, C["lp_pre_0"]:C["lp_pre_0"] + fw.NT]
    got_pp = buf[:n, C["p_post_0"]:C["p_post_0"] + fw.NT]
    d_lp = float((got_lp - m["lp_pre"][:n]).abs().nan_to_num(0).max())
    d_pp = float((got_pp - pp[:n]).abs().nan_to_num(0).max())
    am = bool(torch.equal(buf[:n, C["argmax"]].long(), m["argmax"][:n]))
    ent = float((buf[:n, C["entropy"]] - m["entropy"][:n]).abs().max())
    ml = float((buf[:n, C["mass_lt_0"]:C["mass_lt_0"] + fw.NT] - m["mass_lt"][:n]).abs().nan_to_num(0).max())
    el = bool(torch.equal(buf[:n, C["elig_lo_0"]:C["elig_lo_0"] + fw.NT], m["elig_lo"][:n]))
    return d_lp, d_pp, am, ent, ml, el


def run_scenario(name, mode, W=96, seed=3, co_runner=True, req_seed=11):
    e = Eng()
    prompt = [5000 + (7 * i) % 900 for i in range(37)]
    F = make_history(prompt, W, seed)
    sp = SamplingParams(temperature=1.0, top_p=0.95, seed=req_seed, max_tokens=len(F),
                        extra_args={"forcewin_hooks": F, "local_fw_tag": name, "local_fw_draft": mode})
    e.add("forced", prompt, sp, len(F))
    rids = ["forced"]
    if co_runner:
        sp2 = SamplingParams(temperature=1.0, top_p=0.95, seed=req_seed + 1, max_tokens=10_000)
        e.add("free", [4000 + i for i in range(23)], sp2, 10_000)
        rids.append("free")
    e.prefill(rids)
    idx = e.reqs["forced"]["idx"]
    state_ok = True
    while len(e.reqs["forced"]["out"]) < len(F):
        ib, logits = e.decode(rids)
        e.finish_decode(rids, ib, logits)
        n_out = len(e.reqs["forced"]["out"])
        P = len(prompt)
        tl = int(e.rs.total_len.gpu[idx])
        nc = int(e.rs.num_computed_tokens.gpu[idx])
        ls = int(e.rs.last_sampled_tokens[idx, 0])
        ids = e.rs.all_token_ids.gpu[idx, P:P + n_out].tolist()
        state_ok &= (tl == P + n_out) and (nc == P + n_out - 1) and (ls == F[n_out - 1]) and ids == F[:n_out]
    out = e.reqs["forced"]["out"]
    buf = fw.S.buf[idx, : len(F)].clone()
    nmeas = fw.S.nmeas[idx, : len(F)].clone()
    m, pp = reference(prompt, F)
    d_lp, d_pp, am, ent, ml, el = compare(buf, m, pp, len(F))
    acc = [a for (_, a) in e.commits["forced"][1:]]
    hist = np.bincount([a for a in acc if a is not None and a >= 0], minlength=K + 1)
    sum_commit = sum(n for (n, _) in e.commits["forced"])
    dec = [(n, a) for (n, a) in e.commits["forced"][1:] if a is not None and a >= 0]
    bonus_ok = all(n == a + 1 for (n, a) in dec[:-1]) and (not dec or dec[-1][0] <= dec[-1][1] + 1)
    inv = fw.S.inv[idx].tolist()
    fw.remove_request(e.runner, "forced")          # writes req_<tag>_r0.pt (the analysis input)
    p = os.path.join(fw.OUT_DIR, "req_%s_r0.pt" % fw._safe(name))
    saved = torch.load(p, weights_only=False) if os.path.exists(p) else None
    file_ok = saved is not None and torch.equal(torch.nan_to_num(saved["metrics"], 9.0), torch.nan_to_num(buf, 9.0)) \
        and torch.equal(saved["nmeas"], nmeas)
    return dict(file_ok=file_ok, F=F, out=out, buf=buf, nmeas=nmeas, d_lp=d_lp, d_pp=d_pp, am=am, ent=ent, ml=ml, el=el,
                hist=hist.tolist(), state_ok=state_ok, sum_commit=sum_commit, bonus_ok=bonus_ok, inv=inv,
                e=e, prompt=prompt, oon=int(torch.nansum(buf[:, fw.C["out_of_nucleus"]])))


def t_scenarios(only=None):
    expect = {"natural": None, "oracle": "all_K", "poison": "zero", "reject:1": 0, "reject:2": 1,
              "reject:3": 2, "sched:7": "mixed"}
    if only:
        expect = {k: v for k, v in expect.items() if k in only}
    res = {}
    for mode, ex in expect.items():
        r = run_scenario(mode, mode)
        res[mode] = r
        W = len(r["F"])
        ok_ids = r["out"] == r["F"]
        once = bool((r["nmeas"] == 1).all())
        tol = 1e-5
        vals = r["d_lp"] <= tol and r["d_pp"] <= tol and r["am"] and r["ent"] <= 1e-4 and r["ml"] <= tol and r["el"]
        h = r["hist"]
        steps = sum(h)
        if ex is None:
            sched_ok = steps > 0
        elif ex == "all_K":
            sched_ok = h[K] >= steps - 2
        elif ex == "zero":
            sched_ok = h[0] == steps
        elif ex == "mixed":
            sched_ok = all(x > 0 for x in h)
        else:
            sched_ok = h[ex] >= steps - 2
        check("F1[%s] committed ids == history (%d tokens)" % (mode, W), ok_ids)
        check("F2[%s] every window position measured exactly once" % mode, once,
              "min/max nmeas %d/%d" % (int(r["nmeas"].min()), int(r["nmeas"].max())))
        check("F3[%s] measured metrics == single-token reference on the exact prefix" % mode, vals,
              "max|dlp| %.2e max|dpost| %.2e argmax_eq %s" % (r["d_lp"], r["d_pp"], r["am"]))
        check("F4[%s] accepted-count histogram as scheduled" % mode, sched_ok, "hist(a=0..%d) %s" % (K, h))
        check("F5[%s] bonus accounting: each step commits a+1 (clamped), sum == W" % mode,
              r["bonus_ok"] and r["sum_commit"] == W, "sum %d W %d" % (r["sum_commit"], W))
        check("F6[%s] integer committed state after every step (total_len, num_computed, last, all_token_ids)"
              % mode, r["state_ok"] and r["inv"][1] == 0, "inv %s" % r["inv"])
    check("F6c per-request record written at removal equals the device buffers (all schedules)",
          all(r["file_ok"] for r in res.values()))
    if "oracle" in res:
        check("F6b out-of-nucleus history tokens flagged (2 planted)", res["oracle"]["oon"] == 2,
              "flagged %d" % res["oracle"]["oon"])
    return res


def _committed(x, b):
    """What the engine consumes from a sampler output for batch row b: the first num_sampled tokens, num_sampled
    and num_rejected. Entries past num_sampled are uninitialised scratch (torch.empty in rejection_sample) and
    differ between two identical hook-off calls, so they are not compared (tests/out/diag_f7.log)."""
    n = int(x.num_sampled[b])
    return x.sampled_token_ids[b, :n].tolist(), n, int(x.num_rejected[b])


def t_corunner(steps=4):
    e = Eng()
    prompt = [6000 + i for i in range(31)]
    F = make_history(prompt, 40, 5)
    sp = SamplingParams(temperature=1.0, top_p=0.95, seed=1, max_tokens=len(F),
                        extra_args={"forcewin_hooks": F, "local_fw_draft": "natural"})
    e.add("forced", prompt, sp, len(F))
    e.add("free", [4100 + i for i in range(19)], SamplingParams(temperature=1.0, top_p=0.95, seed=2,
                                                                 max_tokens=1000), 1000)
    rids = ["forced", "free"]
    e.prefill(rids)
    same, caught, forced_hist, detail = True, True, True, []
    orig = fw.forced_override
    for step in range(steps):
        ib, logits = e.decode(rids, hook=True)
        sc = fw.S.sc
        a = e.rej(logits.clone(), ib, None)             # hook on
        fw.S.sc = None
        b = e.rej(logits.clone(), ib, None)             # hook off, same step and inputs
        ca, cb = _committed(a, 1), _committed(b, 1)
        same &= ca == cb
        detail.append((ca[1], ca[2], cb[1], cb[2]))
        n0 = len(e.reqs["forced"]["out"])
        forced_hist &= _committed(a, 0)[0] == F[n0: n0 + int(a.num_sampled[0])]

        def planted(*args, **kw):                        # negative control: perturb the co-runner's row
            s_, n_, v_, o_, acc_ = orig(*args, **kw)
            s_ = s_.clone()
            s_[1, 0] = (s_[1, 0] + 1) % V
            return s_, n_, v_, o_, acc_

        fw.S.sc = sc
        fw.forced_override = planted
        try:
            c = e.rej(logits.clone(), ib, None)
        finally:
            fw.forced_override = orig
        caught &= _committed(c, 1) != cb
        fw.S.sc = sc
        e.finish_decode(rids, ib, logits)                # advance the step with the hook on
    check("F7 unforced co-runner's committed tokens, num_sampled, num_rejected equal the real sampler output "
          "(hook on vs off, same step, %d steps)" % steps, same, "(n,rej) on/off per step %s" % detail)
    check("F7n negative control: a planted change of the co-runner's committed token is detected", caught)
    check("F7b forced request commits history tokens in the same step", forced_hist)


def t_order():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(8, 1000, generator=g)
    x[x < -1.0] = float("-inf")
    before = x.clone()
    cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    ds = torch.randint(0, 1000, (8,), generator=g)
    pos = torch.arange(8, dtype=torch.int64) + 100
    idx = torch.tensor([0, 1], dtype=torch.int32)
    eidx = torch.tensor([0] * 4 + [1] * 4, dtype=torch.int32)
    lp = torch.tensor([0, 1, 2, 3] * 2, dtype=torch.int32)
    rejection_sample(x, None, ds, cu, pos, idx, eidx, lp, torch.ones(2), torch.tensor([3, 4]), 3)
    check("O1 rejection_sample leaves the processed logits it reads unchanged", torch.equal(
        torch.nan_to_num(x, neginf=-1e30), torch.nan_to_num(before, neginf=-1e30)))
    import vllm.v1.worker.gpu.model_runner as MR
    import vllm.v1.worker.gpu.sample.sampler as SM
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler as RS

    def order(src, names):
        pos_ = {}
        for n in names:
            k = src.find(n)
            pos_[n] = k
        return pos_

    s = inspect.getsource(RS.RejectionSampler._verify)
    o = order(s, ["_fw.on_processed", "rejection_sample("])
    ok1 = 0 <= o["_fw.on_processed"] < o["rejection_sample("]
    s = inspect.getsource(RS.RejectionSampler.__call__)
    o = order(s, ["_fw.override", "get_num_sampled_and_rejected("])
    ok2 = 0 <= o["_fw.override"] < o["get_num_sampled_and_rejected("]
    s = inspect.getsource(SM.Sampler.sample)
    o = order(s, ["apply_top_k_top_p(processed_logits", "_fw.on_processed", "gumbel_sample("])
    ok3 = 0 <= o["apply_top_k_top_p(processed_logits"] < o["_fw.on_processed"] < o["gumbel_sample("]
    s = inspect.getsource(MR.GPUModelRunner.sample)
    o = order(s, ["_fw.on_logits", "self.sampler(logits", "self.rejection_sampler("])
    ok4 = 0 <= o["_fw.on_logits"] < o["self.sampler(logits"] and o["_fw.on_logits"] < o["self.rejection_sampler("]
    s = inspect.getsource(MR.GPUModelRunner.sample_tokens)
    o = order(s, ["self.sample(", "self.postprocess_sampled(", "_fw.after_commit", "self.speculator.propose(",
                  "_fw.after_propose", "set_draft_tokens("])
    ok5 = 0 <= o["self.sample("] < o["self.postprocess_sampled("] < o["_fw.after_commit"] \
        < o["self.speculator.propose("] < o["_fw.after_propose"] < o["set_draft_tokens("]
    check("O2 static order: measure (raw, post-mask) < decide < override < commit < propose < draft override",
          ok1 and ok2 and ok3 and ok4 and ok5, "verify %s call %s sampler %s runner %s tokens %s"
          % (ok1, ok2, ok3, ok4, ok5))


def t_seed_independence(res):
    r2 = run_scenario("natural-seed2", "natural", req_seed=97)
    r1 = res["natural"]
    same = torch.equal(torch.nan_to_num(r1["buf"], 7.0), torch.nan_to_num(r2["buf"], 7.0)) and r1["out"] == r2["out"]
    check("O3 measured values independent of the sampler's random draws (request seeds 11 vs 97)", same)


def t_negative_controls():
    orig = fw.row_plan

    def off_by_one(cu, pos, prompt_len_r, is_forced_r, forced_len_r):
        req, j, o, fwin = orig(cu, pos, prompt_len_r, is_forced_r, forced_len_r)
        return req, j, o + 1, fwin & (o + 1 < forced_len_r[req])

    fw.row_plan = off_by_one
    try:
        r = run_scenario("neg-offbyone", "natural", W=60)
    finally:
        fw.row_plan = orig
    det = (r["d_lp"] > 0.5) or not r["am"] or not bool((r["nmeas"] == 1).all())
    check("N1 negative control: an off-by-one measurement index is detected", det,
          "max|dlp| %.2f argmax_eq %s" % (r["d_lp"], r["am"]))
    orig_fo = fw.forced_override

    def all_rows(*a, **k):
        s, n, valid, oi, acc = orig_fo(*a, **k)
        R, K1 = valid.shape
        return s, n, (acc[:, None] >= 0) & torch.ones_like(valid), oi, acc

    fw.forced_override = all_rows
    try:
        r = run_scenario("neg-allrows", "reject:1", W=60)
    finally:
        fw.forced_override = orig_fo
    det = not bool((r["nmeas"] == 1).all()) or r["d_lp"] > 1e-5
    check("N2 negative control: rows conditioned on rejected drafts counted as measurements are detected", det,
          "max nmeas %d" % int(r["nmeas"].max()))


def t_prefill_ref():
    e = Eng()
    prompt = [5000 + (7 * i) % 900 for i in range(37)]
    F = make_history(prompt, 90, 3)
    full = prompt + F
    sp = SamplingParams(temperature=1.0, top_p=0.95, seed=5, max_tokens=1,
                        extra_args={"local_fw_ref": len(prompt), "local_fw_tag": "ref"})
    e.add("ref", full, sp, 1)
    e.full[e.reqs["ref"]["key"]] = full
    idx = e.reqs["ref"]["idx"]
    while not e.reqs["ref"]["prefilled"]:
        e.prefill(["ref"], chunk=50, ref_hidden=True)
    buf = fw.S.buf[idx, : len(F)].clone()
    m, pp = reference(prompt, F)
    d_lp, d_pp, am, ent, ml, el = compare(buf, m, pp, len(F))
    once = bool((fw.S.nmeas[idx, : len(F)] == 1).all())
    check("R1 prefill reference (chunked, every prompt row) == single-token reference at every index",
          d_lp <= 1e-5 and d_pp <= 1e-5 and am and once, "max|dlp| %.2e once %s" % (d_lp, once))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("FW_THREADS", max(1, os.cpu_count() // 2))))
    part = sys.argv[1] if len(sys.argv) > 1 else "all"     # all | <schedule> | rest (parallel shards)
    if part == "all":
        res = t_scenarios()
        t_corunner()
        t_order()
        t_seed_independence(res)
        t_negative_controls()
        t_prefill_ref()
    elif part == "rest":
        t_corunner()
        t_order()
        t_negative_controls()
        t_prefill_ref()
    elif part == "seed":
        t_seed_independence(t_scenarios(only=["natural"]))
    else:
        t_scenarios(only=[part])
    ok = summary("test_forcing")
    sys.exit(0 if ok else 1)
