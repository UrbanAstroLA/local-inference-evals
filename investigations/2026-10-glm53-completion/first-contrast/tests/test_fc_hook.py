#!/usr/bin/env python3
"""CPU hook tests for the first-contrast envelope (PLAN.md amendment 22 items 4 and 9; PROPOSAL.md s.7.4). The hook is the
validated module (hook-validation/hook/forcewin_hooks.py, sha256 15e93e23...), UNCHANGED; these tests run it with
FORCEWIN_WMAX=8448 (the first-contrast setting) against the installed B1 sampler code, the real RequestState
and post_update, exactly as hook-validation/tests/test_forcing.py does (CPU container, Triton interpreter, toy model).

  W1  WMAX = 8448: module constant, forced-id table and measurement buffer widths (8,449 output indices)
  W2  an 8,199-id forced list (the longest first-contrast list) under the oracle schedule: committed ids == history,
      every output index measured exactly once, metrics == the single-token reference at every index, integer
      committed state after every step, the request record written at removal equals the device buffers
  W2n natural drafts over the history's last 600 ids at high absolute positions (7,636-8,235) on a cached prefix of
      4,352 tokens: committed ids, measured once, equal to the reference
  W3  a list of 8,450 ids (> WMAX + 1) is silently truncated to 8,449 by add_request: the plan check (P1) is the
      guard; documents why runner/plan_check.py refuses such a list
  W4  forcing on a CACHED prefix: prompt 4,400 tokens of which 4,352 are cached (num_computed_prefill_tokens starts
      at 4,352, so the first prefill row sits at absolute position 4,352): committed ids, one measurement per
      output index, values == reference, o = absolute position + 1 - prompt length; with an unforced co-runner;
      natural and reject:2 schedules; and a cached prefix at a multiple of 4,352 with a 3,224-token remainder
      prefilled in two chunks (the largest first-contrast remainder)
  W5  prefill reference on a cached prefix: ref >= cached boundary -> every row measured once and equal to the
      reference; ref < cached boundary -> rows below the boundary are NOT measured (nmeas 0): references must be
      built cold (LW1-LW3 are: cached_expected 0, gated)
  W6  row_plan with absolute positions starting at a cached offset: o = pos + 1 - prompt_len for every row
  W7  on_selection at output indices 1,000 (not designated), 8,128 and 8,193 for a 112,024-token prompt: records
      land at (request, o, layer) for the designated ones only; the selection record at o = 8,193
"""
import os
import sys
import time
import types

os.environ["FORCEWIN"] = "on"
os.environ["FORCEWIN_WMAX"] = "8448"
os.environ.setdefault("FORCEWIN_DIR", "/tmp/fwtest-fc")
sys.path.insert(0, "/hv/tests")

import test_forcing as TF  # noqa: E402  (the validated scaffolding: Eng, make_history, reference, compare)
from fwtest_common import check, summary, toy_logits  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vllm import forcewin_hooks as fw  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.worker.gpu.input_batch import post_update  # noqa: E402
from vllm.v1.worker.gpu.states import RequestState  # noqa: E402

K = TF.K
MAXLEN = 16384


class EngL(TF.Eng):
    """TF.Eng with room for long histories and an optional cached prefix (num_computed starts at `cached`)."""

    def __init__(self):
        super().__init__()
        self.rs = RequestState(TF.MAXR, MAXLEN, 4096, K, TF.V, TF.CPU)
        from vllm.v1.worker.gpu.sample.sampler import Sampler
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
        self.sampler = Sampler(TF.MAXR, TF.V, TF.CPU, self.rs, num_speculative_tokens=K)
        spec = types.SimpleNamespace(num_speculative_tokens=K, rejection_sample_method="standard",
                                     synthetic_acceptance_rates=None)
        self.rej = RejectionSampler(self.sampler, spec, TF.CPU)
        self.runner.req_states = self.rs

    def add(self, rid, prompt, sp, max_tokens, cached=0):
        self.rs.add_request(rid, len(prompt), prompt, cached, max_tokens)
        idx = self.rs.req_id_to_index[rid]
        self.sampler.add_request(idx, len(prompt), sp)
        fw.add_request(self.runner, rid, idx, len(prompt), types.SimpleNamespace(sampling_params=sp))
        self.rs.apply_staged_writes()
        self.sampler.apply_staged_writes()
        self.reqs[rid] = dict(idx=idx, prompt=list(prompt), out=[], max=max_tokens, prefilled=False,
                              key=len(self.reqs), pdone=cached)
        self.full[self.reqs[rid]["key"]] = list(prompt)
        self.commits[rid] = []
        return idx


def fast_history(prompt, n, seed):
    """Long histories: the toy's argmax with a 10 % chance of the second choice (cheap; no top_p sampling)."""
    g = np.random.default_rng(seed)
    h = []
    for i in range(n):
        x = toy_logits(prompt + h)
        t2 = torch.topk(x, 2).indices.tolist()
        h.append(t2[1] if g.random() < 0.1 else t2[0])
    return h


def scenario(mode, prompt, F, cached=0, co_runner=False, chunk=None, name=None, measure_from=0):
    e = EngL()
    sp = SamplingParams(temperature=1.0, top_p=0.95, seed=11, max_tokens=len(F),
                        extra_args={"forcewin_hooks": F, "local_fw_tag": name or mode, "local_fw_draft": mode})
    e.add("forced", prompt, sp, len(F), cached=cached)
    rids = ["forced"]
    if co_runner:
        e.add("free", [4000 + i for i in range(23)], SamplingParams(temperature=1.0, top_p=0.95, seed=12,
                                                                     max_tokens=100000), 100000)
        rids.append("free")
    if co_runner:
        while not e.reqs["free"]["prefilled"]:
            e.prefill(["free"])
    while not e.reqs["forced"]["prefilled"]:
        e.prefill(["forced"], chunk=chunk)
    idx = e.reqs["forced"]["idx"]
    P = len(prompt)
    state_ok = True
    while len(e.reqs["forced"]["out"]) < len(F):
        ib, logits = e.decode(rids)
        e.finish_decode(rids, ib, logits)
        n_out = len(e.reqs["forced"]["out"])
        tl = int(e.rs.total_len.gpu[idx])
        nc = int(e.rs.num_computed_tokens.gpu[idx])
        state_ok &= (tl == P + n_out) and (nc == P + n_out - 1) and \
            e.rs.all_token_ids.gpu[idx, P + n_out - 1].item() == F[n_out - 1]
    buf = fw.S.buf[idx, : len(F)].clone()
    nmeas = fw.S.nmeas[idx, : len(F)].clone()
    lo = measure_from
    raw = torch.stack([toy_logits(prompt + F[:o]) for o in range(lo, len(F))])
    ft = torch.tensor(F[lo:], dtype=torch.int64)
    m = fw.row_metrics(raw, ft)
    Cc = fw.C
    d_lp = float((buf[lo:, Cc["lp_pre_0"]:Cc["lp_pre_0"] + fw.NT] - m["lp_pre"]).abs().nan_to_num(0).max())
    am = bool(torch.equal(buf[lo:, Cc["argmax"]].long(), m["argmax"]))
    fw.remove_request(e.runner, "forced")
    p = os.path.join(fw.OUT_DIR, "req_%s_r0.pt" % fw._safe(name or mode))
    saved = torch.load(p, weights_only=False) if os.path.exists(p) else None
    file_ok = saved is not None and torch.equal(torch.nan_to_num(saved["metrics"], 9.0), torch.nan_to_num(buf, 9.0))
    return dict(out=e.reqs["forced"]["out"], F=F, nmeas=nmeas, d_lp=d_lp, am=am, state_ok=state_ok, file_ok=file_ok,
                inv=fw.S.inv[idx].tolist())


def t_w1():
    e = EngL()
    fw.register(e.runner)                      # the capture-time allocation of every persistent table
    check("W1 WMAX = 8448: module constant, forced table [R, 8449], buffer [R + 1, 8449, NM]",
          fw.WMAX == 8448 and tuple(fw.S.forced.shape) == (TF.MAXR, 8449) and fw.S.buf.shape[1] == 8449
          and fw.S.nmeas.shape[1] == 8449, "WMAX %d forced %s buf %s" % (fw.WMAX, tuple(fw.S.forced.shape),
                                                                      tuple(fw.S.buf.shape)))
    return e


def t_w2():
    prompt = [5000 + (7 * i) % 900 for i in range(37)]
    t = time.time()
    F = fast_history(prompt, 8199, 3)
    r = scenario("oracle", prompt, F, name="w2-oracle-8199")
    once = bool((r["nmeas"] == 1).all())
    check("W2 8,199-id list, oracle: committed ids == history, every o in [0, 8198] measured once, metrics == reference,"
          " integer state every step, record == buffers", r["out"] == F and once and r["d_lp"] <= 1e-5 and r["am"]
          and r["state_ok"] and r["file_ok"], "n %d once %s max|dlp| %.2e argmax %s state %s file %s (%.0f s)" % (
              len(r["out"]), once, r["d_lp"], r["am"], r["state_ok"], r["file_ok"], time.time() - t))
    return prompt, F


def t_w2n(prompt, F):
    """natural drafts over the history's last 600 ids: prompt = prompt + F[:7599] (4,352 cached), list = F[7599:]."""
    pre = prompt + F[:7599]
    r = scenario("natural", pre, F[7599:], cached=4352, name="w2n-natural-tail")
    once = bool((r["nmeas"] == 1).all())
    check("W2n natural drafts at absolute positions 7,636-8,235 on a cached prefix: ids, once, reference",
          r["out"] == F[7599:] and once and r["d_lp"] <= 1e-5 and r["am"] and r["state_ok"],
          "once %s max|dlp| %.2e" % (once, r["d_lp"]))


def t_w3():
    e = EngL()
    F = list(range(1000, 1000 + 8450))
    sp = SamplingParams(temperature=1.0, top_p=0.95, seed=1, max_tokens=len(F),
                        extra_args={"forcewin_hooks": F, "local_fw_tag": "w3"})
    e.add("w3", [5, 6, 7], sp, len(F))
    r = fw.S.reqs.get(e.reqs["w3"]["idx"])
    check("W3 a list > WMAX + 1 (8,450) is truncated to 8,449 by add_request (plan check P1 refuses such lists)",
          r is not None and len(r.F) == 8449 and int(fw.S.flen[e.reqs["w3"]["idx"]]) == 8449,
          "kept %s" % (len(r.F) if r else None))


def t_w4():
    prompt = [5000 + (7 * i) % 900 for i in range(4400)]
    F = fast_history(prompt, 120, 5)
    for mode in ("natural", "reject:2"):
        r = scenario(mode, prompt, F, cached=4352, co_runner=True, name="w4-" + mode)
        once = bool((r["nmeas"] == 1).all())
        check("W4[%s] forcing on a cached prefix (4,352 of 4,400 cached), with a co-runner: ids, once, reference, state"
              % mode, r["out"] == F and once and r["d_lp"] <= 1e-5 and r["am"] and r["state_ok"] and r["inv"][1] == 0,
              "once %s max|dlp| %.2e argmax %s state %s inv %s" % (once, r["d_lp"], r["am"], r["state_ok"], r["inv"]))
    prompt = [3000 + (11 * i) % 700 for i in range(4352 + 3224)]
    F = fast_history(prompt, 96, 9)
    r = scenario("natural", prompt, F, cached=4352, chunk=2048, name="w4-rem3224")
    once = bool((r["nmeas"] == 1).all())
    check("W4b cached prefix 4,352 + 3,224-token remainder prefilled in 2,048-token chunks: ids, once, reference",
          r["out"] == F and once and r["d_lp"] <= 1e-5 and r["am"], "once %s max|dlp| %.2e" % (once, r["d_lp"]))


def t_w5():
    for ref, cached, expect_all in ((4370, 4352, True), (4300, 4352, False)):
        e = EngL()
        prompt = [5000 + (7 * i) % 900 for i in range(4352 + 120)]
        sp = SamplingParams(temperature=1.0, top_p=0.95, seed=5, max_tokens=1,
                            extra_args={"local_fw_ref": ref, "local_fw_tag": "w5-%d" % ref})
        e.add("ref", prompt, sp, 1, cached=cached)
        idx = e.reqs["ref"]["idx"]
        while not e.reqs["ref"]["prefilled"]:
            e.prefill(["ref"], chunk=64, ref_hidden=True)
        n = len(prompt) - ref + 1
        nm = fw.S.nmeas[idx, :n]
        if expect_all:
            raw = torch.stack([toy_logits(prompt[: ref + k]) for k in range(n - 1)])
            ft = torch.tensor(prompt[ref: ref + n - 1], dtype=torch.int64)
            m = fw.row_metrics(raw, ft)
            got = fw.S.buf[idx, : n - 1, fw.C["lp_pre_0"]:fw.C["lp_pre_0"] + fw.NT]
            d = float((got - m["lp_pre"]).abs().nan_to_num(0).max())
            check("W5a prefill reference with ref >= the cached boundary: every row k in [0, %d] measured once and == "
                  "the single-token reference" % (n - 1), bool((nm == 1).all()) and d <= 1e-5, "max|dlp| %.2e" % d)
        else:
            below = cached - ref + 1          # rows k = 0 .. cached - ref predict positions <= the boundary
            check("W5b prefill reference with ref below the cached boundary: rows predicting cached positions are NOT "
                  "measured (references must be cold; LW cached_expected 0 is gated)",
                  bool((nm[:below] == 0).all()) and bool((nm[below:] == 1).all()),
                  "unmeasured %d of %d below the boundary" % (int((nm[:below] == 0).sum()), below))


def t_w6():
    cu = torch.tensor([0, 4, 8])
    pos = torch.tensor([112023, 112024, 112025, 112026, 4351, 4352, 4353, 4354])
    plen = torch.tensor([112024, 4352])
    req, j, o, fwin = fw.row_plan(cu, pos, plen, torch.tensor([True, True]), torch.tensor([8194, 3]))
    check("W6 row_plan on absolute positions after a cached prefix: o = pos + 1 - prompt_len, window flags",
          o.tolist() == [0, 1, 2, 3, 0, 1, 2, 3] and fwin.tolist() == [True] * 4 + [True, True, True, False],
          "o %s fwin %s" % (o.tolist(), fwin.tolist()))


def t_w7():
    e = EngL()
    P = 112024
    F = list(range(2000, 2000 + 8194))
    des = sorted(set(range(512, 8194, 64)) | {8193})
    sp = SamplingParams(temperature=1.0, top_p=0.95, seed=1, max_tokens=len(F),
                        extra_args={"forcewin_hooks": F, "local_fw_tag": "w7", "local_fw_desig": des,
                                    "local_fw_selrec": [8128, 8193]})
    # direct registration (a 112k prompt does not fit the toy RequestState): the hook's add_request only needs the
    # prompt length and the sampling params
    fw.add_request(e.runner, "w7", 0, P, types.SimpleNamespace(sampling_params=sp))
    idx = 0
    fw.S.layer_slot.clear()
    os_ = [1000, 8128, 8193]
    rows = len(os_)
    S = fw.S
    S.tokreq.fill_(-1)
    S.tokreq[:rows] = idx
    S.xstep.fill_(77)
    S.xarmed.fill_(1)
    pids = torch.stack([torch.arange(511) + (P // 4 - 200) for _ in range(rows)])
    dec_seq = torch.tensor([P + o for o in os_])
    fw.on_selection("layers.3.self_attn.indexer", pids, dec_seq, rows, 1)
    ds = [int(S.dslot[idx, o]) for o in os_]
    got = [int(S.xbuf[idx, k, 0, 0]) if k >= 0 else None for k in ds]
    ss = int(S.sslot[idx, 8193])
    sel_ok = ss >= 0 and int(S.selbuf[idx, ss, 0, 0]) == 77 and int(S.selbuf[idx, ss, 0, 1]) == P // 4 - 200
    check("W7 on_selection at o = 1,000 / 8,128 / 8,193 for a 112,024-token prompt: exposure and selection records at "
          "their (request, o, layer); an undesignated o writes nothing", ds[0] == -1 and got[1] == 77 and got[2] == 77
          and sel_ok and int((S.xbuf[idx, :, 0, 0] == 77).sum()) == 2,
          "slots %s steps %s selection %s" % (ds, got, sel_ok))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("FW_THREADS", max(1, os.cpu_count() // 2))))
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("all", "fast"):
        t_w1()
        t_w3()
        t_w6()
        t_w7()
        t_w4()
        t_w5()
    if part in ("all", "long"):
        prompt, F = t_w2()
        t_w2n(prompt, F)
    ok = summary("test_fc_hook[%s]" % part)
    sys.exit(0 if ok else 1)
