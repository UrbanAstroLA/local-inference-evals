#!/usr/bin/env python3
"""F7 evidence (CPU): does the hook change anything the engine CONSUMES for an unforced co-runner?
 (a) same step, hook on vs off: processed logits (the served top_p output the rejection sampler draws from),
     raw logits after the call, committed tokens / num_sampled / num_rejected, and the scratch tail;
 (b) two engines over 6 steps, A with the hook forcing request 0, B with the hook off for everyone:
     the co-runner's committed sequence and every RequestState value for its row (num_computed, total_len,
     last_sampled, all_token_ids, draft_tokens) after each step."""
import os, sys
sys.argv = [sys.argv[0], "none"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_forcing as T
import torch
fw = T.fw


def setup():
    e = T.Eng()
    prompt = [6000 + i for i in range(31)]
    F = T.make_history(prompt, 40, 5)
    sp = T.SamplingParams(temperature=1.0, top_p=0.95, seed=1, max_tokens=len(F),
                          extra_args={"forcewin_hooks": F, "local_fw_draft": "natural"})
    e.add("forced", prompt, sp, len(F))
    e.add("free", [4100 + i for i in range(19)], T.SamplingParams(temperature=1.0, top_p=0.95, seed=2, max_tokens=1000), 1000)
    return e, ["forced", "free"]


# (a) same step, processed logits captured inside the real RejectionSampler call
e, rids = setup()
e.prefill(rids)
CAP = []
orig_aps = e.sampler.apply_sampling_params
def cap(*a, **k):
    out = orig_aps(*a, **k)
    CAP.append(out.clone())
    return out
e.sampler.apply_sampling_params = cap
for step in range(3):
    ib, logits = e.decode(rids, hook=True)
    sc = fw.S.sc
    L0 = logits.clone()
    on = e.rej(logits, ib, None)
    raw_unchanged_on = torch.equal(logits, L0)
    p_on = CAP[-1]
    fw.S.sc = None
    off = e.rej(L0.clone(), ib, None)
    p_off = CAP[-1]
    rows = slice(4, 8)        # co-runner's logits rows
    n = int(off.num_sampled[1])
    print("a step", step, "| processed logits co-runner rows bitwise equal on/off:",
          torch.equal(torch.nan_to_num(p_on[rows], neginf=-1e30), torch.nan_to_num(p_off[rows], neginf=-1e30)),
          "| forced rows too:", torch.equal(torch.nan_to_num(p_on[:4], neginf=-1e30), torch.nan_to_num(p_off[:4], neginf=-1e30)),
          "| raw logits unchanged by the hooked call:", raw_unchanged_on,
          "| committed on/off:", on.sampled_token_ids[1, :n].tolist(), off.sampled_token_ids[1, :n].tolist(),
          "num_sampled", int(on.num_sampled[1]), int(off.num_sampled[1]), "num_rejected", int(on.num_rejected[1]), int(off.num_rejected[1]),
          "| scratch past num_sampled on/off:", on.sampled_token_ids[1, n:].tolist(), off.sampled_token_ids[1, n:].tolist())
    fw.S.sc = sc
    e.finish_decode(rids, ib, L0)

# (b) two engines: hook forcing request 0 vs hook off for every request
def run(hook):
    e, rids = setup()
    if not hook:
        fw.S.isf.zero_()                 # nobody forced: the hook's tables say "unkeyed"
        fw.S.reqs.clear(); fw.S.by_id.clear()
    e.prefill(rids)
    idx = e.reqs["free"]["idx"]
    rs = e.rs
    trace = []
    for step in range(6):
        ib, logits = e.decode(rids, hook=hook)
        e.finish_decode(rids, ib, logits)
        P = len(e.reqs["free"]["prompt"])
        trace.append(dict(out=list(e.reqs["free"]["out"]), nc=int(rs.num_computed_tokens.gpu[idx]),
                          tl=int(rs.total_len.gpu[idx]), last=int(rs.last_sampled_tokens[idx, 0]),
                          ids=rs.all_token_ids.gpu[idx, :int(rs.total_len.gpu[idx])].tolist(),
                          drafts=rs.draft_tokens[idx].tolist()))
    return trace, e.reqs["forced"]["out"]
A, fa = run(True)
B, fb = run(False)
for s, (x, y) in enumerate(zip(A, B)):
    print("b step", s, "co-runner state equal (out, num_computed, total_len, last, all_token_ids, drafts):",
          {k: x[k] == y[k] for k in x}, "| co-runner committed so far:", len(x["out"]))
print("b forced request differs between A and B as expected (forced vs sampled):", fa != fb)
