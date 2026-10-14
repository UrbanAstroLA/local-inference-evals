#!/usr/bin/env python3
"""F7 diagnosis (CPU): which entries of the unforced co-runner's sampler output differ between hook-on and
hook-off calls on the same step, and whether hook-off vs hook-off differs too (uninitialised scratch)."""
import os, sys
sys.argv = [sys.argv[0], "none"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_forcing as T
import torch
fw = T.fw
for trial in range(3):
    e = T.Eng()
    prompt = [6000 + i for i in range(31)]
    F = T.make_history(prompt, 40, 5)
    sp = T.SamplingParams(temperature=1.0, top_p=0.95, seed=1, max_tokens=len(F),
                          extra_args={"forcewin_hooks": F, "local_fw_draft": "natural"})
    e.add("forced", prompt, sp, len(F))
    e.add("free", [4100 + i for i in range(19)], T.SamplingParams(temperature=1.0, top_p=0.95, seed=2, max_tokens=1000), 1000)
    rids = ["forced", "free"]
    e.prefill(rids)
    ib, logits = e.decode(rids, hook=True)
    sc = fw.S.sc
    on = e.rej(logits.clone(), ib, None)
    fw.S.sc = None
    off1 = e.rej(logits.clone(), ib, None)
    off2 = e.rej(logits.clone(), ib, None)
    fw.S.sc = sc
    on2 = e.rej(logits.clone(), ib, None)
    for name, x in (("on", on), ("off1", off1), ("off2", off2), ("on2", on2)):
        print(trial, name, "free row", x.sampled_token_ids[1].tolist(), "num_sampled", int(x.num_sampled[1]),
              "num_rejected", int(x.num_rejected[1]), "| forced row", x.sampled_token_ids[0].tolist(),
              int(x.num_sampled[0]), "ptr", x.sampled_token_ids.data_ptr())
    n = int(off1.num_sampled[1])
    print(trial, "committed prefix equal on/off:", on.sampled_token_ids[1, :n].tolist() == off1.sampled_token_ids[1, :n].tolist(),
          "| num equal:", int(on.num_sampled[1]) == n, int(on.num_rejected[1]) == int(off1.num_rejected[1]),
          "| full rows equal on/off:", torch.equal(on.sampled_token_ids[1], off1.sampled_token_ids[1]),
          "off1/off2:", torch.equal(off1.sampled_token_ids[1], off2.sampled_token_ids[1]))
