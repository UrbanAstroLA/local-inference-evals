#!/usr/bin/env python3
"""P4 acceptance schedule `sched` (PROPOSAL.md section 4: "the accepted counts of a recorded natural run").  CPU only.

Runs INSIDE a CPU container of the stock image (torch is needed to read the forcewin records), via run_select.sh.
Source (read only, mounted at /hv): hook-validation/run-a2/forcewin/req_V2a-natural-*_r0.pt -- the forcewin natural
mode on the B1 image (DFlash2 x3, 8 in flight; the only recorded natural forced runs with per-step accepted counts),
exact-id histories.  Rule (fixed before reading): for every V2a-natural rank-0 record, in sorted tag order, take the
verify rows (kind 0) that were measured, group them by step and keep one accepted count per step, in step order;
concatenate the requests' sequences.  sched = the first 1,024 entries.  Every P4 forced request uses the same list,
indexed by its own verify-step counter, so all 8 requests of a batch advance in lock step and both arms see the
same shapes step for step.

Writes /priv/sched.json (0600; the list) and /out/sched-summary.json (public: length, histogram, mean, sha256).
"""
import glob
import hashlib
import json
import os
import sys

import torch

os.umask(0o077)
N = 1024


def main():
    files = sorted(glob.glob("/hv/req_V2a-natural-*_r0.pt"))
    if not files:
        raise SystemExit("no V2a-natural records under /hv")
    seq, per = [], []
    for f in files:
        rec = torch.load(f, map_location="cpu", weights_only=False)
        cols = rec["cols"]
        m = rec["metrics"].numpy()
        ci = {c: i for i, c in enumerate(cols)}
        steps = {}
        for row in m:
            if row[ci["kind"]] != 0.0 or row[ci["step"]] != row[ci["step"]]:   # verify rows only; NaN = unmeasured
                continue
            st = int(row[ci["step"]])
            a = int(row[ci["accepted"]])
            if st in steps and steps[st] != a:
                raise SystemExit(f"{os.path.basename(f)}: step {st} has two accepted counts")
            steps[st] = a
        s = [steps[k] for k in sorted(steps)]
        per.append(dict(file=os.path.basename(f), steps=len(s)))
        seq += s
    sched = seq[:N]
    if len(sched) < N:
        raise SystemExit(f"only {len(sched)} steps available")
    blob = json.dumps(sched).encode()
    path = "/priv/sched.json"
    if os.path.exists(path):
        raise SystemExit(f"{path} exists (never overwritten)")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(json.dumps(dict(note="P4 acceptance schedule (accepted drafts per verify step)", source=per,
                                 sched=sched)).encode())
    hist = {str(k): sched.count(k) for k in range(0, 4)}
    out = dict(what="P4 sched summary (public)", n=len(sched), hist=hist, mean_accepted=round(sum(sched) / len(sched), 4),
               mean_tokens_per_step=round(1 + sum(sched) / len(sched), 4), sha256_list=hashlib.sha256(blob).hexdigest(),
               sources=per, rule="windows/extract_sched.py docstring")
    json.dump(out, open("/out/sched-summary.json", "w"), indent=1)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
