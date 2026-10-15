#!/usr/bin/env python3
"""F7 [R-1] negative controls for the run-fc1 reproduction: the step comparison must DETECT a wrong configuration.
Each variant changes one input of the replay and must produce at least one mismatching step (and report where)."""
import json
import os
import sys

sys.path.insert(0, "/replay")
import engine_replay as E  # noqa: E402
import trace_replay as T  # noqa: E402
import reproduce_fc1 as R  # noqa: E402

V = os.environ.get("VARIANT", "")


def main():
    recs = [r for r in T.jl(R.RUN + "/mapguard/ms_p274.jsonl") if r["kind"] in ("sched", "upd")]
    S, U, info = T.prepass(recs)
    reqs, creq, salts = R.build_requests(recs, info)
    kw = {}
    mto = R.unrecorded_max_tokens(info, reqs)
    if V == "blocks1274":
        kw["num_blocks"] = 1274
    elif V == "eagle_mla_only":
        kw["mla_prefix"] = "model.layers."      # only group 0 carries an unwrapped name -> eagle groups {0}
    elif V == "salt_dropped":
        for d in reqs.values():
            d["salt"] = None
    elif V == "probe_max_tokens_95":
        mto = {q: (95 if v == 96 else v) for q, v in mto.items()}
    elif V == "boundary_lookup_1":
        pass   # set through REPLAY_ENV=-e GLM53_DFLASH_BOUNDARY_LOOKUP=1
    tr = T.TraceReplay(recs, reqs, max_tokens_override=mto, **kw)
    ok = tr.run(stop_on_mismatch=True)
    hits = {d["label"]: tr.eng.lookups.get(q, {}).get("hit") for q, d in reqs.items() if d["label"].startswith("C1")}
    out = dict(variant=V or "none", env_boundary_lookup=os.environ.get("GLM53_DFLASH_BOUNDARY_LOOKUP"),
               exact=ok, steps_before_first_mismatch=tr.n_steps - (0 if ok else 1),
               first_mismatch=tr.mismatch[:1], c1_hits_if_reached=hits)
    print(json.dumps(out, default=str)[:900])
    with open("/out/neg_controls.jsonl", "a") as f:
        f.write(json.dumps(out, default=str) + "\n")


if __name__ == "__main__":
    main()
