#!/usr/bin/env python3
"""Carry-over record for the attempt-2 analysis (PA2-11; CPU, host python, json only). POST-INSPECTION, labelled.

Attempt 2 does not re-run V1 (free runs: the in-session unhooked decode-vs-prefill null for H2) or V2b / the
reject and sched schedules (H3). Their attempt-1 results carry over, with provenance (DIAGNOSIS-a1 s.8.4: the forcing
code is unchanged; the image differs only in on_selection/_shadow_step; V0 re-checks unkeyed equality):
  * H2 null quantiles: B1's own free-run logprobs vs the prefill reference (run-a1, V1), as computed by
    diagnosis-a1/d_h2_corrected.py (identical to the frozen null_gap() up to dropping non-finite entries);
  * H2 under PI-1/PI-2 on attempt 1 (aligned quantiles, top-1, argmax shift control);
  * H3 as recorded by the frozen analysis in run-a1/verdict.json (PASS).
Writes analysis/carryover-a1.json with the sha256 of every source file.

PA2-18 (amendment 21g item 3): every carried value is labelled "carried-over evidence" with its provenance, and the
H2 values are those of the CORRECTED join (PI-1: reference row k = o + P - ref_from; top-1 on finite reference rows),
not the frozen attempt-1 join. Adds `scope`: the image-scope check (prep-a2/scope_a1_a2.sh ->
prep-a2/out-21g/image-scope.json: every file of the installed vllm package compared between the attempt-1 and
attempt-2 images; only vllm/forcewin_hooks.py differs, the five hooked engine files' .pyc differ in the 16-byte
mtime header only). The analysis admits the carried rows only if this scope holds and the current-image V0, forcing,
attribution, ledger and state checks pass (CARRY-admissible)."""
import hashlib
import json
import os

HV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = {
    "h2_corrected": os.path.join(HV, "diagnosis-a1", "out", "h2_corrected.json"),
    "verdict_a1": os.path.join(HV, "run-a1", "verdict.json"),
    "client_a1": os.path.join(HV, "run-a1", "client.jsonl"),
    "image_scope": os.path.join(HV, "prep-a2", "out-21g", "image-scope.json"),
}
sha = {k: hashlib.sha256(open(p, "rb").read()).hexdigest() for k, p in SRC.items()}
h2 = json.load(open(SRC["h2_corrected"]))
v = json.load(open(SRC["verdict_a1"]))
h3 = [c for c in v["checks"] if c["name"].startswith("H3 ")][0]
scope = json.load(open(SRC["image_scope"]))
out = dict(
    label="POST-INSPECTION carry-over from hook-validation attempt 1 (run-a1); not re-measured in attempt 2",
    evidence_class="carried-over evidence (amendment 21g item 3): attempt-1 measurements on image "
                   "sha256:1b9f8b15..., admitted for attempt 2 only under CARRY-admissible",
    sources={k: dict(path=os.path.relpath(p, HV), sha256=sha[k]) for k, p in SRC.items()},
    h2_null=dict(n=h2["null_n"], quantiles=h2["null"],
                 note="V1: B1's own free-run logprobs vs prefill reference, run-a1 (in-session unhooked null of "
                      "attempt 1)"),
    h2_attempt1_PI=dict(n=h2["n"], aligned=h2["aligned"], top1=h2["top1"], shift=h2["shift"],
                        per_request_top1_min=h2["per_request_top1_min"],
                        per_request_shifted_top1_max=h2["per_request_shifted_top1_max"],
                        H2a_pass=h2["H2a_corrected_pass"], H2c_argmax_failing=h2["H2c_argmax_rule_failing"]),
    h3_attempt1=dict(ok=h3["ok"], detail=h3["detail"], h3=v.get("h3"),
                     join="none needed: V2a schedules compared with each other at the same output index"),
    h2_join="corrected (PI-1): reference row k = o + P - ref_from, top-1 on finite reference rows "
            "(diagnosis-a1/d_h2_corrected.py); the frozen attempt-1 join is NOT used for any carried value",
    scope=dict(image_a1=scope["image_a1"], image_a2=scope["image_a2"],
               only_hook_module_changed=scope["only_hook_module_changed"],
               differing_sources=scope["differing_sources"],
               pyc_header_only=sorted(f for f, x in scope["pyc"].items() if x["body_equal"]),
               files_compared=scope["files_a1"], listing_sha256=scope["listing_sha256"]),
)
json.dump(out, open(os.path.join(HV, "analysis", "carryover-a1.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:2500])
