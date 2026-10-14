#!/usr/bin/env python3
"""PA2 change log with before/after hashes (CPU, host python). Attempt-1 frozen files are never overwritten in place:
each modified file's attempt-1 version is kept as <file>.a1 and must hash to its SHA256SUMS.a1 entry (checked here).
Writes prep-a2/PA2-changes.json. Run after every PA2 edit and before the final make_sums.sh.

PA2-25 (amendment 21g): the PA2-1..15 rows now give `after` = the hash frozen in SHA256SUMS.pre21g (the package the
21g review saw, sha256 2ab92f14...) and `current` = today's hash, with `changed_by` naming the later PA2 item when they
differ. PA2-16 onward give `before` = the SHA256SUMS.pre21g hash (or "(new file)") and `after` = today's hash; every
file they modify keeps its pre-21g version as <file>.pre21g, which must hash to its SHA256SUMS.pre21g entry (checked).
PA2-26 (amendment 21h) is logged the same way against SHA256SUMS.pre21h (the 21g package, sha256 160015e5...), with
<file>.pre21h copies. Attempt-1 files and outputs are untouched: every SHA256SUMS.a1 entry that is not a PA2-modified file (whose .a1 copy is
checked instead) must still hash as frozen (checked)."""
import hashlib
import json
import os

HV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A1 = {ln[66:].strip()[2:]: ln[:64] for ln in open(os.path.join(HV, "SHA256SUMS.a1")) if ln.strip()}
PRE = {ln[66:].strip()[2:]: ln[:64] for ln in open(os.path.join(HV, "SHA256SUMS.pre21g")) if ln.strip()}
PRE21H = {ln[66:].strip()[2:]: ln[:64] for ln in open(os.path.join(HV, "SHA256SUMS.pre21h")) if ln.strip()}
H = lambda p: hashlib.sha256(open(os.path.join(HV, p), "rb").read()).hexdigest()  # noqa: E731
ITEMS = [
    ("PA2-1", "hook: on_selection maps selection rows to requests through a per-token table (flattened layout)",
     ["hook/forcewin_hooks.py"]),
    ("PA2-2", "hook: shadow ledger stores the step's real request/token counts (padded FULL-graph steps)",
     ["hook/forcewin_hooks.py"]),
    ("PA2-3", "tests: X2f/X2fn flattened layout (+ X0 no hook errors); stand-in carries query_start_loc_np",
     ["tests/test_exposure.py", "tests/errscan_forcing.sh"]),
    ("PA2-4", "tests: D7p/D7pn padded steps through the hook ledger (+ D0); stand-in carries query_start_loc_np",
     ["tests/test_d2_inject.py"]),
    ("PA2-5", "tests: CPU test runner writes tests/out-a2 and pins the attempt-2 image", ["tests/run_cpu_tests.sh"]),
    ("PA2-6", "build: attempt-2 image under a new tag (build-a2/)", ["build-a2/make_build.sh", "build-a2/Dockerfile",
                                                                     "build-a2/image-id.txt",
                                                                     "build-a2/manifest-diff.txt",
                                                                     "build-a2/manifest-diff-vs-a1.txt"]),
    ("PA2-7", "analysis: original criteria verbatim (a1 code) + labelled fixes PA2-7a..f + DRAFT criteria",
     ["analysis/analyze_hookval.py"]),
    ("PA2-8", "tests: synthetic attempt-2 runs (real poll format; attempt-1 non-defects built in; 10 planted defects)",
     ["tests/make_synthetic_run.py"]),
    ("PA2-9", "runner: scoped attempt-2 client (V0, V2a nat+oracle, V2c, V4 + V4n nulls, V3, optional poison)",
     ["runner/client_hookval.py"]),
    ("PA2-10", "runner: attempt 2 of 2, cumulative budget gate, new image pin, MIN_WALL 1500",
     ["runner/run_hookval.sh", "runner/images.txt"]),
    ("PA2-11", "analysis: attempt-1 carry-over (H2 null, H3) with source hashes",
     ["prep-a2/make_carryover.py", "analysis/carryover-a1.json"]),
    ("PA2-12", "runner: attempt-2 cost estimate from attempt-1 phase times; client plan dry run",
     ["runner/cost_a2.py", "runner/cost_a2.json", "runner/plan-dryrun-a2/plan.json"]),
    ("PA2-13", "tests: runner DRY_RUN and refusal tests with shims", ["tests/test_runner_a2.sh",
                                                                       "tests/out-a2/runner-tests.log"]),
    ("PA2-14", "sums: make_sums.sh exclusions; SHA256SUMS regenerated (hash recorded outside the package)",
     ["make_sums.sh"]),
    ("PA2-15", "criteria basis (attempt-1 data) and attempt-1 replay of the attempt-2 analysis",
     ["prep-a2/criteria_basis.py", "prep-a2/out/criteria_basis.json", "prep-a2/replay-a1/verdict.json",
      "prep-a2/replay-a1/verdict-a2.json"]),
]
ITEMS21G = [
    ("PA2-16", "analysis: amendment 21g criteria (index-K content vs nulls matched per unit, frozen aggregation, "
               "per fixture x layer tables, fixture and arm-asymmetry flags; IDX-attrib; KDA H4-slot / H4b-r / "
               "H4b-sep / H4b-disagree; carry-over rows CARRY-scope / CARRY-admissible; coverage without silent "
               "drops; H1f-unmasked outside-nucleus mass); H5d-r judged on all units",
     ["analysis/analyze_hookval.py", "analysis/analyze_hookval.py.pre21g"]),
    ("PA2-17", "image scope for the carry-over: every installed vllm file, attempt-1 vs attempt-2 image",
     ["prep-a2/scope_a1_a2.sh", "prep-a2/out-21g/image-scope.json", "prep-a2/out-21g/image-scope-diff.txt"]),
    ("PA2-18", "carry-over record: evidence class, corrected-join statement, image scope",
     ["prep-a2/make_carryover.py", "prep-a2/make_carryover.py.pre21g", "analysis/carryover-a1.json",
      "analysis/carryover-a1.json.pre21g"]),
    ("PA2-19", "synthetic runs: 21g fields (image id, matrix/d2set, dumpranges, accepted counts, snapshot step) and "
               "five new plants; wrong-page plant moved to the solo request",
     ["tests/make_synthetic_run.py", "tests/make_synthetic_run.py.pre21g"]),
    ("PA2-20", "client: records only (dump_src_sha, coverage record, V3 coverage_gap) and the optional poison "
               "phase's time requirement; plan regenerated (payloads identical); V4n matched-null plan check",
     ["runner/client_hookval.py", "runner/client_hookval.py.pre21g", "runner/plan-dryrun-a2/plan.json",
      "runner/plan-dryrun-a2/client.jsonl", "tests/check_v4n_matched.py"]),
    ("PA2-21", "offline fault injection on attempt-1 captures under the revised analysis",
     ["prep-a2/faultinj_a1.py", "prep-a2/run_faultinj_a1.sh", "prep-a2/faultinj-a1/summary.json",
      "prep-a2/faultinj-a1/build.json"]),
    ("PA2-22", "21g CPU suites driver and logs; attempt-1 replay under the revised analysis",
     ["tests/run_21g.sh", "tests/out-21g/run_21g.log", "prep-a2/replay-a1-21g/verdict.json",
      "prep-a2/replay-a1-21g/verdict-a2.json"]),
    ("PA2-23", "cost recomputed (unchanged: no GPU request added)", ["runner/cost_a2.json"]),
    ("PA2-24", "documents: ATTEMPT2-21G.md; superseded-by notes in ATTEMPT2-PREP.md and ATTEMPT2-CRITERIA-DRAFT.md",
     ["ATTEMPT2-21G.md", "ATTEMPT2-PREP.md", "ATTEMPT2-CRITERIA-DRAFT.md"]),
    ("PA2-25", "change log (pre-21g hashes), sums (SHA256SUMS.pre21g kept; new exclusions), runner gate tests rerun",
     ["prep-a2/pa2_changes.py", "prep-a2/pa2_changes.py.pre21g", "make_sums.sh", "make_sums.sh.pre21g",
      "SHA256SUMS.pre21g", "tests/out-a2/runner-tests.log", "tests/out-a2/runner-tests.log.pre21g"]),
]
ITEMS21H = [
    ("PA2-26", "amendment 21h: H5e-r gates on the POOLED matched null of the eight-fixture set; the comparison with "
               "the solo fixture's own matched null is a FLAG (H5e-fixture); pre-21g pooled form kept reported; "
               "dry runs, attempt-1 replay and fault injection rerun",
     ["analysis/analyze_hookval.py", "analysis/analyze_hookval.py.pre21h", "ATTEMPT2-21G.md", "ATTEMPT2-21G.md.pre21h",
      "prep-a2/pa2_changes.py", "prep-a2/pa2_changes.py.pre21h", "SHA256SUMS.pre21h", "tests/out-21g/run_21g.log",
      "prep-a2/replay-a1-21g/verdict-a2.json", "prep-a2/faultinj-a1/summary.json",
      "runner/plan-dryrun-a2/plan.json", "runner/plan-dryrun-a2/client.jsonl"]),   # plan regenerated by the rerun
]
out, bad, bad21 = [], [], []
LATER = {f: pid for pid, _, fs in ITEMS21G for f in fs}
for pid, what, files in ITEMS:
    rows = []
    for f in files:
        before = A1.get(f)
        cur = H(f)
        after = PRE.get(f, cur)
        extra = dict(current=cur, changed_by=LATER.get(f)) if cur != after else {}
        if before is not None:
            kept = f + ".a1"
            ok = os.path.exists(os.path.join(HV, kept)) and H(kept) == before
            if not ok:
                bad.append(f)
            rows.append(dict(file=f, before=before, after=after, a1_copy=kept, a1_copy_verified=ok, **extra))
        else:
            rows.append(dict(file=f, before="(new file)", after=after, **extra))
    out.append(dict(id=pid, label="post-inspection", what=what, files=rows))
LATER21H = {f: pid for pid, _, fs in ITEMS21H for f in fs}
for pid, what, files in ITEMS21G:
    rows = []
    for f in files:
        before = PRE.get(f)
        cur = H(f)
        after = PRE21H.get(f, cur)
        extra = dict(current=cur, changed_by=LATER21H.get(f)) if cur != after else {}
        if before is not None and os.path.exists(os.path.join(HV, f + ".pre21g")):
            kept = f + ".pre21g"
            ok = H(kept) == before
            if not ok:
                bad21.append(f)
            rows.append(dict(file=f, before=before, after=after, pre21g_copy=kept, pre21g_copy_verified=ok, **extra))
        else:
            rows.append(dict(file=f, before=before or "(new file)", after=after, **extra))
    out.append(dict(id=pid, label="post-inspection, amendment 21g", what=what, files=rows))
for pid, what, files in ITEMS21H:
    rows = []
    for f in files:
        before = PRE21H.get(f)
        if before is not None and os.path.exists(os.path.join(HV, f + ".pre21h")):
            kept = f + ".pre21h"
            ok = H(kept) == before
            if not ok:
                bad21.append(f)
            rows.append(dict(file=f, before=before, after=H(f), pre21h_copy=kept, pre21h_copy_verified=ok))
        else:
            rows.append(dict(file=f, before=before or "(new file)", after=H(f)))
    out.append(dict(id=pid, label="post-inspection, amendment 21h", what=what, files=rows))
modified = {f for _, _, fs in ITEMS for f in fs} | {f for _, _, fs in ITEMS21G for f in fs} | \
    {f for _, _, fs in ITEMS21H for f in fs}
a1_untouched = [f for f in A1 if f not in modified and not f.endswith((".a1",)) and
                (not os.path.exists(os.path.join(HV, f)) or H(f) != A1[f])]
res = dict(note="PA2-1..15: before = attempt-1 frozen hash (SHA256SUMS.a1, sha256 67d8e54f...), after = the hash frozen "
                "in SHA256SUMS.pre21g (current/changed_by when a later item changed the file); PA2-16..25: before = "
                "SHA256SUMS.pre21g hash, after = the hash frozen in SHA256SUMS.pre21h (current/changed_by when PA2-26 "
                "changed it); PA2-26: before = SHA256SUMS.pre21h hash (21g package, sha256 160015e5...), after = "
                "current file",
           items=out, a1_copies_all_verified=not bad, a1_copy_failures=bad,
           pre21g_copies_all_verified=not bad21, pre21g_copy_failures=bad21,
           a1_files_untouched=not a1_untouched, a1_files_changed=a1_untouched,
           sha256sums_a1=hashlib.sha256(open(os.path.join(HV, "SHA256SUMS.a1"), "rb").read()).hexdigest(),
           sha256sums_pre21g=hashlib.sha256(open(os.path.join(HV, "SHA256SUMS.pre21g"), "rb").read()).hexdigest(),
           sha256sums_pre21h=hashlib.sha256(open(os.path.join(HV, "SHA256SUMS.pre21h"), "rb").read()).hexdigest())
json.dump(res, open(os.path.join(HV, "prep-a2", "PA2-changes.json"), "w"), indent=1)
for it in out:
    for r in it["files"]:
        print("%-6s %-42s %s -> %s%s" % (it["id"], r["file"], r["before"][:12], r["after"][:12],
                                         f" (now {r['current'][:12]}, {r['changed_by']})" if r.get("current") else ""))
print("a1 copies verified:", not bad, bad, "| pre21g copies verified:", not bad21, bad21,
      "| other attempt-1 files untouched:", not a1_untouched, a1_untouched[:5])
