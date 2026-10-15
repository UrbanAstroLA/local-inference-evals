#!/bin/bash
# F8 [R-8]: run-fc1's as-run gate kept as a regression fixture. CPU container of the B1 image (--network none, no GPU).
#  1. the FROZEN attempt-1 analysis/gate_fc.py, unchanged, re-run on run-fc1 (read-only, through a symlink tree in the
#     container's /tmp) must reproduce diagnosis-fc1/gate-asrun/gate1.json row for row (verdict, ids, statuses,
#     details): the original verdict stays unchanged;
#  2. the newly specified checks that C1 can be evaluated on are computed and labelled POST-INSPECTION (C1 is a
#     long-context instrument-characterization run: not pooled, not paired, not evidence about D2): F2 cache rows,
#     F6 null-relative index-K rows (N_dd absent), and the attempt-1 hook-level rows it passed.
# usage: test_f8_regression.sh  -> retry/calibration/f8_regression.json, .out; rc 0 iff part 1 reproduces exactly
RETRY=$(cd "$(dirname "$0")/.." && pwd)
FC=$(cd "$RETRY/.." && pwd)
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a
OUTD=$RETRY/calibration
docker run --rm --network none --name "fcr-f8-$$" --user "$(id -u):$(id -g)" -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= \
  -v "$FC":/fc:ro -v "$OUTD":/out --entrypoint sh "$IMG" -c '
mkdir -p /tmp/run && for f in /fc/run-fc1/*; do ln -s "$f" /tmp/run/; done
python3 /fc/analysis/gate_fc.py /tmp/run /fc > /tmp/gate1-console.txt 2>&1; echo "gate_fc rc=$?" > /out/f8_gate_rc.txt
cp /tmp/run/gate1.json /out/f8_gate1_rerun.json
python3 - <<PY
import json, sys
sys.path.insert(0, "/fc/retry/analysis")
a = json.load(open("/out/f8_gate1_rerun.json"))
b = json.load(open("/fc/diagnosis-fc1/gate-asrun/gate1.json"))
same = a == b
rows_a = [(r["id"], r["status"]) for r in a["rows"]]
rows_b = [(r["id"], r["status"]) for r in b["rows"]]
import cache_gate as cg, fcr_geometry as geo
RUN = "/fc/run-fc1"
reqs = [r for r in cg.jl(RUN + "/client.jsonl") if r.get("kind") == "request"]
for r in reqs:
    r["cached_expected"] = 0 if r["phase"] in ("LW1", "LW2", "warm-s1") else geo.rule_repeat(r["P"])
info = cg.sched_requests(cg.load_sched(RUN))
hook = {x["req"]: x["tag"] for x in cg.jl(RUN + "/forcewin/fw_r0_p370.jsonl") if x.get("kind") == "add"}
rows = cg.evaluate(reqs, info, cg.map_client(reqs, info, hook))
f6 = json.load(open("/out/f6_calibration.json"))["C1_vs_LW1_post_inspection"]
passed = [r["id"] for r in b["rows"] if r["status"] == "PASS"]
post = dict(label="POST-INSPECTION: C1 of run-fc1 shown against the newly specified checks; instrument "
                  "characterization only; not pooled with or paired against any retry run; no D2 evidence",
            F2=dict(safe=all(x["safe"] for x in rows), classes={x["tag"]: x["cls"] for x in rows if x["phase"] == "C1"},
                    note="M3 is a safe miss; under the retry gate it would stop the session before injected traffic "
                         "(R-cache-model), as the replay explains (warm order vs admission order)"),
            F6=dict(ok=f6["ok"], pooled=f6["pooled"]["gross"], limit=f6["pooled"]["limit"],
                    pages_failing=f6["pages"]["n_failing"], early_gross=f6["early"]["gross_n"],
                    original=f6["original"], dd_absent=True),
            attempt1_rows_passed_on_C1=passed)
out = dict(regression=dict(identical=same, verdict_rerun=a["verdict"], verdict_asrun=b["verdict"],
                           rows_equal=rows_a == rows_b, n_rows=len(rows_a)), post_inspection=post)
json.dump(out, open("/out/f8_regression.json", "w"), indent=1)
print("F8 part 1: gate_fc re-run identical to gate-asrun: %s (verdict %s, %d rows)" % (same, a["verdict"], len(rows_a)))
print("F8 part 2 (POST-INSPECTION, C1 only): F2 safe %s classes %s; F6 ok %s (original criteria pass %s); attempt-1 rows "
      "passed: %s" % (post["F2"]["safe"], post["F2"]["classes"], post["F6"]["ok"], post["F6"]["original"]["orig_pass"], passed))
sys.exit(0 if same else 1)
PY
' 2>&1 | grep -v cpp_extension | tee "$OUTD/f8_regression.out"
exit ${PIPESTATUS[0]}
