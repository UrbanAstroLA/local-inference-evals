#!/bin/bash
# PA2-22: the CPU suites affected by the amendment-21g changes, with their expected outcomes asserted.
# CPU only: CPU containers of the B1 image (--network none, no --gpus, CUDA_VISIBLE_DEVICES=, run as the invoking
# user), host python for json-only steps. Nothing is launched on a GPU; the runner is exercised only through
# tests/test_runner_a2.sh (DRY_RUN with shims), which is run separately after SHA256SUMS is regenerated.
# The hook and the image are unchanged by 21g (hook/forcewin_hooks.py and build-a2/image-id.txt hash as in
# SHA256SUMS.pre21g), so the hook suites (forcing, D2, exposure, inertness, image) are not affected and not rerun.
# Writes tests/out-21g/*. No `set -e`: every step's result is checked; rc 0 iff all pass.
HV=$(cd "$(dirname "$0")/.." && pwd)
OUT=$HV/tests/out-21g
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a   # B1 local/glm53-v0.9.1-b1-d7:local
mkdir -p "$OUT/analysis-dryrun"
DR=(docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e CUDA_VISIBLE_DEVICES=
    -e PYTHONDONTWRITEBYTECODE=1)
FAILS=0
res() { if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2"; FAILS=$((FAILS + 1)); fi; }
{
echo "== 21g CPU suites $(date -u +%FT%TZ)"
echo "hook  $(sha256sum "$HV/hook/forcewin_hooks.py" | cut -c1-64) (pre-21g: $(grep ' ./hook/forcewin_hooks.py$' "$HV/SHA256SUMS.pre21g" | cut -c1-64))"
echo "image $(cat "$HV/build-a2/image-id.txt")"
cmp -s <(grep ' ./hook/forcewin_hooks.py$' "$HV/SHA256SUMS.pre21g" | cut -c1-64) <(sha256sum "$HV/hook/forcewin_hooks.py" | cut -c1-64)
res $? "S0 hook module unchanged by 21g"

# ---- S1 synthetic analysis dry runs
for M in good bad; do
  "${DR[@]}" -v "$HV/hook":/hv/hook:ro -v "$HV/tests":/hv/tests:ro -v "$HV/analysis":/hv/analysis:ro \
    -v "$OUT/analysis-dryrun":/dry --entrypoint sh "$IMG" -c \
    "rm -rf /dry/$M && python3 /hv/tests/make_synthetic_run.py /dry/$M $M && python3 /hv/analysis/analyze_hookval.py /dry/$M" \
    2>&1 | grep -v "Warning\|cpp_extension" > "$OUT/analysis-dryrun-$M.log"
done
python3 - "$OUT/analysis-dryrun" <<'EOF'
import json, sys
d = sys.argv[1]
g = json.load(open(f"{d}/good/verdict-a2.json"))
b = json.load(open(f"{d}/bad/verdict-a2.json"))
exp = {"H1c", "H1f-r", "H2c-r", "H3-r", "H4-slot", "H4b-r", "H4b-sep", "IDX-attrib", "IDX-page", "IDX-early",
       "IDX-content", "H5e-r", "H5f-B1", "H5g-r", "H5h-r", "CARRY-admissible"}
bf = {r["id"] for r in b["rows"] if r["status"] == "FAIL"}
gf = [r["id"] for r in g["rows"] if r["status"] == "FAIL"]
fl = json.loads(json.dumps(b.get("fixture_flags", {})))
flagged = {k: sorted(x["fixture"] for x in v) for k, v in fl.items()}
print(f"good: revised {g['verdict']} (failing {gf}), original {g['original_verdict']}")
print(f"bad:  revised {b['verdict']}, failing {sorted(bf)}; missing {sorted(exp - bf)}; unexpected {sorted(bf - exp)}")
print(f"bad:  fixture flags {flagged}")
ok = g["verdict"] == "VALID" and not gf and b["verdict"] == "INVALID" and bf == exp and \
    flagged.get("b1") == ["u4", "u5"] and flagged.get("arm_asymmetry") == ["u3"]
sys.exit(0 if ok else 1)
EOF
res $? "S1 synthetic dry runs: good VALID; bad fails exactly the 16 planted checks; flags u4, u5 (own null) and u3 (arm asymmetry)"

# ---- S2 V4n matched nulls in the client plan (and a tampered plan must fail)
rm -f "$HV/runner/plan-dryrun-a2/client.jsonl" "$HV/runner/plan-dryrun-a2/plan.json"
python3 "$HV/runner/client_hookval.py" "$HV/runner/plan-dryrun-a2" 0 --plan > "$OUT/plan.log" 2>&1
res $? "S2a client plan dry run ($(tail -n 1 "$OUT/plan.log"))"
python3 "$HV/tests/check_v4n_matched.py" > "$OUT/v4n-matched.log" 2>&1
res $? "S2b V4n produces matched nulls for every tested fixture ($(tail -n 1 "$OUT/v4n-matched.log" | cut -c1-60))"
TP=$(mktemp) && python3 - "$HV/runner/plan-dryrun-a2/plan.json" "$TP" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
for r in p:
    if str(r.get("tag", "")).startswith("V4n-rebuild2-"):
        r["dumprange"] = [r["dumprange"][0] + 4, r["dumprange"][1] + 4]   # a misplaced second rebuild
        break
json.dump(p, open(sys.argv[2], "w"))
EOF
python3 "$HV/tests/check_v4n_matched.py" "$TP" > "$OUT/v4n-matched-negative.log" 2>&1
[ $? -ne 0 ]; res $? "S2c negative control: a misplaced V4n-rebuild2 window fails the plan check"
rm -f "$TP"
grep -q '"kind": "coverage"' "$HV/runner/plan-dryrun-a2/client.jsonl"
res $? "S2d client writes its explicit coverage record"

# ---- S3 attempt-1 replay: original verdict reproduced check-for-check; revised checks as the diagnosis says
rm -rf "$HV/prep-a2/replay-a1-21g"; mkdir -p "$HV/prep-a2/replay-a1-21g"
"${DR[@]}" -e HV_ANALYSIS_OUT=/o -v "$HV/run-a1":/run:ro -v "$HV/analysis":/a:ro -v "$HV/prep-a2/replay-a1-21g":/o \
  --entrypoint python3 "$IMG" /a/analyze_hookval.py /run 2>&1 | grep -v cpp_extension > "$HV/prep-a2/replay-a1-21g/console.txt"
python3 - "$HV" <<'EOF'
import json, sys
hv = sys.argv[1]
o = json.load(open(f"{hv}/prep-a2/replay-a1-21g/verdict.json"))
a1 = json.load(open(f"{hv}/run-a1/verdict.json"))
same = [(c["name"], c["ok"], c["detail"]) for c in o["checks"]] == [(c["name"], c["ok"], c["detail"]) for c in a1["checks"]]
v = json.load(open(f"{hv}/prep-a2/replay-a1-21g/verdict-a2.json"))
st = {r["id"]: r["status"] for r in v["rows"]}
want_pass = ["V0", "H1f-r", "H2a-r", "H2b-r", "H2c-r", "H3-r", "H4a", "H4-slot", "H4b-r", "H4b-sep", "IDX-stable", "H5g-r"]
want_fail = ["H5h-r", "H5f-B1",                                    # the two hook defects (DIAGNOSIS-a1)
             "coverage-a2", "IDX-attrib", "IDX-page", "IDX-early",  # no V4n phases in attempt 1
             "IDX-content", "H5e-r", "H5f-D2",                      # no matched nulls in attempt 1
             "CARRY-scope", "CARRY-admissible"]                     # attempt-1 image; hook defects
bad = [(k, st.get(k)) for k in want_pass if st.get(k) != "PASS"] + [(k, st.get(k)) for k in want_fail if st.get(k) != "FAIL"]
attrib = v["idx_attrib"]
print(f"original verdict {o['verdict']} reproduced check-for-check: {same} ({len(o['checks'])} checks)")
print(f"revised verdict {v['verdict']}; unexpected {bad}")
print(f"IDX-attrib on attempt 1: {attrib['n_violations']} attribution violations; population "
      f"{ {k: (x['compared'], x['expected']) for k, x in attrib['population'].items()} }")
print(f"H5d-r {st.get('H5d-r')}; H1f-unmasked outside-nucleus {v['h1f']['unmasked_outside_nucleus']}")
print(f"KDA: {v['h4_kda']['n']} comparisons, layers 0-17 {v['h4_kda']['ident_0_17']}, 18+ {v['h4_kda']['ident_18_plus']}, "
      f"disagreements {len(v['h4_kda']['disagreements'])}; H4-slot {v['h4_slot']['snapshots']} snapshots, "
      f"{len(v['h4_slot']['violations'])} violations; metadata diag {v['h4_slot']['layer_metadata_prev_step']}")
sys.exit(0 if same and not bad and attrib["n_violations"] == 0 else 1)
EOF
res $? "S3 attempt-1 replay: original verdict reproduced; revised checks as DIAGNOSIS-a1; zero attribution violations"

# ---- S4 fault injection on attempt-1 captures under the revised analysis
"$HV/prep-a2/run_faultinj_a1.sh" > "$OUT/faultinj.log" 2>&1
res $? "S4 fault injection: base passes every targeted check; every wrong-page / attribution / ledger fault fails its targets"

# ---- S5 cost
python3 "$HV/runner/cost_a2.py" > "$OUT/cost.log" 2>&1
python3 - "$HV/runner/cost_a2.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
print(f"mandatory {c['mandatory']}; + optional poison {c['mandatory_plus_optional_poison']}; wall cap {c['wall_cap_s']} s; "
      f"worst case cumulative {c['worst_case_cumulative_gpu_h']} GPU-h")
sys.exit(0 if c["wall_cap_s"] <= 2138 and c["worst_case_cumulative_gpu_h"] <= 2.4 and
         c["mandatory_plus_optional_poison"]["cumulative_gpu_h"] <= 2.4 else 1)
EOF
res $? "S5 cost: item total <= 2.4 GPU-h at the wall cap; wall cap <= 2,138 s"
echo "21g suites: $FAILS failing"
} 2>&1 | tee "$OUT/run_21g.log"
grep -q "21g suites: 0 failing" "$OUT/run_21g.log"
