#!/usr/bin/env bash
# Refusal / cleanup tests for ../run_k_selection.sh. CPU only. Every scenario except R07 puts the
# shims first on PATH, so no real container is started and no GPU is queried; R07 uses the real
# docker only for `image inspect` under DRY_RUN=1 (it stops before the GPU pre-flight; even without
# DRY_RUN the running GPQA container would make the pre-flight refuse). No `set -e`.
set -u
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K="$(dirname "$T")"
R="$K/run_k_selection.sh"
PASS=0; FAIL=0
: > "$T/results.txt"
scen() { # id description expect(REFUSED|DRY|RAN) expected-text ...env/cmd via function body
  local id="$1" desc="$2" expect="$3" pat="$4"; shift 4
  local d="$T/$id"; rm -rf "$d"; mkdir -p "$d"
  export SHIM_LOG="$d/shim.log"; : > "$SHIM_LOG"
  ( cd "$d" && "$@" ) > "$d/out.txt" 2>&1
  local rc=$?
  local ok=1
  grep -q -- "$pat" "$d/out.txt" || ok=0
  case "$expect" in
    REFUSED) [ $rc -eq 2 ] || ok=0; [ ! -s "$d/costs.jsonl" ] || ok=0; ! grep -q "^docker run" "$SHIM_LOG" || ok=0 ;;
    DRY)     [ $rc -eq 0 ] || ok=0; [ ! -s "$d/costs.jsonl" ] || ok=0 ;;
    *)       : ;;
  esac
  if [ $ok = 1 ]; then PASS=$((PASS+1)); r=PASS; else FAIL=$((FAIL+1)); r=FAIL; fi
  printf '%s %s rc=%s  %s\n' "$r" "$id" "$rc" "$desc" | tee -a "$T/results.txt"
  printf '      %s\n' "$(grep -m1 -E 'REFUSED|DRY_RUN|cost:|results in' "$d/out.txt")" | tee -a "$T/results.txt"
}
SHIMP="$T/shim:$PATH"
envrun() { env PATH="$SHIMP" K_TEST_ROOT="$PWD" "$@" bash "$R"; }

scen R01 "no approval" REFUSED "set APPROVED_GPU_H" envrun
scen R02 "approval below the 0.5 ceiling" REFUSED "> approved 0.4" envrun APPROVED_GPU_H=0.4
scen R03 "approval not a number" REFUSED "not a number" envrun APPROVED_GPU_H=abc
pre_spent() { echo '{"item": "k-selection", "gpu_hours": 0.3}' > "$PWD/costs.jsonl"; envrun APPROVED_GPU_H=0.5; local rc=$?; : > "$PWD/costs.jsonl"; return $rc; }
scen R04 "0.3 already spent on K, approval 0.5" REFUSED "spent 0.3 + ceiling 0.5 > approved 0.5" pre_spent
tamper() {
  local c="$PWD/copy"; mkdir -p "$c/kselect" "$c/cost"
  cp "$K/run_k_selection.sh" "$K/SHA256SUMS" "$K/plan.json" "$c/"; cp "$K/kselect/"* "$c/kselect/"; cp "$K/cost/cost.py" "$c/cost/"
  ln -s "$K/bank" "$c/bank"
  echo "# tampered" >> "$c/kselect/kref.py"
  env PATH="$SHIMP" K_TEST_ROOT="$PWD" APPROVED_GPU_H=0.5 bash "$c/run_k_selection.sh"
}
scen R05 "a frozen file was edited" REFUSED "differ from SHA256SUMS" tamper
scen R06 "image id differs from the pin" REFUSED "not present" envrun APPROVED_GPU_H=0.5 SHIM_BAD_IMAGE=1
realdry() { env K_TEST_ROOT="$PWD" APPROVED_GPU_H=0.5 DRY_RUN=1 bash "$R"; }
scen R07 "DRY_RUN with the REAL docker: frozen files, budget and both real image pins pass" DRY "DRY_RUN: frozen files" realdry
scen R08 "inference container running" REFUSED "inference/fixture container is running" envrun APPROVED_GPU_H=0.5 SHIM_BUSY=1
scen R09 "GPU memory/utilisation busy" REFUSED "not idle" envrun APPROVED_GPU_H=0.5 SHIM_GPU_MEM_BUSY=1
scen R10 "a non-desktop graphics client on the GPU" REFUSED "not idle" envrun APPROVED_GPU_H=0.5 SHIM_GPU_PROC=1
scen R11 "bad GPU index" REFUSED "K_GPU_INDEX must be 0 or 1" envrun APPROVED_GPU_H=0.5 K_GPU_INDEX=2
outside() { env PATH="$SHIMP" K_TEST_ROOT=/tmp/elsewhere APPROVED_GPU_H=0.5 bash "$R"; }
scen R12 "test root outside refusal-tests" REFUSED "must lie under a refusal-tests" outside

# ---- run-path scenarios (shim docker run): cost line, container removal, report
ranok() {
  envrun APPROVED_GPU_H=0.5 SHIM_GPU_DESKTOP=1 SHIM_RUN_SLEEP=0
  local rc=$?
  python3 - "$PWD" <<'PY' || return 9
import json, sys, os
d = sys.argv[1]
lines = [json.loads(l) for l in open(os.path.join(d, "costs.jsonl")) if l.strip()]
assert len(lines) == 1 and lines[0]["status"] == "done" and lines[0]["item"] == "k-selection", lines
log = open(os.path.join(d, "shim.log")).read()
assert log.count("docker run") == 2 and "--network none" in log and "--gpus device=0" in log, log
for e in ("b1", "v070"):
    assert f"docker rm -f kselect-{e}-" in log
runs = [x for x in os.listdir(d) if x.startswith("run-")]
assert runs and os.path.exists(os.path.join(d, runs[0], "REPORT.txt"))
print("CHECKED: one cost line (done), two shimmed containers, both removed, REPORT.txt written")
PY
  return $rc
}
scen R13 "idle GPU (desktop client only): run path completes" RAN "CHECKED: one cost line (done)" ranok
again() {
  # R13's own cost line, re-billed as 36 s (0.01 GPU-h): the gate must read the runner's format back
  sed 's/"gpu_hours": [0-9.]*/"gpu_hours": 0.01/' "$T/R13/costs.jsonl" > "$PWD/costs.jsonl"
  envrun APPROVED_GPU_H=0.5 SHIM_GPU_DESKTOP=1; local rc=$?
  ! grep -q "^docker run" "$SHIM_LOG" || return 9
  [ "$(wc -l < "$PWD/costs.jsonl")" -eq 1 ] || return 9
  return $rc
}
scen R14 "second run after a billed run (runner's own cost-line format): cumulative budget refuses" RAN "spent 0.01 + ceiling 0.5 > approved 0.5" again
interrupt() {
  env PATH="$SHIMP" K_TEST_ROOT="$PWD" APPROVED_GPU_H=0.5 SHIM_RUN_SLEEP=60 bash "$R" &
  local pid=$! t0; t0=$(date +%s)
  sleep 4; kill -TERM "$pid"; wait "$pid"; local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  python3 - "$PWD" "$rc" "$dt" <<'PY' || return 9
import json, sys, os
d, rc, dt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lines = [json.loads(l) for l in open(os.path.join(d, "costs.jsonl")) if l.strip()]
assert len(lines) == 1 and lines[0]["status"] == "interrupted", lines
assert "docker rm -f kselect-b1-" in open(os.path.join(d, "shim.log")).read()
assert rc == 130 and dt < 15, (rc, dt)
print(f"CHECKED: TERM -> exit 130 after {dt}s, cost line 'interrupted', container removed")
PY
}
scen R15 "TERM during a run: immediate cleanup and cost line" RAN "CHECKED: TERM -> exit 130" interrupt
wallcap() {
  envrun APPROVED_GPU_H=0.5 SHIM_RUN_SLEEP=20 K_TEST_CAP_S=2
  python3 - "$PWD" <<'PY' || return 9
import json, sys, os
d = sys.argv[1]
lines = [json.loads(l) for l in open(os.path.join(d, "costs.jsonl")) if l.strip()]
assert len(lines) == 1 and lines[0]["status"].startswith("wall-cap"), lines
log = open(os.path.join(d, "shim.log")).read()
assert "docker rm -f kselect-b1-" in log and "docker rm -f kselect-v070-" in log
print(f"CHECKED: wall cap -> status {lines[0]['status']}, both containers removed, {lines[0]['wall_s']} s billed")
PY
}
scen R16 "per-container wall cap (test cap 2 s)" RAN "CHECKED: wall cap" wallcap
echo "refusal/cleanup tests: $PASS pass, $FAIL fail" | tee -a "$T/results.txt"
