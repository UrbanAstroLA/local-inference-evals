#!/usr/bin/env bash
# Analysis dry runs on synthetic run folders (CHECKS.md). CPU only. Usage: run_analysis_dryruns.sh SCRATCH_DIR
# Each scenario: build a synthetic run folder, run the frozen analysis in the CPU-only container, compare the state /
# reading and exit code with the expectation. Synthetic folders are deleted afterwards.
W=<runs>/long-reply/bplus
S=${1:?scratch dir}
mkdir -p "$S"
PASS=0; FAIL=0
while read -r SCN EXP_RC EXP_STATE EXP_READ; do
  [ -z "$SCN" ] && continue
  D="$S/synth-$SCN"
  python3 -B "$W/tests/make_synthetic_run.py" "$SCN" "$D" > /dev/null || { echo "FAIL $SCN: generator"; FAIL=$((FAIL+1)); continue; }
  RUN_DIR="$D" bash "$W/scripts/run_cpu.sh" v070 scripts/analysis_bplus.py "/work/tests/out/dryrun-$SCN.json" > "$W/tests/out/dryrun-$SCN.md" 2> "$W/tests/out/dryrun-$SCN.err"
  RC=$(grep -oE '^exit [0-9]+' "$W/tests/out/dryrun-$SCN.md" | tail -1 | cut -d' ' -f2)
  GOT=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['state'], d.get('reading'))" "$W/tests/out/dryrun-$SCN.json" 2>/dev/null)
  if [ "$RC" = "$EXP_RC" ] && [ "$GOT" = "$EXP_STATE $EXP_READ" ]; then echo "PASS $SCN: rc $RC, $GOT"; PASS=$((PASS+1));
  else echo "FAIL $SCN: rc $RC (expected $EXP_RC), got '$GOT' (expected '$EXP_STATE $EXP_READ')"; FAIL=$((FAIL+1)); fi
  rm -rf "$D"
done <<'LIST'
all-finish 0 COMPLETE R-I
none-finish 0 COMPLETE R-I
engine-effect 0 COMPLETE R-E
state-dominated 0 COMPLETE R-S
null 0 COMPLETE R-I
c1-only 5 C1-ONLY R-E
neg-wrong-seed 3 INVALID None
neg-prompt-hash 3 INVALID None
neg-raw-mismatch 3 INVALID None
neg-image 3 INVALID None
neg-no-ready 3 INVALID None
neg-missing-request 4 INCOMPLETE None
neg-watchdog 4 INCOMPLETE None
LIST
echo "dry runs: $PASS pass, $FAIL fail"
