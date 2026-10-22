#!/usr/bin/env bash
set -u
# [E copy of bplus/tests/run_analysis_dryruns.sh: set -u; scratch fixed inside e-cross/PRIVATE/ and guarded; E scenarios;
#  also checks the E1 screen / PREFIX_HIGH flags and that the cross table was computed. 22ah: KV floor cases replace the
#  KV identity cases.]
# Analysis dry runs on synthetic run folders (CHECKS.md). CPU only. Usage: run_analysis_dryruns.sh > tests/analysis-dryruns.txt
# Each scenario: build a synthetic run folder, run the frozen analysis (scripts/analysis_e.py) in the CPU-only container,
# compare the state / reading / exit code (and, where stated, the E1 screen) with the expectation. Synthetic folders are
# deleted afterwards.
H=$(cd "$(dirname "$0")" && pwd); W=$(dirname "$H")
EC=<runs>/long-reply/e-cross
case "$W" in "$EC") ;; *) echo "ABORT: unexpected package path $W"; exit 2 ;; esac
S=$W/PRIVATE/scratch-dryruns
case "$S" in "$EC"/PRIVATE/scratch-dryruns) ;; *) echo "ABORT: scratch $S not inside PRIVATE/"; exit 2 ;; esac
mkdir -p "$S" "$W/tests/out" || exit 1
chmod 700 "$S"
PASS=0; FAIL=0
while read -r SCN EXP_RC EXP_STATE EXP_READ EXP_EXTRA; do
  [ -z "$SCN" ] && continue
  D="$S/synth-$SCN"
  case "$D" in "$EC"/PRIVATE/scratch-dryruns/synth-*) ;; *) echo "ABORT: bad scratch path $D"; exit 2 ;; esac
  python3 -B "$W/tests/make_synthetic_run_e.py" "$SCN" "$D" > /dev/null || { echo "FAIL $SCN: generator"; FAIL=$((FAIL+1)); continue; }
  RUN_DIR="$D" bash "$W/scripts/run_cpu.sh" v070 scripts/analysis_e.py "/work/tests/out/dryrun-$SCN.json" > "$W/tests/out/dryrun-$SCN.md" 2> "$W/tests/out/dryrun-$SCN.err"
  RC=$(grep -oE '^exit [0-9]+' "$W/tests/out/dryrun-$SCN.md" | tail -1 | cut -d' ' -f2)
  GOT=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['state'], d.get('reading'))" "$W/tests/out/dryrun-$SCN.json" 2>/dev/null)
  EX=$(python3 -c "
import json,sys
d=json.load(open(sys.argv[1])); p=d.get('primary') or {}; c=d.get('cross') or {}
print(f\"screen={p.get('E1_screen')},prefix_high={p.get('PREFIX_HIGH')},cross={'error' not in c and bool(c)}\")" "$W/tests/out/dryrun-$SCN.json" 2>/dev/null)
  OKX=1; [ "$EXP_EXTRA" != "-" ] && { printf '%s' "$EX" | grep -qF -- "$EXP_EXTRA" || OKX=0; }
  if [ "$RC" = "$EXP_RC" ] && [ "$GOT" = "$EXP_STATE $EXP_READ" ] && [ "$OKX" = 1 ]; then
    echo "PASS $SCN: rc $RC, $GOT; $EX$( [ "$EXP_STATE" = INVALID ] && echo "; failed: $(python3 -c "import json,sys; print([c['check'][:90] for c in json.load(open(sys.argv[1]))['checks'] if not c['ok']])" "$W/tests/out/dryrun-$SCN.json")")"
    PASS=$((PASS+1))
  else echo "FAIL $SCN: rc $RC (expected $EXP_RC), got '$GOT' (expected '$EXP_STATE $EXP_READ'), extra '$EX' (expected '$EXP_EXTRA')"; FAIL=$((FAIL+1)); fi
  case "$D" in "$EC"/PRIVATE/scratch-dryruns/synth-*) rm -rf -- "$D" ;; esac
done <<'LIST'
all-finish 0 COMPLETE R-I screen=SMALL,prefix_high=False,cross=True
none-finish 0 COMPLETE R-I screen=SMALL,prefix_high=False,cross=True
engine-effect 0 COMPLETE R-E screen=EFFECT
reverse 0 COMPLETE R-E- screen=REVERSE
state-dominated 0 COMPLETE R-I screen=SMALL,prefix_high=True,cross=True
null 0 COMPLETE R-I -
kv-2749599 0 COMPLETE R-E -
kv-bplus-log 0 COMPLETE R-E -
kv-unseen-above-floor 0 COMPLETE R-E -
kv-unseen-v091 0 COMPLETE R-E -
neg-kv-below-floor 3 INVALID None -
neg-kv-v091-below-floor 3 INVALID None -
neg-wrong-seed 3 INVALID None -
neg-seed-reused 3 INVALID None -
neg-prompt-hash 3 INVALID None -
neg-raw-mismatch 3 INVALID None -
neg-image 3 INVALID None -
neg-no-ready 3 INVALID None -
neg-missing-request 4 INCOMPLETE None -
neg-watchdog 4 INCOMPLETE None -
neg-v091-budget-stop 4 INCOMPLETE None -
LIST
echo "dry runs: $PASS pass, $FAIL fail"
case "$S" in "$EC"/PRIVATE/scratch-dryruns) rmdir -- "$S" 2>/dev/null ;; esac
[ "$FAIL" = 0 ]
