#!/bin/bash
# capture-p0 CPU tests.  Every engine-side test runs in a CPU container: docker run --rm --network none, no --gpus,
# CUDA_VISIBLE_DEVICES=, read-only mounts except tests/out.  No `set -e`: each suite's rc and PASS/FAIL counts are
# reported.  Synthetic data only; nothing is written under a PRIVATE path.
#   run_cpu_tests.sh [suite ...]   suites: layout hooked image enginepath precheck analysis runner gates attempt1 s0regress
#                                  (default: layout hooked enginepath precheck analysis runner gates attempt1)
T=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$T/.." && pwd)
TILE=$(cd "$PKG/.." && pwd)
S0IMG=sha256:7408f893dfe35f1fbd330153e34ea81e83460fe5e2fff200ab87ea992565d3c7
CAPIMG=$(cat "$PKG/build/image-id.txt" 2>/dev/null); [ -n "$CAPIMG" ] || CAPIMG=missing-cap0-image
A1IMG=sha256:5831fc274d037dad3e61f18ba8cfc3e334278dbdcd74d1df9f3f327e706d8a9d     # the attempt-1 image (defective module)
INV=<runs>
OUT=${OUT_OVERRIDE:-$T/out}; mkdir -p "$OUT"
summ() { echo "== $1: rc=$2; $(grep -c '^PASS' "$3") PASS, $(grep -c '^FAIL' "$3") FAIL"; grep -E "^FAIL|Traceback|Error:" "$3" | head -n 8; }
crun() {  # name image env... -- script args
  local NAME=$1 I=$2 VAR=hooked; shift 2
  { [ "$I" = "$CAPIMG" ] || [ "$I" = "$A1IMG" ]; } && VAR=image
  local ENVS=(); while [ "$1" != "--" ]; do ENVS+=(-e "$1"); shift; done; shift
  rm -rf "$OUT/$NAME.d"; mkdir -p "$OUT/$NAME.d"
  echo "== $NAME ($VAR) $(date -u +%FT%TZ)" > "$OUT/$NAME.log"
  docker run --rm --pull=never --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    "${ENVS[@]}" -v "$PKG":/c:ro -v "$TILE/build/files/vllm":/s0p4:ro -v "$OUT/$NAME.d":/o -e TRITON_INTERPRET=1 \
    -v "$TILE/tests":/t:ro -v "$INV/hook-validation/tests":/hv/tests:ro -v "$TILE/precheck":/p4pc:ro \
    --entrypoint /bin/bash "$I" /c/tests/inner_cap0.sh "$VAR" "$@" >> "$OUT/$NAME.log" 2>&1
  local RC=$?
  echo "rc=$RC" >> "$OUT/$NAME.log"
  summ "$NAME ($VAR)" $RC "$OUT/$NAME.log"
}
CAPENV=(GLM53_P4=on GLM53_P4_S0=on GLM53_CAP0=on GLM53_CAP0_LOC=on GLM53_CAP0_LOC_TOKENS=200 GLM53_P4_DIR=/tmp/cap0test CAP0_TEST_OUT=/o/synthetic-loc)
EPENV=(FORCEWIN=on GLM53_P4=on GLM53_P4_S0=on GLM53_CAP0=on GLM53_P4_DIR=/tmp/p4s0test FORCEWIN_DIR=/tmp/p4s0test/fw FW_THREADS=4 PYTHONPATH=/hv/tests:/t:/p4pc)
[ "$#" -eq 0 ] && set -- layout hooked enginepath precheck analysis runner gates attempt1
for S in "$@"; do
  case $S in
    layout)   python3 "$T/test_layout.py" > "$OUT/layout.log" 2>&1; summ layout $? "$OUT/layout.log" ;;
    hooked)   crun hook_hooked "$S0IMG" "${CAPENV[@]}" -- test_hook_cpu.py
              crun hook_off_hooked "$S0IMG" GLM53_P4=on GLM53_P4_S0=on GLM53_P4_DIR=/tmp/x -- test_switch_off.py ;;
    image)    crun hook_image "$CAPIMG" "${CAPENV[@]}" -- test_hook_cpu.py
              crun hook_off_image "$CAPIMG" GLM53_P4=on GLM53_P4_S0=on GLM53_P4_DIR=/tmp/x -- test_switch_off.py
              crun precheck_dry_image "$CAPIMG" P4_PRECHECK_DEV=cpu -- /c/precheck/precheck_cap0.py /o/precheck.json ;;
    enginepath) # the real forcewin -> p4_hooks -> cap0_hooks path with keyed, NON-forced requests only
              crun engine_path_hooked "$S0IMG" "${EPENV[@]}" -- test_engine_path_cpu.py
              [ -n "$(docker image inspect "$CAPIMG" --format '{{.Id}}' 2>/dev/null)" ] && crun engine_path_image "$CAPIMG" "${EPENV[@]}" -- test_engine_path_cpu.py
              # negative control: the attempt-1 image must reproduce the missing-flush defect
              crun engine_path_attempt1_image "$A1IMG" "${EPENV[@]}" EXPECT_DEFECT=1 -- test_engine_path_cpu.py ;;
    attempt1) # read-only re-analysis of the real attempt-1 records with the current analysis (results to tests/out only)
              A1=$PKG/runs/session-attempt-1; P1=$TILE/PRIVATE-capture-p0/session-attempt-1/p4
              rm -rf "$OUT/attempt1-reanalysis"; mkdir -p "$OUT/attempt1-reanalysis"
              docker run --rm --pull=never --network none -e CUDA_VISIBLE_DEVICES= --user "$(id -u):$(id -g)" -e HOME=/tmp \
                -v "$PKG":/c:ro -v "$A1":/ao:ro -v "$P1":/po:ro -v "$OUT/attempt1-reanalysis":/o --entrypoint python3 "$S0IMG" \
                /c/analysis/analyze_cap0.py /ao /po --out /o > "$OUT/attempt1.log" 2>&1
              echo "rc=$?" >> "$OUT/attempt1.log"
              python3 "$T/check_attempt1.py" "$OUT/attempt1-reanalysis/cap0-result.json" >> "$OUT/attempt1.log" 2>&1
              summ attempt1 $? "$OUT/attempt1.log" ;;
    precheck) crun precheck_dry_hooked "$S0IMG" P4_PRECHECK_DEV=cpu -- /c/precheck/precheck_cap0.py /o/precheck.json ;;
    analysis) rm -rf "$OUT/analysis.d"; mkdir -p "$OUT/analysis.d"
              docker run --rm --pull=never --network none -e CUDA_VISIBLE_DEVICES= --user "$(id -u):$(id -g)" -e HOME=/tmp \
                -v "$PKG":/c:ro -v "$OUT/analysis.d":/o --entrypoint sh "$S0IMG" -c \
                "python3 /c/tests/make_fake_cap0.py /o && python3 /c/tests/check_analysis_cap0.py /o" > "$OUT/analysis.log" 2>&1
              summ analysis $? "$OUT/analysis.log" ;;
    runner)   OUT_OVERRIDE="$OUT" bash "$T/test_runner_cap0.sh" > "$OUT/runner.log" 2>&1; summ runner $? "$OUT/runner.log" ;;
    gates)    python3 "$T/test_gates_cap0.py" > "$OUT/gates.log" 2>&1; summ gates $? "$OUT/gates.log" ;;
    s0regress) # the stage-0 CPU suite against the cap0 image with GLM53_CAP0 unset (switch off == stage-0 behaviour)
              ( cd "$TILE" && IMG_S0="$CAPIMG" OUT_OVERRIDE="$OUT/s0regress" bash tests/run_cpu_tests.sh s0image ) > "$OUT/s0regress.log" 2>&1
              echo "== s0regress: rc=$?"; grep -E "^== " "$OUT/s0regress.log" ;;
  esac
done
