#!/bin/bash
# P4 CPU tests.  Every engine-side test runs in a CPU container: docker run --rm --network none, no --gpus,
# CUDA_VISIBLE_DEVICES=, TRITON_INTERPRET=1, read-only mounts except tests/out.  Variant "hooked" = the stock image
# with build/files installed over its (ephemeral) site-packages; "image" = the built P4 image as is.  No `set -e`.
#   run_cpu_tests.sh [suite ...]   suites: gen units list modes forcing inert gates runner analysis image (default: all
#                                  but image)
T=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$T/.." && pwd)
INV=<runs>
STOCK=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
P4IMG=$(awk '{print $1}' "$ROOT/build/image-id.txt" 2>/dev/null); [ -n "$P4IMG" ] || P4IMG=missing-p4-image
OUT=${OUT_OVERRIDE:-$T/out}; mkdir -p "$OUT"
run() {  # name variant env... -- dir script args
  local NAME=$1 VAR=$2; shift 2
  local ENVS=(); while [ "$1" != "--" ]; do ENVS+=(-e "$1"); shift; done; shift
  local I=$STOCK; [ "$VAR" = image ] && I=$P4IMG
  echo "== $NAME ($VAR) $(date -u +%FT%TZ)" > "$OUT/$NAME.log"
  docker run --rm --pull=never --network none -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 -e HOME=/tmp "${ENVS[@]}" \
    -v "$ROOT/build/files/vllm":/p4files:ro -v "$T":/t:ro -v "$INV/hook-validation/tests":/hv/tests:ro -v "$ROOT/precheck":/p4pc:ro \
    --entrypoint /bin/bash "$I" /t/inner.sh "$VAR" "$@" >> "$OUT/$NAME.log" 2>&1
  local RC=$?
  echo "rc=$RC" >> "$OUT/$NAME.log"
  echo "== $NAME ($VAR): rc=$RC; $(grep -c '^PASS' "$OUT/$NAME.log") PASS, $(grep -c '^FAIL' "$OUT/$NAME.log") FAIL"
  grep -E "^FAIL|Traceback|Error:" "$OUT/$NAME.log" | grep -v Warning | head -n 8
}
P4ENV=(FORCEWIN=on GLM53_P4=on GLM53_P4_DIR=/tmp/p4test FORCEWIN_DIR=/tmp/p4test/fw FW_THREADS=4)
[ "$#" -eq 0 ] && set -- gen units precheck list modes forcing inert gates runner analysis
for S in "$@"; do
  case $S in
    gen)      { echo "== gen $(date -u +%FT%TZ)"; ( cd "$ROOT/build" && python3 make_hooked.py --check && python3 port_forcewin.py --check ); echo "rc=$?"; } > "$OUT/gen.log" 2>&1
              echo "== gen: $(grep -c '^PASS' "$OUT/gen.log") PASS, $(grep -c '^FAIL' "$OUT/gen.log") FAIL; $(tail -1 "$OUT/gen.log")" ;;
    units)    run p4_units hooked "${P4ENV[@]}" -- t test_p4_cpu.py units ;;
    precheck) run precheck_dry hooked GLM53_P4=on GLM53_P4_DIR=/tmp/p4pre-cpu -- t test_precheck_cpu.py ;;
    list)     run p4_list hooked "${P4ENV[@]}" -- t test_p4_cpu.py list ;;
    modes)    run p4_modes hooked "${P4ENV[@]}" -- t test_p4_cpu.py modes ;;
    forcing)  # the validated hook-validation forcing suite against the PORTED module (GLM53_P4 unset and set)
              for P in natural oracle poison reject:1 reject:2 reject:3 sched:7 seed rest; do
                ( run "fw_${P/:/}" hooked FORCEWIN=on FW_THREADS=2 -- hv test_forcing.py "$P" > "$OUT/.fw_${P/:/}.sum" ) &
              done; wait
              ( run fw_rest_p4on hooked FORCEWIN=on GLM53_P4=on GLM53_P4_DIR=/tmp/x FW_THREADS=2 -- hv test_forcing.py rest > "$OUT/.fw_p4on.sum" )
              cat "$OUT"/.fw_*.sum; rm -f "$OUT"/.fw_*.sum ;;
    inert)    run inert_orig orig -- hv test_inert.py orig
              run inert_hooked_off hooked -- hv test_inert.py hooked-off
              run inert_hooked_on_unkeyed hooked FORCEWIN=on GLM53_P4=on GLM53_P4_DIR=/tmp/x -- hv test_inert.py hooked-on-unkeyed
              [ -f "$ROOT/build/image-id.txt" ] && run image_inert_off image -- hv test_inert.py image-off
              python3 "$INV/hook-validation/tests/compare_inert.py" "$OUT" | tee "$OUT/inert-compare.txt" ;;
    gates)    python3 "$T/test_gates.py" > "$OUT/gates.log" 2>&1; echo "== gates: rc=$?; $(grep -c '^PASS' "$OUT/gates.log") PASS, $(grep -c '^FAIL' "$OUT/gates.log") FAIL" ;;
    runner)   bash "$T/test_runner.sh" > "$OUT/runner.log" 2>&1; echo "== runner: rc=$?; $(grep -c '^PASS' "$OUT/runner.log") PASS, $(grep -c '^FAIL' "$OUT/runner.log") FAIL" ;;
    analysis) rm -rf "$OUT/fake-root"
              docker run --rm --pull=never --network none -e CUDA_VISIBLE_DEVICES= --user "$(id -u):$(id -g)" -e HOME=/tmp \
                -v "$ROOT":/p4:ro -v "$OUT":/o --entrypoint sh "$STOCK" -c \
                "python3 /p4/tests/make_fake_sessions.py /o/fake-root && python3 /p4/tests/check_analysis.py /o/fake-root" \
                > "$OUT/analysis.log" 2>&1
              echo "== analysis: rc=$?; $(grep -c '^PASS' "$OUT/analysis.log") PASS, $(grep -c '^FAIL' "$OUT/analysis.log") FAIL"
              grep -E "^FAIL|Traceback" "$OUT/analysis.log" | head ;;
    image)    run image_units image "${P4ENV[@]}" -- t test_p4_cpu.py units
              run image_precheck_dry image GLM53_P4=on GLM53_P4_DIR=/tmp/p4pre-cpu -- t test_precheck_cpu.py
              run image_list image "${P4ENV[@]}" -- t test_p4_cpu.py list
              run image_fw_rest image FORCEWIN=on FW_THREADS=2 -- hv test_forcing.py rest
              run image_inert_off image -- hv test_inert.py image-off
              python3 "$INV/hook-validation/tests/compare_inert.py" "$OUT" | tee "$OUT/inert-compare.txt" ;;
  esac
done
