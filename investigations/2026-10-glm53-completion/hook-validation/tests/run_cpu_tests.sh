#!/bin/bash
# CPU-only tests of the forcewin hook in the B1 image: docker --network none, no --gpus, CUDA_VISIBLE_DEVICES=,
# TRITON_INTERPRET=1, read-only mounts except tests/out. No image is built. No `set -e` (each test's rc recorded).
HV=<runs>/hook-validation
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a     # B1 local/glm53-v0.9.1-b1-d7:local
OUT=$HV/tests/out; mkdir -p "$OUT"
IMGF=sha256:1b9f8b15d63197ad8b2ef2410321baedf9dcd55a112603bc77b887e8dac293d5  # local/glm53-v0.9.1-b1-forcehook-mg:local (built here)
run() {  # name variant env... -- script args
  local NAME=$1 VAR=$2; shift 2
  local ENVS=(); while [ "$1" != "--" ]; do ENVS+=(-e "$1"); shift; done; shift
  echo "== $NAME ($VAR) $(date -u +%FT%TZ)"
  local I=$IMG; [ "$VAR" = image ] && I=$IMGF
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 "${ENVS[@]}" \
    -v $HV/hook:/hv/hook:ro -v $HV/tests:/hv/tests:ro \
    -v <runs>/d7/step2/guard:/hv/guard:ro \
    --entrypoint /bin/bash "$I" /hv/tests/inner.sh "$VAR" "$@" > "$OUT/$NAME.log" 2>&1
  local RC=$?
  grep -E "^(PASS|FAIL)|checks passed|Traceback|Error" "$OUT/$NAME.log" | grep -v Warning | tail -n 60
  echo "rc=$RC" | tee -a "$OUT/$NAME.log"
}
[ "$#" -eq 0 ] && set -- forcing d2 exposure inert
for T in "$@"; do
  case $T in
    forcing)  run test_forcing hooked FORCEWIN=on -- test_forcing.py ;;
    forcing-par)   # the same checks as "forcing", in parallel shards (one container per schedule + rest + seed)
              for P in natural oracle poison reject:1 reject:2 reject:3 sched:7 seed rest; do
                ( run "test_forcing_${P/:/}" hooked FORCEWIN=on FW_THREADS=2 -- test_forcing.py "$P" > /dev/null ) &
              done; wait
              grep -hE "^(PASS|FAIL)|checks passed|Traceback" "$OUT"/test_forcing_*.log ;;
    d2)       run test_d2_inject hooked FORCEWIN=on FORCEWIN_D2=1 FORCEWIN_SHADOW=1 -- test_d2_inject.py ;;
    exposure) run test_exposure hooked FORCEWIN=on -- test_exposure.py ;;
    inert)    run inert_orig orig -- test_inert.py orig
              run inert_hooked_off hooked -- test_inert.py hooked-off
              run inert_hooked_on_unkeyed hooked FORCEWIN=on -- test_inert.py hooked-on-unkeyed
              python3 $HV/tests/compare_inert.py $OUT | tee $OUT/inert-compare.txt ;;
    mapguard) run test_patch_mapguard hooked-mapguard FORCEWIN=on -- test_inert.py hooked-mapguard-import ;;
    image)    run image_installed image -- check_installed.py
              run image_mapguard_import image FORCEWIN=on -- test_inert.py hooked-mapguard-import
              run image_inert_off image -- test_inert.py image-off
              run image_d2_inject image FORCEWIN=on FORCEWIN_D2=1 FORCEWIN_SHADOW=1 -- test_d2_inject.py
              run image_exposure image FORCEWIN=on -- test_exposure.py ;;
    analysis) for M in good bad; do
                docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v $HV/hook:/hv/hook:ro -v $HV/tests:/hv/tests:ro \
                  -v $HV/analysis:/hv/analysis:ro -v $OUT/analysis-dryrun:/dry --entrypoint sh "$IMG" -c \
                  "rm -rf /dry/$M && python3 /hv/tests/make_synthetic_run.py /dry/$M $M && python3 /hv/analysis/analyze_hookval.py /dry/$M" \
                  2>&1 | grep -v "Warning\|cpp_extension" > "$OUT/analysis-dryrun-$M.log"
                echo "== analysis dry run ($M)"; grep -E "VERDICT|FAIL" "$OUT/analysis-dryrun-$M.log" | cut -c1-160
              done ;;
  esac
done
