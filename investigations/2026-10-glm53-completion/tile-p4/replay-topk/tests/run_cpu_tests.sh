#!/bin/bash
# replay-topk CPU test suite (no GPU: every container runs with --network none -e CUDA_VISIBLE_DEVICES=).
#   tests/run_cpu_tests.sh            all parts; summary to tests/RESULTS.txt
# Parts: host syntax; served-argument checks in each image; CPU dry run of the replay in each image; harness test with
# CPU stand-ins (v0.9.1 image); analysis with planted defects (host); runner gates with fake docker / nvidia-smi
# (host; needs SHA256SUMS current); public-text check.  No `set -e`: each part's exit code is recorded.
set -u
T=$(cd "$(dirname "$0")" && pwd)
PKG=$(cd "$T/.." && pwd)
OUT=$T/out
mkdir -p "$OUT"
LOG=$OUT/suite.log
: > "$LOG"
declare -A IMG=( [v091]=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
                 [v070]=sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629 )
CPU=(--rm --network none -e CUDA_VISIBLE_DEVICES= --user "$(id -u):$(id -g)" -e HOME=/tmp)
declare -A RC
part() { echo "=== $1" | tee -a "$LOG"; }
part "syntax (host)"
ok=0
for f in "$PKG"/scripts/*.py "$PKG"/analysis/*.py "$PKG"/runner/*.py "$PKG"/tests/*.py; do
  python3 -m py_compile "$f" 2>>"$LOG" || { echo "FAIL syntax $f" | tee -a "$LOG"; ok=1; }
done
for f in "$PKG"/scripts/*.sh "$PKG"/runner/*.sh "$PKG"/tests/*.sh "$PKG"/make_sums.sh; do
  bash -n "$f" 2>>"$LOG" || { echo "FAIL syntax $f" | tee -a "$LOG"; ok=1; }
done
[ $ok = 0 ] && echo "PASS syntax" | tee -a "$LOG"
RC[syntax]=$ok
for v in v091 v070; do
  part "served arguments ($v image, CPU)"
  docker run "${CPU[@]}" -v "$PKG/scripts":/s:ro -v "$T":/t:ro --entrypoint python3 "${IMG[$v]}" /t/test_served_args.py "$v" \
    2>&1 | grep -E "^(PASS|FAIL|  INFO)|failure" | tee -a "$LOG"
  RC[served_$v]=${PIPESTATUS[0]}
done
for v in v091 v070; do
  part "dry run ($v image, CPU, TRITON_INTERPRET=1)"
  rm -rf "$OUT/dry/$v"; mkdir -p "$OUT/dry/$v"
  docker run "${CPU[@]}" -v "$PKG/scripts":/s:ro -v "$OUT/dry/$v":/o --entrypoint python3 "${IMG[$v]}" /s/replay.py \
    --version "$v" --out /o --dry 2>&1 | grep -E "^\[|^\{" | tee -a "$LOG"
  RC[dry_$v]=${PIPESTATUS[0]}
done
part "harness with CPU stand-ins (v091 image)"
rm -rf "$OUT/harness"; mkdir -p "$OUT/harness"
docker run "${CPU[@]}" -v "$PKG/scripts":/s:ro -v "$T":/t:ro -v "$OUT/harness":/o --entrypoint python3 "${IMG[v091]}" \
  /t/test_harness_cpu.py /o 2>&1 | grep -E "^(PASS|FAIL|INFO)|failure|Traceback|Error" | tee -a "$LOG"
RC[harness]=${PIPESTATUS[0]}
part "analysis (host): stand-in cases and planted defects"
rm -rf "$OUT/planted"
python3 "$T/test_analysis.py" "$OUT/harness" "$OUT/planted" 2>&1 | tee -a "$LOG"
RC[analysis]=${PIPESTATUS[0]}
part "runner gates (host, fake docker / nvidia-smi)"
bash "$T/test_runner.sh" "$OUT/harness" 2>&1 | tee -a "$LOG"
RC[runner]=${PIPESTATUS[0]}
part "public text (no hostnames, user names, home paths, e-mail, other projects)"
PAT='<withheld>'  # public-text check: run locally against a withheld deny-list
hits=$(cd "$PKG" && grep -rlE "$PAT" --exclude-dir=out --exclude-dir=runs --exclude-dir=cache --exclude-dir=__pycache__ \
       --exclude=SHA256SUMS --exclude=run_cpu_tests.sh . 2>/dev/null)
if [ -z "$hits" ]; then echo "PASS public text" | tee -a "$LOG"; RC[public]=0; else echo "FAIL public text: $hits" | tee -a "$LOG"; RC[public]=1; fi
part "summary"
NP=$(grep -c "^PASS" "$LOG"); NF=$(grep -c "^FAIL" "$LOG")
{
  echo "replay-topk CPU test suite, $(date -u +%FT%TZ)"
  echo "images: v091 ${IMG[v091]}, v070 ${IMG[v070]} (CPU containers, --network none, CUDA hidden)"
  for k in syntax served_v091 served_v070 dry_v091 dry_v070 harness analysis runner public; do echo "part $k: exit ${RC[$k]}"; done
  echo "checks: $NP PASS, $NF FAIL"
  echo
  grep -E "^(=== |PASS|FAIL|  INFO|\[|\{|v0[79][01] R|harness:|analysis:|runner:|v0[79][01]: )" "$LOG"
} > "$T/RESULTS.txt"
tail -n +1 "$T/RESULTS.txt" | head -12
bad=0; for k in "${!RC[@]}"; do [ "${RC[$k]}" = 0 ] || bad=1; done
exit $bad
