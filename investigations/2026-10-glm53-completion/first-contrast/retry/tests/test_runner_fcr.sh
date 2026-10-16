#!/bin/bash
# Retry runner gates [R-8]: DRY_RUN and refusal tests with shims (derived from ../../tests/test_runner_fc.sh). CPU only;
# nothing is launched.
#   * the runner under test is a byte-identical copy of retry/runner/run_fcr.sh inside a shim tree holding exactly the
#     files listed in retry/SHA256SUMS and in the attempt-1 ../SHA256SUMS (both frozen-hash gates run for real) plus a
#     shim cost ledger;
#   * PATH shims: `docker` answers only `image inspect` (real, read-only, or a planted id via SHIM_IMG_ID) and `info`;
#     it REFUSES every other subcommand; `nvidia-smi` and `curl` refuse; `df` is real unless SHIM_DF_GB plants a value.
#     Every case is refused by a gate or exits at DRY_RUN, before the pre-flight; a shim refusal fails the case.
# usage: test_runner_fcr.sh   -> retry/tests/out/runner-tests.log, rc 0 iff every case behaves as expected. No `set -e`.
RETRY=$(cd "$(dirname "$0")/.." && pwd)
FC=$(cd "$RETRY/.." && pwd)
LOG=$RETRY/tests/out/runner-tests.log
SCR=${TMPDIR:-/tmp}/fcr-runner-shim.$$
rm -rf "$SCR"; mkdir -p "$SCR/bin" "$SCR/fc/retry" || exit 1
trap 'rm -rf "$SCR"' EXIT
REAL_DOCKER=$(command -v docker)
REAL_DF=$(command -v df)
cat > "$SCR/bin/docker" <<EOF
#!/bin/bash
if [ "\$1" = image ] && [ "\$2" = inspect ]; then
  if [ -n "\${SHIM_IMG_ID:-}" ] && [ "\${4:-}" = --format ]; then echo "\$SHIM_IMG_ID"; exit 0; fi
  exec "$REAL_DOCKER" "\$@"
fi
if [ "\$1" = info ]; then exec "$REAL_DOCKER" "\$@"; fi
echo "SHIM-REFUSED: docker \$*" >&2; echo "SHIM-REFUSED docker \$*" >> "$SCR/refusals"; exit 97
EOF
cat > "$SCR/bin/df" <<EOF
#!/bin/bash
if [ -n "\${SHIM_DF_GB:-}" ]; then echo "Avail"; echo "\${SHIM_DF_GB}G"; exit 0; fi
exec "$REAL_DF" "\$@"
EOF
for c in nvidia-smi curl; do
  printf '#!/bin/bash\necho "SHIM-REFUSED: %s $*" >&2; echo "SHIM-REFUSED %s" >> "%s/refusals"; exit 97\n' "$c" "$c" "$SCR" > "$SCR/bin/$c"
done
chmod +x "$SCR/bin/"*
( cd "$FC" && cut -c67- SHA256SUMS | while read -r f; do cp --parents -p "$f" "$SCR/fc/" || echo "COPY FAIL $f"; done; cp -p SHA256SUMS "$SCR/fc/" )
( cd "$RETRY" && cut -c67- SHA256SUMS | while read -r f; do cp --parents -p "$f" "$SCR/fc/retry/" || echo "COPY FAIL $f"; done; cp -p SHA256SUMS "$SCR/fc/retry/" )
cmp -s "$SCR/fc/retry/runner/run_fcr.sh" "$RETRY/runner/run_fcr.sh" || { echo "runner copy differs" | tee "$LOG"; exit 1; }
mkdir -p "$(dirname "$LOG")"; : > "$LOG"
echo "runner under test: sha256 $(sha256sum "$RETRY/runner/run_fcr.sh" | cut -d' ' -f1) (copy identical)" | tee -a "$LOG"
PASSN=0; FAILN=0
A_TRAFFIC='{"item":"fcr1","gpu_hours":1.9,"attempt":1,"rc":10}'
A_PRE='{"item":"fcr1","gpu_hours":0.41,"attempt":1,"rc":7}'
OTHER='{"item":"fc1","gpu_hours":1.0861,"attempt":1}'
run_case() {   # name expected_rc regex attempt costs_content [env...]
  local NAME=$1 ERC=$2 RX=$3 ATT=$4 COSTS=$5; shift 5
  printf '%s' "$COSTS" > "$SCR/fc/retry/costs.jsonl"
  local OUTT; OUTT=$(env -i PATH="$SCR/bin:/usr/local/bin:/usr/bin:/bin" HOME="$HOME" DRY_RUN=1 "$@" \
                     bash "$SCR/fc/retry/runner/run_fcr.sh" $ATT 2>&1); local RC=$?
  local OK=1
  [ "$RC" = "$ERC" ] || OK=0
  echo "$OUTT" | grep -qE "$RX" || OK=0
  if [ -s "$SCR/refusals" ]; then OK=0; fi
  if [ $OK = 1 ]; then PASSN=$((PASSN + 1)); echo "PASS $NAME (rc $RC)" | tee -a "$LOG"
  else FAILN=$((FAILN + 1)); echo "FAIL $NAME (rc $RC, expected $ERC /$RX/)" | tee -a "$LOG"; fi
  echo "$OUTT" | sed 's/^/    | /' >> "$LOG"
  [ -s "$SCR/refusals" ] && { sed 's/^/    shim: /' "$SCR/refusals" >> "$LOG"; : > "$SCR/refusals"; }
  rm -rf "$SCR/fc/retry/run-fcr1" "$SCR/fc/retry/run-fcr2"
}
B="APPROVED_GPU_H=6.2 VARIANT=B APPROVED_POLICY=single"   # [R-16] PLAN.md amendment 22h
run_case R1-dryrun-B 0 "DRY_RUN: .*plan check \(B\): 224 requests, 34 phases, 0 failures" "" "" $B
run_case R2-budget-line-B 0 "variant B policy single attempt 1; spent 0.0 \+ this attempt's cap 6.2 \(wall 11160s\) <= approved 6.2" "" "" $B
run_case R3-dryrun-A0 0 "plan check \(A0\): 176 requests, 28 phases, 0 failures" "" "" APPROVED_GPU_H=3.72 VARIANT=A0 APPROVED_POLICY=single
run_case R4-no-approval 2 "REFUSED: set APPROVED_GPU_H" "" "" VARIANT=B APPROVED_POLICY=single
run_case R5-over-ceiling-and-allowance 2 "REFUSED: APPROVED_GPU_H=6.7 is not in \(0, 6.2 \(\+ 0.46" "" "" APPROVED_GPU_H=6.7 VARIANT=B APPROVED_POLICY=single
run_case R6-allowance-needs-policy 2 "REFUSED: APPROVED_GPU_H above the B ceiling 6.2 requires APPROVED_POLICY=invalid-start" "" "" APPROVED_GPU_H=6.21 VARIANT=B APPROVED_POLICY=single
run_case R7-no-policy 2 "REFUSED: set APPROVED_POLICY=single or invalid-start" "" "" APPROVED_GPU_H=6.2 VARIANT=B
run_case R8-single-attempt-recorded 2 "REFUSED: 1 attempt\(s\) already recorded for fcr1 \(policy single" 2 "$A_PRE
" $B
run_case R9-invalid-start-after-traffic 2 "REFUSED: invalid-start policy allows attempt 2 only after an attempt that stopped before any client traffic" 2 "$A_TRAFFIC
" APPROVED_GPU_H=6.66 VARIANT=B APPROVED_POLICY=invalid-start
run_case R10-invalid-start-after-pre-traffic-stop 0 "attempt 2; spent 0.41 \+ this attempt's cap 6.2 \(wall 11160s\) <= approved 6.66" 2 "$A_PRE
" APPROVED_GPU_H=6.66 VARIANT=B APPROVED_POLICY=invalid-start
run_case R11-invalid-start-wrong-attempt-number 2 "REFUSED: invalid-start policy allows attempt 2 only" "" "$A_PRE
" APPROVED_GPU_H=6.66 VARIANT=B APPROVED_POLICY=invalid-start
run_case R12-attempt-2-without-attempt-1 2 "REFUSED: no attempt recorded; this must be attempt 1" 2 "" $B
run_case R13-attempt-3 2 "REFUSED: attempt must be 1 or 2" 3 "" $B
run_case R14-below-min-wall 2 "REFUSED: approved 2.5 GPU-h \(spent 0.0\) gives 4500s wall < 4942s" "" "" APPROVED_GPU_H=2.5 VARIANT=B APPROVED_POLICY=single
run_case R15-bad-variant 2 "REFUSED: VARIANT must be B or A0" "" "" APPROVED_GPU_H=6.2 VARIANT=X APPROVED_POLICY=single
run_case R16-wrong-image-id 2 "ABORT: local/glm53-v0.9.1-b1-forcehook-mg2:local is sha256:dead, expected" "" "" $B SHIM_IMG_ID=sha256:dead
run_case R17-disk-insufficient 2 "ABORT: /data has 150 GB free < 200 GB" "" "" $B SHIM_DF_GB=150
run_case R18-negative-approval 2 "REFUSED: APPROVED_GPU_H=-1 is not in" "" "" APPROVED_GPU_H=-1 VARIANT=B APPROVED_POLICY=single
run_case R19-other-items-ignored 0 "spent 0.0 \+ this attempt's cap 6.2" "" "$OTHER
" $B
# [R-16] 22h budget: a lower approval sets a lower cap; above 6.2 needs invalid-start; the superseded 4.18 is no ceiling
run_case R24-lower-approval-lower-cap 0 "spent 0.0 \+ this attempt's cap 4.179 \(wall 7523s\) <= approved 4.18" "" "" APPROVED_GPU_H=4.18 VARIANT=B APPROVED_POLICY=single
run_case R25-just-above-6.2-refused 2 "REFUSED: APPROVED_GPU_H above the B ceiling 6.2 requires APPROVED_POLICY=invalid-start" "" "" APPROVED_GPU_H=6.2001 VARIANT=B APPROVED_POLICY=single
run_case R26-old-4.18-limit-gone 0 "cap 5.0 \(wall 9000s\) <= approved 5.0" "" "" APPROVED_GPU_H=5.0 VARIANT=B APPROVED_POLICY=single
run_case R27-A0-unchanged 2 "REFUSED: APPROVED_GPU_H above the A0 ceiling 3.72 requires APPROVED_POLICY=invalid-start" "" "" APPROVED_GPU_H=4.0 VARIANT=A0 APPROVED_POLICY=single
cp -p "$SCR/fc/retry/analysis/gate_fcr.py" "$SCR/g.bak"; echo "# tamper" >> "$SCR/fc/retry/analysis/gate_fcr.py"
run_case R20-retry-hash-mismatch 2 "ABORT: frozen hashes differ" "" "" $B
cp -p "$SCR/g.bak" "$SCR/fc/retry/analysis/gate_fcr.py"
cp -p "$SCR/fc/analysis/gate_fc.py" "$SCR/g1.bak"; echo "# tamper" >> "$SCR/fc/analysis/gate_fc.py"
run_case R21-attempt1-import-tampered 2 "ABORT: attempt-1 frozen files \(imported read-only\) differ" "" "" $B
cp -p "$SCR/g1.bak" "$SCR/fc/analysis/gate_fc.py"
cp -p "$SCR/fc/fixtures/ids/L4.json.gz" "$SCR/ids.bak"; printf 'x' >> "$SCR/fc/fixtures/ids/L4.json.gz"
run_case R22-frozen-ids-tampered 2 "ABORT: attempt-1 frozen files \(imported read-only\) differ" "" "" $B
cp -p "$SCR/ids.bak" "$SCR/fc/fixtures/ids/L4.json.gz"
mkdir -p "$SCR/fc/retry/run-fcr1"; printf '' > "$SCR/fc/retry/costs.jsonl"
OUTT=$(env -i PATH="$SCR/bin:/usr/local/bin:/usr/bin:/bin" HOME="$HOME" DRY_RUN=1 APPROVED_GPU_H=6.2 VARIANT=B APPROVED_POLICY=single bash "$SCR/fc/retry/runner/run_fcr.sh" 2>&1); RC=$?
if [ "$RC" = 2 ] && echo "$OUTT" | grep -q "ABORT: .*run-fcr1 exists"; then PASSN=$((PASSN + 1)); echo "PASS R23-existing-run-dir (rc $RC)" | tee -a "$LOG"
else FAILN=$((FAILN + 1)); echo "FAIL R23-existing-run-dir (rc $RC)" | tee -a "$LOG"; fi
echo "$OUTT" | sed 's/^/    | /' >> "$LOG"
rm -rf "$SCR/fc/retry/run-fcr1"
echo "runner tests: $PASSN passed, $FAILN failed" | tee -a "$LOG"
[ "$FAILN" = 0 ]
