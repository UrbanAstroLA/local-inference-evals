#!/bin/bash
# First-contrast runner gates [A22-9]: DRY_RUN and refusal tests with shims (derived from hook-validation/tests/
# test_runner_a2.sh). CPU only; nothing is launched.
#   * the runner under test is a byte-identical copy of runner/run_fc.sh inside a shim tree holding exactly the files
#     listed in SHA256SUMS (so the frozen-hash gate runs for real) plus a shim cost ledger;
#   * PATH shims: `docker` answers only `image inspect` (delegated to the real docker, read-only, or a planted id via
#     SHIM_IMG_ID) and `info` (read-only); it REFUSES every other subcommand (run/exec/ps/stop...); `nvidia-smi` and
#     `curl` refuse; `df` is real unless SHIM_DF_GB plants a free-space figure. Every case is refused by a gate or
#     exits at DRY_RUN, before the pre-flight; a shim refusal would show and fail the case.
# usage: test_runner_fc.sh   -> tests/out/runner-tests.log, rc 0 iff every case behaves as expected. No `set -e`.
FC=$(cd "$(dirname "$0")/.." && pwd)
LOG=$FC/tests/out/runner-tests.log
SCR=${TMPDIR:-/tmp}/fc-runner-shim.$$
rm -rf "$SCR"; mkdir -p "$SCR/bin" "$SCR/fc" || exit 1
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
cmp -s "$SCR/fc/runner/run_fc.sh" "$FC/runner/run_fc.sh" || { echo "runner copy differs" | tee "$LOG"; exit 1; }
mkdir -p "$(dirname "$LOG")"; : > "$LOG"
echo "runner under test: sha256 $(sha256sum "$FC/runner/run_fc.sh" | cut -d' ' -f1) (copy identical)" | tee -a "$LOG"
PASSN=0; FAILN=0
FC1='{"item":"fc1","gpu_hours":1.2,"attempt":1}'
OTHER='{"item":"hookval","gpu_hours":0.795,"attempt":2}'
run_case() {   # name expected_rc regex attempt costs_content [env...]
  local NAME=$1 ERC=$2 RX=$3 ATT=$4 COSTS=$5; shift 5
  printf '%s' "$COSTS" > "$SCR/fc/costs.jsonl"
  local OUTT; OUTT=$(env -i PATH="$SCR/bin:/usr/local/bin:/usr/bin:/bin" HOME="$HOME" DRY_RUN=1 "$@" \
                     bash "$SCR/fc/runner/run_fc.sh" $ATT 2>&1); local RC=$?
  local OK=1
  [ "$RC" = "$ERC" ] || OK=0
  echo "$OUTT" | grep -qE "$RX" || OK=0
  if [ -s "$SCR/refusals" ]; then OK=0; fi
  if [ $OK = 1 ]; then PASSN=$((PASSN + 1)); echo "PASS $NAME (rc $RC)" | tee -a "$LOG"
  else FAILN=$((FAILN + 1)); echo "FAIL $NAME (rc $RC, expected $ERC /$RX/)" | tee -a "$LOG"; fi
  echo "$OUTT" | sed 's/^/    | /' >> "$LOG"
  [ -s "$SCR/refusals" ] && { sed 's/^/    shim: /' "$SCR/refusals" >> "$LOG"; : > "$SCR/refusals"; }
  rm -rf "$SCR/fc/run-fc1"
}
run_case R1-dryrun-pass 0 "DRY_RUN: budget gate, single attempt, disk, frozen hashes, image pin, client plan and plan check passed \(plan_check: 240 requests, 0 failures\)" "" "" APPROVED_GPU_H=4.29
run_case R2-budget-line 0 "spent 0.0 \+ this attempt's cap 4.286 \(wall 7714s\) <= approved 4.29 GPU-h" "" "" APPROVED_GPU_H=4.29
run_case R3-no-approval 2 "REFUSED: set APPROVED_GPU_H" "" ""
run_case R4-over-ceiling 2 "REFUSED: APPROVED_GPU_H=4.3 is not in \(0, 4.29\]" "" "" APPROVED_GPU_H=4.3
run_case R5-attempt-2 2 "REFUSED: single attempt only \(got attempt '2'\)" 2 "" APPROVED_GPU_H=4.29
run_case R6-attempt-recorded 2 "REFUSED: 1 attempt\(s\) already recorded for fc1 \(single attempt; no automatic additional attempt\)" "" "$FC1
" APPROVED_GPU_H=4.29
run_case R7-budget-below-min-wall 2 "REFUSED: approved 2.4 GPU-h gives 4320s wall < 4477s needed" "" "" APPROVED_GPU_H=2.4
run_case R8-other-items-ignored 0 "spent 0.0 \+ this attempt's cap 4.286" "" "$OTHER
" APPROVED_GPU_H=4.29
run_case R9-wallcap-env-capped 0 "\(wall 7722s\)" "" "" APPROVED_GPU_H=4.29 WALL_CAP=99999
run_case R10-wrong-image-id 2 "ABORT: local/glm53-v0.9.1-b1-forcehook-mg2:local is sha256:dead, expected" "" "" APPROVED_GPU_H=4.29 SHIM_IMG_ID=sha256:dead
run_case R11-disk-insufficient 2 "ABORT: /data has 150 GB free < 200 GB" "" "" APPROVED_GPU_H=4.29 SHIM_DF_GB=150
run_case R12-negative-approval 2 "REFUSED: APPROVED_GPU_H=-1 is not in" "" "" APPROVED_GPU_H=-1
cp -p "$SCR/fc/analysis/analyze_fc.py" "$SCR/analyze.bak"; echo "# tamper" >> "$SCR/fc/analysis/analyze_fc.py"
run_case R13-frozen-hash-mismatch 2 "ABORT: frozen hashes differ" "" "" APPROVED_GPU_H=4.29
cp -p "$SCR/analyze.bak" "$SCR/fc/analysis/analyze_fc.py"
cp -p "$SCR/fc/fixtures/ids/L4.json.gz" "$SCR/ids.bak"; printf 'x' >> "$SCR/fc/fixtures/ids/L4.json.gz"
run_case R14-frozen-ids-tampered 2 "ABORT: frozen hashes differ" "" "" APPROVED_GPU_H=4.29
cp -p "$SCR/ids.bak" "$SCR/fc/fixtures/ids/L4.json.gz"
mkdir -p "$SCR/fc/run-fc1"; printf '' > "$SCR/fc/costs.jsonl"
OUTT=$(env -i PATH="$SCR/bin:/usr/local/bin:/usr/bin:/bin" HOME="$HOME" DRY_RUN=1 APPROVED_GPU_H=4.29 bash "$SCR/fc/runner/run_fc.sh" 2>&1); RC=$?
if [ "$RC" = 2 ] && echo "$OUTT" | grep -q "ABORT: .*run-fc1 exists"; then PASSN=$((PASSN + 1)); echo "PASS R15-existing-run-dir (rc $RC)" | tee -a "$LOG"
else FAILN=$((FAILN + 1)); echo "FAIL R15-existing-run-dir (rc $RC)" | tee -a "$LOG"; fi
echo "$OUTT" | sed 's/^/    | /' >> "$LOG"
rm -rf "$SCR/fc/run-fc1"
echo "runner tests: $PASSN passed, $FAILN failed" | tee -a "$LOG"
[ "$FAILN" = 0 ]
