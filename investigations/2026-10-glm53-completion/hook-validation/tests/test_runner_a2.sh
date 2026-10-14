#!/bin/bash
# Attempt-2 runner gates (PA2-13): DRY_RUN and refusal tests with shims. CPU only; nothing is launched.
#   * the runner under test is a byte-identical copy of runner/run_hookval.sh inside a shim tree that holds exactly
#     the files listed in SHA256SUMS (so the frozen-hash gate runs for real) plus a shim cost ledger;
#   * PATH shims: `docker` answers only `image inspect` (delegated to the real docker, read-only, or a planted id
#     via SHIM_IMG_ID) and REFUSES every other subcommand (run/exec/ps/stop...); `nvidia-smi` and `curl` refuse.
#     Every case is either refused by a gate or exits at DRY_RUN, before the pre-flight; a shim refusal would show.
# usage: test_runner_a2.sh            -> tests/out-a2/runner-tests.log, rc 0 iff every case behaves as expected
# No `set -e`: every case's rc is checked.
HV=$(cd "$(dirname "$0")/.." && pwd)
LOG=$HV/tests/out-a2/runner-tests.log
SCR=${TMPDIR:-/tmp}/hv-runner-shim.$$
rm -rf "$SCR"; mkdir -p "$SCR/bin" "$SCR/hv" || exit 1
trap 'rm -rf "$SCR"' EXIT
REAL_DOCKER=$(command -v docker)
cat > "$SCR/bin/docker" <<EOF
#!/bin/bash
if [ "\$1" = image ] && [ "\$2" = inspect ]; then
  if [ -n "\${SHIM_IMG_ID:-}" ] && [ "\${4:-}" = --format ]; then echo "\$SHIM_IMG_ID"; exit 0; fi
  exec "$REAL_DOCKER" "\$@"
fi
echo "SHIM-REFUSED: docker \$*" >&2; echo "SHIM-REFUSED docker \$*" >> "$SCR/refusals"; exit 97
EOF
for c in nvidia-smi curl; do
  printf '#!/bin/bash\necho "SHIM-REFUSED: %s $*" >&2; echo "SHIM-REFUSED %s" >> "%s/refusals"; exit 97\n' "$c" "$c" "$SCR" > "$SCR/bin/$c"
done
chmod +x "$SCR/bin/"*
# shim tree: exactly the SHA256SUMS-listed files (+ SHA256SUMS)
( cd "$HV" && cut -c67- SHA256SUMS | while read -r f; do cp --parents -p "$f" "$SCR/hv/" || echo "COPY FAIL $f"; done; cp -p SHA256SUMS "$SCR/hv/" )
cmp -s "$SCR/hv/runner/run_hookval.sh" "$HV/runner/run_hookval.sh" || { echo "runner copy differs" | tee "$LOG"; exit 1; }
A1LINE=$(grep '"item":"hookval"' "$HV/costs.jsonl" | grep '"attempt":1')
[ -n "$A1LINE" ] || { echo "no attempt-1 cost line" | tee "$LOG"; exit 1; }
: > "$LOG"
echo "runner under test: sha256 $(sha256sum "$HV/runner/run_hookval.sh" | cut -d' ' -f1) (copy identical)" | tee -a "$LOG"
PASSN=0; FAILN=0
run_case() {   # name expected_rc regex attempt costs_content [env...]
  local NAME=$1 ERC=$2 RX=$3 ATT=$4 COSTS=$5; shift 5
  printf '%s' "$COSTS" > "$SCR/hv/costs.jsonl"
  local OUTT; OUTT=$(env -i PATH="$SCR/bin:/usr/local/bin:/usr/bin:/bin" HOME="$HOME" DRY_RUN=1 "$@" \
                     bash "$SCR/hv/runner/run_hookval.sh" "$ATT" 2>&1); local RC=$?
  local OK=1
  [ "$RC" = "$ERC" ] || OK=0
  echo "$OUTT" | grep -qE "$RX" || OK=0
  if [ -s "$SCR/refusals" ]; then OK=0; fi
  if [ $OK = 1 ]; then PASSN=$((PASSN + 1)); echo "PASS $NAME (rc $RC)" | tee -a "$LOG"
  else FAILN=$((FAILN + 1)); echo "FAIL $NAME (rc $RC, expected $ERC /$RX/)" | tee -a "$LOG"; fi
  echo "$OUTT" | sed 's/^/    | /' >> "$LOG"
  [ -s "$SCR/refusals" ] && { sed 's/^/    shim: /' "$SCR/refusals" >> "$LOG"; : > "$SCR/refusals"; }
  rm -rf "$SCR/hv/run-a2"
}
K_LINE='{"item":"k-selection","gpu_hours":0.1417,"attempt":1}'
A1_HIGH=$(echo "$A1LINE" | sed 's/"gpu_hours":[0-9.]*/"gpu_hours":1.9/')
A2_FAKE=$(echo "$A1LINE" | sed 's/"attempt":1/"attempt":2/')
run_case T1-dryrun-pass 0 "DRY_RUN: budget gate, attempt count, frozen hashes, image pin and client plan passed" 2 "$A1LINE
" APPROVED_GPU_H=2.4
run_case T1b-budget-line 0 "spent 1.2117 \+ this attempt's cap 1.188 \(wall 2138s\) <= approved 2.4" 2 "$A1LINE
" APPROVED_GPU_H=2.4
run_case T2-no-approval 2 "REFUSED: set APPROVED_GPU_H" 2 "$A1LINE
"
run_case T3-over-ceiling 2 "REFUSED: APPROVED_GPU_H=2.5 exceeds" 2 "$A1LINE
" APPROVED_GPU_H=2.5
run_case T4-attempt-1 2 "REFUSED: this runner variant is attempt 2 only" 1 "$A1LINE
" APPROVED_GPU_H=2.4
run_case T5-attempt-3 2 "REFUSED: this runner variant is attempt 2 only" 3 "$A1LINE
" APPROVED_GPU_H=2.4
run_case T6-two-attempts-recorded 2 "REFUSED: 2 attempts already recorded" 2 "$A1LINE
$A2_FAKE
" APPROVED_GPU_H=2.4
run_case T7-ledger-missing-a1 2 "REFUSED: cost ledger holds 0 hookval attempt\(s\); attempt 2 needs exactly 1" 2 "" APPROVED_GPU_H=2.4
run_case T8-spent-too-high 2 "REFUSED: spent 1.9 of 2.4 GPU-h leaves 900s wall < 1500s" 2 "$A1_HIGH
" APPROVED_GPU_H=2.4
run_case T9-cumulative-gate 2 "REFUSED: spent 1.2117 of 2.0 GPU-h leaves 1418s wall < 1500s" 2 "$A1LINE
" APPROVED_GPU_H=2.0
run_case T10-other-items-ignored 0 "spent 1.2117 \+ this attempt's cap 1.188" 2 "$A1LINE
$K_LINE
" APPROVED_GPU_H=2.4
run_case T11-wallcap-env-capped 0 "\(wall 2138s\)" 2 "$A1LINE
" APPROVED_GPU_H=2.4 WALL_CAP=99999
run_case T12-wrong-image-id 2 "ABORT: local/glm53-v0.9.1-b1-forcehook-mg2:local is sha256:dead, expected" 2 "$A1LINE
" APPROVED_GPU_H=2.4 SHIM_IMG_ID=sha256:dead
cp -p "$SCR/hv/analysis/analyze_hookval.py" "$SCR/analyze.bak"; echo "# tamper" >> "$SCR/hv/analysis/analyze_hookval.py"
run_case T13-frozen-hash-mismatch 2 "ABORT: frozen hashes differ" 2 "$A1LINE
" APPROVED_GPU_H=2.4
cp -p "$SCR/analyze.bak" "$SCR/hv/analysis/analyze_hookval.py"
mkdir -p "$SCR/hv/run-a2"
printf '%s\n' "$A1LINE" > "$SCR/hv/costs.jsonl"
OUTT=$(env -i PATH="$SCR/bin:/usr/local/bin:/usr/bin:/bin" HOME="$HOME" DRY_RUN=1 APPROVED_GPU_H=2.4 bash "$SCR/hv/runner/run_hookval.sh" 2 2>&1); RC=$?
if [ "$RC" = 2 ] && echo "$OUTT" | grep -q "ABORT: .*run-a2 exists"; then PASSN=$((PASSN + 1)); echo "PASS T14-existing-run-dir (rc $RC)" | tee -a "$LOG"
else FAILN=$((FAILN + 1)); echo "FAIL T14-existing-run-dir (rc $RC)" | tee -a "$LOG"; fi
echo "$OUTT" | sed 's/^/    | /' >> "$LOG"
rm -rf "$SCR/hv/run-a2"
# default attempt argument is 2
printf '%s\n' "$A1LINE" > "$SCR/hv/costs.jsonl"
OUTT=$(env -i PATH="$SCR/bin:/usr/local/bin:/usr/bin:/bin" HOME="$HOME" DRY_RUN=1 APPROVED_GPU_H=2.4 bash "$SCR/hv/runner/run_hookval.sh" 2>&1); RC=$?
if [ "$RC" = 0 ] && [ ! -s "$SCR/refusals" ]; then PASSN=$((PASSN + 1)); echo "PASS T15-default-attempt-is-2 (rc $RC)" | tee -a "$LOG"
else FAILN=$((FAILN + 1)); echo "FAIL T15-default-attempt-is-2 (rc $RC)" | tee -a "$LOG"; fi
echo "$OUTT" | sed 's/^/    | /' >> "$LOG"
echo "runner tests: $PASSN passed, $FAILN failed" | tee -a "$LOG"
[ "$FAILN" = 0 ]
