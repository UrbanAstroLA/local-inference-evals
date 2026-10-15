#!/bin/bash
# GATE-1 step test [A22-4]: runner/gate_step.sh against a STAND-IN container (CPU only: the B1 image running `sleep`,
# --network none, no --gpus, name fcgate-test-*, never glm53*), holding a synthetic run's hook and guard records in
# /tmp/forcewin and /tmp/mapguard exactly where the serving container keeps them. Checks: the stage-A file list is
# copied completely (and nothing of later phases), the gate runs in its CPU container, and the verdict / rc match the
# synthetic variant (good -> PASS rc 0; g-content -> FAIL rc 1). The stand-in is removed on every exit path.
# usage: test_gate_step.sh  -> tests/out/gate-step-tests.log. No `set -e`.
FC=$(cd "$(dirname "$0")/.." && pwd)
IMG=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a
LOG=$FC/tests/out/gate-step-tests.log
: > "$LOG"
PASSN=0; FAILN=0
CNS=()
trap 'for c in "${CNS[@]}"; do docker rm -f "$c" >/dev/null 2>&1; done' EXIT
for V in good g-content; do
  SRC=$FC/tests/out/synth-$V/full
  CN=fcgate-test-$$-$V
  CNS+=("$CN")
  W=${TMPDIR:-/tmp}/fc-gate-step.$$.$V
  rm -rf "$W"; mkdir -p "$W"
  docker run -d --name "$CN" --network none -e CUDA_VISIBLE_DEVICES= --entrypoint sleep "$IMG" 900 >/dev/null \
    || { echo "FAIL stand-in start" | tee -a "$LOG"; FAILN=$((FAILN+1)); continue; }
  docker exec "$CN" mkdir -p /tmp/forcewin /tmp/mapguard
  docker cp "$SRC/forcewin/." "$CN:/tmp/forcewin/" >/dev/null && docker cp "$SRC/mapguard/." "$CN:/tmp/mapguard/" >/dev/null
  # the client log as it stands when the gate starts: stage A only (V0, LW1, LW2, warm-s1, C1, warm-s2, C2, CV)
  grep -v -E '"phase": "(rewarm|warm-s3|LC|LD|D[0-9]|PS|C3|LW3)' "$SRC/client.jsonl" > "$W/client.jsonl"
  OUTT=$(FC_CONTAINER="$CN" bash "$FC/runner/gate_step.sh" "$W" 2>&1); RC=$?
  NEXP=$(wc -l < "$W/gate1/files.txt" 2>/dev/null); NGOT=$(ls "$W/gate1/forcewin" 2>/dev/null | wc -l)
  LATER=$(ls "$W/gate1/forcewin" 2>/dev/null | grep -cE '^(req|snap)_(LC|LD|D[0-9]|PS|C3|LW3)')
  V_=$(python3 -c "import json;print(json.load(open('$W/gate1/gate1.json'))['verdict'])" 2>/dev/null)
  ERC=0; EV=PASS; [ "$V" = g-content ] && { ERC=1; EV=FAIL; }
  if [ "$RC" = "$ERC" ] && [ "$V_" = "$EV" ] && [ "$NEXP" = "$NGOT" ] && [ "$LATER" = 0 ] && [ "${NEXP:-0}" -gt 100 ]; then
    PASSN=$((PASSN+1)); echo "PASS gate_step[$V]: rc $RC verdict $V_, copied $NGOT of $NEXP stage-A files, later-phase files $LATER" | tee -a "$LOG"
  else
    FAILN=$((FAILN+1)); echo "FAIL gate_step[$V]: rc $RC (want $ERC) verdict $V_ (want $EV), copied $NGOT of $NEXP, later $LATER" | tee -a "$LOG"
  fi
  echo "$OUTT" | sed 's/^/    | /' >> "$LOG"
  docker rm -f "$CN" >/dev/null 2>&1; rm -rf "$W"
done
docker ps -a --format '{{.Names}}' | grep -q "^fcgate-test-$$" && echo "WARNING: stand-in left behind" | tee -a "$LOG"
echo "gate-step tests: $PASSN passed, $FAILN failed" | tee -a "$LOG"
[ "$FAILN" = 0 ]
