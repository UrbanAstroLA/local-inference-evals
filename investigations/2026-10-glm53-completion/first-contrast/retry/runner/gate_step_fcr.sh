#!/bin/bash
# GATE-1R step of the retry runner (PLAN.md amendment 22d s.9; RETRY-PROPOSAL.md s.6). Derived from
# ../../runner/gate_step.sh (attempt 1): called by client_fcr.py after stage A (V0, LW1, LW2, the s1 and s2 builds and
# touch passes, LC1, LC2, C1, CV, C2, PS-a, PS-b). CPU only: copies stage A's hook records AND the guard's scheduler
# log out of the serving container (docker exec + tar; nothing in the container is modified), then runs the frozen
# analysis/gate_fcr.py in a CPU container of the B1 image (--network none, no --gpus, CUDA_VISIBLE_DEVICES=, host uid).
# The serving engine is idle meanwhile (the client waits: s2's build must survive to D2). rc 0 iff verdict PASS.
# No `set -e`: every step's rc is checked. usage: gate_step_fcr.sh OUTDIR
OUT=$1
HERE=$(cd "$(dirname "$0")" && pwd)
RETRY=$(cd "$HERE/.." && pwd)
FC=$(cd "$RETRY/.." && pwd)
CN=${FC_CONTAINER:-glm53-flash-exl3-b12x-vllm}     # FC_CONTAINER: test override only (tests/test_gate_step_fcr.sh)
IMG_CPU=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a   # local/glm53-v0.9.1-b1-d7:local
G=$OUT/gate1r
T0=$(date +%s)
rm -rf "$G"; mkdir -p "$G/forcewin" "$G/mapguard" || exit 20
cp "$OUT/client.jsonl" "$G/client.jsonl" || exit 21
docker exec "$CN" sh -c 'cd /tmp/forcewin && ls' > "$G/container-files.txt" 2>"$G/copy.err"; R1=$?
grep -E '^(req|snap)_(LW1|LW2|LC1|LC2|C1|C2|CV|PS-a|PS-b)-' "$G/container-files.txt" > "$G/files.txt"
grep -E '^fw_r[0-9]+_p[0-9]+\.jsonl$' "$G/container-files.txt" >> "$G/files.txt"
docker exec -i "$CN" sh -c 'cd /tmp/forcewin && tar -cf - -T -' < "$G/files.txt" 2>>"$G/copy.err" \
  | tar -C "$G/forcewin" -xf - 2>>"$G/copy.err"; R2=${PIPESTATUS[0]}
docker exec "$CN" sh -c 'cd /tmp/mapguard && tar -cf - .' 2>>"$G/copy.err" | tar -C "$G/mapguard" -xf - 2>>"$G/copy.err"
R3=${PIPESTATUS[0]}
NEXP=$(wc -l < "$G/files.txt"); NGOT=$(ls "$G/forcewin" | wc -l); NMS=$(ls "$G/mapguard" | grep -c '^ms_p')
echo "copy: rc $R1/$R2/$R3, files $NGOT of $NEXP, scheduler logs $NMS, $(du -sh "$G" | cut -f1), $(( $(date +%s) - T0 )) s" | tee "$G/copy.txt"
if [ "$R1" != 0 ] || [ "$R2" != 0 ] || [ "$R3" != 0 ] || [ "$NGOT" -ne "$NEXP" ] || [ "$NEXP" -lt 100 ] || [ "$NMS" -lt 1 ]; then
  echo "GATE-1R copy incomplete: verdict FAIL"
  printf '{"gate":"GATE-1R","verdict":"FAIL","reading":"stage-A records or scheduler log not copied","stop_classes":["missing coverage"]}\n' > "$G/gate1r.json"; exit 22
fi
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= -v "$G":/run \
  -v "$FC":/fc:ro --entrypoint python3 "$IMG_CPU" /fc/retry/analysis/gate_fcr.py /run /fc 2>&1 | grep -v cpp_extension \
  > "$G/gate1r-console.txt"
RC=${PIPESTATUS[0]}
echo "gate_fcr rc $RC after $(( $(date +%s) - T0 )) s" | tee -a "$G/copy.txt"
head -1 "$G/gate1r-console.txt"
exit $RC
