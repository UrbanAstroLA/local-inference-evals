#!/bin/bash
# GATE-1 step of the first-contrast runner (PLAN.md amendment 22 item 4): called by client_fc.py after stage A (V0,
# LW1, LW2, C1, C2, CV) while stage L runs. CPU only: copies stage A's hook records out of the serving container
# (docker exec + tar; nothing in the container is modified), then runs the frozen analysis/gate_fc.py in a CPU
# container of the B1 image (--network none, no --gpus, CUDA_VISIBLE_DEVICES=, host uid). rc 0 iff verdict PASS.
# No `set -e`: every step's rc is checked. usage: gate_step.sh OUTDIR
OUT=$1
HERE=$(cd "$(dirname "$0")" && pwd)
FC=$(cd "$HERE/.." && pwd)
CN=${FC_CONTAINER:-glm53-flash-exl3-b12x-vllm}     # FC_CONTAINER: test override only (tests/test_gate_step.sh)
IMG_CPU=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a   # local/glm53-v0.9.1-b1-d7:local
G=$OUT/gate1
T0=$(date +%s)
rm -rf "$G"; mkdir -p "$G/forcewin" "$G/mapguard" || exit 20
cp "$OUT/client.jsonl" "$G/client.jsonl" || exit 21
docker exec "$CN" sh -c 'cd /tmp/forcewin && ls' > "$G/container-files.txt" 2>"$G/copy.err"; R1=$?
grep -E '^(req|snap)_(LW1|LW2|C1|C2|CV)-' "$G/container-files.txt" > "$G/files.txt"
grep -E '^fw_r[0-9]+_p[0-9]+\.jsonl$' "$G/container-files.txt" >> "$G/files.txt"
docker exec -i "$CN" sh -c 'cd /tmp/forcewin && tar -cf - -T -' < "$G/files.txt" 2>>"$G/copy.err" \
  | tar -C "$G/forcewin" -xf - 2>>"$G/copy.err"; R2=${PIPESTATUS[0]}
docker exec "$CN" sh -c 'cd /tmp/mapguard && tar -cf - .' 2>>"$G/copy.err" | tar -C "$G/mapguard" -xf - 2>>"$G/copy.err"
R3=${PIPESTATUS[0]}
NEXP=$(wc -l < "$G/files.txt"); NGOT=$(ls "$G/forcewin" | wc -l)
echo "copy: rc $R1/$R2/$R3, files $NGOT of $NEXP, $(du -sh "$G" | cut -f1), $(( $(date +%s) - T0 )) s" | tee "$G/copy.txt"
if [ "$R1" != 0 ] || [ "$R2" != 0 ] || [ "$NGOT" -ne "$NEXP" ] || [ "$NEXP" -lt 100 ]; then
  echo "GATE-1 copy incomplete: verdict FAIL"; printf '{"gate":"GATE-1","verdict":"FAIL","reading":"stage-A records not copied"}\n' > "$G/gate1.json"; exit 22
fi
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= -v "$G":/run \
  -v "$FC":/fc:ro --entrypoint python3 "$IMG_CPU" /fc/analysis/gate_fc.py /run /fc 2>&1 | grep -v cpp_extension \
  > "$G/gate1-console.txt"
RC=${PIPESTATUS[0]}
echo "gate_fc rc $RC after $(( $(date +%s) - T0 )) s" | tee -a "$G/copy.txt"
head -1 "$G/gate1-console.txt"
exit $RC
