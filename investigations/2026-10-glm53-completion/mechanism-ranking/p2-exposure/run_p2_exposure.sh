#!/bin/bash
# CPU only (no GPU, no network): tail-ring exposure through each image's real kernels under TRITON_INTERPRET=1.
#   ./run_p2_exposure.sh          batch 1: P2 (ring 4) reintroduced on B1's kernels; 0.7.0 at served acceptance
#   STEP6=1 ./run_p2_exposure.sh  batch 2 (added after batch 1): 0.9.1 kernels under 0.7.0's DCP2 layout, x3
D=$(cd "$(dirname "$0")" && pwd); O=$D/out; mkdir -p "$O"
B1=local/glm53-v0.9.1-b1-forcehook-mg2@sha256:ac05230bd94b3114cb6fb202b06402f9e2d45a61b20f041f314e045f967e97ee
V070=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629
V091=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
run() { docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -v "$D":/w:ro -v "$O":/o --entrypoint python3 "$@" 2>&1 \
  | grep -v "^WARNING\|cpp_extension\|No CUDA runtime"; }
S="--conc 1 2 8 --seeds 0 1 2"
if [ "${STEP6:-0}" = 1 ]; then
  run $V091 /w/p2_exposure.py --engine 091 --dcp 2 --ring 8 --K 3 --subpages 14 --accept 0.77 $S --out /o/v091-dcp2-r8-k3-a0.77.json > $O/v091-dcp2-r8-k3-a0.77.log
  tail -n 1 $O/v091-dcp2-r8-k3-a0.77.log
  exit 0
fi
run $B1 /w/p2_exposure.py --engine b1 --ring 8 --K 3 --accept 0.77 --arms positions $S --out /o/b1-r8-k3-a0.77.json > $O/b1-r8-k3-a0.77.log &
run $B1 /w/p2_exposure.py --engine b1 --ring 4 --K 3 --accept 0.77 --arms positions $S --out /o/b1-r4-k3-a0.77.json > $O/b1-r4-k3-a0.77.log &
run $B1 /w/p2_exposure.py --engine b1 --ring 4 --K 3 --accept 0.6  --arms positions $S --out /o/b1-r4-k3-a0.6.json  > $O/b1-r4-k3-a0.6.log &
run $B1 /w/p2_exposure.py --engine b1 --ring 4 --K 3 --accept 0.85 --arms positions $S --out /o/b1-r4-k3-a0.85.json > $O/b1-r4-k3-a0.85.log &
run $B1 /w/p2_exposure.py --engine b1 --ring 4 --K 5 --accept 0.72 --arms positions $S --out /o/b1-r4-k5-a0.72.json > $O/b1-r4-k5-a0.72.log &
run $B1 /w/p2_exposure.py --engine b1 --ring 8 --K 5 --accept 0.72 --arms positions $S --out /o/b1-r8-k5-a0.72.json > $O/b1-r8-k5-a0.72.log &
run $V070 /w/p2_exposure.py --engine 070 --accept 0.72 $S --out /o/v070-a0.72.json > $O/v070-a0.72.log &
wait
tail -n 1 "$O"/*.log
