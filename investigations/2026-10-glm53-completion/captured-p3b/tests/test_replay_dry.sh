#!/bin/bash
# CPU-only (--dry) tests of the replay scripts on fabricated captures.  Every container is
# `docker run --rm --network none -e CUDA_VISIBLE_DEVICES=` (no --gpus) from the stock images by digest.
# Writes tests/out/replay-dry.txt; assertions in tests/check_dry.py.
set -u
T=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$T/.." && pwd)
O=$T/out
mkdir -p "$O"
LOG=$O/replay-dry.txt
: > "$LOG"
IMG91=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
IMG70=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629
P3=<runs>/moe-plan-p3
MR=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1
U="$(id -u):$(id -g)"
cpu() { docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --user "$U" -e HOME=/tmp -v "$ROOT":/p:ro -v "$P3/harness":/p3h:ro \
          -v "$P3/out/attempt-2":/p3out:ro -v "$O":/o "$@" 2>&1 | grep -v -iE "warn|cpp_extension"; }
run() { echo "----- $1" >> "$LOG"; shift; "$@" >> "$LOG" 2>&1; echo "rc=$?" >> "$LOG"; }
MOD=/p/build/files/vllm/p3b_capture_hooks.py
rm -rf "$O/fake" "$O/fake-ls" "$O/dry-"*.json "$O/fake-priv" "$O/fake-ls-priv" "$O/e2e"
run "make fake capture" cpu -e TRITON_INTERPRET=1 --entrypoint python3 "$IMG91" /p/tests/make_fake_capture.py "$MOD" /p3h/moe_plan_eq.py /o/fake
run "make fake capture (length-sensitive reference kernel)" cpu -e TRITON_INTERPRET=1 -e P3B_DRY_ATTN_LENSENSITIVE=1 \
  --entrypoint python3 "$IMG91" /p/tests/make_fake_capture.py "$MOD" /p3h/moe_plan_eq.py /o/fake-ls
run "attn dry" cpu -e TRITON_INTERPRET=1 --entrypoint python3 "$IMG91" /p/replay/replay_attn.py --dry --cap /o/fake/cap \
  --caplog /o/fake/caplog --out /o/dry-attn.json --priv /o/fake-priv
run "attn dry, length-sensitive kernel, minima 1 (DIFFERENCES path)" cpu -e TRITON_INTERPRET=1 -e P3B_DRY_ATTN_LENSENSITIVE=1 \
  -e P3B_DRY_MIN_ROWS_1=1 --entrypoint python3 "$IMG91" /p/replay/replay_attn.py --dry --cap /o/fake-ls/cap \
  --caplog /o/fake-ls/caplog --out /o/dry-attn-ls.json --priv /o/fake-ls-priv
run "attn refuses dry switches without --dry" cpu -e P3B_DRY_MIN_ROWS_1=1 --entrypoint python3 "$IMG91" /p/replay/replay_attn.py \
  --cap /o/fake/cap --caplog /o/fake/caplog --out /o/dry-attn-refused.json --priv /o/fake-priv
run "moe v091 dry (+ forced tile)" cpu --entrypoint python3 "$IMG91" /p/replay/replay_moe.py --engine v091 --dry --dry-skip-weights \
  --forced-tile --cap /o/fake/cap --caplog /o/fake/caplog --out /o/dry-moe-v091.json
run "moe v091 dry, forced tile computes as decode (PREDICTION FAILS path)" cpu -e P3B_DRY_FORCED_DIFF=1 --entrypoint python3 "$IMG91" \
  /p/replay/replay_moe.py --engine v091 --dry --dry-skip-weights --forced-tile --cap /o/fake/cap --caplog /o/fake/caplog \
  --out /o/dry-moe-v091-forceddiff.json
run "moe refuses the dry switch without --dry" cpu -e P3B_DRY_FORCED_DIFF=1 --entrypoint python3 "$IMG91" /p/replay/replay_moe.py \
  --engine v091 --cap /o/fake/cap --caplog /o/fake/caplog --out /o/dry-moe-refused.json
run "moe v070 dry" cpu --entrypoint python3 "$IMG70" /p/replay/replay_moe.py --engine v070 --dry --dry-skip-weights \
  --cap /o/fake/cap --caplog /o/fake/caplog --out /o/dry-moe-v070.json
run "moe v091 dry forced-tile probe (real layer-3 weights on CPU through the image code)" cpu -v "$MR":/model-repo:ro \
  --entrypoint python3 "$IMG91" /p/replay/replay_moe.py --engine v091 --dry --dry-forced-probe --forced-tile \
  --p3-json /p3out/v091.json --layers 3 --cap /o/fake/cap --caplog /o/fake/caplog --out /o/dry-moe-v091-probe.json
run "moe v070 with the v091 image is refused (exl3 hash)" cpu -v "$MR":/model-repo:ro --entrypoint python3 "$IMG91" \
  /p/replay/replay_moe.py --engine v070 --dry --cap /o/fake/cap --caplog /o/fake/caplog --out /o/dry-moe-wrongimage.json
# end to end: the in-container driver with production-like mounts (as run_replay.sh mounts them), --dry
mkdir -p "$O/e2e/o" "$O/e2e/po"
run "in-container driver, v091, --dry" docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 --user "$U" \
  -e HOME=/tmp -v "$ROOT/replay":/r:ro -v "$P3/harness":/p3h:ro -v "$P3/out/attempt-2":/p3out:ro -v "$MR":/model-repo:ro \
  -v "$O/fake/cap":/cap:ro -v "$O/fake/caplog":/caplog:ro -v "$O/e2e/o":/o -v "$O/e2e/po":/po \
  --entrypoint sh "$IMG91" /r/in_container.sh v091 600 attn,moe,forced --dry
python3 "$T/check_dry.py" "$O" | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
