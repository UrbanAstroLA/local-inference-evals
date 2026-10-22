#!/usr/bin/env bash
set -u
# Long-reply pipeline for d091 (PROPOSAL.md sections 6-7). ZERO GPU: host stdlib steps plus CPU-only containers
# (--network none, CUDA_VISIBLE_DEVICES empty, no --gpus, inputs read-only); never starts a model server and sends no request.
#   1 index     scripts/index_d091.py (host)                       -> WORK/out/rep070F_index.json (the d091 index; name as discovery)
#   2 metrics   scripts/trajectory.py, TAG=all (container)          -> WORK/out/replies_all.json, windows_all.jsonl   [unchanged copy]
#   3 post hoc  scripts/periods.py (container)                      -> WORK/out/periods_all.json                      [unchanged copy]
#   4 tables    scripts/compare_d091.py (host)                      -> WORK/out/tables_d091.md, tables_d091.json
#   5 select    scripts/prefix_select_d091.py (host)                -> WORK/out/selection.json
#   6 build     scripts/prefix_build_d091.py build, 0.7.0 image      -> WORK/PRIVATE/prefixes.json, prompts.json (0600); WORK/out/prefix_manifest.json
#   7 verify    prefix_build_d091.py verify, 0.9.1 and B1 images     -> WORK/checks/prefix-verify-v091.json, -b1.json
#   8 privacy   checks/privacy_scan.py (host)                        -> WORK/out/privacy-scan.txt
# Not set -e: every step's exit code is recorded in WORK/out/pipeline.log; a failed step stops the steps that need it.
# usage: run_pipeline.sh [WORK RAW_DIR REQLOG_NAME SEED ANALYSIS_JSON]
#   defaults (the real run): WORK = this package, RAW_DIR = WORK/PRIVATE/raw, REQLOG_NAME = the d091 session-1 request log,
#   SEED = 70705, ANALYSIS_JSON = WORK/run/result-d091.json. Tests pass a scratch WORK under PRIVATE/ and a stand-in raw.
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H")
case "$PKG" in <runs>/long-reply/d091b) ;; *) echo "ABORT: unexpected package path $PKG"; exit 1 ;; esac
WORK=${1:-$PKG}; RAWD=${2:-$PKG/PRIVATE/raw}; REQ=${3:-glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091b.s1.requests.jsonl}; SEED=${4:-70705}
AN=${5:-$PKG/run/result-d091.json}
case "$WORK" in "$PKG"|"$PKG"/PRIVATE/*) ;; *) echo "ABORT: WORK must be the package or a folder under its PRIVATE/"; exit 1 ;; esac
[ -n "$WORK" ] && [ -d "$WORK" ] && [ -d "$WORK/PRIVATE" ] && [ -d "$WORK/out" ] && [ -d "$WORK/checks" ] || { echo "ABORT: WORK needs PRIVATE/, out/ and checks/"; exit 1; }
[ "$(stat -c %a "$WORK/PRIVATE")" = 700 ] || { echo "ABORT: $WORK/PRIVATE must be mode 0700"; exit 1; }
[ -d "$RAWD" ] || { echo "ABORT: raw directory $RAWD missing"; exit 1; }
for f in trajectory.py periods.py build_prefixes.py prefix_build_d091.py; do [ -f "$WORK/scripts/$f" ] || { echo "ABORT: $WORK/scripts/$f missing"; exit 1; }; done
B=<runs>
REPO=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1
IMG_TRAJ=sha256:f68e47a0eedc4895478c28750620e9529fe6fb76964913fa3aef83ea4ab1adc0      # local/glm53-v0.7.0-kpoolring:local (discovery's)
IMG_V070=sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629      # B+ build image
IMG_V091=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
IMG_B1=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a
LOG="$WORK/out/pipeline.log"
echo "$(date -Is) pipeline start; WORK $WORK; raw $RAWD; request log $REQ; seed $SEED" >> "$LOG"
st() { echo "$(date -Is) step $1 rc=$2" >> "$LOG"; }
ctr() { # image script envs... -- args
  local img=$1 s=$2; shift 2
  local envs=() ; while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=(-e "$1"); shift; done; [ $# -gt 0 ] && shift
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 "${envs[@]}" \
    --user "$(id -u):$(id -g)" --entrypoint python3 \
    -v "$RAWD":/raw:ro -v "$REPO":/repo:ro -v <gpqa-runs>:/gpqa:ro \
    -v "$B/d7/exposure/gpqa_prompt_tokens.json":/d7/gpqa_prompt_tokens.json:ro \
    -v "$WORK/out":/disc:ro -v "$WORK":/work "$img" "/work/scripts/$s" "$@"
}
python3 -B "$PKG/scripts/index_d091.py" --raw-dir "$RAWD" --files s1 --reqlogs "<gpqa-runs>/$REQ" --seeds "$SEED" --expect 198 \
  --out-dir "$WORK/out" > "$WORK/out/index.log" 2>&1; rc=$?; st index $rc
[ "$rc" = 0 ] || { echo "index failed: stopping" >> "$LOG"; exit 1; }
ctr "$IMG_TRAJ" trajectory.py TAG=all > "$WORK/out/run_all.log" 2>&1; rc=$?; echo "exit $rc" >> "$WORK/out/run_all.log"; st trajectory $rc
[ "$rc" = 0 ] || { echo "trajectory failed: stopping" >> "$LOG"; exit 1; }
ctr "$IMG_TRAJ" periods.py > "$WORK/out/run_periods.log" 2>&1; rc=$?; echo "exit $rc" >> "$WORK/out/run_periods.log"; st periods $rc
[ "$rc" = 0 ] || { echo "periods failed: stopping" >> "$LOG"; exit 1; }
python3 -B "$PKG/scripts/compare_d091.py" --d091-out "$WORK/out" --analysis "$AN" --out-md "$WORK/out/tables_d091.md" \
  --out-json "$WORK/out/tables_d091.json" ${PIPE_LABEL:+--label "$PIPE_LABEL"} > "$WORK/out/compare.log" 2>&1; st compare $?
python3 -B "$PKG/scripts/prefix_select_d091.py" "$WORK/out" > "$WORK/out/select.log" 2>&1; rc=$?; st select $rc
N=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['summary']['selected'])" "$WORK/out/selection.json" 2>/dev/null || echo 0)
if [ "$rc" = 0 ] && [ "$N" -gt 0 ]; then
  ( umask 077; ctr "$IMG_V070" prefix_build_d091.py ENGINE=v070 -- build --seed "$SEED" --reqlog "$REQ" ) > "$WORK/out/build.log" 2>&1; rc=$?; st build $rc
  if [ "$rc" = 0 ]; then
    ctr "$IMG_V091" prefix_build_d091.py ENGINE=v091 -- verify --seed "$SEED" --reqlog "$REQ" > "$WORK/out/verify-v091.log" 2>&1; st verify-v091 $?
    if docker image inspect "$IMG_B1" > /dev/null 2>&1; then
      ctr "$IMG_B1" prefix_build_d091.py ENGINE=b1 -- verify --seed "$SEED" --reqlog "$REQ" > "$WORK/out/verify-b1.log" 2>&1; st verify-b1 $?
    else echo "B1 image not present: verify-b1 skipped" >> "$LOG"; fi
  fi
else
  echo "no eligible prefix selected (selected $N): build skipped" >> "$LOG"
fi
for f in "$WORK/PRIVATE"/*.json; do [ -f "$f" ] && chmod 600 "$f"; done
python3 -B "$PKG/checks/privacy_scan.py" "$WORK" "$RAWD" > "$WORK/out/privacy-scan.txt" 2>&1; st privacy $?
echo "$(date -Is) pipeline done" >> "$LOG"
tail -12 "$LOG"
