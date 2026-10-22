#!/usr/bin/env bash
set -u
# E check (rewritten under amendment 22ah: inputs from ref/ and PRIVATE/src-d091/, rendering on both images added): D's
# prefixes P01-P08 rebuilt byte-identically with the unchanged build code, and rendered identically
# on both engines' images. ZERO GPU: CPU-only containers (--network none, CUDA_VISIBLE_DEVICES empty, no --gpus, inputs
# read-only), exactly as d091's run_pipeline.sh steps 6-7.
#   build   0.7.0 image: B+'s build_prefixes.py (1ebbec2e..., unchanged) through d091's wrapper prefix_build_d091.py
#           (unchanged; REQLOGS set by its --seed 70704 --reqlog <d091 request log> arguments) on d091's out/selection.json,
#           raw bodies and index. The rebuilt out/prefix_manifest.json must be byte-identical to d091's (= ref/d091-prefix_manifest.json),
#           and the rebuilt PRIVATE files must hash to d091's (= PRIVATE/src-d091/) and to the manifest's private_*_sha256.
#   verify  the same wrapper in verify mode in the 0.7.0 and 0.9.1 images (each image's libraries, the template and snapshot
#           the engine serves): every check must pass; the outputs are kept as checks/prefix-verify-<engine>.json (hashes only)
#           and the 0.9.1 one must be byte-identical to d091's checks/prefix-verify-v091.json.
# The scratch WORK lies under PRIVATE/ and is shredded on every exit path. Prints hashes and counts only.
# usage: tests/prefix_build_repro.sh > tests/prefix-build-repro.txt
umask 077
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H")
EC=<runs>/long-reply/e-cross
case "$PKG" in "$EC") ;; *) echo "ABORT: unexpected package path $PKG"; exit 2 ;; esac
B=<runs>
D2=$B/long-reply/d091
SEED=70704; REQ=glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091.s1.requests.jsonl
S=$PKG/PRIVATE/repro-build
case "$S" in "$EC"/PRIVATE/repro-build) ;; *) echo "ABORT: scratch $S not inside PRIVATE/"; exit 2 ;; esac
[ -e "$S" ] && { echo "ABORT: $S exists (earlier attempt); remove it first"; exit 2; }
cleanup() {
  case "$S" in "$EC"/PRIVATE/repro-build)
    if [ -d "$S" ]; then find "$S" -type f -exec shred -u {} + 2>/dev/null; rm -rf -- "$S"; fi ;;
  esac
}
trap cleanup EXIT INT TERM
mkdir -p "$S/scripts" "$S/out" "$S/checks" "$S/PRIVATE" || exit 1
chmod 700 "$S/PRIVATE"
for f in trajectory.py build_prefixes.py prefix_build_d091.py; do cp "$PKG/scripts/$f" "$S/scripts/$f" || exit 1; done
cp "$PKG/ref/d091-selection.json" "$S/out/selection.json" || exit 1
echo "# D prefix construction repro and rendering, $(date -Is); CPU only"
for f in scripts/trajectory.py scripts/build_prefixes.py out/selection.json; do
  a=$(sha256sum "$S/$f" | cut -c1-64); b=$(sha256sum "$D2/$f" | cut -c1-64)
  echo "input $f sha256 ${a:0:16} (d091 ${b:0:16}) $([ "$a" = "$b" ] && echo identical || echo DIFFERENT)"
done
echo "input scripts/prefix_build_d091.py sha256 $(sha256sum "$S/scripts/prefix_build_d091.py" | cut -c1-16) (d091's wrapper, unchanged; seed and request log passed as arguments)"
ctr() { # image engine mode
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e ENGINE="$2" \
    --user "$(id -u):$(id -g)" --entrypoint python3 \
    -v "$D2/PRIVATE/raw":/raw:ro -v <models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1:/repo:ro \
    -v <gpqa-runs>:/gpqa:ro -v "$B/d7/exposure/gpqa_prompt_tokens.json":/d7/gpqa_prompt_tokens.json:ro \
    -v "$D2/out":/disc:ro -v "$S":/work \
    "$1" /work/scripts/prefix_build_d091.py "$3" --seed "$SEED" --reqlog "$REQ"
}
IMG070=sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629
IMG091=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
ctr "$IMG070" v070 build > "$S/build.log" 2>&1; rcb=$?
echo "build rc=$rcb: $(grep -E 'checks pass' "$S/build.log")"
ctr "$IMG070" v070 verify > "$S/verify-v070.log" 2>&1; rc70=$?
echo "verify v070 rc=$rc70: $(grep -E 'checks pass' "$S/verify-v070.log")"
ctr "$IMG091" v091 verify > "$S/verify-v091.log" 2>&1; rc91=$?
echo "verify v091 rc=$rc91: $(grep -E 'checks pass' "$S/verify-v091.log")"
grep -hE '^FAIL' "$S"/*.log | cut -c1-200
for e in v070 v091; do [ -f "$S/checks/prefix-verify-$e.json" ] && cp "$S/checks/prefix-verify-$e.json" "$PKG/checks/prefix-verify-$e.json" && chmod 644 "$PKG/checks/prefix-verify-$e.json"; done
PASS=0; FAIL=0
t() { if [ "$2" = "$3" ]; then PASS=$((PASS+1)); echo "PASS $1 (${2:0:16})"; else FAIL=$((FAIL+1)); echo "FAIL $1 (${2:0:16} vs ${3:0:16})"; fi; }
hs() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
t "rebuilt out/prefix_manifest.json byte-identical to d091's" "$(hs "$S/out/prefix_manifest.json")" "$(hs "$D2/out/prefix_manifest.json")"
t "rebuilt out/prefix_manifest.json byte-identical to ref/d091-prefix_manifest.json" "$(hs "$S/out/prefix_manifest.json")" "$(hs "$PKG/ref/d091-prefix_manifest.json")"
t "rebuilt PRIVATE/prefixes.json = d091's = PRIVATE/src-d091/" "$(hs "$S/PRIVATE/prefixes.json")" "$(hs "$D2/PRIVATE/prefixes.json")"
t "rebuilt PRIVATE/prompts.json = d091's = PRIVATE/src-d091/" "$(hs "$S/PRIVATE/prompts.json")" "$(hs "$D2/PRIVATE/prompts.json")"
t "PRIVATE/src-d091 = d091's private files" "$(hs "$PKG/PRIVATE/src-d091/prefixes.json") $(hs "$PKG/PRIVATE/src-d091/prompts.json")" "$(hs "$D2/PRIVATE/prefixes.json") $(hs "$D2/PRIVATE/prompts.json")"
M=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['private_file_sha256'], d['private_prompts_sha256'])" "$D2/out/prefix_manifest.json")
t "d091 manifest private_*_sha256 = rebuilt private files" "$M" "$(hs "$S/PRIVATE/prefixes.json") $(hs "$S/PRIVATE/prompts.json")"
t "build exit code" "$rcb" "0"
t "verify exit code, 0.7.0 image" "$rc70" "0"
t "verify exit code, 0.9.1 image" "$rc91" "0"
t "verify 0.9.1 output byte-identical to d091's checks/prefix-verify-v091.json" "$(hs "$PKG/checks/prefix-verify-v091.json")" "$(hs "$D2/checks/prefix-verify-v091.json")"
echo "D prefix build repro and rendering: $PASS pass, $FAIL fail"
[ "$FAIL" = 0 ]
