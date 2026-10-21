#!/usr/bin/env bash
set -u
# Proof that B+'s prefix construction code runs unchanged through scripts/prefix_build_d091.py: with REQLOGS left at B+'s
# value (--bplus-defaults) and B+'s inputs mounted (rep070-F raw, discovery/out, B+'s selection.json), the wrapper rebuilds
# B+'s 12 prefixes; out/prefix_manifest.json must be byte-identical to bplus/out/prefix_manifest.json, and the verify runs
# in the 0.9.1 and B1 images must reproduce bplus/checks/prefix-verify-v091.json and -b1.json. CPU-only containers
# (--network none, CUDA_VISIBLE_DEVICES empty, no --gpus). The scratch work dir is under PRIVATE/ (token ids and prompt
# messages are written there, mode 0600) and is shredded on every exit path; only comparison lines are kept.
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H")
case "$PKG" in <runs>/long-reply/d091) ;; *) echo "ABORT: unexpected package path"; exit 1 ;; esac
B=<runs>
BP="$B/long-reply/bplus"
W="$PKG/PRIVATE/repro-bplus-build"; REPORT="$H/prefix-build-repro.txt"
[ -n "$W" ] && case "$W" in <runs>/long-reply/d091/PRIVATE/repro-bplus-build) ;; *) exit 1 ;; esac
[ -e "$W" ] && { echo "ABORT: $W exists"; exit 1; }
cleanup() { [ -n "$W" ] && [ -d "$W" ] && case "$W" in "$PKG"/PRIVATE/repro-bplus-build) find "$W" -type f -exec shred -u {} + 2>/dev/null; rm -rf -- "$W" ;; esac; }
trap cleanup EXIT
( umask 077; mkdir -p "$W/scripts" "$W/out" "$W/PRIVATE" "$W/checks" ) || exit 1
for f in trajectory.py build_prefixes.py prefix_build_d091.py; do cp -p "$PKG/scripts/$f" "$W/scripts/$f" || exit 1; done
cp -p "$BP/out/selection.json" "$W/out/selection.json" || exit 1
ctr() { # image engine mode
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e ENGINE="$2" \
    --user "$(id -u):$(id -g)" --entrypoint python3 \
    -v "$B/rep070/run-F/raw":/raw:ro -v <models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1:/repo:ro \
    -v <gpqa-runs>:/gpqa:ro -v "$B/d7/exposure/gpqa_prompt_tokens.json":/d7/gpqa_prompt_tokens.json:ro \
    -v "$B/long-reply/discovery/out":/disc:ro -v "$W":/work "$1" /work/scripts/prefix_build_d091.py "$3" --bplus-defaults
}
{
echo "# B+ prefix-construction reproduction $(date -Is) (CPU-only containers; zero GPU)"
echo "build_prefixes.py: package $(sha256sum "$PKG/scripts/build_prefixes.py" | cut -c1-16), B+ $(sha256sum "$BP/scripts/build_prefixes.py" | cut -c1-16)"
( umask 077; ctr sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629 v070 build ) > "$W/build.log" 2>&1; echo "build (0.7.0 image) rc=$?; $(grep -E 'checks pass' "$W/build.log")"
cmp -s "$W/out/prefix_manifest.json" "$BP/out/prefix_manifest.json" && echo "SAME out/prefix_manifest.json (byte-identical to bplus/out/prefix_manifest.json)" || echo "DIFF out/prefix_manifest.json"
[ "$(sha256sum "$W/PRIVATE/prefixes.json" | cut -c1-64)" = "$(sha256sum "$BP/PRIVATE/prefixes.json" | cut -c1-64)" ] && echo "SAME PRIVATE/prefixes.json (sha256 equal to B+'s; ids not shown)" || echo "DIFF PRIVATE/prefixes.json"
[ "$(stat -c %a "$W/PRIVATE/prefixes.json")" = 600 ] && echo "mode 0600 PRIVATE/prefixes.json" || echo "WRONG MODE PRIVATE/prefixes.json"
for e in v070 v091 b1; do
  case $e in v070) I=sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629 ;; v091) I=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07 ;; b1) I=sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a ;; esac
  ctr "$I" "$e" verify > "$W/verify-$e.log" 2>&1; echo "verify ($e image) rc=$?; $(grep -E 'checks pass' "$W/verify-$e.log")"
  cmp -s "$W/checks/prefix-verify-$e.json" "$BP/checks/prefix-verify-$e.json" && echo "SAME checks/prefix-verify-$e.json (byte-identical to B+'s)" || echo "DIFF checks/prefix-verify-$e.json"
done
} > "$REPORT"
cat "$REPORT"
! grep -qE '^DIFF|WRONG|rc=[1-9]' "$REPORT"
