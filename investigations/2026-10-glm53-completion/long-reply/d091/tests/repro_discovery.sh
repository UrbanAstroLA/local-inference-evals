#!/usr/bin/env bash
set -u
# Proof that the discovery pipeline is unchanged: run byte-identical copies of discovery's build_index.py, trajectory.py,
# periods.py and analyse.py on the rep070-F raw file, in a scratch work dir inside this package, and compare every output
# with long-reply/discovery/out byte for byte. Zero GPU: trajectory.py and periods.py run in the same CPU-only container as
# discovery (--network none, CUDA_VISIBLE_DEVICES empty, no --gpus, inputs read-only); build_index.py and analyse.py run on
# the host (stdlib). Writes ids, metrics and comparison lines only (discovery's outputs hold no text).
# Not set -e: every exit code is checked. Usage: tests/repro_discovery.sh   (about 20-25 min of CPU)
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H")
case "$PKG" in <runs>/long-reply/d091) ;; *) echo "ABORT: unexpected package path $PKG"; exit 1 ;; esac
DISC=<runs>/long-reply/discovery
RAW=<runs>/rep070/run-F/raw
REPO=<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1
IMG_TAG=local/glm53-v0.7.0-kpoolring:local
IMG_ID=sha256:f68e47a0eedc4895478c28750620e9529fe6fb76964913fa3aef83ea4ab1adc0
W="$PKG/tests/repro-discovery"
REPORT="$PKG/tests/repro-discovery.txt"
[ -n "$W" ] && case "$W" in "$PKG"/tests/repro-discovery) ;; *) echo "ABORT: bad work dir"; exit 1 ;; esac
[ -e "$W" ] && { echo "ABORT: $W exists (remove it first)"; exit 1; }
GOT=$(docker image inspect "$IMG_TAG" --format '{{.Id}}' 2>/dev/null)
[ "$GOT" = "$IMG_ID" ] || { echo "ABORT: $IMG_TAG is '$GOT', not $IMG_ID"; exit 1; }
mkdir -p "$W/scripts" "$W/out" || exit 1
for f in build_index.py trajectory.py periods.py analyse.py; do cp -p "$DISC/scripts/$f" "$W/scripts/$f" || exit 1; done
cp -p "$DISC/PRESPEC.md" "$W/PRESPEC.md" || exit 1
{
echo "# discovery reproduction run $(date -Is) (CPU only; zero GPU)"
echo "## byte identity of the copies (this package's scripts/ vs discovery/scripts/ vs the run copies)"
for f in build_index.py trajectory.py periods.py analyse.py; do
  a=$(sha256sum "$DISC/scripts/$f" | cut -c1-64); b=$(sha256sum "$PKG/scripts/$f" | cut -c1-64); c=$(sha256sum "$W/scripts/$f" | cut -c1-64)
  [ "$a" = "$b" ] && [ "$b" = "$c" ] && echo "SAME $f $a" || echo "DIFF $f $a $b $c"
done
a=$(sha256sum "$DISC/PRESPEC.md" | cut -c1-64); b=$(sha256sum "$PKG/PRESPEC.md" | cut -c1-64)
[ "$a" = "$b" ] && [ "$a" = 97040a5c6260b2447917dfaf48c87605c066ebc57de0199f42571ea01c96c4b7 ] && echo "SAME PRESPEC.md $a" || echo "DIFF PRESPEC.md $a $b"
} > "$REPORT"
python3 -B "$W/scripts/build_index.py" > "$W/build_index.log" 2>&1; echo "build_index rc=$?" >> "$REPORT"
run_ctr() { # script, extra env...
  local s=$1; shift
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp "$@" --user "$(id -u):$(id -g)" \
    --entrypoint python3 -v "$RAW":/raw:ro -v "$REPO":/repo:ro -v "$W":/work "$IMG_ID" "/work/scripts/$s"
}
run_ctr trajectory.py -e ONLY=capped -e TAG=capped > "$W/run_capped.log" 2>&1; rc=$?; echo "exit $rc" >> "$W/run_capped.log"; echo "trajectory capped rc=$rc" >> "$REPORT"
run_ctr trajectory.py -e TAG=all > "$W/out/run_all.log" 2>&1; rc=$?; echo "exit $rc" >> "$W/out/run_all.log"; echo "trajectory all rc=$rc" >> "$REPORT"
run_ctr periods.py > "$W/out/run_periods.log" 2>&1; rc=$?; echo "exit $rc" >> "$W/out/run_periods.log"; echo "periods rc=$rc" >> "$REPORT"
python3 -B "$W/scripts/analyse.py" > "$W/analyse.log" 2>&1; echo "analyse rc=$?" >> "$REPORT"
{
echo "## outputs vs discovery/out (byte for byte)"
N=0; S=0
for f in $(cd "$DISC/out" && ls); do
  [ "$f" = prespec.sha256 ] && { echo "N/A $f (written once when PRESPEC.md was frozen; not an output of the scripts)"; continue; }
  N=$((N+1))
  if cmp -s "$DISC/out/$f" "$W/out/$f"; then S=$((S+1)); echo "SAME $f $(sha256sum "$W/out/$f" | cut -c1-16)"; else echo "DIFF $f"; fi
done
X=$(cd "$W/out" && ls | while read -r f; do [ -e "$DISC/out/$f" ] || echo "$f"; done)
[ -n "$X" ] && echo "EXTRA files not in discovery/out: $X"
echo "identical: $S of $N discovery output files"
} >> "$REPORT"
cat "$REPORT"
grep -q "^identical: $N of $N" "$REPORT" && ! grep -q '^DIFF' "$REPORT"
