#!/bin/sh
# INSIDE the P4 image (launched by g4/run_g4.sh on GPU0): G4a (env unset) then G4b (GLM53_P4_TILE set), each with a
# soft deadline from SOFT_S.  Mounts: /g g4/ (ro), /r captured-p3b/replay (ro), /p3h P3 harness (ro), /model-repo (ro),
# /cap captured tensors (ro), /caplog capture logs (ro), /o public outputs, /cc compile cache.
#   in_container_g4.sh SOFT_S
SOFT=$1
T0=$(date +%s)
left() { echo $(( SOFT - ($(date +%s) - T0) )); }
umask 077
unset GLM53_P4 GLM53_P4_TILE
A=$(left); [ "$A" -gt 400 ] && A=400
echo "== G4a soft ${A}s"
python3 /g/g4_moe.py --engine v091 --cap /cap --caplog /caplog --out /o/moe-g4a.json --deadline-s "$A"
echo "G4a rc=$?"
B=$(left)
echo "== G4b soft ${B}s"
[ "$B" -gt 60 ] && GLM53_P4_TILE=64,256,64,256 python3 /g/g4_moe.py --engine v091 --cap /cap --caplog /caplog \
    --out /o/moe-g4b.json --deadline-s "$B" --layers 3,23,44
echo "G4b rc=$?"
echo "in-container elapsed $(( $(date +%s) - T0 ))s"
