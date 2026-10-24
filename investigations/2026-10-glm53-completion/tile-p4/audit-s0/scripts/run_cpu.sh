#!/bin/bash
# Audit-s0 CPU runner: python3 from the stock image (numpy/torch), no GPU, no network.
# tile-p4 is mounted READ-ONLY at /p4; only audit-s0 (/out) and PRIVATE-audit-s0 (/pout) are writable.
#   audit-s0/scripts/run_cpu.sh SCRIPT.py [args]      (SCRIPT path relative to audit-s0/scripts)
#   audit-s0/scripts/run_cpu.sh -                      (script on stdin)
T=<runs>/tile-p4
STOCK=sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
S="$1"; shift
[ "$S" = "-" ] || S="/out/scripts/$S"
exec docker run --rm -i --pull=never --network none -e CUDA_VISIBLE_DEVICES= -e HOME=/tmp \
  --user "$(id -u):$(id -g)" -v "$T":/p4:ro -v "$T/audit-s0":/out -v "$T/PRIVATE-audit-s0":/pout \
  --entrypoint python3 "$STOCK" "$S" "$@"
