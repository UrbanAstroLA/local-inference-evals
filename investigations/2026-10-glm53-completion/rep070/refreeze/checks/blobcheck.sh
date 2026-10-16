#!/usr/bin/env bash
# verify each snapshot file's blob content against its blob name (sha256 for LFS, git blob sha1 otherwise)
for SNAP in <models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots/701cd7456c13d87bf0147ad946f828a999afb59c <models>/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a; do
  cd "$SNAP" || continue
  find . -type l | sort | while read f; do
    name=$(basename "$(readlink "$f")")
    if [ ${#name} = 64 ]; then got=$(sha256sum -L "$f" 2>/dev/null | cut -c1-64); [ -z "$got" ] && got=$(sha256sum "$(readlink -f "$f")" | cut -c1-64)
    else got=$(git hash-object "$(readlink -f "$f")"); fi
    [ "$got" = "$name" ] && echo "OK $SNAP $f" || echo "MISMATCH $SNAP $f $got"
  done
done
echo BLOBCHECK_DONE
