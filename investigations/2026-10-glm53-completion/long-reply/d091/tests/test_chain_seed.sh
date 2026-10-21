#!/usr/bin/env bash
set -u
# Evaluates the seed lines of the chain copy (and of the original chain) for d091's single pass. No process other than bash
# runs. The copy is rep070's (byte-identical, sha256 81ef3386...); the runner passes GPQA_SEED_LIST="70704", GPQA_PASSES=1.
H=$(cd "$(dirname "$0")" && pwd); COPY="$H/../runner/local-eval-chain-rep070.sh"; ORIG=<stacks>/glm53-flash-v0.8.0/eval-chain.sh
REPC=<runs>/rep070/refreeze/runner/local-eval-chain-rep070.sh
LC=$(grep -E '^\s*(SEEDARG=""|\[ -n "\$\{GPQA_SEED_LIST)' "$COPY"); LO=$(grep -E '^\s*SEEDARG=""' "$ORIG")
[ "$(echo "$LC" | wc -l)" = 2 ] && [ "$(echo "$LO" | wc -l)" = 1 ] || { echo "FAIL: seed lines not found"; exit 1; }
F=0
ev() { # label lines expect ; env given by caller; pass 1 only (GPQA_PASSES=1)
  local lab=$1 lines=$2 exp=$3; local got="" pass=1
  eval "$lines"; got="[$SEEDARG]"
  [ "$got" = "[$exp]" ] && echo "PASS $lab: $got" || { echo "FAIL $lab: got $got expected [$exp]"; F=1; return 1; }
}
( unset GPQA_SEED_PER_PASS; GPQA_SEED_LIST="70704"; ev "copy, d091 list (runner): pass 1 sends seed 70704" "$LC" ",seed=70704" ) || F=1
( unset GPQA_SEED_PER_PASS GPQA_SEED_LIST; ev "original, no seed variables = no seed argument (the 0.9.1 record's pass 1: default 1234)" "$LO" "" ) || F=1
LP=$(grep -E -- '--seed \$\(\(1234\+pass\)\)' "$COPY" | head -1)
[ -n "$LP" ] && echo "PASS copy: lm-eval process seed is 1234+pass = 1235 for pass 1 (as the records' pass 1)" || { echo "FAIL process seed line"; F=1; }
grep -q 'max_gen_toks=${GPQA_MAX_TOKENS:-327680}' "$COPY" && grep -q 'num_concurrent=${GPQA_CONCURRENCY:-8}' "$COPY" && grep -q 'timeout=${GPQA_TIMEOUT:-10800},max_retries=3' "$COPY" \
  && echo "PASS copy: cap 327,680, concurrency 8, timeout 10,800, max_retries 3 (protocol defaults)" || { echo "FAIL protocol defaults"; F=1; }
diff "$ORIG" "$COPY" | cmp -s - "$H/../runner/local-eval-chain-rep070.diff" && echo "PASS chain copy differs from the original exactly by the frozen diff" || { echo "FAIL diff"; F=1; }
cmp -s "$COPY" "$REPC" && echo "PASS chain copy byte-identical to rep070's copy" || { echo "FAIL not rep070's copy"; F=1; }
[ "$F" = 0 ] && echo "chain seed tests: all passed" || { echo "chain seed tests: FAILED"; exit 1; }
