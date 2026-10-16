#!/usr/bin/env bash
# Evaluates the seed lines of the chain copy (and of the original chain) for passes 1-3. No process other than bash runs.
H=$(cd "$(dirname "$0")" && pwd); COPY="$H/../runner/local-eval-chain-rep070.sh"; ORIG=<stacks>/glm53-flash-v0.8.0/eval-chain.sh
LC=$(grep -E '^\s*(SEEDARG=""|\[ -n "\$\{GPQA_SEED_LIST)' "$COPY"); LO=$(grep -E '^\s*SEEDARG=""' "$ORIG")
[ "$(echo "$LC" | wc -l)" = 2 ] && [ "$(echo "$LO" | wc -l)" = 1 ] || { echo "FAIL: seed lines not found"; exit 1; }
F=0
ev() { # label lines expect... ; env given by caller
  local lab=$1 lines=$2; shift 2; local got=""
  for pass in 1 2 3; do eval "$lines"; got="$got[$SEEDARG]"; done
  local exp; exp=$(printf '[%s]' "$@")
  [ "$got" = "$exp" ] && echo "PASS $lab: $got" || { echo "FAIL $lab: got $got expected $exp"; F=1; return 1; }
}
( unset GPQA_SEED_PER_PASS; GPQA_SEED_LIST="1234 1235 1236"; ev "copy, variant R list (runner)" "$LC" ",seed=1234" ",seed=1235" ",seed=1236" ) || F=1
( unset GPQA_SEED_PER_PASS; GPQA_SEED_LIST="70701 70702 70703"; ev "copy, variant F list (runner)" "$LC" ",seed=70701" ",seed=70702" ",seed=70703" ) || F=1
( GPQA_SEED_PER_PASS=1; unset GPQA_SEED_LIST; ev "original, GPQA_SEED_PER_PASS=1 (record passes 2-3)" "$LO" ",seed=1234" ",seed=1235" ",seed=1236" ) || F=1
( GPQA_SEED_PER_PASS=1; unset GPQA_SEED_LIST; ev "copy without a list = original behaviour" "$LC" ",seed=1234" ",seed=1235" ",seed=1236" ) || F=1
( unset GPQA_SEED_PER_PASS GPQA_SEED_LIST; ev "copy, no seed variables = no seed argument (record pass 1)" "$LC" "" "" "" ) || F=1
diff "$ORIG" "$COPY" | cmp -s - "$H/../runner/local-eval-chain-rep070.diff" && echo "PASS chain copy differs from the original exactly by the frozen diff" || { echo "FAIL diff"; F=1; }
[ "$F" = 0 ] && echo "chain seed tests: all passed" || { echo "chain seed tests: FAILED"; exit 1; }
