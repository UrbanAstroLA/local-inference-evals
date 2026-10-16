#!/usr/bin/env bash
# Refusal tests of runner/run_rep070.sh, all with DRY_RUN=1. Docker is touched read-only only (docker ps, docker image
# inspect): the stubs below answer 'ps' / 'image inspect' themselves or pass them to the real binary, and refuse every other
# subcommand (exit 99), so no container can be created, started or stopped. GPU state is read with nvidia-smi only.
# Usage: test_runner_refusals.sh [STUB_DIR]. Prints one line per test and a summary; exit 0 only if every test passed.
H=$(cd "$(dirname "$0")" && pwd); RUN="$H/../runner/run_rep070.sh"
S=${1:-$(mktemp -d)}; mkdir -p "$S" || exit 1
trap 'rm -rf "$S"' EXIT
REAL_DOCKER=$(command -v docker); REAL_NSMI=$(command -v nvidia-smi)
mkstub() { # dir kind
  mkdir -p "$S/$1"
  case "$2" in
    glm-running) cat > "$S/$1/docker" <<EOF
#!/usr/bin/env bash
case "\$1" in
  ps) echo "glm53-flash-exl3-b12x-vllm ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629" ;;
  image) [ "\$2" = inspect ] && exec "$REAL_DOCKER" "\$@"; echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
  *) echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
esac
EOF
    ;;
    wrong-digest) cat > "$S/$1/docker" <<EOF
#!/usr/bin/env bash
case "\$1" in
  ps) exec "$REAL_DOCKER" "\$@" ;;
  image) [ "\$2" = inspect ] && { echo "sha256:0000000000000000000000000000000000000000000000000000000000000000"; exit 0; }; echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
  *) echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
esac
EOF
    ;;
    readonly) cat > "$S/$1/docker" <<EOF
#!/usr/bin/env bash
case "\$1" in
  ps) exec "$REAL_DOCKER" "\$@" ;;
  image) [ "\$2" = inspect ] && exec "$REAL_DOCKER" "\$@"; echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
  *) echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
esac
EOF
    ;;
    gpu-busy) cat > "$S/$1/nvidia-smi" <<EOF
#!/usr/bin/env bash
case "\$*" in
  *--query-compute-apps*) echo "999999" ;;
  *) exec "$REAL_NSMI" "\$@" ;;
esac
EOF
      cp "$S/readonly/docker" "$S/$1/docker" ;;
  esac
  chmod +x "$S/$1"/*
}
mkstub readonly readonly; mkstub glm glm-running; mkstub digest wrong-digest; mkstub gpu gpu-busy
PASS=0; FAIL=0
t() { # name expected-substring env...   (runs the runner under env -i with a minimal environment plus the given variables)
  local name=$1 exp=$2; shift 2
  local out; out=$(env -i HOME="$HOME" PATH="$S/readonly:/usr/local/bin:/usr/bin:/bin" DRY_RUN=1 "$@" bash "$RUN" 2>&1); local rc=$?
  if printf '%s' "$out" | grep -qF -- "$exp"; then PASS=$((PASS+1)); echo "PASS [$name] rc=$rc: $(printf '%s' "$out" | grep -F -- "$exp" | head -1 | cut -c1-150)"
  else FAIL=$((FAIL+1)); echo "FAIL [$name] rc=$rc expected '$exp'; got: $(printf '%s' "$out" | tail -3 | tr '\n' ' ' | cut -c1-300)"; fi
}
t "variant unset" "ABORT: VARIANT must be R or F" APPROVED_GPU_H=26.45
t "variant invalid" "ABORT: VARIANT must be R or F" VARIANT=X APPROVED_GPU_H=26.45
t "budget unset" "ABORT: APPROVED_GPU_H is not set" VARIANT=R
t "budget above plan maximum (36.51)" "ABORT: APPROVED_GPU_H=36.51 must be a number in (0, 36.5]" VARIANT=F APPROVED_GPU_H=36.51
t "budget at the plan maximum (36.5) accepted, variant F" "DRY_RUN: all refusal checks passed" VARIANT=F APPROVED_GPU_H=36.5
t "budget not a number" "ABORT: APPROVED_GPU_H=abc must be a number" VARIANT=F APPROVED_GPU_H=abc
t "budget zero" "ABORT: APPROVED_GPU_H=0 must be a number" VARIANT=F APPROVED_GPU_H=0
t "IMAGE override" "ABORT: override variables set in the environment (not the shipped configuration): IMAGE" VARIANT=R APPROVED_GPU_H=26.45 IMAGE=foo
t "PORT override" "ABORT: override variables set in the environment (not the shipped configuration): PORT" VARIANT=R APPROVED_GPU_H=26.45 PORT=0.0.0.0:8001
t "seed list override" "GPQA_SEED_LIST" VARIANT=F APPROVED_GPU_H=26.45 GPQA_SEED_LIST="1 2 3"
t "GLM container running (stub)" "ABORT: a GLM container is running" VARIANT=R APPROVED_GPU_H=26.45 PATH="$S/glm:/usr/local/bin:/usr/bin:/bin"
t "wrong image digest (stub)" "ABORT: image ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629 has Id 'sha256:0000" VARIANT=F APPROVED_GPU_H=26.45 PATH="$S/digest:/usr/local/bin:/usr/bin:/bin"
t "compute process on the GPUs (stub)" "ABORT: compute process 999999 on the GPUs" VARIANT=R APPROVED_GPU_H=26.45 PATH="$S/gpu:/usr/local/bin:/usr/bin:/bin"
( exec -a "/tmp/rep070-test/bin/lm_eval --model local-chat-completions" sleep 20 ) & DUMMY=$!; sleep 1
t "lm_eval process running (dummy process)" "ABORT: an lm_eval process is running" VARIANT=R APPROVED_GPU_H=26.45
kill $DUMMY 2>/dev/null; wait $DUMMY 2>/dev/null
( exec -a "<tools>/lm-eval/lm_eval/models/openai_completions.py" sleep 20 ) & DUMMY=$!; sleep 1
t "a process merely mentioning lm_eval does not block (dummy)" "DRY_RUN: all refusal checks passed" VARIANT=F APPROVED_GPU_H=26.45
kill $DUMMY 2>/dev/null; wait $DUMMY 2>/dev/null
t "real machine, variant R" "DRY_RUN: all refusal checks passed" VARIANT=R APPROVED_GPU_H=26.45
t "real machine, variant F" "DRY_RUN: all refusal checks passed" VARIANT=F APPROVED_GPU_H=30
# nothing of the run tags may exist after the dry runs
LEFT=$(ls -d <gpqa-runs>/*+rep070-* <data-runs>/benchmarks/*+rep070-* "$H/../../run-R" "$H/../../run-F" 2>/dev/null)
if [ -z "$LEFT" ]; then PASS=$((PASS+1)); echo "PASS [no files created by DRY_RUN]"; else FAIL=$((FAIL+1)); echo "FAIL [files created]: $LEFT"; fi
echo "runner refusal tests: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
