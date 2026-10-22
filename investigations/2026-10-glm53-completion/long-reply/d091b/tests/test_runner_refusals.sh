#!/usr/bin/env bash
set -u
# Refusal tests of runner/run_d091.sh, all with DRY_RUN=1 (after rep070/refreeze/tests/test_runner_refusals.sh). Docker is
# touched read-only only (docker ps, docker image inspect): the stubs answer 'ps' / 'image inspect' themselves or pass them
# to the real binary, and refuse every other subcommand (exit 99), so no container can be created, started or stopped. GPU
# state is read with nvidia-smi only. Run-state refusals use temporary marker files inside this package (removed at once).
# Usage: tests/test_runner_refusals.sh. Prints one line per test and a summary; exit 0 only if every test passed.
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H"); RUN="$PKG/runner/run_d091.sh"
case "$PKG" in <runs>/long-reply/d091b) ;; *) echo "ABORT: unexpected package path"; exit 1 ;; esac
S="$PKG/tests/stubs-tmp"
[ -n "$S" ] && case "$S" in <runs>/long-reply/d091b/tests/stubs-tmp) ;; *) exit 1 ;; esac
[ -e "$S" ] && { echo "ABORT: $S exists"; exit 1; }
[ -e "$PKG/run" ] && { echo "ABORT: $PKG/run exists before the tests"; exit 1; }
cleanup() {
  [ -d "$S" ] && rm -rf -- "$S"
  [ -f "$PKG/run/.refusal-test-marker" ] && rm -f -- "$PKG/run/.refusal-test-marker" && rmdir -- "$PKG/run" 2>/dev/null
  [ -f "$PKG/out/.refusal-test-marker" ] && rm -f -- "$PKG/out/.refusal-test-marker"
  [ -f "$PKG/SHA256SUMS.refusal-test-backup" ] && mv -f -- "$PKG/SHA256SUMS.refusal-test-backup" "$PKG/SHA256SUMS"
  chmod 700 "$PKG/PRIVATE"
}
trap cleanup EXIT
mkdir -p "$S" || exit 1
REAL_DOCKER=$(command -v docker); REAL_NSMI=$(command -v nvidia-smi)
mkstub() { # dir kind
  mkdir -p "$S/$1"
  case "$2" in
    glm-running) cat > "$S/$1/docker" <<EOF
#!/usr/bin/env bash
set -u
case "\$1" in
  ps) echo "glm53-flash-exl3-b12x-vllm ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07" ;;
  image) [ "\$2" = inspect ] && exec "$REAL_DOCKER" "\$@"; echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
  *) echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
esac
EOF
    ;;
    wrong-digest) cat > "$S/$1/docker" <<EOF
#!/usr/bin/env bash
set -u
case "\$1" in
  ps) exec "$REAL_DOCKER" "\$@" ;;
  image) [ "\$2" = inspect ] && { echo "sha256:0000000000000000000000000000000000000000000000000000000000000000"; exit 0; }; echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
  *) echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
esac
EOF
    ;;
    readonly) cat > "$S/$1/docker" <<EOF
#!/usr/bin/env bash
set -u
case "\$1" in
  ps) exec "$REAL_DOCKER" "\$@" ;;
  image) [ "\$2" = inspect ] && exec "$REAL_DOCKER" "\$@"; echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
  *) echo "STUB REFUSED: docker \$*" >&2; exit 99 ;;
esac
EOF
    ;;
    gpu-busy) cat > "$S/$1/nvidia-smi" <<EOF
#!/usr/bin/env bash
set -u
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
t "budget unset" "ABORT: APPROVED_GPU_H is not set"
t "budget above the ceiling (10.01)" "ABORT: APPROVED_GPU_H=10.01 must be a number in (0, 10.0]" APPROVED_GPU_H=10.01
t "budget not a number" "ABORT: APPROVED_GPU_H=abc must be a number" APPROVED_GPU_H=abc
t "budget zero" "ABORT: APPROVED_GPU_H=0 must be a number" APPROVED_GPU_H=0
t "IMAGE override" "ABORT: override variables set in the environment (not the shipped configuration): IMAGE" APPROVED_GPU_H=10 IMAGE=foo
t "PORT override" "ABORT: override variables set in the environment (not the shipped configuration): PORT" APPROVED_GPU_H=10 PORT=0.0.0.0:8001
t "GPU_MEMORY_UTILIZATION override" "GPU_MEMORY_UTILIZATION" APPROVED_GPU_H=10 GPU_MEMORY_UTILIZATION=0.97
t "seed list override" "GPQA_SEED_LIST" APPROVED_GPU_H=10 GPQA_SEED_LIST="1234"
t "launcher variable override (MAX_NUM_SEQS)" "MAX_NUM_SEQS" APPROVED_GPU_H=10 MAX_NUM_SEQS=8
t "GLM container running (stub)" "ABORT: a GLM container is running" APPROVED_GPU_H=10 PATH="$S/glm:/usr/local/bin:/usr/bin:/bin"
t "wrong image digest (stub)" "has Id 'sha256:0000" APPROVED_GPU_H=10 PATH="$S/digest:/usr/local/bin:/usr/bin:/bin"
t "compute process on the GPUs (stub)" "ABORT: compute process 999999 on the GPUs" APPROVED_GPU_H=10 PATH="$S/gpu:/usr/local/bin:/usr/bin:/bin"
( exec -a "/tmp/d091-test/bin/lm_eval --model local-chat-completions" sleep 20 ) & DUMMY=$!; sleep 1
t "lm_eval process running (dummy process)" "ABORT: an lm_eval process is running" APPROVED_GPU_H=10
kill $DUMMY 2>/dev/null; wait $DUMMY 2>/dev/null
( exec -a "<tools>/lm-eval/lm_eval/models/openai_completions.py" sleep 20 ) & DUMMY=$!; sleep 1
t "a process merely mentioning lm_eval does not block (dummy)" "DRY_RUN: all refusal checks passed" APPROVED_GPU_H=10
kill $DUMMY 2>/dev/null; wait $DUMMY 2>/dev/null
python3 -c "import socket,time; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', 8002)); s.listen(1); time.sleep(15)" & LIS=$!; sleep 1
t "port 8002 in use (dummy listener)" "ABORT: port 8001 or 8002 is in use" APPROVED_GPU_H=10
kill $LIS 2>/dev/null; wait $LIS 2>/dev/null
mkdir "$PKG/run" && : > "$PKG/run/.refusal-test-marker"
t "run directory exists (earlier attempt)" "ABORT: $PKG/run exists" APPROVED_GPU_H=10
rm -f -- "$PKG/run/.refusal-test-marker"; rmdir -- "$PKG/run"
: > "$PKG/out/.refusal-test-marker"
t "out/ not empty (earlier pipeline run)" "is not empty" APPROVED_GPU_H=10
rm -f -- "$PKG/out/.refusal-test-marker"
chmod 755 "$PKG/PRIVATE"
t "PRIVATE/ not mode 0700" "PRIVATE must exist with mode 0700" APPROVED_GPU_H=10
chmod 700 "$PKG/PRIVATE"
cp -p "$PKG/SHA256SUMS" "$PKG/SHA256SUMS.refusal-test-backup" && echo "0000000000000000000000000000000000000000000000000000000000000000  PROPOSAL.md" >> "$PKG/SHA256SUMS"
t "package SHA256SUMS does not verify" "ABORT: $PKG/SHA256SUMS does not verify" APPROVED_GPU_H=10
mv -f -- "$PKG/SHA256SUMS.refusal-test-backup" "$PKG/SHA256SUMS"
t "real machine, budget at the ceiling (10.0) accepted" "DRY_RUN: all refusal checks passed" APPROVED_GPU_H=10.0
t "real machine, budget 8" "DRY_RUN: all refusal checks passed" APPROVED_GPU_H=8
LEFT=$(ls -d <gpqa-runs>/*+d091b* <data-runs>/benchmarks/*+d091b* "$PKG/run" "$PKG/PRIVATE/raw" 2>/dev/null; ls -A "$PKG/out" 2>/dev/null)
if [ -z "$LEFT" ]; then PASS=$((PASS+1)); echo "PASS [no files created by DRY_RUN]"; else FAIL=$((FAIL+1)); echo "FAIL [files created]: $LEFT"; fi
echo "runner refusal tests: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
