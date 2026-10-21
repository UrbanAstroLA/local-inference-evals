#!/usr/bin/env bash
# Refusal tests of runner/run_bplus.sh, all with DRY_RUN=1. Docker is touched read-only only (docker ps, docker image
# inspect): the stubs answer 'ps' / 'image inspect' themselves or pass them to the real binary and refuse every other
# subcommand (exit 99), so no container can be created, started or stopped. GPU state is read with nvidia-smi only.
# Package-level refusals run on copies of the package in a scratch folder.
# Usage: test_runner_refusals.sh SCRATCH_DIR   (SCRATCH_DIR must be a new or disposable folder; it is deleted at the end)
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H"); RUN0="$PKG/runner/run_bplus.sh"
S=${1:?usage: test_runner_refusals.sh SCRATCH_DIR}
case "$S" in /tmp/*|<data-runs>/*) ;; *) echo "REFUSED: scratch dir must be under /tmp or <data-runs>"; exit 2 ;; esac
[ -f "$RUN0" ] || { echo "REFUSED: runner not found at $RUN0"; exit 2; }
rm -rf "$S"; mkdir -p "$S" || exit 1
REAL_DOCKER=$(command -v docker); REAL_NSMI=$(command -v nvidia-smi)
mkdir -p "$S/readonly" "$S/glm" "$S/digest" "$S/tag" "$S/gpu" "$S/gpumem" || exit 1
printf '%s\n' '#!/usr/bin/env bash' \
  "case \"\$1\" in ps) exec \"$REAL_DOCKER\" \"\$@\" ;; image) [ \"\$2\" = inspect ] && exec \"$REAL_DOCKER\" \"\$@\" ;; esac" \
  'echo "STUB REFUSED: docker $*" >&2; exit 99' > "$S/readonly/docker"
printf '%s\n' '#!/usr/bin/env bash' \
  "case \"\$1\" in ps) echo 'glm53-flash-exl3-b12x-vllm ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c0'; exit 0 ;; image) [ \"\$2\" = inspect ] && exec \"$REAL_DOCKER\" \"\$@\" ;; esac" \
  'echo "STUB REFUSED: docker $*" >&2; exit 99' > "$S/glm/docker"
printf '%s\n' '#!/usr/bin/env bash' \
  "case \"\$1\" in ps) exec \"$REAL_DOCKER\" \"\$@\" ;; image) [ \"\$2\" = inspect ] && { echo sha256:0000000000000000000000000000000000000000000000000000000000000000; exit 0; } ;; esac" \
  'echo "STUB REFUSED: docker $*" >&2; exit 99' > "$S/digest/docker"
printf '%s\n' '#!/usr/bin/env bash' \
  "case \"\$1\" in ps) exec \"$REAL_DOCKER\" \"\$@\" ;; image) [ \"\$3\" = local/glm53-v0.9.1-b1-d7:local ] && { echo sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548; exit 0; }; [ \"\$2\" = inspect ] && exec \"$REAL_DOCKER\" \"\$@\" ;; esac" \
  'echo "STUB REFUSED: docker $*" >&2; exit 99' > "$S/tag/docker"
cp "$S/readonly/docker" "$S/gpu/docker"; cp "$S/readonly/docker" "$S/gpumem/docker"
printf '%s\n' '#!/usr/bin/env bash' \
  "case \"\$*\" in *--query-compute-apps*) echo 999999 ;; *) exec \"$REAL_NSMI\" \"\$@\" ;; esac" > "$S/gpu/nvidia-smi"
printf '%s\n' '#!/usr/bin/env bash' \
  "case \"\$*\" in *--query-gpu=index,memory.used,utilization.gpu*) echo '0, 30000, 0'; echo '1, 2000, 0' ;; *) exec \"$REAL_NSMI\" \"\$@\" ;; esac" > "$S/gpumem/nvidia-smi"
chmod +x "$S/readonly/docker" "$S/glm/docker" "$S/digest/docker" "$S/tag/docker" "$S/gpu/docker" "$S/gpumem/docker" "$S/gpu/nvidia-smi" "$S/gpumem/nvidia-smi"
copy() { local d="$S/pkg-$1"; rm -rf "$d"; cp -a "$PKG" "$d" && rm -rf "$d/run" && echo "$d"; }
PASS=0; FAIL=0
t() { # name expected-substring stubdir runner env...
  local name=$1 exp=$2 stub=$3 run=$4; shift 4
  local out; out=$(env -i HOME="$HOME" PATH="$S/$stub:/usr/local/bin:/usr/bin:/bin" DRY_RUN=1 "$@" bash "$run" 2>&1); local rc=$?
  if printf '%s' "$out" | grep -qF -- "$exp" && { [ "$exp" = "DRY_RUN: all refusal checks passed" ] || [ $rc != 0 ]; }; then
    PASS=$((PASS+1)); echo "PASS [$name] rc=$rc: $(printf '%s' "$out" | grep -F -- "$exp" | head -1 | cut -c1-160)"
  else FAIL=$((FAIL+1)); echo "FAIL [$name] rc=$rc expected '$exp'; got: $(printf '%s' "$out" | tail -3 | tr '\n' ' ' | cut -c1-300)"; fi
}
t "approval unset" "ABORT: APPROVED_GPU_H is not set" readonly "$RUN0"
t "approval zero" "must be a number in (0, 31.6]" readonly "$RUN0" APPROVED_GPU_H=0
t "approval above the ceiling" "must be a number in (0, 31.6]" readonly "$RUN0" APPROVED_GPU_H=40
t "approval not a number" "must be a number in (0, 31.6]" readonly "$RUN0" APPROVED_GPU_H=abc
t "launcher override MAX_NUM_SEQS" "override variables set in the environment" readonly "$RUN0" APPROVED_GPU_H=31.6 MAX_NUM_SEQS=8
t "launcher override IMAGE" "(not the shipped configuration): IMAGE" readonly "$RUN0" APPROVED_GPU_H=31.6 IMAGE=x
t "launcher override PORT" "(not the shipped configuration): PORT" readonly "$RUN0" APPROVED_GPU_H=31.6 PORT=0.0.0.0:8001
t "launcher override ENABLE_PREFIX_CACHING" "(not the shipped configuration): ENABLE_PREFIX_CACHING" readonly "$RUN0" APPROVED_GPU_H=31.6 ENABLE_PREFIX_CACHING=0
t "GLM container running" "ABORT: a GLM container is running" glm "$RUN0" APPROVED_GPU_H=31.6
t "image digest mismatch" "has Id 'sha256:0000" digest "$RUN0" APPROVED_GPU_H=31.6
t "B1 tag moved" "(tag moved)" tag "$RUN0" APPROVED_GPU_H=31.6
t "compute process on the GPUs" "ABORT: GPU pre-flight: compute process 999999" gpu "$RUN0" APPROVED_GPU_H=31.6
t "GPU memory in use" "ABORT: GPU pre-flight: GPU0 busy" gpumem "$RUN0" APPROVED_GPU_H=31.6
python3 -c "import socket,time; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', 8001)); s.listen(); time.sleep(25)" &
LP=$!; sleep 1
t "port 8001 in use" "ABORT: port 8001 is in use" readonly "$RUN0" APPROVED_GPU_H=31.6
kill $LP 2>/dev/null; wait $LP 2>/dev/null
( exec -a "python3 -B $PKG/scripts/client_bplus.py v070 x" sleep 20 ) &
CP=$!; sleep 0.5
t "B+ client process running" "ABORT: a B+ client process is running" readonly "$RUN0" APPROVED_GPU_H=31.6
kill $CP 2>/dev/null; wait $CP 2>/dev/null
P=$(copy sums) || exit 1; echo "# tampered" >> "$P/scripts/plan.py"
t "package file changed after freezing" "SHA256SUMS does not verify" readonly "$P/runner/run_bplus.sh" APPROVED_GPU_H=31.6
P=$(copy nosums) || exit 1; rm -f "$P/SHA256SUMS"
t "package not frozen" "SHA256SUMS missing" readonly "$P/runner/run_bplus.sh" APPROVED_GPU_H=31.6
P=$(copy priv) || exit 1; python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['prefixes']['P01'][-1]+=1; json.dump(d,open(p,'w'))" "$P/PRIVATE/prefixes.json"; ( cd "$P" && ./make_sums.sh > /dev/null )
t "PRIVATE prefixes differ from the manifest" "PRIVATE/prefixes.json or prompts.json does not match" readonly "$P/runner/run_bplus.sh" APPROVED_GPU_H=31.6
P=$(copy mode) || exit 1; chmod 755 "$P/PRIVATE"
t "PRIVATE folder mode" "PRIVATE/ must be mode 0700" readonly "$P/runner/run_bplus.sh" APPROVED_GPU_H=31.6
P=$(copy pin) || exit 1; sed -i 's/\]=e1196cbcd6b4744cb41fdebf3d12d8586dbea464e702af5de47617837efe0e5e/]=0000000000000000000000000000000000000000000000000000000000000000/' "$P/runner/run_bplus.sh"; ( cd "$P" && ./make_sums.sh > /dev/null )
t "pinned launcher hash differs" "ABORT: pinned file changed: <stacks>/glm53-flash-v0.9.0/start.sh" readonly "$P/runner/run_bplus.sh" APPROVED_GPU_H=31.6
P=$(copy snap) || exit 1; echo "./extra -> x" >> "$P/checks/snapshot-0490d2f7.txt"; ( cd "$P" && ./make_sums.sh > /dev/null )
t "snapshot listing differs" "ABORT: snapshot listing differs" readonly "$P/runner/run_bplus.sh" APPROVED_GPU_H=31.6
P=$(copy rundir) || exit 1; mkdir "$P/run"
t "run folder exists (no automatic retry)" "exists (an earlier attempt; no automatic retry)" readonly "$P/runner/run_bplus.sh" APPROVED_GPU_H=31.6
t "real configuration: every check passes" "DRY_RUN: all refusal checks passed" readonly "$RUN0" APPROVED_GPU_H=31.6
echo "refusal tests: $PASS passed, $FAIL failed"
rm -rf "$S"
[ "$FAIL" = 0 ]
