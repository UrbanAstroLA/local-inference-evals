#!/usr/bin/env bash
set -u
# [E copy of bplus/tests/test_runner_refusals.sh: set -u; scratch fixed inside e-cross/PRIVATE/ and guarded; package copies
#  by tar (the scratch lies inside the package); E's ceiling (19.9 under amendment 22ah); the B1-tag test replaced by a v091 digest test and an
#  approval at B+'s 31.6 ceiling; stubs written with set -u.]
# Refusal tests of runner/run_e.sh, all with DRY_RUN=1. Docker is touched read-only only (docker ps, docker image
# inspect): the stubs answer 'ps' / 'image inspect' themselves or pass them to the real binary and refuse every other
# subcommand (exit 99), so no container can be created, started or stopped. GPU state is read with nvidia-smi only.
# Package-level refusals run on copies of the package in a scratch folder under PRIVATE/ (deleted at the end).
# Usage: test_runner_refusals.sh > OUT   (needs a frozen package: SHA256SUMS must verify for the real-configuration test)
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H"); RUN0="$PKG/runner/run_e.sh"
EC=<runs>/long-reply/e-cross
case "$PKG" in "$EC") ;; *) echo "REFUSED: unexpected package path $PKG"; exit 2 ;; esac
S=$PKG/PRIVATE/scratch-refusals
case "$S" in "$EC"/PRIVATE/scratch-refusals) ;; *) echo "REFUSED: scratch $S"; exit 2 ;; esac
[ -f "$RUN0" ] || { echo "REFUSED: runner not found at $RUN0"; exit 2; }
cleanup() { case "$S" in "$EC"/PRIVATE/scratch-refusals) rm -rf -- "$S" ;; esac; }
trap cleanup EXIT
cleanup; mkdir -p "$S" || exit 1; chmod 700 "$S"
REAL_DOCKER=$(command -v docker); REAL_NSMI=$(command -v nvidia-smi)
for d in readonly glm digest digest91 gpu gpumem; do mkdir -p "$S/$d" || exit 1; done
printf '%s\n' '#!/usr/bin/env bash' 'set -u' \
  "case \"\${1:-}\" in ps) exec \"$REAL_DOCKER\" \"\$@\" ;; image) [ \"\${2:-}\" = inspect ] && exec \"$REAL_DOCKER\" \"\$@\" ;; esac" \
  'echo "STUB REFUSED: docker $*" >&2; exit 99' > "$S/readonly/docker"
printf '%s\n' '#!/usr/bin/env bash' 'set -u' \
  "case \"\${1:-}\" in ps) echo 'glm53-flash-exl3-b12x-vllm ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c0'; exit 0 ;; image) [ \"\${2:-}\" = inspect ] && exec \"$REAL_DOCKER\" \"\$@\" ;; esac" \
  'echo "STUB REFUSED: docker $*" >&2; exit 99' > "$S/glm/docker"
printf '%s\n' '#!/usr/bin/env bash' 'set -u' \
  "case \"\${1:-}\" in ps) exec \"$REAL_DOCKER\" \"\$@\" ;; image) [ \"\${2:-}\" = inspect ] && { echo sha256:0000000000000000000000000000000000000000000000000000000000000000; exit 0; } ;; esac" \
  'echo "STUB REFUSED: docker $*" >&2; exit 99' > "$S/digest/docker"
printf '%s\n' '#!/usr/bin/env bash' 'set -u' \
  "case \"\${1:-}\" in ps) exec \"$REAL_DOCKER\" \"\$@\" ;; image) case \"\${3:-}\" in *a9e985c0*) echo sha256:1111111111111111111111111111111111111111111111111111111111111111; exit 0 ;; esac; [ \"\${2:-}\" = inspect ] && exec \"$REAL_DOCKER\" \"\$@\" ;; esac" \
  'echo "STUB REFUSED: docker $*" >&2; exit 99' > "$S/digest91/docker"
cp "$S/readonly/docker" "$S/gpu/docker"; cp "$S/readonly/docker" "$S/gpumem/docker"
printf '%s\n' '#!/usr/bin/env bash' 'set -u' \
  "case \"\$*\" in *--query-compute-apps*) echo 999999 ;; *) exec \"$REAL_NSMI\" \"\$@\" ;; esac" > "$S/gpu/nvidia-smi"
printf '%s\n' '#!/usr/bin/env bash' 'set -u' \
  "case \"\$*\" in *--query-gpu=index,memory.used,utilization.gpu*) echo '0, 30000, 0'; echo '1, 2000, 0' ;; *) exec \"$REAL_NSMI\" \"\$@\" ;; esac" > "$S/gpumem/nvidia-smi"
for d in readonly glm digest digest91 gpu gpumem; do chmod +x "$S/$d/docker"; done
chmod +x "$S/gpu/nvidia-smi" "$S/gpumem/nvidia-smi"
copy() { # package copy without run/ and the scratch folders
  local d="$S/pkg-$1"
  case "$d" in "$EC"/PRIVATE/scratch-refusals/pkg-*) ;; *) return 1 ;; esac
  rm -rf -- "$d"; mkdir -p "$d" || return 1
  tar -C "$PKG" --exclude='./run' --exclude='./PRIVATE/scratch-*' -cf - . | tar -C "$d" -xf - || return 1
  echo "$d"
}
PASS=0; FAIL=0
t() { # name expected-substring stubdir runner env...
  local name=$1 exp=$2 stub=$3 run=$4; shift 4
  local out; out=$(env -i HOME="$HOME" PATH="$S/$stub:/usr/local/bin:/usr/bin:/bin" DRY_RUN=1 "$@" bash "$run" 2>&1); local rc=$?
  if printf '%s' "$out" | grep -qF -- "$exp" && { [ "$exp" = "DRY_RUN: all refusal checks passed" ] || [ $rc != 0 ]; }; then
    PASS=$((PASS+1)); echo "PASS [$name] rc=$rc: $(printf '%s' "$out" | grep -F -- "$exp" | head -1 | cut -c1-160)"
  else FAIL=$((FAIL+1)); echo "FAIL [$name] rc=$rc expected '$exp'; got: $(printf '%s' "$out" | tail -3 | tr '\n' ' ' | cut -c1-300)"; fi
}
t "approval unset" "ABORT: APPROVED_GPU_H is not set" readonly "$RUN0"
t "approval zero" "must be a number in (0, 19.9]" readonly "$RUN0" APPROVED_GPU_H=0
t "approval above the ceiling" "must be a number in (0, 19.9]" readonly "$RUN0" APPROVED_GPU_H=20.0
t "approval at B+'s ceiling" "must be a number in (0, 19.9]" readonly "$RUN0" APPROVED_GPU_H=31.6
t "approval not a number" "must be a number in (0, 19.9]" readonly "$RUN0" APPROVED_GPU_H=abc
t "launcher override MAX_NUM_SEQS" "override variables set in the environment" readonly "$RUN0" APPROVED_GPU_H=19.9 MAX_NUM_SEQS=8
t "launcher override IMAGE" "(not the shipped configuration): IMAGE" readonly "$RUN0" APPROVED_GPU_H=19.9 IMAGE=x
t "launcher override PORT" "(not the shipped configuration): PORT" readonly "$RUN0" APPROVED_GPU_H=19.9 PORT=0.0.0.0:8001
t "launcher override ENABLE_PREFIX_CACHING" "(not the shipped configuration): ENABLE_PREFIX_CACHING" readonly "$RUN0" APPROVED_GPU_H=19.9 ENABLE_PREFIX_CACHING=0
t "GLM container running" "ABORT: a GLM container is running" glm "$RUN0" APPROVED_GPU_H=19.9
t "image digest mismatch (every image)" "has Id 'sha256:0000" digest "$RUN0" APPROVED_GPU_H=19.9
t "image digest mismatch (0.9.1 only)" "v091 image ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c0" digest91 "$RUN0" APPROVED_GPU_H=19.9
t "compute process on the GPUs" "ABORT: GPU pre-flight: compute process 999999" gpu "$RUN0" APPROVED_GPU_H=19.9
t "GPU memory in use" "ABORT: GPU pre-flight: GPU0 busy" gpumem "$RUN0" APPROVED_GPU_H=19.9
python3 -c "import socket,time; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', 8001)); s.listen(); time.sleep(25)" &
LP=$!; sleep 1
t "port 8001 in use" "ABORT: port 8001 is in use" readonly "$RUN0" APPROVED_GPU_H=19.9
kill $LP 2>/dev/null; wait $LP 2>/dev/null
( exec -a "python3 -B $PKG/scripts/client_bplus.py v070 x" sleep 20 ) &
CP=$!; sleep 0.5
t "client process running" "ABORT: a B+ client process is running" readonly "$RUN0" APPROVED_GPU_H=19.9
kill $CP 2>/dev/null; wait $CP 2>/dev/null
P=$(copy sums) || exit 1; echo "# tampered" >> "$P/scripts/plan.py"
t "package file changed after freezing" "SHA256SUMS does not verify" readonly "$P/runner/run_e.sh" APPROVED_GPU_H=19.9
P=$(copy nosums) || exit 1; rm -f -- "$P/SHA256SUMS"
t "package not frozen" "SHA256SUMS missing" readonly "$P/runner/run_e.sh" APPROVED_GPU_H=19.9
P=$(copy priv) || exit 1; python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['prefixes']['P01'][-1]+=1; json.dump(d,open(p,'w'))" "$P/PRIVATE/prefixes.json"; ( cd "$P" && ./make_sums.sh > /dev/null )
t "PRIVATE prefixes differ from the manifest" "PRIVATE/prefixes.json or prompts.json does not match" readonly "$P/runner/run_e.sh" APPROVED_GPU_H=19.9
P=$(copy mode) || exit 1; chmod 755 "$P/PRIVATE"
t "PRIVATE folder mode" "PRIVATE/ must be mode 0700" readonly "$P/runner/run_e.sh" APPROVED_GPU_H=19.9
P=$(copy pin) || exit 1; sed -i 's/\]=e1196cbcd6b4744cb41fdebf3d12d8586dbea464e702af5de47617837efe0e5e/]=0000000000000000000000000000000000000000000000000000000000000000/' "$P/runner/run_e.sh"; ( cd "$P" && ./make_sums.sh > /dev/null )
t "pinned launcher hash differs" "ABORT: pinned file changed: <stacks>/glm53-flash-v0.9.0/start.sh" readonly "$P/runner/run_e.sh" APPROVED_GPU_H=19.9
P=$(copy snap) || exit 1; echo "./extra -> x" >> "$P/checks/snapshot-0490d2f7.txt"; ( cd "$P" && ./make_sums.sh > /dev/null )
t "snapshot listing differs" "ABORT: snapshot listing differs" readonly "$P/runner/run_e.sh" APPROVED_GPU_H=19.9
P=$(copy rundir) || exit 1; mkdir "$P/run"
t "run folder exists (no automatic retry)" "exists (an earlier attempt; no automatic retry)" readonly "$P/runner/run_e.sh" APPROVED_GPU_H=19.9
t "real configuration: every check passes" "DRY_RUN: all refusal checks passed" readonly "$RUN0" APPROVED_GPU_H=19.9
echo "refusal tests: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
