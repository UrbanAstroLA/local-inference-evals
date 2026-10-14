#!/bin/bash
# Hook-validation image for ATTEMPT 2 (PA2-6; hook-validation/ATTEMPT2-PREP.md). Derived from build/make_build.sh
# (attempt 1, kept unchanged with its outputs in build/). Differences: new tag ...-mg2, outputs in build-a2/, and a
# second manifest comparison against the attempt-1 image (only vllm/forcewin_hooks.py may differ).
# B1 mapguard image (B1 + inert observe-only guard, d7/step2) + the local-forcewin hook (inert unless
# FORCEWIN=on AND a request is keyed). NEW TAG ONLY; no existing tag is replaced.
# CPU only: docker build of a file-copy layer, then the in-image gates. No GPU, no server, no network in RUN steps
# beyond the build itself. No `set -e`: each step's rc is checked.
HERE=$(cd "$(dirname "$0")" && pwd)
HV=$(cd "$HERE/.." && pwd)
BASE=sha256:4d7bd73e6f2ef6485289dcb5bcb2f2a65f1846a5959cde7cbd9c275471c72e16   # B1-mapguard (d7/step2/FROZEN.md)
BASETAG=local/glm53-v0.9.1-b1-mapguard:local
A1=sha256:1b9f8b15d63197ad8b2ef2410321baedf9dcd55a112603bc77b887e8dac293d5     # attempt-1 image (build/image-id.txt)
TAG=local/glm53-v0.9.1-b1-forcehook-mg2:local
if docker image inspect "$TAG" >/dev/null 2>&1; then echo "ABORT: $TAG exists (new tags only; remove it deliberately first)"; exit 1; fi
[ "$(docker image inspect "$BASETAG" --format '{{.Id}}' 2>/dev/null)" = "$BASE" ] || { echo "ABORT: $BASETAG is not $BASE"; exit 1; }
docker image inspect "$A1" >/dev/null 2>&1 || { echo "ABORT: attempt-1 image $A1 missing (needed for the manifest check)"; exit 1; }
rm -rf "$HERE/files" && mkdir -p "$HERE/files/vllm" || exit 1
python3 "$HV/hook/make_patch.py" "$HV/src/b1-mapguard" "$HERE/files/vllm" --base b1-mapguard > "$HERE/patch-report.json" || exit 1
cp "$HV/hook/forcewin_hooks.py" "$HERE/files/vllm/forcewin_hooks.py" || exit 1
cd "$HERE" || exit 1
# the five hooked engine files must be byte-identical to attempt 1's (only the module changed)
for f in v1/worker/gpu/model_runner.py v1/worker/gpu/spec_decode/rejection_sampler.py v1/worker/gpu/sample/sampler.py \
         v1/attention/backends/mla/indexer.py model_executor/layers/sparse_attn_indexer_kpool.py; do
  cmp -s "files/vllm/$f" "$HV/build/files/vllm/$f" || { echo "ABORT: hooked $f differs from attempt 1's"; exit 1; }
done
H() { sha256sum "files/vllm/$1" | cut -d' ' -f1; }
F="v1/worker/gpu/model_runner.py v1/worker/gpu/spec_decode/rejection_sampler.py v1/worker/gpu/sample/sampler.py v1/attention/backends/mla/indexer.py model_executor/layers/sparse_attn_indexer_kpool.py"
{
echo "# B1 + inert mapguard (d7/step2) + inert local-forcewin hook, attempt 2 (PA2-1/PA2-2; hook-validation/ATTEMPT2-PREP.md)."
echo "# Base verified to be $BASE before the build. Every forcewin call site is behind _FW_ON (env FORCEWIN=on)."
echo "FROM $BASETAG"
echo "ARG VP=/usr/local/lib/python3.12/dist-packages/vllm"
echo "# Gate 0: the replaced files are the B1-mapguard image's (make_patch.py PRE['b1-mapguard'])."
echo "RUN printf '%s\\n' \\"
python3 - <<'PY'
import sys; sys.path.insert(0, "../hook"); import make_patch as m
for rel, h in m.PRE["b1-mapguard"].items():
    print('    "%s  ${VP}/%s" \\' % (h, rel))
PY
echo "  | sha256sum -c - && ! test -e \${VP}/forcewin_hooks.py"
echo "COPY files/vllm/forcewin_hooks.py \${VP}/forcewin_hooks.py"
for f in $F; do echo "COPY files/vllm/$f \${VP}/$f"; done
echo "RUN printf '%s\\n' \\"
echo "    \"$(H forcewin_hooks.py)  \${VP}/forcewin_hooks.py\" \\"
for f in $F; do echo "    \"$(H $f)  \${VP}/$f\" \\"; done
echo "  | sha256sum -c - \\"
echo " && python3 -m py_compile \${VP}/forcewin_hooks.py $(for f in $F; do printf '${VP}/%s ' $f; done) \\"
for f in $F; do echo " && test \"\$(grep -c local-forcewin \${VP}/$f)\" = $(grep -c local-forcewin files/vllm/$f) \\"; done
echo " && test \"\$(grep -c local-mapguard \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)\" = 9 \\"
echo " && test \"\$(grep -c local-d7 \${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)\" = 2"
echo "LABEL io.local.base=\"B1-mapguard $BASE\" io.local.purpose=\"forced-history hook validation attempt 2, PLAN.md amendment 21\""
} > Dockerfile
docker build -t "$TAG" . > build.log 2>&1; rc=$?
tail -n 4 build.log
[ $rc -eq 0 ] || { echo "build failed rc=$rc"; exit $rc; }
ID=$(docker image inspect "$TAG" --format '{{.Id}}')
python3 - "$BASE" "$TAG" <<'PY' || { echo "ABORT: derived image does not extend the base layers"; exit 1; }
import json, subprocess, sys
L = lambda i: json.loads(subprocess.check_output(["docker", "image", "inspect", i, "--format", "{{json .RootFS.Layers}}"]))
b, d = L(sys.argv[1]), L(sys.argv[2])
sys.exit(0 if d[:len(b)] == b and len(d) > len(b) else 1)
PY
echo "$TAG $ID" | tee image-id.txt
# .py manifest diffs: against the base (the five hooked files + the new module) and against attempt 1 (module only)
M='cd /usr/local/lib/python3.12/dist-packages && find vllm -name "*.py" -type f | sort | xargs sha256sum'
docker run --rm --network none --entrypoint sh "$BASE" -c "$M" > manifest-base.txt
docker run --rm --network none --entrypoint sh "$A1" -c "$M" > manifest-a1.txt
docker run --rm --network none --entrypoint sh "$TAG" -c "$M" > manifest-derived.txt
diff manifest-base.txt manifest-derived.txt | grep '^[<>]' | awk '{print $1, $3}' | sort -k2 > manifest-diff.txt
diff manifest-a1.txt manifest-derived.txt | grep '^[<>]' | awk '{print $1, $3}' | sort -k2 > manifest-diff-vs-a1.txt
N=$(cut -d' ' -f2 manifest-diff.txt | sort -u | wc -l)
NA=$(cut -d' ' -f2 manifest-diff-vs-a1.txt | sort -u | tr '\n' ' ')
echo "py files differing or added vs base: $N (expected 6)"; cat manifest-diff.txt
echo "py files differing vs attempt-1 image: $NA(expected vllm/forcewin_hooks.py only)"; cat manifest-diff-vs-a1.txt
[ "$N" = 6 ] || { echo "ABORT: unexpected manifest difference vs base"; exit 1; }
[ "$NA" = "vllm/forcewin_hooks.py " ] || { echo "ABORT: unexpected manifest difference vs attempt 1"; exit 1; }
# the installed module is the reviewed one
IH=$(docker run --rm --network none --entrypoint sha256sum "$TAG" /usr/local/lib/python3.12/dist-packages/vllm/forcewin_hooks.py | cut -d' ' -f1)
[ "$IH" = "$(sha256sum "$HV/hook/forcewin_hooks.py" | cut -d' ' -f1)" ] || { echo "ABORT: installed module differs from hook/forcewin_hooks.py"; exit 1; }
echo "installed forcewin_hooks.py == hook/forcewin_hooks.py ($IH)"
