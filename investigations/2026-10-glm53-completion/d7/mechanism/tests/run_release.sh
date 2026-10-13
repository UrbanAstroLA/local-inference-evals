#!/bin/bash
# D7 step 0a: run the writer and chunking checks inside each release image (CPU only: no --gpus,
# CUDA_VISIBLE_DEVICES empty, --network none, TRITON_INTERPRET=1). Nothing is mounted over engine files.
# No `set -e`: every exit code is recorded.
M=$(cd "$(dirname "$0")/.." && pwd)
OUT=$M/out/release; mkdir -p "$OUT"
declare -A IMG=(
  [v070]=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629
  [v080]=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:e4d37a01da91ec590df78af13efa9d256fc306a5a88df09908ea12ebefa32423
  [v091]=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07
  [b0]=local/glm53-v0.9.1-bundle-d1234:local
)
# served block size (server logs: "Setting attention block size") and DFlash draft slots K
declare -A BS=([v070]=3584 [v080]=4352 [v091]=4352 [b0]=4352)
declare -A K=([v070]=5 [v080]=3 [v091]=3 [b0]=3)
VP=/usr/local/lib/python3.12/dist-packages/vllm
RUN=(docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 -e PYTHONDONTWRITEBYTECODE=1
     --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$M/tests":/t:ro -v "$OUT":/o --entrypoint python3)
filt() { grep -v "^WARNING\|cpp_extension\|No CUDA runtime\|UserWarning\|warnings.warn"; }
: > "$OUT/summary.txt"
for t in v070 v080 v091 b0; do
  img=${IMG[$t]}
  id=$(docker image inspect "$img" --format '{{.Id}}')
  sums=$(docker run --rm --network none --entrypoint sha256sum "$img" \
         $VP/model_executor/layers/sparse_attn_indexer_kpool.py $VP/models/glm5next/nvidia/ops/kpool_compress.py \
         $VP/v1/core/sched/scheduler.py $VP/v1/attention/backends/mla/compressor_utils.py 2>/dev/null | awk '{print substr($1,1,12)}' | tr '\n' ' ')
  echo "$t $id files(kpool,kpool_compress,scheduler,compressor_utils): $sums" | tee -a "$OUT/summary.txt"
  "${RUN[@]}" "$img" /t/test_release_writer.py "$t" "/o/writer-$t.json" 2>&1 | filt > "$OUT/writer-$t.log"
  echo "$t writer rc=${PIPESTATUS[0]} $(tail -n 1 "$OUT/writer-$t.log")" | tee -a "$OUT/summary.txt"
  "${RUN[@]}" "$img" /t/test_release_chunking.py "$t" "${BS[$t]}" "${K[$t]}" "/o/chunking-$t.json" 2>&1 | filt > "$OUT/chunking-$t.log"
  echo "$t chunking rc=${PIPESTATUS[0]} $(python3 -c "import json;d=json.load(open('$OUT/chunking-$t.json'));print('any_unaligned',d['any_unaligned'],'residues',d['all_residues_seen'],'single',{k:[a for a,_ in v] for k,v in d['single_request_unaligned_starts'].items()})" 2>&1)" | tee -a "$OUT/summary.txt"
done
echo "done $(date -u +%FT%TZ)" | tee -a "$OUT/summary.txt"
