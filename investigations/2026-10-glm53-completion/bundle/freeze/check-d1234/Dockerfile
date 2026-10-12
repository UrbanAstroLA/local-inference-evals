# VERIFICATION image for the bundle gate ONLY (PLAN.md amendment 14 (3)); NOT for the GPQA screen.
# = bundle d1234 + three observe-only observers, each inert unless its variable is set, each previously run on a GPU:
#   col511check.py (gate/freeze, GLM53_COL511CHECK=device; col511fix-check image 3e3d887b, gate PASS)
#   kpool_tailguard.py   (GLM53_TAILGUARD=device; tailfix-u-guard image 301baa73, U-graph-a1)
#   verify13_capture.py    (GLM53_VERIFY13=1; verify13 serving image ff6c8542, Stage A server-a1)
#   d4limit.py     (GLM53_D4_LIMIT_OVERRIDE=1|none or GLM53_D4_INSPECT=1; NEW, CPU-tested only: D4 plan inspection and
#                           reachability override for the gate)
# Engine files changed vs the bundle: sparse_attn_indexer_kpool.py (+28 observer lines) and gate_linear.py (+16), additions only.
FROM local/glm53-v0.9.1-bundle-d1234:local@sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548
ARG VP=/usr/local/lib/python3.12/dist-packages/vllm
RUN printf '%s\n' "9104be37c8cb876a436f5fac11401c4abdb254529a6c37f3dca8121f41a5b61d  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py" "2499ee679a3a8816f45eb4034a039770c90cccdcf720a38dca5d4aa10e275c2a  ${VP}/model_executor/layers/fused_moe/router/gate_linear.py" | sha256sum -c -
COPY col511check.py kpool_tailguard.py verify13_capture.py d4limit.py ${VP}/
COPY sparse_attn_indexer_kpool.py ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py
COPY gate_linear.py ${VP}/model_executor/layers/fused_moe/router/gate_linear.py
COPY test_c511check.py additions_only.py test_tailguard.py test_verify13.py test_fix.py test_d4limit.py d4limit.py bundle_sparse_attn_indexer_kpool.py bundle_gate_linear.py      col511check.py kpool_tailguard.py verify13_capture.py sparse_attn_indexer_kpool.py gate_linear.py      sparse_attn_indexer_kpool.py.observer-additions.diff gate_linear.py.observer-additions.diff tailguard.diff /opt/bundle-check/
RUN printf '%s\n' "90ae3113e5557166fb8022e0e89377e0469986da6f567f7e3e21eccd0958558c  ${VP}/col511check.py"       "414f5b5c1993d772b17ef641b79b683c7980a1c0140ff77e7b02efe7a4a71e6e  ${VP}/kpool_tailguard.py"       "96dcd5e36bd75497b823a817ae2b08602db7eef1e20c0f7ab54d883c2872fe9c  ${VP}/verify13_capture.py"       "fae1127c1e7dee3d8408af9299bf19149e41cc9bc670baacd1230184f0a8c13d  ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py"       "02ba81040d01dfba5a8cbeb58685acdadc75bf79a4cbb4760502983752950a58  ${VP}/model_executor/layers/fused_moe/router/gate_linear.py" \
      "49597e4556dcd1049613cf305cd2e8c2c6f3e39446817a068959e9d76aa4265d  ${VP}/d4limit.py" | sha256sum -c - \
 && python3 -m py_compile ${VP}/d4limit.py ${VP}/col511check.py ${VP}/kpool_tailguard.py ${VP}/verify13_capture.py \
      ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py ${VP}/model_executor/layers/fused_moe/router/gate_linear.py \
 && cd /opt/bundle-check \
 && python3 additions_only.py bundle_sparse_attn_indexer_kpool.py ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py \
 && python3 additions_only.py bundle_gate_linear.py ${VP}/model_executor/layers/fused_moe/router/gate_linear.py \
 && test "$(grep -c '_tailguard.check_call(' ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 2 \
 && test "$(grep -c 'col511-fix' ${VP}/model_executor/layers/sparse_attn_indexer_kpool.py)" = 4 \
 && env -u GLM53_COL511CHECK -u GLM53_TAILGUARD -u GLM53_VERIFY13 -u GLM53_D4_LIMIT_OVERRIDE -u GLM53_D4_INSPECT CUDA_VISIBLE_DEVICES= python3 -c "import sys; import vllm.models.glm5next.nvidia.attention; import vllm.model_executor.layers.sparse_attn_indexer_kpool as k; from vllm.model_executor.layers.fused_moe.router import gate_linear as gl; import vllm.v1.worker.gpu.block_table as b; import vllm.col511check as c, vllm.kpool_tailguard as t; assert not c.ENABLED and not t.ENABLED and not k._C511_ON and not k._TAILGUARD_ON; assert not getattr(b.BlockTables, '_col511check', False) and not getattr(b.BlockTables, '_tailguard', False); assert not getattr(gl.GateLinear, '_v13', False) and 'vllm.verify13_capture' not in sys.modules and 'vllm.d4limit' not in sys.modules and not getattr(gl.GateLinear, '_d4limit', False); print('all observers inert by default: ok')" \
 && { CUDA_VISIBLE_DEVICES= GLM53_COL511CHECK=device GLM53_COL511CHECK_DIR=/tmp/c511-bt python3 test_c511check.py --device cpu > /tmp/c511.log 2>&1; rc=$?; grep -E "^(PASS|FAIL)|passed" /tmp/c511.log; test $rc = 0; } \
 && { CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 GLM53_TAILGUARD=device GLM53_TAILGUARD_DIR=/tmp/tg-bt python3 test_tailguard.py > /tmp/tg.log 2>&1; rc=$?; grep -E "^(PASS|FAIL)|passed" /tmp/tg.log; test $rc = 0; } \
 && { CUDA_VISIBLE_DEVICES= python3 test_verify13.py > /tmp/v13.log 2>&1; rc=$?; grep -E "^(PASS|FAIL)|passed" /tmp/v13.log; test $rc = 0; } \
 && { CUDA_VISIBLE_DEVICES= python3 test_d4limit.py > /tmp/d4.log 2>&1; rc=$?; grep -E "^(PASS|FAIL)|failed" /tmp/d4.log; test $rc = 0; } \
 && { CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 TAILFIX_EXPECT=U python3 test_fix.py > /tmp/fix.log 2>&1; rc=$?; grep -E "^(PASS|FAIL)|met expectation" /tmp/fix.log; test $rc = 0; } \
 && rm -rf /tmp/*.log /tmp/c511-bt /tmp/tg-bt /tmp/v13test_* /tmp/d3test_* /tmp/d5test_* /root/.triton /root/.cache/vllm
LABEL io.local.patch="bundle d1234 + observe-only col511 checker, tail guard, verify13 capture and the D4 limit override (each env-gated, inert by default); verification image for the gate only" \
      io.local.base="local/glm53-v0.9.1-bundle-d1234:local@sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548"
