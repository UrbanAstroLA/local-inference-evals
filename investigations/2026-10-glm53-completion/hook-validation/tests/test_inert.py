#!/usr/bin/env python3
"""Inertness of the hooked files (same deterministic workload, three installs, outputs hashed):
  orig               the B1 image's installed files (no hook module present)
  hooked-off         hooked files + module, FORCEWIN unset
  hooked-on-unkeyed  hooked files + module, FORCEWIN=on, no request keyed
Workload: the real RejectionSampler (K=3, 3 requests, seeded), the real Sampler (single rows), the real
KpoolTailMetadataBuilder.build on a mixed decode / prefill / padded batch. compare_inert.py requires equal hashes.
Variant hooked-mapguard-import: the hooked B1-mapguard index-K file imports with both instrumentations present.
"""
import hashlib
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
VAR = sys.argv[1]
from fwtest_common import V  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CPU = torch.device("cpu")
h = hashlib.sha256()


PARTS = []


def add(t):
    b = t.detach().cpu().contiguous().numpy().tobytes()
    h.update(b)
    PARTS.append(hashlib.sha256(b).hexdigest()[:12])


if VAR == "hooked-mapguard-import":
    import vllm.model_executor.layers.sparse_attn_indexer  # noqa: F401  (import order, as b0emu)
    import vllm.models.glm5next.nvidia.ops.kpool_compress  # noqa: F401
    import vllm.models.glm5next.nvidia.attention  # noqa: F401  (imports the index-K op as the model does)
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as KP
    src = open(KP.__file__).read()
    ok = ("local-forcewin" in src and "local-mapguard" in src and KP._FW_ON and KP._fw is not None
          and hasattr(KP, "_MAPGUARD_ON"))
    print("%s MG1 hooked B1-mapguard index-K file imports with both instrumentations (forcewin on: %s)"
          % ("PASS" if ok else "FAIL", KP._FW_ON))
    sys.exit(0 if ok else 1)

from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.worker.gpu.sample.sampler import Sampler  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler  # noqa: E402
from vllm.v1.worker.gpu.states import RequestState  # noqa: E402
import vllm.v1.attention.backends.mla.indexer as IDX  # noqa: E402
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402

hooked = "local-forcewin" in open(IDX.__file__).read()
if VAR == "image-off":   # built image, env unset: hooked files present, every site inert
    import vllm.model_executor.layers.sparse_attn_indexer  # noqa: F401  (import order as in the model)
    import vllm.models.glm5next.nvidia.ops.kpool_compress  # noqa: F401
    import vllm.models.glm5next.nvidia.attention  # noqa: F401
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as KP
    import vllm.v1.worker.gpu.model_runner as MR
    print("%s IMG2 built image with FORCEWIN unset: files hooked, _FW_ON False everywhere" % (
        "PASS" if hooked and not IDX._FW_ON and not MR._FW_ON and not KP._FW_ON else "FAIL"))
print("variant", VAR, "hooked files", hooked, "env", os.environ.get("FORCEWIN"))
K, R = 3, 3
rs = RequestState(4, 512, 256, K, V, CPU)
smp = Sampler(4, V, CPU, rs, num_speculative_tokens=K)
rej = RejectionSampler(smp, types.SimpleNamespace(num_speculative_tokens=K, rejection_sample_method="standard",
                                                  synthetic_acceptance_rates=None), CPU)
for i in range(R):
    rs.add_request("q%d" % i, 20, list(range(100, 120)), 0, 50)
    smp.add_request(rs.req_id_to_index["q%d" % i], 20, SamplingParams(temperature=1.0, top_p=0.95, seed=40 + i))
rs.apply_staged_writes()
smp.apply_staged_writes()
idx = [rs.req_id_to_index["q%d" % i] for i in range(R)]
g = torch.Generator().manual_seed(9)
L = R * (K + 1)
logits = torch.randn(L, V, generator=g) * 3
ids = torch.randint(0, V, (L,), generator=g).to(torch.int32)
pos = torch.tensor([30 + j for _ in range(R) for j in range(K + 1)], dtype=torch.int64)
ib = types.SimpleNamespace(
    num_reqs=R, idx_mapping=torch.tensor(idx, dtype=torch.int32), idx_mapping_np=np.array(idx),
    input_ids=ids, positions=pos, logits_indices=torch.arange(L), cu_num_logits=torch.tensor([0, 4, 8, 12], dtype=torch.int32),
    cu_num_logits_np=np.array([0, 4, 8, 12], dtype=np.int32),
    expanded_idx_mapping=torch.tensor([i for i in idx for _ in range(K + 1)], dtype=torch.int32),
    expanded_local_pos=torch.tensor(list(range(K + 1)) * R, dtype=torch.int32),
    seq_lens=torch.tensor([34] * R, dtype=torch.int32))
out = rej(logits.clone(), ib, None)
# entries past num_sampled are uninitialised scratch (torch.empty): hash only the committed ones
_ns = out.num_sampled.long()
_mask = torch.arange(K + 1)[None, :] < _ns[:, None]
add(torch.where(_mask, out.sampled_token_ids.long(), torch.full_like(out.sampled_token_ids.long(), -7)))
add(out.num_sampled); add(out.num_rejected)
ib1 = types.SimpleNamespace(**dict(vars(ib), logits_indices=torch.tensor([3, 7, 11]),
                                   cu_num_logits=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
                                   cu_num_logits_np=np.array([0, 1, 2, 3], dtype=np.int32),
                                   expanded_idx_mapping=torch.tensor(idx, dtype=torch.int32),
                                   expanded_local_pos=torch.zeros(R, dtype=torch.int32)))
o1 = smp(logits[[3, 7, 11]].clone(), ib1)
add(o1.sampled_token_ids); add(o1.num_sampled)
b = object.__new__(IDX.KpoolTailMetadataBuilder)
b.kv_cache_spec = types.SimpleNamespace(block_size=8)
bt = torch.zeros(4, 64, dtype=torch.int32)
bt[:3, 0] = torch.tensor([5, 6, 7], dtype=torch.int32)
p2 = torch.tensor([40, 41, 42, 43, 61, 62, 63, 64, 8, 9, 10, 11, 0, 0, 0, 0], dtype=torch.int64)
qsl = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int64)
sm = torch.arange(16, dtype=torch.int64)
cm = CommonAttentionMetadata(query_start_loc=qsl, query_start_loc_cpu=qsl.clone(),
                             seq_lens=torch.tensor([44, 65, 12, 0], dtype=torch.int32), num_reqs=4,
                             num_actual_tokens=16, max_query_len=4, max_seq_len=65, block_table_tensor=bt,
                             slot_mapping=sm, positions=p2)
md = IDX.KpoolTailMetadataBuilder.build(b, 0, cm)
add(md.slot_mapping); add(sm)
print("PARTS", PARTS)
print("HASH", h.hexdigest())
