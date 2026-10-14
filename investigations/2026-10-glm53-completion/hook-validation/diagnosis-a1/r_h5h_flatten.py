"""CPU reproduction (hooked validation image, real forcewin_hooks.on_selection) of H5h.
On SM120 the DSA indexer flattens spec-verify rows (server.log: 'use_flattening=True ... next_n=4'), so the
decode branch calls on_selection(..., batch_size = num_decode_tokens, next_n = 1). The hook maps row t to batch row
t // next_n = t. Same 2-request batch, once in the native layout (next_n=4, as CPU test X2 used) and once flattened."""
import os, types, json
os.environ.setdefault("FORCEWIN", "on"); os.environ.setdefault("FORCEWIN_WMAX", "512")
os.environ.setdefault("FORCEWIN_DIR", "/tmp/fwtest")
import numpy as np, torch
from vllm import forcewin_hooks as fw
from vllm.sampling_params import SamplingParams
CPU = torch.device("cpu")
def run(flat):
    fw.S.__init__()
    runner = types.SimpleNamespace(device=CPU, num_speculative_steps=3, max_num_reqs=4,
                                   model_config=types.SimpleNamespace(get_vocab_size=lambda: 154880))
    PA, PB = 101, 77
    desA = list(range(0, 80)); desB = list(range(0, 80))
    fw.add_request(runner, "a", 2, PA, types.SimpleNamespace(sampling_params=SamplingParams(max_tokens=300, extra_args={
        "forcewin_hooks": [7] * 300, "local_fw_tag": "a", "local_fw_desig": desA})))
    fw.add_request(runner, "b", 0, PB, types.SimpleNamespace(sampling_params=SamplingParams(max_tokens=300, extra_args={
        "forcewin_hooks": [7] * 300, "local_fw_tag": "b", "local_fw_desig": desB})))
    ib = types.SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor([2, 0], dtype=torch.int32), idx_mapping_np=np.array([2, 0]),
                               is_prefilling_np=np.zeros(2, dtype=bool), num_tokens=8, num_draft_tokens=6)
    fw.on_batch(runner, ib)
    pos_a = [PA + 40 + j for j in range(4)]; pos_b = [PB + 10 + j for j in range(4)]
    dec_seq = torch.tensor([p + 1 for p in pos_a + pos_b], dtype=torch.int32)
    pids = torch.full((8, 16), -1, dtype=torch.int64); pids[:, :3] = torch.tensor([1, 2, 3])
    if flat:
        fw.on_selection("model.layers.3.self_attn.indexer.k_cache", pids, dec_seq, 8, 1)
    else:
        fw.on_selection("model.layers.3.self_attn.indexer.k_cache", pids, dec_seq, 2, 4)
    exp = {("a", p + 1 - PA) for p in pos_a} | {("b", p + 1 - PB) for p in pos_b}
    got = set()
    for name, idx, des in (("a", 2, desA), ("b", 0, desB)):
        for k, o in enumerate(des):
            if int(fw.S.xbuf[idx, k, 0, 0]) >= 0:
                got.add((name, o))
    return dict(layout="flattened [B*4,1]" if flat else "native [B,4]", expected=sorted(exp), recorded=sorted(got),
                correct=sorted(exp & got), missing=sorted(exp - got), misattributed=sorted(got - exp))
res = [run(False), run(True)]
for r in res: print(json.dumps(r))
json.dump(res, open("/d/out/h5h_flatten_repro.json", "w"), indent=1)
