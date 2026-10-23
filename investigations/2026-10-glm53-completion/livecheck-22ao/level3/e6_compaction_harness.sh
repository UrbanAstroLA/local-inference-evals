#!/bin/bash
# E6 (level 3, compaction and tail columns): the stock _compact_dropped_selection, _mask_page_table_after_nsa_len and
# _expand_pools_and_append_tail Triton kernels, AST-extracted from the stock 0.9.1 base image and run on CPU under
# TRITON_INTERPRET=1 (no GPU, --network none). Recovered verbatim from the source-reading session; only the output
# redirection differs (to e6_compaction.out instead of tail -40).
docker run --rm -i --network none -e TRITON_INTERPRET=1 -e CUDA_VISIBLE_DEVICES= --entrypoint python3 ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07 - <<'PY' 2>&1 | tee "$(dirname "$0")/e6_compaction.out"
import ast, sys, time, hashlib
SP="/usr/local/lib/python3.12/dist-packages/vllm/"
A=SP+"v1/attention/backends/mla/b12x_mla_sparse.py"
K=SP+"models/glm5next/nvidia/ops/kpool_compress.py"
def grab(path, names):
    src=open(path).read(); print(path.split('/')[-1], hashlib.sha256(src.encode()).hexdigest()[:16])
    tree=ast.parse(src); out=[]
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            out.append(ast.get_source_segment(src, n, padded=False))
            # include decorators
            out[-1]="@triton.jit\n"+out[-1] if any(getattr(d,'attr',None)=='jit' for d in n.decorator_list) else out[-1]
    return out
code=["import torch, triton\nimport triton.language as tl\n"]
code+=grab(A,["_mask_page_table_after_nsa_len_kernel","_mask_page_table_after_nsa_len","_compact_dropped_selection_kernel","_compact_dropped_selection"])
code+=grab(K,["_expand_pools_and_append_tail_kernel","expand_pools_and_append_tail"])
open("/tmp/k.py","w").write("\n\n".join(code))
sys.path.insert(0,"/tmp"); import k, torch
W=2048; CW=64
def model_attended(tab, nsa):
    # per 64-chunk list of (col, idx) the decode kernel treats as valid: col < nsa and idx >= 0
    s=int(min(max(nsa,0),W)); res={}
    for c in range(s):
        if tab[c]>=0: res.setdefault(c//CW,[]).append((c,int(tab[c])))
    return res
t0=time.time(); n=0; mism=0; fired={}; extra_chunks=0
gen=torch.Generator().manual_seed(1)
for order in ("sorted","shuffled"):
  for p1 in range(1,2048):          # p1 = p+1 = indexer seq (dense regime and boundary)
    pool_len=p1//4
    pid=torch.full((1,511),-1,dtype=torch.int64)
    if pool_len:
        ids=torch.arange(min(pool_len,511))
        if order=="shuffled": ids=ids[torch.randperm(len(ids),generator=gen)]
        pid[0,:len(ids)]=ids
    seq=torch.tensor([p1],dtype=torch.int32)
    exp=k.expand_pools_and_append_tail(pid,seq,4)          # (1, 2047)
    buf=torch.full((1,W),-1,dtype=torch.int32); buf[:,:exp.shape[1]]=exp   # buffer reset to -1 each step
    outs=[]
    for r in range(0,4):
        tab=buf.clone(); nsa=torch.tensor([min(p1+r,W)],dtype=torch.int32)
        k._compact_dropped_selection(tab,nsa); k._mask_page_table_after_nsa_len(tab,nsa)
        outs.append((tab[0].clone(), int(nsa[0])))
    base_tab, base_nsa = outs[0]
    att0=model_attended(base_tab, base_nsa)
    for r in range(1,4):
        tab,nsa=outs[r]; n+=1
        same_tab=torch.equal(tab,base_tab)
        att=model_attended(tab,nsa)
        if not same_tab or att!=att0: mism+=1
        if nsa!=base_nsa and (nsa+CW-1)//CW != (base_nsa+CW-1)//CW: extra_chunks+=1
        fired[(nsa==base_nsa)]=fired.get(nsa==base_nsa,0)+1
print("cases",n,"table-or-attended-set mismatches",mism,"cases where post-compaction nsa equal:",fired.get(True,0),"differ:",fired.get(False,0),"of which extra all-masked 64-chunk:",extra_chunks,"secs",round(time.time()-t0,1))
PY