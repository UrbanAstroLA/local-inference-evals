import os,sys
os.environ["TRITON_INTERPRET"]="1"
sys.path.insert(0,"/t")
import torch
import vllm.model_executor.layers.sparse_attn_indexer
from vllm.models.glm5next.nvidia.ops.kpool_compress import kpool_compress_and_write_cache
from d7lib import *
import d7lib
g=torch.Generator().manual_seed(11)
ape=(torch.randn(4,128,generator=g)*0.1).float()
N=6000
k=torch.randn(N,4,128,generator=g).to(torch.bfloat16); s=torch.randn(N,4,128,generator=g).to(torch.bfloat16)
qk,qs=kpool_compress_and_write_cache(torch.zeros(4,64,132,dtype=torch.uint8),k,s,ape,torch.zeros(N,dtype=torch.int64),pool_size=4,head_dim=128,round_scale=True,return_compressed=True,write_cache=False)
bad=[]
for i in range(N):
  r=ref_record(k[i],s[i],ape)
  a=torch.cat([qk[i].view(torch.uint8), qs[i:i+1].view(torch.uint8)])
  if not torch.equal(a,r): bad.append((i,int((a!=r).sum()), bool(torch.equal(a[128:],r[128:]))))
print("mismatch", len(bad), "of", N, bad[:10])
if bad:
  i=bad[0][0]
  kf=k[i].float(); sc=s[i].float()+ape; m=sc.max(0).values
  acc=torch.zeros(128);den=torch.zeros(128)
  for j in range(4):
    p=torch.exp(sc[j]-m); den+=p; acc+=kf[j]*p
  x=d7lib._bf16(acc/den,"interp")
  x2=d7lib._bf16((x@d7lib._H)*torch.tensor(0.08838834764831845),"interp")
  absmax=x2.abs().max(); print("absmax",float(absmax), "absmax/448", float(absmax*torch.tensor(1/448.)), "log2", float(torch.log2(absmax*torch.tensor(1/448.))))
  print("scale kern", float(qs[i]), "ref", float(torch.exp2(torch.ceil(torch.log2(absmax*torch.tensor(1/448.))))))
  a=qk[i].view(torch.uint8); r=ref_record(k[i],s[i],ape)[:128]
  idx=(a!=r).nonzero().flatten()
  print("idx",idx.tolist()[:10]); print("kern",qk[i].float()[idx][:5].tolist(),"ref", r.clone().view(torch.float8_e4m3fn).float()[idx][:5].tolist(), "x/scale", (x2/qs[i])[idx][:5].tolist())
