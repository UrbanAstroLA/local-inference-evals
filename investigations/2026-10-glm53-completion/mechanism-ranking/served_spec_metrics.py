"""Served speculative-decoding acceptance and concurrency from GPQA server logs (counts only; read-only).
Usage: python3 served_spec_metrics.py <server.log> ..."""
import re,sys,collections
for f in sys.argv[1:]:
    A=D=0; pos=None; n=0; npos=0; run=collections.Counter(); kv=0
    for line in open(f,errors='replace'):
        m=re.search(r'Accepted: (\d+) tokens, Drafted: (\d+) tokens, Per-position acceptance rate: ([\d., ]+?), Avg',line)
        if m:
            a,d=int(m[1]),int(m[2]); r=[float(x) for x in m[3].split(',')]
            A+=a;D+=d;n+=1
            K=len(r); w=d/K
            if pos is None: pos=[0.0]*K
            for i,x in enumerate(r): pos[i]+=x*w
            npos+=w
        m=re.search(r'Running: (\d+) reqs',line)
        if m: run[int(m[1])]+=1
        m=re.search(r'GPU KV cache usage: ([\d.]+)%',line)
        if m: kv=max(kv,float(m[1]))
    K=len(pos)
    print(f.split('/')[-1]); print(f"  lines {n} K={K} accepted/drafted={A/D:.3f} mean acceptance length={1+K*A/D:.2f} per-pos={[round(p/npos,3) for p in pos]}")
    tot=sum(run.values()); print(f"  status lines {tot}; share running>=8: {sum(v for k,v in run.items() if k>=8)/tot:.2f}; share running 1-2: {sum(v for k,v in run.items() if 1<=k<=2)/tot:.2f}; share running 3-4: {sum(v for k,v in run.items() if 3<=k<=4)/tot:.2f}; max kv {kv}")
