#!/usr/bin/env python3
"""Cost input: per-request decode rate from server status lines (Avg generation throughput / Running) joined to the
request logs (active requests and their interpolated positions). Host, standard library, read only. Numbers only.
Section 1: low concurrency at >= 100k context (loop-inflated); section 2: >= 7 running, binned by total batch context."""
import json,re,datetime,statistics as st,sys
def ts(l):
    m=re.search(r" (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) ",l)
    return datetime.datetime(2026,*map(int,m.groups()),tzinfo=datetime.timezone.utc).timestamp() if m else None
def run(label, logs, reqlogs):
    reqs=[]
    for f in reqlogs:
        for l in open(f):
            r=json.loads(l); 
            if r.get('secs'): reqs.append((r['t_start'], r['t_start']+r['secs'], r['prompt_tokens'], r['completion_tokens']))
    out=[]
    for lg in logs:
        for l in open(lg,errors='replace'):
            m=re.search(r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs",l)
            if not m: continue
            t=ts(l); G=float(m.group(1)); r=int(m.group(2))
            act=[q for q in reqs if q[0]<=t-10 and q[1]>=t]
            if len(act)!=r or r==0: continue
            ctx=[q[2]+q[3]*(t-q[0])/(q[1]-q[0]) for q in act]
            out.append((r,G,min(ctx),st.mean(ctx)))
    for rr in (1,2,3,4,6,8):
        for lo in (100000,):
            sel=[G/r for r,G,mn,mc in out if r==rr and mn>=lo]
            if sel: print(label,'running',rr,'min ctx>=',lo,'n',len(sel),'per-req tok/s median',round(st.median(sel),1),'p10',round(sorted(sel)[len(sel)//10],1))
    sel=[G/r for r,G,mn,mc in out if r>=7 and mc>=60000]
    if sel: print(label,'running>=7 mean ctx>=60k n',len(sel),'median per-req',round(st.median(sel),1))
G='<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell'
B='<runs>'
run('0.7.0 rep-F',[B+'/rep070/run-F/server-s1.log',B+'/rep070/run-F/server-s2.log'],[G+'+rep070-F.s1.requests.jsonl',G+'+rep070-F.s2.requests.jsonl'])
run('0.9.1',['<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.server.log','<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log'],[G+'+v0.9.1.requests.jsonl',G+'+v0.9.1+ps.requests.jsonl'])
run('B1',[B+'/b1-screen/run-s1236/server-s1.log'],[G+'+v0.9.1+bundle-d1234+d7.s1.requests.jsonl'])

def bins(label, logs, reqlogs):
    reqs=[]
    for f in reqlogs:
        for l in open(f):
            r=json.loads(l)
            if r.get('secs'): reqs.append((r['t_start'], r['t_start']+r['secs'], r['prompt_tokens'], r['completion_tokens'], r['finish_reason']))
    out=[]
    for lg in logs:
        for l in open(lg,errors='replace'):
            m=re.search(r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs",l)
            if not m: continue
            t=ts(l); G=float(m.group(1)); r=int(m.group(2))
            act=[q for q in reqs if q[0]<=t-10 and q[1]>=t]
            if len(act)!=r or r<7: continue
            ctx=[q[2]+q[3]*(t-q[0])/(q[1]-q[0]) for q in act]
            nlong=sum(c>=100000 for c in ctx)
            out.append((G/r, sum(ctx)/1e3, nlong))
    for lo,hi in ((0,200),(200,400),(400,600),(600,800),(800,1000),(1000,1400),(1400,3000)):
        s=[x for x,tot,nl in out if lo<=tot<hi]
        if s: print(label,'total ctx (k tokens over running reqs)',lo,hi,'n',len(s),'per-req median',round(st.median(s),1))
print('== binned by total context across >=7 running requests')
bins('0.7.0',[B+'/rep070/run-F/server-s1.log',B+'/rep070/run-F/server-s2.log'],[G+'+rep070-F.s1.requests.jsonl',G+'+rep070-F.s2.requests.jsonl'])
bins('0.9.1',['<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.server.log','<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log'],[G+'+v0.9.1.requests.jsonl',G+'+v0.9.1+ps.requests.jsonl'])
bins('B1',[B+'/b1-screen/run-s1236/server-s1.log'],[G+'+v0.9.1+bundle-d1234+d7.s1.requests.jsonl'])
