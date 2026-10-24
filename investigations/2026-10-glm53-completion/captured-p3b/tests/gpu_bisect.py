import sys, os, time, importlib.util, json
sys.argv = [sys.argv[0], sys.argv[1], "/o/bisect-unused.json"]
src = open("/t/test.py").read().split("\nt0 = time.time()")[0]   # definitions only
exec(compile(src, "/t/test.py", "exec"))
import torch
def trial(name, fn):
    cap, layers, impls = fresh(); t = static_inputs()
    try:
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn(cap, layers, impls, t)
        torch.cuda.synchronize(); print("OK  ", name, flush=True)
    except Exception as e:
        print("FAIL", name, repr(e)[:160].replace("\n"," "), flush=True)
        try: torch.cuda.synchronize()
        except Exception: pass
    cap.S.stop.set()
W_ = 2048
trial("heavy only", lambda c,l,i,t: torch.tanh(t["a"] @ t["a"]))
trial("moe hook", lambda c,l,i,t: c.hook_moe(l[3], t["x"][3], t["w"][3], t["ids"][3], t["out"][3]))
trial("attn_pre", lambda c,l,i,t: c.hook_attn_pre(i[3], t["logical"], t["sel"], t["L"], t["q"].shape[0]))
def pre_in(c,l,i,t):
    n=t["q"].shape[0]; c.hook_attn_pre(i[3], t["logical"], t["sel"], t["L"], n)
    c.hook_attn_in(i[3], t["q"], t["kv"], t["sel"], t["L"].clamp(max=W_)-1, t["L"].clone(), n)
trial("attn_pre+in", pre_in)
def pre_in_out(c,l,i,t):
    pre_in(c,l,i,t); c.hook_attn_out(i[3], t["aout"], None)
trial("attn_pre+in+out", pre_in_out)
os._exit(0)
