import torch, json
out = {}
x = torch.zeros(4, device="cuda:0"); y = x * 2; torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = x * 3
    for name, fn in [("is_available", torch.cuda.is_available), ("is_current_stream_capturing", torch.cuda.is_current_stream_capturing)]:
        try: out[name] = repr(fn())
        except Exception as e: out[name] = "EXC " + repr(e)[:200]
    try: out["combined"] = repr(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception as e: out["combined"] = "EXC " + repr(e)[:200]
torch.cuda.synchronize()
print(json.dumps(out))
