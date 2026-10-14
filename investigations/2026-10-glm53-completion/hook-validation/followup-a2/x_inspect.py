import torch, sys
p = sys.argv[1]
r = torch.load(p, weights_only=False)
print(type(r), list(r.keys()) if isinstance(r, dict) else None)
for k, v in r.items():
    if k in ("layers",):
        ks = list(v.keys()); print("layers", len(ks), ks[:3]);
        L = v[ks[0]]; print({kk: (type(vv).__name__, getattr(vv, 'shape', None) if not isinstance(vv, dict) else len(vv)) for kk, vv in L.items()})
        idx = [n for n, L in v.items() if L.get("kind") == "paged" and L.get("shape", [0])[-1:] == [132]]
        print("idx layers", len(idx), idx[:3]); Li = v[idx[0]]; print({kk: (vv if not hasattr(vv,'shape') and not isinstance(vv,dict) else (len(vv) if isinstance(vv,dict) else vv.shape)) for kk, vv in Li.items()})
        print("pages keys", sorted(Li["pages"].keys())[:40])
    elif k == "groups":
        for g in v: print("group", {kk: (vv.tolist()[:40] if hasattr(vv,'tolist') else vv) for kk, vv in g.items()})
    else:
        print(k, v if not hasattr(v, 'shape') else v.shape)
