import torch, glob, json
p="/run/forcewin/snap_V4-b1-cbac17c70af8_F%swin_r0.pt"
import os
fs=sorted(glob.glob("/run/forcewin/snap_V4-b1-cbac17c70af8_*win_r0.pt"))+sorted(glob.glob("/run/forcewin/snap_V4-rebuild-cbac17c70af8_*win_r0.pt"))
for f in fs:
    a=torch.load(f,weights_only=False)
    print(f, {k:(v if not isinstance(v,(dict,list)) else type(v).__name__) for k,v in a.items()})
    for g in a["groups"]:
        print("  group", g["g"], g["block_size"], g["kernel_block"], g["bpk"], g["ids"].shape, g["ids"][:8].tolist())
    for n,L in list(a["layers"].items()):
        print("  ", n, L.get("type"), L.get("kind"), L.get("shape"), L.get("dtype"), len(L.get("pages",{})) if "pages" in L else "", sorted(L.get("pages",{}).keys())[:6] if "pages" in L else "")
    break
