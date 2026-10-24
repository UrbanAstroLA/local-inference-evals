import numpy as np, glob, os, collections, sys
P = "/p4/PRIVATE-runs-stage0/stage0-attempt-1"
files = sorted(glob.glob(os.path.join(P, "**", "entry_*_r*.npz"), recursive=True))
print("entry files", len(files))
by = {}
for f in files:
    b = os.path.basename(f)[len("entry_"):-4]          # e.g. V1-R01-early-r1_r0
    tag, rank = b.rsplit("_r", 1)
    var, rest = tag.split("-", 1); wid, rep = rest.rsplit("-r", 1)
    by[(var, wid, int(rep), int(rank))] = f
summ = collections.defaultdict(lambda: [0, 0])   # (var, kind, tensor_family) -> [diff blocks, compared blocks]
first_pt = collections.Counter()
for (var, wid, rep, rank), f1 in sorted(by.items()):
    if rep != 1: continue
    f2 = by.get((var, wid, 2, rank))
    if not f2: continue
    e1, e2 = np.load(f1, allow_pickle=False), np.load(f2, allow_pickle=False)
    d1, d2, grp, tens = e1["edig"], e2["edig"], e1["tensor_group"], list(e1["tensors"])
    pts = e1["points"]
    for pt in range(pts.shape[0]):
        kind = "entry" if int(pts[pt][1]) == 1 else "prefill"
        for ti, tn in enumerate(tens):
            g = int(grp[ti])
            if g < 0: continue
            fam = str(tn).split(".self_attn.")[-1] if ".self_attn." in str(tn) else str(tn)
            L = str(tn).split("layers.")[1].split(".")[0] if "layers." in str(tn) else "?"
            nlog = int((e1["eblk"][pt, g] >= 0).sum())
            for j in range(nlog):
                b1, b2 = int(e1["eblk"][pt, g, j]), int(e2["eblk"][pt, g, j])
                if b1 == 0 or b2 == 0: continue
                eq = np.array_equal(d1[pt, ti, j], d2[pt, ti, j])
                k = (var, kind, pt, fam)
                summ[k][1] += 1; summ[k][0] += int(not eq)
                if not eq: first_pt[(var, kind, pt, fam, L)] += 1
for k in sorted(summ):
    a, n = summ[k]; print(k, f"differ {a}/{n}")
print("--- differing blocks by layer (top)")
for k, v in first_pt.most_common(25): print(k, v)
print("points sample:", np.load(files[0])["points"][:6].tolist())
