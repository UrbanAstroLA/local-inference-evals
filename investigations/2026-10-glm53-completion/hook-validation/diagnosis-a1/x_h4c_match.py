import json, collections, numpy as np
H4 = json.load(open("/d/out/h4c_idx.json")); NL = json.load(open("/d/out/h4c_nulls.json"))["P"]
keys = {(r["tag"], r["layer"], r["pool"]) for r in H4}
m = [r["rel"] for r in NL if (r["g"], r["layer"], r["pool"]) in keys]
a = np.array(m); b = np.array([r["rel"] for r in H4])
print("H4c pools: dec-vs-prefill n %d gross %.4f q50 %.4f | same pools prefill-vs-prefill n %d gross %.4f q50 %.4f q999 %.3f" % (
    b.size, np.mean(b > .25), np.median(b), a.size, np.mean(a > .25), np.median(a), np.quantile(a, .999)))
# window-only pools (decode-written) in the null overall
P0 = {}
for r in H4: P0[r["tag"]] = r["P"]
w = np.array([r["rel"] for r in NL if r["pool"] * 4 >= P0.get(r["g"], 10**9)])
print("prefill-vs-prefill restricted to window positions (>= P): n %d gross %.4f q999 %.3f" % (w.size, np.mean(w > .25), np.quantile(w, .999)))
