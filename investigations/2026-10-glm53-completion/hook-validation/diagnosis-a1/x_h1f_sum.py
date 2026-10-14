import json, collections
V = json.load(open("/d/out/h1f_rows.json"))
print("n", len(V), collections.Counter(v["kind"] for v in V), collections.Counter(v["id"] for v in V))
print("tags", collections.Counter(v["tag"] for v in V))
print("distinct (tag,o)", len({(v["tag"], v["o"]) for v in V}))
print("max p_post", max(v["p_post"] for v in V), "max p_pre", max(v["p_pre"] for v in V), "min top1_p", min(v["top1_p"] for v in V))
print("max mass_le", max(v["mass_le"] for v in V))
print("rows/batch", collections.Counter((v["n_rows"], v["batch_tokens"], v["kind_row"]) for v in V))
print("p_post==p_pre (unmasked)?", sum(abs(v["p_post"] - v["p_pre"]) <= 1e-3 * max(v["p_pre"], 1e-45) for v in V))
for v in V[:3] + V[-3:]:
    print({k: v[k] for k in ("tag", "o", "id", "p_pre", "p_post", "mass_le", "top1_p")})
