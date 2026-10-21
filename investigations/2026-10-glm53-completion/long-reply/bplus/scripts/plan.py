#!/usr/bin/env python3
"""B+ request plan (PROPOSAL.md sections 3.1-3.3). Host, standard library. Writes out/plan.json: ids, seeds, hashes.

Seeds   seed = 8,160,000 + 10,000 * engine_index + 100 * prefix_number + j   (engine_index 1 = 0.7.0, 2 = 0.9.1, 3 = B1;
        prefix_number 1-12; j = 1-4): 144 distinct seeds, none used before (checks/seed-inventory.json, seed_scan.py run
        unchanged with these 144 candidates; checks/seed-formula-scan.txt). Never shared across engines or prefixes.
Order   one fixed submission order, identical on every engine: 4 rounds (j = 1..4); in each round every prefix appears
        once; the round's order is the two strata each shuffled with random.Random(ORDER_SEED + j) and interleaved
        capped/finished, the round's first stratum alternating (capped first in odd rounds). The first 8 in flight are
        therefore 8 different prefixes, 4 per stratum; seeds 2-4 of a prefix are submitted after its seed 1.
Body    {model, prompt: <prefix token ids>, max_tokens: 65536, temperature: 1.0, top_p: 0.95, seed, stop: ["</s>"]
         (the GPQA task's stop string, as in every logged chat request), return_token_ids: true, stream: false}
"""
import hashlib
import json
import random
import sys

HERE = sys.path[0]
PKG = HERE.rsplit("/", 1)[0]
ENGINES = [("v070", 1, "stock 0.7.0"), ("v091", 2, "stock 0.9.1"), ("b1", 3, "B1 = 0.9.1 + D1-D4 + D7")]
SEED_BASE = 8160000
ORDER_SEED = 20261016
MAX_TOKENS = 65536
MODEL = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1"
PARAMS = {"max_tokens": MAX_TOKENS, "temperature": 1.0, "top_p": 0.95, "stop": ["</s>"], "return_token_ids": True,
          "stream": False}


def params_sha(model=MODEL, params=PARAMS):
    return hashlib.sha256(json.dumps({"model": model, **params}, sort_keys=True).encode()).hexdigest()


def seed_of(engine_index, prefix_number, j):
    return SEED_BASE + 10000 * engine_index + 100 * prefix_number + j


def main():
    man = json.load(open(f"{PKG}/out/prefix_manifest.json"))["prefixes"]
    caps = [m["prefix_id"] for m in man if m["stratum"] == "capped"]
    fins = [m["prefix_id"] for m in man if m["stratum"] == "finished"]
    order = []
    for j in (1, 2, 3, 4):
        rng = random.Random(ORDER_SEED + j)
        c, f = caps[:], fins[:]
        rng.shuffle(c)
        rng.shuffle(f)
        a, b = (c, f) if j % 2 else (f, c)
        for x, y in zip(a, b):
            order += [(x, j), (y, j)]
    mm = {m["prefix_id"]: m for m in man}
    plan = {"what": "B+ request plan (ids, seeds, hashes; no text)", "model": MODEL, "params": PARAMS,
            "params_sha256": params_sha(), "seed_base": SEED_BASE, "order_seed": ORDER_SEED,
            "engines": [{"engine": e, "index": i, "label": lab} for e, i, lab in ENGINES],
            "order": [{"k": k, "prefix_id": p, "j": j, "stratum": mm[p]["stratum"], "prefix_len": mm[p]["prefix_len"],
                       "prefix_ids_sha256": mm[p]["prefix_ids_sha256"],
                       "seeds": {e: seed_of(i, int(p[1:]), j) for e, i, _ in ENGINES}} for k, (p, j) in enumerate(order, 1)]}
    seeds = [s for r in plan["order"] for s in r["seeds"].values()]
    assert len(seeds) == 144 and len(set(seeds)) == 144
    assert len(plan["order"]) == 48 and len({(r["prefix_id"], r["j"]) for r in plan["order"]}) == 48
    first8 = plan["order"][:8]
    assert len({r["prefix_id"] for r in first8}) == 8 and sum(r["stratum"] == "capped" for r in first8) == 4
    json.dump(plan, open(f"{PKG}/out/plan.json", "w"), indent=1)
    print("order:", " ".join(f"{r['prefix_id']}.{r['j']}" for r in plan["order"]))
    print("seeds:", min(seeds), "-", max(seeds), "distinct", len(set(seeds)), "params", plan["params_sha256"][:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
