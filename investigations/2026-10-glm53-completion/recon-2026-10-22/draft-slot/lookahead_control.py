#!/usr/bin/env python3
"""Deterministic lookahead positive control (22an step 2). Runs inside the stock 0.9.1 image like harness.py.

Construction: every draft accepted (a = K), so the committed length c grows by K + 1 = 4 per decode step. With lookahead
K (the plant) the draft-group allocation covers positions < c + 1 + K + K, i.e. up to c + 2K; the last query row writes
c + 2K + 1. The plant therefore bites exactly when (c + 2K + 1) % 1088 == 0, i.e. c ≡ 1081 (mod 1088) for K = 3.
A prompt of P = 1041 tokens (P ≡ 1081 mod 4) reaches c = 1081 after 10 decode steps and recurs every 272 steps.
Expected with the plant: one I4 (beyond horizon) and one block-0 / not-own write per crossing per request.
Expected without the plant, same alignment: 0.
"""
import json, random, sys
sys.path.insert(0, "/w")
import harness as H  # noqa: E402

P, GEN, NREQ = 1041, 2400, 4
rng = random.Random(97)
prompts = [[rng.randrange(1000, 150000) for _ in range(P)] for _ in range(NREQ)]
gens = [GEN] * NREQ
crossings_expected = sum(1 for t in range(GEN // 4 + 1) if (P + 4 * t + 2 * H.K + 1) % H.BS == 0) * NREQ
out = {"P": P, "gen": GEN, "requests": NREQ, "crossings_expected": crossings_expected, "results": []}
for name, faults, asyn in [("PLANT lookahead K, aligned, async", frozenset({"lookahead_K"}), True),
                           ("PLANT lookahead K, aligned, sync", frozenset({"lookahead_K"}), False),
                           ("CONTROL lookahead K+1, aligned, async", frozenset(), True)]:
    r = H.scenario(name, 21, NREQ, prompts, gens, lambda g: H.K, faults=faults, async_sched=asyn, kernels=("stock",))
    out["results"].append(r)
    print(name, "expected crossings", crossings_expected, r["stock"]["violations"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("written", sys.argv[1])
