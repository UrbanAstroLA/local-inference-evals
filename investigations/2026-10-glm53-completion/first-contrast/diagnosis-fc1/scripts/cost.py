#!/usr/bin/env python3
"""Post-inspection cost sketch for a NEW first-contrast attempt (diagnosis-fc1). Inputs are the run-fc1 measurements;
everything else is labelled as an assumption. Prints a table; writes ../cost.json. Not an authorization estimate:
any new attempt is re-priced by its own frozen proposal.
"""
import json

INV = "<runs>/first-contrast"
G = json.load(open(INV + "/fixtures/position_plan.json"))
P = {f["fixture"]: int(f["P"]) for f in G["fixtures"]}
W = {f["fixture"]: int(f["W"]) for f in G["fixtures"]}
B, LP = 4352, 7168

# measured in run-fc1 (client.jsonl phase records, run-fc1.out)
STARTUP = 732.0                       # readiness after container start
V0 = 6.2
LW1, LW2 = 297.9, 301.4               # cold keyed references, 8 prompts each (P + 7,168 / P + 8,192)
WARM_COLD = 266.7                     # warm-s1: 738,896 prompt tokens cold
C_BATCH = 299.6                       # C1: 8 x ~8.2k forced tokens, remainder prefill incl. the EAGLE block (+M3 cold)
DUMP = 6.0                            # client sleep after each phase dump
TAIL = 1955 - STARTUP - (1792087172.496 - 1792085964.473)   # cleanup + client start overhead, measured
PF = 738896 / WARM_COLD               # prompt tok/s, warm-s1 (cold, serial chunks)
DEC = 8196 / (1792087098.788 - 1792086874.964)              # C1-L3 decode tok/s (first decode step -> done)

# derived
rem_eagle = sum(P[n] % B + B for n in P)                    # remainder + recomputed EAGLE block per batch
rem_eagle_L = sum((P[n] + LP) % B + B for n in P)
warm_hit = rem_eagle / PF + DUMP                            # a re-warm that hits: only the uncached tail
lp_batch = rem_eagle_L / PF + max(W[n] - LP + 1 for n in W) / DEC + DUMP
warm_lp_cold = sum(P[n] + LP for n in P) / PF + DUMP
cv_cold = (4 * (113152 - 63) + 4 * (95744 - 63)) / PF + 72 / DEC * 4 + DUMP   # worst case: 8 cold CV prompts
cv_hit = 8 * (4289 + B) / PF + 72 / DEC * 4 + DUMP
ps_set_cold = sum(P.values()) / PF + 8 * 1.0 + DUMP          # one at a time, cold
ps_set_hit = rem_eagle / PF + 8 * 1.0 + DUMP
GATE = 54.0                                                 # gate_fc.py on stage-A data, CPU, this diagnosis (overlapped)


def total(parts):
    return sum(v for _, v in parts)


scen = {}
# A: stage-A-only validation session (V0, LW1, LW2, s1 warm + C1, s2 warm + C2, CV), GATE-1 offline afterwards
scen["A_stageA_only"] = [("startup", STARTUP), ("V0", V0 + DUMP), ("LW1+LW2", LW1 + LW2 + 2 * DUMP),
                         ("warm s1, s2 (cold)", 2 * (WARM_COLD + DUMP)), ("C1, C2", 2 * (C_BATCH + DUMP)),
                         ("CV (cold, worst case)", cv_cold), ("tail", TAIL)]
# B: full option B as frozen, every re-warm / CV / PS a miss (eviction unfixed; accounting fixed as an inequality)
lp_part = [("LC1, LC2 re-warms cold", 2 * warm_lp_cold), ("LC1, LC2", 2 * lp_batch),
           ("LD2, LD1 re-warms cold", 2 * warm_lp_cold), ("LD2, LD1", 2 * lp_batch),
           ("LW3 + LD3 + re-warm l3 cold + LC3", LW1 + DUMP + lp_batch + warm_lp_cold + lp_batch)]
scen["B_full_all_misses"] = scen["A_stageA_only"][:-2] + [("CV (cold)", cv_cold)] + lp_part + [
    ("D2, D1 re-warms cold", 2 * (WARM_COLD + DUMP)), ("D2, D1", 2 * (C_BATCH + DUMP)),
    ("warm s3 cold", WARM_COLD + DUMP), ("PS x3 (cold)", 3 * ps_set_cold), ("D3", C_BATCH + DUMP),
    ("re-warm s3 cold", WARM_COLD + DUMP), ("C3", C_BATCH + DUMP), ("tail", TAIL)]
# C: full option B with eviction designed out (every re-warm / CV / PS hits on the corrected rule)
lp_hit = [("LC1, LC2 re-warms hit", 2 * warm_hit), ("LC1, LC2", 2 * lp_batch), ("LD2, LD1 re-warms hit", 2 * warm_hit),
          ("LD2, LD1", 2 * lp_batch), ("LW3 + LD3 + re-warm l3 hit + LC3", LW1 + DUMP + lp_batch + warm_hit + lp_batch)]
scen["C_full_hits"] = scen["A_stageA_only"][:-2] + [("CV (hit)", cv_hit)] + lp_hit + [
    ("D2, D1 re-warms hit", 2 * warm_hit), ("D2, D1", 2 * (C_BATCH + DUMP)), ("warm s3 cold", WARM_COLD + DUMP),
    ("PS x3 (hit)", 3 * ps_set_hit), ("D3", C_BATCH + DUMP), ("re-warm s3 hit", warm_hit), ("C3", C_BATCH + DUMP),
    ("tail", TAIL)]

out = dict(inputs=dict(startup=STARTUP, V0=V0, LW1=LW1, LW2=LW2, warm_cold=WARM_COLD, C_batch=C_BATCH, dump=DUMP,
                       tail=round(TAIL, 1), prefill_tok_s=round(PF), decode_tok_s_per_request=round(DEC, 1),
                       gate_cpu_s=GATE),
           scenarios={k: dict(parts=[(a, round(b, 1)) for a, b in v], seconds=round(total(v)),
                              gpu_hours=round(total(v) * 2 / 3600, 2)) for k, v in scen.items()})
json.dump(out, open(INV + "/diagnosis-fc1/cost.json", "w"), indent=1)
print("inputs", out["inputs"])
for k, v in out["scenarios"].items():
    print("%-20s %6d s  %.2f GPU-h" % (k, v["seconds"], v["gpu_hours"]))
    for a, b in v["parts"]:
        print("      %-36s %7.1f" % (a, b))
