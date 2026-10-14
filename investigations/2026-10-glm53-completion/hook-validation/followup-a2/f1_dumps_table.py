"""Follow-up a2, check 1: the dump files each B1 comparison actually read (as analyze_hookval.py win_dump resolves
them: one window snapshot per tag per rank), with sha256 prefixes, request seed / batch, engine step at the dump,
and the group-0 block-table hash. -> out/f1_dumps_table.md"""
import json
D = "<runs>/hook-validation/followup-a2/out/"
M = json.load(open(D + "dump_meta.json"))
tids = sorted({k.split("|")[0].rsplit("-", 1)[-1] for k in M})
L = ["| Fixture | Arm (tag) | sha256 r0 / r1 (first 12) | seed | batch | step | block table |", "|---|---|---|---:|---|---:|---|"]
for t in tids:
    for tag in (f"V4-b1-{t}", f"V4n-b1b-{t}", f"V4-rebuild-{t}", f"V4n-rebuild2-{t}"):
        a, b = M[f"{tag}|r0"], M[f"{tag}|r1"]
        L.append(f"| {t} | {tag} | {a['sha256'][:12]} / {b['sha256'][:12]} | {a['client']['seed']} | "
                 f"{a['client']['batch'] or '(none recorded)'} | {a['step']} | {a['g0_ids_sha']} |")
open(D + "f1_dumps_table.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
