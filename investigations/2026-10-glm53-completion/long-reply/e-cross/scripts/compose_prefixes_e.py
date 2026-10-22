#!/usr/bin/env python3
"""E prefix set under amendments 22ad / 22ag (PROPOSAL.md section 2). Host, standard library. Writes only inside e-cross.

Rule (fixed by 22ag): D's 8 prefixes keep their place and ids P01-P08; D2's selected replies on questions not already in
D's set are appended in D2's selection order and numbered P09 onwards. With D2's selection that is q69p5, q120p5,
q186p5 (capped) and q88p5, q71p5 (finished): 13 prefixes. Nothing is rebuilt or re-tokenized here: every entry is copied
verbatim from its source package's byte-identical files, with only its prefix id changed (and the source recorded):
  D   ref/d091-selection.json, ref/d091-prefix_manifest.json, PRIVATE/src-d091/{prefixes,prompts}.json
  D2  ref/d091b-selection.json, ref/d091b-prefix_manifest.json, PRIVATE/src-d091b/{prefixes,prompts}.json
Each source's private files must hash to its manifest's private_*_sha256, and every copied id list must hash to its
manifest entry's prefix_ids_sha256 (checked here). Outputs: PRIVATE/prefixes.json and PRIVATE/prompts.json (0600; token ids
and prompt messages), out/prefix_manifest.json and out/selection.json (ids, hashes, lengths, covariates; no text).
usage: compose_prefixes_e.py
"""
import hashlib
import json
import os
import sys

os.umask(0o077)
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EC = "<runs>/long-reply/e-cross"
SRC = {"d091": dict(sel="ref/d091-selection.json", man="ref/d091-prefix_manifest.json", priv="PRIVATE/src-d091", seed=70704),
       "d091b": dict(sel="ref/d091b-selection.json", man="ref/d091b-prefix_manifest.json", priv="PRIVATE/src-d091b", seed=70705)}
EXPECT_NEW = ["q69p5", "q120p5", "q186p5", "q88p5", "q71p5"]       # amendment 22ag


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def main():
    if not (PKG == EC or PKG.startswith(EC + "/")):
        sys.exit(f"package path {PKG} is not e-cross")
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)[:200]})

    S = {}
    for k, s in SRC.items():
        man = json.load(open(f"{PKG}/{s['man']}"))
        sel = json.load(open(f"{PKG}/{s['sel']}"))
        pre = json.load(open(f"{PKG}/{s['priv']}/prefixes.json"))
        pro = json.load(open(f"{PKG}/{s['priv']}/prompts.json"))
        chk(f"{k}: private files hash to the manifest's private_*_sha256",
            fsha(f"{PKG}/{s['priv']}/prefixes.json") == man["private_file_sha256"] and fsha(f"{PKG}/{s['priv']}/prompts.json") == man["private_prompts_sha256"])
        chk(f"{k}: selection and manifest list the same replies in the same order",
            [x["id"] for x in sel["selected"]] == [m["id"] for m in man["prefixes"]])
        S[k] = dict(man=man, sel=sel, pre=pre["prefixes"], pro=pro["prompts"], cut=pre["cut"],
                    man_sha=fsha(f"{PKG}/{s['man']}"), sel_sha=fsha(f"{PKG}/{s['sel']}"))
    d_docs = {m["doc"] for m in S["d091"]["man"]["prefixes"]}
    order = [("d091", m) for m in S["d091"]["man"]["prefixes"]]
    order += [("d091b", m) for m in S["d091b"]["man"]["prefixes"] if m["doc"] not in d_docs]
    chk("D2 replies on new questions are exactly amendment 22ag's (in D2's selection order)",
        [m["id"] for k, m in order if k == "d091b"] == EXPECT_NEW, [m["id"] for k, m in order if k == "d091b"])
    pre, pro, man_out, sel_out = {}, {}, [], []
    selby = {k: {x["id"]: x for x in S[k]["sel"]["selected"]} for k in S}
    for n, (k, m) in enumerate(order, 1):
        pid = f"P{n:02d}"
        ids = S[k]["pre"][m["prefix_id"]]
        chk(f"{pid} ({k} {m['prefix_id']} {m['id']}): ids hash to the source manifest entry and length matches",
            ids_sha(ids) == m["prefix_ids_sha256"] and len(ids) == m["prefix_len"])
        if k == "d091":
            chk(f"{pid}: D's prefix keeps its id", pid == m["prefix_id"])
        pre[pid] = ids
        pro[pid] = S[k]["pro"][m["prefix_id"]]
        e = dict(m)
        e.update(prefix_id=pid, source=k, source_prefix_id=m["prefix_id"], source_request_seed=SRC[k]["seed"])
        man_out.append(e)
        x = dict(selby[k][m["id"]])
        x.update(prefix_id=pid, source=k, source_prefix_id=m["prefix_id"])
        sel_out.append(x)
    chk("13 prefixes on 13 distinct questions", len(man_out) == 13 and len({m["doc"] for m in man_out}) == 13)
    chk("strata: 8 capped, 5 finished", sum(m["stratum"] == "capped" for m in man_out) == 8 and sum(m["stratum"] == "finished" for m in man_out) == 5)
    chk("both sources use the same cut", S["d091"]["cut"] == S["d091b"]["cut"] == 100000)
    for key in ("engine_built_with", "libraries", "template", "template_sha256", "snapshot", "hash_convention"):
        chk(f"both source manifests agree on {key}", S["d091"]["man"][key] == S["d091b"]["man"][key])
    bad = [c for c in checks if not c["ok"]]
    if bad:
        for c in bad:
            print("FAIL", c["check"], c["detail"])
        return 1
    pp, qp = f"{PKG}/PRIVATE/prefixes.json", f"{PKG}/PRIVATE/prompts.json"
    with open(pp, "w") as fo:
        json.dump({"note": "PRIVATE: GPQA-derived token ids (E, amendment 22ah: D's 8 + D2's 5)", "cut": 100000, "prefixes": pre}, fo)
    os.chmod(pp, 0o600)
    with open(qp, "w") as fo:
        json.dump({"note": "PRIVATE: GPQA prompt messages (for the run-time server-side render check)", "prompts": pro}, fo)
    os.chmod(qp, 0o600)
    base = S["d091"]["man"]
    out = {"what": "E prefix manifest, amendment 22ah: D's 8 (P01-P08) + D2's 5 on new questions (P09-P13); entries copied "
                   "verbatim from the source manifests with only prefix_id changed (hashes, lengths, checks, covariates; no text)",
           **{k: base[k] for k in ("engine_built_with", "libraries", "template", "template_sha256", "snapshot", "hash_convention")},
           "sources": {k: {"manifest": SRC[k]["man"], "manifest_sha256": S[k]["man_sha"], "selection": SRC[k]["sel"],
                           "selection_sha256": S[k]["sel_sha"], "request_seed": SRC[k]["seed"]} for k in S},
           "prefixes": man_out, "private_file_sha256": fsha(pp), "private_prompts_sha256": fsha(qp), "compose_checks": checks,
           "source_build_checks": {k: S[k]["man"]["checks"] for k in S}}
    json.dump(out, open(f"{PKG}/out/prefix_manifest.json", "w"), indent=1)
    os.chmod(f"{PKG}/out/prefix_manifest.json", 0o644)
    json.dump({"what": "E prefix selection, amendment 22ah (ids, counts and metrics only): D's selection + D2's selected replies on "
                       "questions not already present (amendment 22ag)",
               "rule": "22ad: D's 8 keep their place; D2 adds questions not already present; ids in D2's selection order",
               "sources": out["sources"], "selected": sel_out,
               "summary": {"selected": len(sel_out), "by_stratum": {s: sum(x["stratum"] == s for x in sel_out) for s in ("capped", "finished")},
                           "distinct_questions": len({x["doc"] for x in sel_out}), "by_source": {k: sum(x["source"] == k for x in sel_out) for k in S}}},
              open(f"{PKG}/out/selection.json", "w"), indent=1)
    print(f"compose: {len(checks)} / {len(checks)} checks pass")
    for m in man_out:
        print(m["prefix_id"], m["id"], m["stratum"], m["source"], m["source_prefix_id"], "prefix", m["prefix_len"], m["prefix_ids_sha256"][:12])
    print("private prefixes sha256", out["private_file_sha256"][:16], "prompts", out["private_prompts_sha256"][:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
