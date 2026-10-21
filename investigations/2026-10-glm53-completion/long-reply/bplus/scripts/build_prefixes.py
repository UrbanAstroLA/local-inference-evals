#!/usr/bin/env python3
"""B+ prefix construction and template/tokenizer verification (PROPOSAL.md sections 2.2-2.3). CPU-only container,
--network none, CUDA_VISIBLE_DEVICES empty, no GPU. Run through scripts/run_cpu.sh.

  build   (0.7.0 image): for each selected reply, read the original request (messages, chat_template_kwargs) and
          response reasoning from the rep070-F raw file; render the prompt with the 0.7.0 engine's chat template
          (checkpoint 701cd745 chat_template.jinja; the 0.7.0 server passes no --chat-template) and tokenizer
          (add_special_tokens=False, as the vLLM chat path); re-tokenize the reasoning text (tokenizers, K3.25
          tokenizer.json, add_special_tokens=False, as discovery); prefix = prompt ids + reasoning ids[:CUT].
          Writes PRIVATE/prefixes.json (token ids; mode 0600) and out/prefix_manifest.json (hashes, lengths, checks,
          covariates; no text, no reasoning ids).
  verify  (any image, ENGINE=v070|v091|b1): rebuild every prefix with THAT image's tokenizer library and the template and
          snapshot THAT engine serves (0.9.1 / B1: /opt/glm53/templates/glm53-zai-a5b45eb.jinja, snapshot 0490d2f7) and
          compare every hash with the frozen manifest. Writes checks/prefix-verify-<ENGINE>.json. Nothing private.

Cut rule: the reasoning text is tokenized whole and the prefix keeps its first CUT = 100,000 token ids; the cut is a
token boundary of that tokenization (the 100,000th id may be a word piece; ids are sent, never re-tokenized text).
Hash convention: ids_sha256 = sha256 of the ASCII string ",".join(str(i) for i in ids). The d7 convention
(sha256(json.dumps(ids))) is also computed for the prompt part to compare with d7/exposure/gpqa_prompt_tokens.json.
"""
import gzip
import hashlib
import json
import os
import sys

os.umask(0o077)
WORK = "/work"            # bplus/
RAW = "/raw"              # rep070/run-F/raw (ro)
REPO = "/repo"            # models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1 (ro)
GPQA = "/gpqa"            # <gpqa-runs> (ro): request logs
D7TOK = "/d7/gpqa_prompt_tokens.json"
DISC = "/disc"            # long-reply/discovery/out (ro)
CUT = 100000
STOP_IDS = (154820, 154827, 154829)
THINK_OPEN, THINK_CLOSE, ASSISTANT = 154841, 154842, 154828
SNAP = {"v070": "701cd7456c13d87bf0147ad946f828a999afb59c", "v091": "0490d2f708b12145f6516555ab066aaeb401cd21",
        "b1": "0490d2f708b12145f6516555ab066aaeb401cd21"}
TEMPLATE = {"v070": f"{REPO}/snapshots/701cd7456c13d87bf0147ad946f828a999afb59c/chat_template.jinja",
            "v091": "/opt/glm53/templates/glm53-zai-a5b45eb.jinja", "b1": "/opt/glm53/templates/glm53-zai-a5b45eb.jinja"}
TEMPLATE_SHA = {"v070": "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891",
                "v091": "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5",
                "b1": "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5"}
TOKJSON_SHA = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
TRAJ_SHA = "c24267b1a9641fc2a3bf4b9bca23950d9530d787002d74286bef8b717f5ac4c6"
REQLOGS = {70701: "glm53-flash-exl3-k3.25@tpurtell+rep070-F.s1.requests.jsonl",
           70702: "glm53-flash-exl3-k3.25@tpurtell+rep070-F.s2.requests.jsonl",
           70703: "glm53-flash-exl3-k3.25@tpurtell+rep070-F.s2.requests.jsonl"}


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def d7_sha(ids):
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


def messages_hash(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def load_raw(file, line):
    with gzip.open(f"{RAW}/{file}.raw.jsonl.gz", "rt") as fi:
        for ln, l in enumerate(fi):
            if ln == line:
                return json.loads(l)
    raise KeyError((file, line))


def main():
    mode = sys.argv[1]
    engine = os.environ.get("ENGINE", "v070")
    assert mode in ("build", "verify") and engine in SNAP
    assert fsha(f"{WORK}/scripts/trajectory.py") == TRAJ_SHA, "trajectory.py is not the discovery's frozen file"
    sys.path.insert(0, f"{WORK}/scripts")
    import numpy as np
    import trajectory as TJ
    import tokenizers
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    snapdir = f"{REPO}/snapshots/{SNAP[engine]}"
    tpl_path = TEMPLATE[engine]
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    chk(f"{engine}: served chat template sha256 {TEMPLATE_SHA[engine][:12]}", fsha(tpl_path) == TEMPLATE_SHA[engine], fsha(tpl_path)[:16])
    chk(f"{engine}: tokenizer.json sha256 {TOKJSON_SHA[:12]}", fsha(f"{snapdir}/tokenizer.json") == TOKJSON_SHA, fsha(f"{snapdir}/tokenizer.json")[:16])
    template = open(tpl_path).read()
    hf = AutoTokenizer.from_pretrained(snapdir)
    tk = Tokenizer.from_file(f"{snapdir}/tokenizer.json")
    sel = json.load(open(f"{WORK}/out/selection.json"))["selected"]
    index = {r["id"]: r for r in json.load(open(f"{DISC}/rep070F_index.json"))}
    d7 = {d["doc_id"]: d for d in json.load(open(D7TOK))["docs"]}
    reqlog = {}
    for seed, f in REQLOGS.items():
        for l in open(f"{GPQA}/{f}"):
            r = json.loads(l)
            if r["seed"] == seed:
                reqlog[(seed, r["messages_sha256"])] = r
    prefixes, man, prompts = {}, [], {}
    for s in sel:
        ix = index[s["id"]]
        d = load_raw(ix["file"], ix["line"])
        req, resp = d["request"], d["response"]
        msgs = req["messages"]
        kw = req.get("chat_template_kwargs") or {}
        mh = messages_hash(msgs)
        rl = reqlog.get((s["seed"], mh))
        text = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, chat_template=template, **kw)
        p_hf = hf.encode(text, add_special_tokens=False)
        p_tk = tk.encode(text, add_special_tokens=False).ids
        reasoning = resp["choices"][0]["message"].get("reasoning") or ""
        enc = tk.encode(reasoning, add_special_tokens=False)
        r_tk = enc.ids
        r_hf = hf.encode(reasoning, add_special_tokens=False)
        cut = r_tk[:CUT]
        prefix = list(p_tk) + list(cut)
        offs = enc.offsets
        cut_char = offs[CUT - 1][1]
        ptxt = reasoning[:cut_char]
        # covariates on the 100k reasoning prefix (counts only)
        nd = sum(ch.isdigit() for ch in ptxt)
        waits = sum(1 for m in TJ.CHK_WAIT.finditer(ptxt))
        verifs = sum(1 for m in TJ.CHK_VERIFY.finditer(ptxt))
        ids = np.array(cut, dtype=np.int64)
        rep50, _ = TJ.rep_and_prev(TJ.ngram_hash(ids, 50))
        r50 = rep50.astype(np.float64)
        cs = np.concatenate([[0.0], np.cumsum(r50)])
        R50 = (cs[TJ.W:] - cs[:-TJ.W]) / TJ.W
        onset, eps = TJ.onset_and_episodes(R50, TJ.THR)
        rec = {"prefix_id": s["prefix_id"], "id": s["id"], "doc": s["doc"], "stratum": s["stratum"],
               "messages_sha256_equals_raw_and_log": mh == d["messages_sha256"] and rl is not None,
               "chat_template_kwargs": kw,
               "prompt_len": len(p_tk), "prompt_tokens_server_raw": resp["usage"]["prompt_tokens"],
               "prompt_tokens_request_log": rl["prompt_tokens"] if rl else None,
               "prompt_ids_sha256": ids_sha(p_tk), "prompt_ids_d7_sha256": d7_sha(p_tk),
               "prompt_rendered_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "prompt_hf_equals_tokenizers": list(p_hf) == list(p_tk),
               "prompt_ends_assistant_think": list(p_tk[-2:]) == [ASSISTANT, THINK_OPEN],
               "reasoning_tokens_full": len(r_tk), "reasoning_hf_equals_tokenizers": list(r_hf) == list(r_tk),
               "reasoning_full_ids_sha256": ids_sha(r_tk), "reasoning_text_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
               "discovery_N_reasoning": s["N_reasoning"],
               "cut": CUT, "cut_ids_sha256": ids_sha(cut), "cut_char_offset": cut_char,
               "prefix_len": len(prefix), "prefix_ids_sha256": ids_sha(prefix),
               "prompt_stop_ids": sum(1 for t in p_tk if t in STOP_IDS),
               "reasoning_cut_stop_ids": sum(1 for t in cut if t in STOP_IDS),
               "reasoning_cut_think_close": sum(1 for t in cut if t == THINK_CLOSE),
               "reasoning_cut_added_ids_ge_154820": sum(1 for t in cut if t >= 154820),
               "covariates_100k": {"chars_per_token": round(len(ptxt) / CUT, 4), "digit_share": round(nd / max(1, len(ptxt)), 4),
                                   "interjection_per1k": round(1000 * waits / CUT, 3), "verify_per1k": round(1000 * verifs / CUT, 3),
                                   "R50_backward_at_cut": round(float(R50[-1]), 4), "R50_max_in_prefix": round(float(R50.max()), 4),
                                   "episodes_in_prefix": len(eps), "terminal_onset_in_prefix": onset}}
        if engine == "v070":
            dd = d7.get(s["doc"], {}).get("A_ckpt_701cd745", {})
        else:
            dd = d7.get(s["doc"], {}).get("B_zai_0490d2f7", {})
        rec["d7_reference_prompt_tokens"] = dd.get("tokens")
        rec["prompt_equals_d7_independent_tokenizer"] = dd.get("token_ids_sha256") == rec["prompt_ids_d7_sha256"]
        man.append(rec)
        prefixes[s["prefix_id"]] = prefix
        prompts[s["prefix_id"]] = {"messages": msgs, "chat_template_kwargs": kw}
        p = s["prefix_id"]
        chk(f"{engine} {p}: messages hash equals the raw record and the rep070-F request log", rec["messages_sha256_equals_raw_and_log"])
        chk(f"{engine} {p}: rendered prompt length equals the server's prompt_tokens (raw response and request log)",
            rec["prompt_len"] == rec["prompt_tokens_server_raw"] == rec["prompt_tokens_request_log"],
            f"{rec['prompt_len']} / {rec['prompt_tokens_server_raw']} / {rec['prompt_tokens_request_log']}")
        chk(f"{engine} {p}: prompt ids equal d7's independent pure-Python tokenization", rec["prompt_equals_d7_independent_tokenizer"])
        chk(f"{engine} {p}: transformers and tokenizers give identical prompt and reasoning ids",
            rec["prompt_hf_equals_tokenizers"] and rec["reasoning_hf_equals_tokenizers"])
        chk(f"{engine} {p}: reasoning re-tokenization length equals discovery's N_reasoning", len(r_tk) == s["N_reasoning"],
            f"{len(r_tk)} vs {s['N_reasoning']}")
        chk(f"{engine} {p}: prompt ends <|assistant|><think>; no stop id and no </think> in the reasoning part of the prefix",
            rec["prompt_ends_assistant_think"] and rec["reasoning_cut_stop_ids"] == 0 and rec["reasoning_cut_think_close"] == 0,
            f"prompt carries {rec['prompt_stop_ids']} stop-class id(s) (the template's <|user|> turn marker; stop ids act on generated tokens only)")
        chk(f"{engine} {p}: prefix length = prompt + {CUT}", len(prefix) == len(p_tk) + CUT, len(prefix))
    lib = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "numpy": np.__version__}
    if mode == "build":
        assert engine == "v070"
        path = f"{WORK}/PRIVATE/prefixes.json"
        with open(path, "w") as fo:
            json.dump({"note": "PRIVATE: GPQA-derived token ids", "cut": CUT, "prefixes": prefixes}, fo)
        os.chmod(path, 0o600)
        ppath = f"{WORK}/PRIVATE/prompts.json"
        with open(ppath, "w") as fo:
            json.dump({"note": "PRIVATE: GPQA prompt messages (for the run-time server-side render check)", "prompts": prompts}, fo)
        os.chmod(ppath, 0o600)
        out = {"what": "B+ prefix manifest (hashes, lengths, checks, covariates; no text, no reasoning ids)", "engine_built_with": engine,
               "libraries": lib, "template": tpl_path, "template_sha256": TEMPLATE_SHA[engine], "snapshot": SNAP[engine],
               "hash_convention": "sha256 of ','.join(str(id)) over the id list", "prefixes": man,
               "private_file_sha256": fsha(path), "private_prompts_sha256": fsha(ppath), "checks": checks}
        json.dump(out, open(f"{WORK}/out/prefix_manifest.json", "w"), indent=1)
    else:
        frozen = {m["prefix_id"]: m for m in json.load(open(f"{WORK}/out/prefix_manifest.json"))["prefixes"]}
        priv = json.load(open(f"{WORK}/PRIVATE/prefixes.json"))["prefixes"]
        for m in man:
            f = frozen[m["prefix_id"]]
            for k in ("prompt_ids_sha256", "reasoning_full_ids_sha256", "cut_ids_sha256", "prefix_ids_sha256", "prefix_len"):
                chk(f"{engine} {m['prefix_id']}: {k} equals the frozen 0.7.0-built manifest", m[k] == f[k], f"{str(m[k])[:16]} vs {str(f[k])[:16]}")
            chk(f"{engine} {m['prefix_id']}: rebuilt ids equal PRIVATE/prefixes.json", prefixes[m["prefix_id"]] == priv[m["prefix_id"]])
            chk(f"{engine} {m['prefix_id']}: rendered prompt text identical to the 0.7.0 template's", m["prompt_rendered_sha256"] == f["prompt_rendered_sha256"])
        out = {"engine": engine, "libraries": lib, "template": tpl_path, "snapshot": SNAP[engine], "checks": checks,
               "prefix_hashes": {m["prefix_id"]: m["prefix_ids_sha256"] for m in man}}
        json.dump(out, open(f"{WORK}/checks/prefix-verify-{engine}.json", "w"), indent=1)
    bad = [c for c in checks if not c["ok"]]
    for c in checks:
        if not c["ok"]:
            print("FAIL", c["check"], c["detail"])
    print(f"{mode} {engine}: {len(checks) - len(bad)}/{len(checks)} checks pass; libraries {lib}")
    for m in man:
        print(m["prefix_id"], m["id"], "prompt", m["prompt_len"], "prefix", m["prefix_len"], m["prefix_ids_sha256"][:16],
              "cov", m["covariates_100k"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
