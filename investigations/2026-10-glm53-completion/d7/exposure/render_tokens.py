#!/usr/bin/env python3
"""D7 exposure accounting (PLAN.md amendment 18), step 2a: EXACT rendered prompt-token counts for the 198 GPQA prompts.

LOCAL ONLY while running (reads restricted GPQA prompt text from lm-eval samples); WRITES doc ids, hashes and token counts
only -- never question, answer or reply text.

No tokenizer library is installed on this host (no `tokenizers`/`transformers`; containers are not used for this task), so
the K3.25 tokenizer.json is implemented here in pure Python and VERIFIED, not assumed: every rendered count is compared with
the prompt_tokens the record proxy logged from the serving engine for the same messages hash, for every logged pass.
  - added tokens: matched first (leftmost, longest), never split (normalized false, no lstrip/rstrip)
  - pre-tokenizer: Split(regex, Isolated) then ByteLevel(use_regex false); \\p{L}/\\p{N} classes built from unicodedata,
    \\s from the Unicode White_Space set (the Rust `regex` crate's \\s), since Python's `re` has neither
  - model: byte-level BPE, ignore_merges true (a pre-token that is itself in the vocab is emitted as one token)
  - chat template rendered with jinja2's ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks, loopcontrols) as in
    transformers/vLLM; add_generation_prompt true; chat_template_kwargs {"enable_thinking": true} (gpqa_fixture.LOCAL.json)
  - template B (0.9.1 family: served --chat-template glm53-zai-a5b45eb.jinja == snapshot 0490d2f7 chat_template.jinja,
    sha256 0c4099f3...); template A (0.7.0: no --chat-template, checkpoint revision 701cd745's chat_template.jinja)
  - tokenized without added BOS (vLLM chat path: add_special_tokens=False; the template itself emits [gMASK]<sop>)
"""
import glob
import hashlib
import json
import os
import re
import sys
import unicodedata
from functools import lru_cache

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

SNAP = "<models>/models--wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1/snapshots"
TEMPLATES = {"B_zai_0490d2f7": f"{SNAP}/0490d2f708b12145f6516555ab066aaeb401cd21/chat_template.jinja",
             "A_ckpt_701cd745": f"{SNAP}/701cd7456c13d87bf0147ad946f828a999afb59c/chat_template.jinja"}
TOKJSON = f"{SNAP}/0490d2f708b12145f6516555ab066aaeb401cd21/tokenizer.json"   # byte-identical in both snapshots
G = "<gpqa-runs>"
SAMPLES = glob.glob(f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1__pass1/**/samples_*.jsonl", recursive=True)[0]
KWARGS = {"enable_thinking": True}
# request logs -> template the serving engine used (0.7.0 record: checkpoint template; 0.9.1 family: zai template)
REQLOGS = {
    "glm53-flash-exl3-k3.25@tpurtell+ps.requests.jsonl": "A_ckpt_701cd745",
    "glm53-flash-exl3-k3.25@tpurtell+v0.9.1.requests.jsonl": "B_zai_0490d2f7",
    "glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.requests.jsonl": "B_zai_0490d2f7",
    "glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s1.requests.jsonl": "B_zai_0490d2f7",
    "glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s2.requests.jsonl": "B_zai_0490d2f7",
    "glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234.s1.requests.jsonl": "B_zai_0490d2f7",
    "glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234.s2.requests.jsonl": "B_zai_0490d2f7",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpqa_prompt_tokens.json")


# ------------------------------------------------------------------ tokenizer
def _ranges(pred):
    out, lo, prev = [], None, None
    for cp in range(0x110000):
        if pred(cp):
            if lo is None:
                lo = cp
            prev = cp
        elif lo is not None:
            out.append((lo, prev))
            lo = None
    if lo is not None:
        out.append((lo, prev))
    return "".join(f"\\U{a:08x}" if a == b else f"\\U{a:08x}-\\U{b:08x}" for a, b in out)


def build_pretok_regex(pattern):
    L = _ranges(lambda c: unicodedata.category(chr(c)).startswith("L"))
    N = _ranges(lambda c: unicodedata.category(chr(c)).startswith("N"))
    WS = "\\t\\n\\x0b\\x0c\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000"  # Unicode White_Space
    p = pattern
    p = p.replace("[^\\r\\n\\p{L}\\p{N}]", f"[^\\r\\n{L}{N}]")
    p = p.replace("[^\\s\\p{L}\\p{N}]", f"[^{WS}{L}{N}]")
    p = p.replace("\\p{L}", f"[{L}]").replace("\\p{N}", f"[{N}]")
    p = p.replace("\\s", f"[{WS}]").replace("\\S", f"[^{WS}]")
    assert "\\p{" not in p
    return re.compile(p)


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class BPE:
    def __init__(self, path):
        tj = json.load(open(path))
        m = tj["model"]
        assert m["type"] == "BPE" and tj["normalizer"] is None
        assert tj["pre_tokenizer"]["type"] == "Sequence"
        sp, bl = tj["pre_tokenizer"]["pretokenizers"]
        assert sp["type"] == "Split" and sp["behavior"] == "Isolated" and not sp["invert"]
        assert bl["type"] == "ByteLevel" and not bl["use_regex"] and not bl["add_prefix_space"]
        self.vocab = m["vocab"]
        self.ignore_merges = m.get("ignore_merges", False)
        self.ranks = {tuple(mg) if isinstance(mg, list) else tuple(mg.split(" ", 1)): i for i, mg in enumerate(m["merges"])}
        self.rx = build_pretok_regex(sp["pattern"]["Regex"])
        self.b2u = bytes_to_unicode()
        self.added = sorted(((a["content"], a["id"]) for a in tj["added_tokens"]), key=lambda x: -len(x[0]))
        for a in tj["added_tokens"]:
            assert not a["lstrip"] and not a["rstrip"] and not a["normalized"] and not a["single_word"]
        self.added_rx = re.compile("|".join(re.escape(c) for c, _ in self.added))
        self.added_id = dict(self.added)

    @lru_cache(maxsize=None)
    def bpe(self, word):
        if self.ignore_merges and word in self.vocab:
            return (self.vocab[word],)
        parts = list(word)
        while len(parts) > 1:
            best, bi = None, None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]
        return tuple(self.vocab[p] for p in parts)

    def encode_plain(self, text):
        ids = []
        for m in self.rx.finditer(text):
            piece = m.group(0)
            if not piece:
                continue
            w = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            ids.extend(self.bpe(w))
        # Split/Isolated keeps unmatched gaps as their own pieces; the pattern ends in \s+ alternatives, so check coverage
        return ids

    def encode(self, text):
        ids, pos = [], 0
        for m in self.added_rx.finditer(text):
            if m.start() > pos:
                ids.extend(self._plain_checked(text[pos:m.start()]))
            ids.append(self.added_id[m.group(0)])
            pos = m.end()
        if pos < len(text):
            ids.extend(self._plain_checked(text[pos:]))
        return ids

    def _plain_checked(self, seg):
        covered = sum(len(m.group(0)) for m in self.rx.finditer(seg))
        assert covered == len(seg), "pre-tokenizer regex left a gap"
        return self.encode_plain(seg)


def make_env():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    def raise_exception(msg):
        raise Exception(msg)
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, separators=None, sort_keys=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, separators=separators, sort_keys=sort_keys)
    return env


def messages_hash(messages):  # = record_proxy.py / analysis_col511.py
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def main():
    tok = BPE(TOKJSON)
    env = make_env()
    tpl = {k: env.from_string(open(p).read()) for k, p in TEMPLATES.items()}
    tsha = {k: hashlib.sha256(open(p, "rb").read()).hexdigest() for k, p in TEMPLATES.items()}
    docs = {}
    for line in open(SAMPLES):
        r = json.loads(line)
        if r["doc_id"] in docs:
            continue
        msgs = json.loads(r["arguments"]["gen_args_0"]["arg_0"][0])
        docs[r["doc_id"]] = msgs
    assert sorted(docs) == list(range(198))
    rows = {}
    for d, msgs in sorted(docs.items()):
        row = {"doc_id": d, "messages_sha256": messages_hash(msgs)}
        for k, t in tpl.items():
            text = t.render(messages=msgs, add_generation_prompt=True, tools=None, **KWARGS)
            ids = tok.encode(text)
            row[k] = {"tokens": len(ids), "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                      "rendered_sha256": hashlib.sha256(text.encode()).hexdigest()}
        rows[d] = row
    # ---- cross-check against the engines' own counts (record proxy prompt_tokens)
    by_hash = {r["messages_sha256"]: d for d, r in rows.items()}
    checks = {}
    for f, k in REQLOGS.items():
        p = os.path.join(G, f)
        recs = [json.loads(l) for l in open(p) if l.strip()]
        res = {"template": k, "requests": len(recs), "matched_docs": 0, "equal": 0, "diff": []}
        for q in recs:
            d = by_hash.get(q.get("messages_sha256"))
            if d is None:
                res["diff"].append({"unknown_messages_sha256": q.get("messages_sha256")})
                continue
            res["matched_docs"] += 1
            if q.get("prompt_tokens") == rows[d][k]["tokens"]:
                res["equal"] += 1
            else:
                res["diff"].append({"doc_id": d, "seed": q.get("seed"), "logged": q.get("prompt_tokens"), "rendered": rows[d][k]["tokens"]})
        checks[f] = res
    out = {"note": "doc ids, hashes and token counts only; no prompt text",
           "samples_file": os.path.relpath(SAMPLES, G), "samples_sha256": hashlib.sha256(open(SAMPLES, "rb").read()).hexdigest(),
           "tokenizer_json_sha256": hashlib.sha256(open(TOKJSON, "rb").read()).hexdigest(), "templates_sha256": tsha,
           "chat_template_kwargs": KWARGS, "docs": [rows[d] for d in sorted(rows)], "crosscheck_vs_proxy_prompt_tokens": checks}
    json.dump(out, open(OUT, "w"), indent=1)
    for f, r in checks.items():
        print(f"{f}: {r['equal']}/{r['requests']} equal (template {r['template']}); first diffs {r['diff'][:3]}")
    for k in TEMPLATES:
        ts = sorted(rows[d][k]["tokens"] for d in rows)
        print(k, "min", ts[0], "median", ts[len(ts) // 2], "max", ts[-1], "n>1088", sum(t > 1088 for t in ts),
              "n>1996", sum(t > 1996 for t in ts), "n>2045", sum(t > 2045 for t in ts))
    same = sum(rows[d]["A_ckpt_701cd745"]["token_ids_sha256"] == rows[d]["B_zai_0490d2f7"]["token_ids_sha256"] for d in rows)
    print(f"token sequences identical under both templates: {same}/198")


if __name__ == "__main__":
    sys.exit(main())
