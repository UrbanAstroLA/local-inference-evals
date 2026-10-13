#!/usr/bin/env python3
"""Fixture inventory scan (PLAN.md amendment 20 / 20b, zero GPU). Runs in a CPU-only container
(--network none, no --gpus) that has `tokenizers`, `transformers`, `numpy`.

Reads (read-only mounts): /runs = <data-runs>, /model = the K3.25 HF repo dir.
Writes /out/scan.json and /out/termination_tokens.json: ids, positions, lengths, metrics and hashes only.
Never writes reasoning, content or GPQA text.

Per text history (client-saved *.reasoning.txt + sibling .json):
  - re-tokenize the reasoning text with the K3.25 tokenizer (add_special_tokens=False), as fixtureB/prep_requests.py
    did for phase-2 requests; hash text and ids; drift against the logged completion_tokens.
  - repetition state per token position (amendment 20b), with the indicators the clients/pilot used:
      zlib30k  = len(zlib(last 30,000 reasoning chars)) / len(...)   (client zr(); loop if < 0.15 at end, early stop
                 after three consecutive periodic checks < 0.10)
      rep200   = share of 200-character windows in the last 30,000 chars whose exact text already occurred earlier in
                 the history (reconstruction of the pilot's "repeated 200-character windows"; its script was not kept)
  - answer-like commitment events (regex classes, positions only) and the reasoning end (= natural close position when
    the reply closed).
Per exact-id trace (step5/6 decode-prefill traces): first </think>, first terminator, active segment metrics.
"""
import glob, gzip, hashlib, json, os, re, sys, zlib
from multiprocessing import Pool

import numpy as np

SNAP = "/model/snapshots/0490d2f708b12145f6516555ab066aaeb401cd21"
SNAP070 = "/model/snapshots/701cd7456c13d87bf0147ad946f828a999afb59c"
RUNS = "/runs"
OUT = "/out"
SAMPLES = glob.glob(f"{RUNS}/gpqa/glm53-flash-exl3-k4@tpurtell+v0.8.0__pass1/**/samples_*.jsonl", recursive=True)[0]
THINK_OPEN, THINK_CLOSE = 154841, 154842
TERMINATORS = {154820: "<|endoftext|>", 154827: "<|user|>", 154829: "<|observation|>"}
CHECKPOINTS = [4000, 8000, 16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000, 160000, 200000, 250000, 300000]
WIN_CHARS = 30000
REPEAT_LOOP_ZLIB = 0.15      # client end-of-request loop threshold
REPEAT_REP200 = 0.50         # share of already-seen 200-char windows that marks repetition at a position
L200 = 200

COMMIT = {
    # explicit choice of a lettered option (letter itself is never recorded)
    "answer_letter": re.compile(r"(?i:answer)\s*(?:(?i:is|would be|should be|must be|will be)\s*|[:=]\s*)\**\s*\(?\s*[ABCD]\b"),
    "go_with": re.compile(r"(?i:\b(?:go with|going with|i(?:'ll| will) (?:choose|pick|select)|settle on|answer with))\s+(?:(?i:option|choice|answer)\s+)?\**\(?[ABCD]\b"),
    "final_answer": re.compile(r"(?i)final answer"),
    "boxed": re.compile(r"\\boxed"),
    # tentative conclusion phrase without a required letter
    "so_answer": re.compile(r"(?i)\b(?:so|thus|therefore|hence),? (?:the )?answer\b"),
}
STRONG = ("answer_letter", "go_with", "final_answer", "boxed")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def ids_sha(ids):
    return sha(json.dumps(ids))  # same convention as fixtureB/prep_requests.py


_tok = None


def tok():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(f"{SNAP}/tokenizer.json")
    return _tok


def seen_before_mask(text):
    """bool per char start i (len n-L+1): window text[i:i+L] occurred at an earlier start j < i."""
    n = len(text)
    if n < L200 + 1:
        return np.zeros(max(n - L200 + 1, 0), dtype=bool)
    c = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.uint64)
    # polynomial rolling hash with uint64 wraparound, two independent bases to make collisions negligible
    out = []
    for base in (np.uint64(1000003), np.uint64(9176)):
        pw = np.ones(L200, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for k in range(1, L200):
                pw[k] = pw[k - 1] * base
        pw = pw[::-1].copy()
        h = np.zeros(n - L200 + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for k in range(L200):
                h += c[k:k + n - L200 + 1] * pw[k]
        out.append(h)
    key = out[0] ^ (out[1] * np.uint64(0x9E3779B97F4A7C15))
    _, first_idx, inv = np.unique(key, return_index=True, return_inverse=True)
    return first_idx[inv] < np.arange(len(key))


def window_metrics(text, rep_cum, char_end):
    """Indicators over the WIN_CHARS characters preceding char_end."""
    a = max(0, char_end - WIN_CHARS)
    seg = text[a:char_end]
    if len(seg) < 2000:
        return None
    zr = round(len(zlib.compress(seg.encode())) / max(1, len(seg.encode())), 4)
    # windows fully inside [a, char_end)
    lo, hi = a, max(a, char_end - L200 + 1)
    nwin = hi - lo
    rep = round(float(rep_cum[hi] - rep_cum[lo]) / nwin, 4) if nwin > 0 else None
    repeating = (zr < REPEAT_LOOP_ZLIB) or (rep is not None and rep >= REPEAT_REP200)
    return {"zlib30k": zr, "rep200": rep, "repeating": bool(repeating)}


def analyse_text(text, starts):
    """starts: token start char offsets (np array). Returns repetition + commitment metrics by token position."""
    ntok = len(starts)
    mask = seen_before_mask(text)
    rep_cum = np.concatenate([[0], np.cumsum(mask)])

    def char_of_tok(p):  # char offset where token p starts (p may equal ntok -> end)
        return len(text) if p >= ntok else int(starts[p])

    def tok_of_char(ci):
        return int(np.searchsorted(starts, ci, side="right") - 1)

    cps = {}
    for p in CHECKPOINTS:
        if p <= ntok:
            cps[str(p)] = window_metrics(text, rep_cum, char_of_tok(p))
    end = window_metrics(text, rep_cum, len(text))
    # loop onset: earliest checkpoint on an 8k-token grid after which every grid point is repeating
    grid = list(range(8000, ntok + 1, 4000)) + [ntok]
    states = [(p, window_metrics(text, rep_cum, char_of_tok(p))) for p in grid]
    onset = None
    if states and states[-1][1] and states[-1][1]["repeating"]:
        k = len(states) - 1
        while k > 0 and states[k - 1][1] and states[k - 1][1]["repeating"]:
            k -= 1
        onset = states[k][0]
    first_rep = next((p for p, m in states if m and m["repeating"]), None)
    # whole-reply repeated share (all windows)
    rep_all = round(float(mask.mean()), 4) if len(mask) else None
    commits = {}
    events = []
    for name, rx in COMMIT.items():
        pos = [tok_of_char(m.start()) for m in rx.finditer(text)]
        commits[name] = {"n": len(pos), "first": pos[0] if pos else None, "last": pos[-1] if pos else None}
        if name in STRONG:
            events += [(p, name) for p in pos]
    events.sort()
    # per strong event: repetition state at that point
    ev_out = []
    for p, name in events:
        m = window_metrics(text, rep_cum, char_of_tok(p))
        ev_out.append([p, name, None if m is None else m["zlib30k"], None if m is None else m["rep200"],
                       None if m is None else m["repeating"]])
    return {"checkpoints": cps, "end": end, "loop_onset_tok": onset, "first_repeating_tok": first_rep,
            "rep200_all": rep_all, "commit_counts": commits, "strong_events": ev_out}


def render_prompts():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(SNAP)
    tmpl = open(f"{SNAP}/chat_template.jinja").read()
    prompts = {}
    for l in open(SAMPLES):
        r = json.loads(l)
        d = r["doc_id"]
        if d in prompts:
            continue
        msgs = json.loads(r["arguments"]["gen_args_0"]["arg_0"][0])
        txt = t.apply_chat_template(msgs, chat_template=tmpl, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=True)
        ids = t.encode(txt, add_special_tokens=False)
        assert ids[-1] == THINK_OPEN
        prompts[d] = ids
    return prompts


def text_row(args):
    tf, prompt_len_by_doc = args
    jf = tf[: -len(".reasoning.txt")] + ".json"
    r = json.load(open(jf))
    text = open(tf, encoding="utf-8").read()
    T = tok()
    enc = T.encode(text, add_special_tokens=False)
    ids = enc.ids
    starts = np.array([o[0] for o in enc.offsets], dtype=np.int64)
    content = r.get("content") or ""
    cchars = r.get("content_chars", len(content))
    content_full = ("content_chars" in r) and (len(content) == cchars)
    n_content = len(T.encode(content, add_special_tokens=False).ids) if content else 0
    u = r.get("usage") or {}
    ctoks = u.get("completion_tokens", r.get("tokens", r.get("tokens_seen")))
    ptoks = u.get("prompt_tokens")
    doc = int(r["doc"])
    fin = r.get("finish_reason")
    closed = cchars > 0
    if fin == "stop" and closed and content_full and ctoks:
        drift = ctoks - (len(ids) + 1 + n_content + 1)
    elif not closed and ctoks:
        drift = ctoks - len(ids)
    else:
        drift = None
    rec = {
        "path": os.path.relpath(tf, RUNS), "doc": doc, "rep": r.get("rep", r.get("pass_")), "seed": r.get("seed"),
        "finish_reason": fin, "client_cls": r.get("cls"), "stopped_early": r.get("stopped_early"),
        "cut": r.get("cut"), "max_tokens": r.get("max_tokens"),
        "completion_tokens": ctoks, "prompt_tokens_logged": ptoks,
        "prompt_tokens_rendered": prompt_len_by_doc.get(doc),
        "reasoning_chars": len(text), "content_chars": cchars, "content_full": content_full,
        "content_tokens_retok": n_content if content_full else None,
        "n_reasoning_tokens_retok": len(ids), "retok_drift": drift,
        "closed": closed,
        "sha256_reasoning_text": sha(text), "sha256_reasoning_ids": ids_sha(ids),
        "literal_in_text": {k: text.count(k) for k in ("</think>", "<think>", "</s>", "<|user|>", "<|endoftext|>", "<|observation|>")},
        "special_ids_in_retok": {str(k): ids.count(k) for k in (THINK_OPEN, THINK_CLOSE, 154820, 154827, 154829)},
        "tail_zlib_logged": r.get("tail_zlib_ratio", r.get("tail_zlib")),
    }
    rec.update(analyse_text(text, starts))
    return rec


def trace_row(gzf):
    d = json.load(gzip.open(gzf))
    gen = d["gen_ids"]
    T = tok()
    close = gen.index(THINK_CLOSE) if THINK_CLOSE in gen else None
    term = next((i for i, t in enumerate(gen) if t in TERMINATORS), None)
    n_terms = sum(1 for t in gen if t in TERMINATORS)
    n_close = gen.count(THINK_CLOSE)
    active = gen[:close] if close is not None else gen
    text = T.decode(active, skip_special_tokens=False)
    re_ids = T.encode(text, add_special_tokens=False).ids
    roundtrip = re_ids == active
    enc = T.encode(text, add_special_tokens=False)
    starts = np.array([o[0] for o in enc.offsets], dtype=np.int64)
    lp = d["decode_token_logprobs"]
    top = d["decode_top_logprobs"]

    def lp_at(i, tid):
        if i is None or top[i] is None:
            return None
        return top[i].get(str(tid))

    rec = {
        "path": os.path.relpath(gzf, RUNS), "prompt_id": d["prompt_id"], "seed": d["seed"], "n_gen": len(gen),
        "prompt_len": len(d["prompt_ids"]), "sha256_prompt_ids": ids_sha(d["prompt_ids"]),
        "sha256_gen_ids": ids_sha(gen), "sha256_active_ids": ids_sha(active),
        "first_close": close, "n_close": n_close, "first_terminator": term,
        "first_terminator_id": gen[term] if term is not None else None, "n_terminators": n_terms,
        "tokens_close_to_terminator": (term - close) if (term is not None and close is not None) else None,
        "token_before_close": gen[close - 1] if close else None,
        "token_before_close_text_is_ws": (T.decode([gen[close - 1]]).strip() == "") if close else None,
        "decode_lp_close_at_close": lp[close] if close is not None else None,
        "decode_lp_terminator_at_term": lp[term] if term is not None else None,
        "close_in_top20_at_close": (lp_at(close, THINK_CLOSE) is not None) if close is not None else None,
        "active_roundtrip_exact": roundtrip,
    }
    # where </think> first entered the top-20 before the close (within the last 2,048 active tokens)
    if close is not None:
        hits = [i for i in range(max(0, close - 2048), close) if top[i] is not None and str(THINK_CLOSE) in top[i]]
        rec["close_in_top20_positions_last2048"] = hits[-50:]
        rec["close_in_top20_count_last2048"] = len(hits)
    # positions come from the re-encoding of the decoded active text; exact when the round trip is exact
    rec["active_retok_len"] = len(re_ids)
    rec["active_metrics"] = analyse_text(text, starts)
    return rec


def termination_facts():
    T = tok()
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(SNAP)
    f = {"added_tokens": {str(i): T.id_to_token(i) for i in range(154820, 154856)},
         "hf_eos_token": hf.eos_token, "hf_eos_token_id": hf.eos_token_id,
         "hf_all_special_ids": sorted(hf.all_special_ids),
         "generation_config": json.load(open(f"{SNAP}/generation_config.json")),
         "generation_config_070_identical": open(f"{SNAP}/generation_config.json", "rb").read() == open(f"{SNAP070}/generation_config.json", "rb").read(),
         "tokenizer_json_070_identical": open(f"{SNAP}/tokenizer.json", "rb").read() == open(f"{SNAP070}/tokenizer.json", "rb").read(),
         "tokenizer_config_070_identical": open(f"{SNAP}/tokenizer_config.json", "rb").read() == open(f"{SNAP070}/tokenizer_config.json", "rb").read(),
         "encode": {}, "decode_skip_special": {}}
    for s in ["</think>", "<think>", "</s>", "<|user|>", "<|endoftext|>", "<|observation|>", "\n</think>", "</think>\n",
              "\n\n</think>\n\n", "</think>\n\n", "</ think>", "</think", "<tool_call>"]:
        f["encode"][repr(s)] = T.encode(s, add_special_tokens=False).ids
    for i in (154820, 154827, 154829, 154841, 154842):
        f["decode_skip_special"][str(i)] = repr(hf.decode([i], skip_special_tokens=True))
    # ordinary vocabulary entries whose decoded text contains the stop string or close text
    vocab = T.get_vocab()
    hits_s, hits_t = [], []
    for tid in range(154820):
        s = T.decode([tid])
        if "</s>" in s:
            hits_s.append(tid)
        if "</think>" in s:
            hits_t.append(tid)
    f["ordinary_ids_containing_stopstring"] = hits_s
    f["ordinary_ids_containing_think_close_text"] = hits_t
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    fac = termination_facts()
    prompts = render_prompts()
    fac["rendered_prompt_len"] = {str(k): len(v) for k, v in sorted(prompts.items())}
    fac["sha256_rendered_prompt_ids"] = {str(k): ids_sha(v) for k, v in sorted(prompts.items())}
    json.dump(fac, open(f"{OUT}/termination_tokens.json", "w"), indent=1)
    plen = {k: len(v) for k, v in prompts.items()}
    # phase-2 requests: verify they equal rendered prompt + re-tokenized pilot trace prefix
    p2 = json.load(open(f"{RUNS}/investigations/2026-10-glm53-completion/phase2-stage2/requests.json"))
    T = tok()
    p2out = []
    cache = {}
    for q in p2["requests"]:
        if q["trace"] not in cache:
            cache[q["trace"]] = T.encode(open(f"{RUNS}/investigations/2026-10-glm53-completion/pilot/{q['trace']}.reasoning.txt", encoding="utf-8").read(), add_special_tokens=False).ids
        mine = prompts[q["doc"]] + cache[q["trace"]][: q["L"]]
        p2out.append({"id": q["id"], "doc": q["doc"], "trace": q["trace"], "L": q["L"], "total": q["total_tokens"],
                      "sha16_logged": q["sha"], "sha256_ids": ids_sha(q["ids"]), "ids_equal_recomputed": mine == q["ids"],
                      "prompt_equal_rendered": q["ids"][: len(prompts[q["doc"]])] == prompts[q["doc"]],
                      "trace_tokens_logged": q["trace_tokens"], "trace_tokens_recomputed": len(cache[q["trace"]])})
    files = sorted(glob.glob(f"{RUNS}/**/*.reasoning.txt", recursive=True))
    if os.environ.get("LIMIT"):
        files = [f for f in files if "pilot/" in f][: int(os.environ["LIMIT"])]
    traces = sorted(glob.glob(f"{RUNS}/investigations/2026-10-glm53-completion/traces/step*/*/prompt*.json.gz"))
    if os.environ.get("LIMIT"):
        traces = traces[:2]
    nproc = int(os.environ.get("NPROC", "6"))
    with Pool(nproc) as pool:
        rows = pool.map(text_row, [(f, plen) for f in files], chunksize=4)
        trs = pool.map(trace_row, traces)
    # does each trace prompt equal the rendered GPQA prompt of doc = prompt_id?
    for tr in trs:
        p = prompts.get(int(tr["prompt_id"]))
        tr["prompt_equals_rendered_gpqa_doc"] = (p is not None and ids_sha(p) == tr["sha256_prompt_ids"])
    json.dump({"text_rows": rows, "exact_traces": trs, "phase2_requests": p2out,
               "params": {"WIN_CHARS": WIN_CHARS, "L200": L200, "REPEAT_LOOP_ZLIB": REPEAT_LOOP_ZLIB,
                          "REPEAT_REP200": REPEAT_REP200, "CHECKPOINTS": CHECKPOINTS,
                          "commit_regex": {k: v.pattern for k, v in COMMIT.items()}, "strong": STRONG,
                          "samples_for_prompts": os.path.relpath(SAMPLES, RUNS)}},
              open(f"{OUT}/scan.json", "w"))
    print("rows", len(rows), "traces", len(trs), "p2", len(p2out))


if __name__ == "__main__":
    main()
