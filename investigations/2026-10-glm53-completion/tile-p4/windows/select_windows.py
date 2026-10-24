#!/usr/bin/env python3
"""P4 window selection and fixture build (PROPOSAL.md section 4; PROPOSAL-AMENDMENTS.md A1-A3).  CPU only.

Runs INSIDE a CPU container of the local stock 0.9.1 image (a9e985c0; `docker run --rm --network none`,
CUDA_VISIBLE_DEVICES empty), through windows/run_select.sh.  Mounts:
  /lr     long-reply/ (read only): d091 and d091b raw bodies (PRIVATE/raw/s1.raw.jsonl.gz) and their run metadata
  /repo   the K3.25 model repo (read only): tokenizer.json of snapshot 0490d2f7
  /out    windows/out/  (public: hashes, counts, classes; no ids, no text, no positions)
  /priv   PRIVATE-fixtures/ (0700; files 0600): token ids, window positions, the batch plan

THE RULE (fixed before any reply text or id was read; only the public row metadata -- finish_reason and logged
completion_tokens -- was looked at to count candidates):
  Sources   the two stock 0.9.1 GPQA passes with full reply text retained (long-reply/d091 seed 70704 and d091b
            seed 70705; image a9e985c0; max_tokens 327,680).  No stock 0.9.1 record on this machine keeps generated
            token ids beyond 5k tokens (fixture-inventory/INVENTORY.md "Gaps"), so reasoning text is re-tokenized
            with the served tokenizer exactly as long-reply B+/D/E built their prefixes (amendment A1).
  Classes   finished: finish_reason "stop" and logged completion_tokens >= 100,000;
            capped:   finish_reason "length" (the 327,680 cap).
  Choice    up to N_PER = 8 per class.  If a class has more, a seeded sample (random.Random(20261023).sample) over the
            class list sorted by (source, doc_id).  Fewer than 8 is reported as a shortfall, never filled.
  Prompt    rendered with the served chat template (/opt/glm53/templates/glm53-zai-a5b45eb.jinja) and the K3.25
            tokenizer (add_special_tokens=False); must end <|assistant|><think> and must equal the server's logged
            prompt_tokens.  Reasoning = message["reasoning"], tokenized whole (tokenizers, add_special_tokens=False).
  Windows   W = 512 forced tokens each; o = index into the reasoning ids (= output index after <think>);
            the absolute prefix length is P = prompt_len + o_start.
              early  P = the value with P = 1 (mod 4352) nearest to prompt_len + 16,000 (ties: the lower);
              deep   the same around prompt_len + 96,000;
              close  finished: o_start = N_reasoning - 512 (the 512 tokens before </think>);
                     capped:   o_start = 131,072 - 512 (amendment A2: the 512 tokens before output index 131,072,
                     not before the 327,680 cap; the literal rule is costed in PROPOSAL-AMENDMENTS.md).
            P = 1 (mod 4352) makes the uncached remainder of a warm prefix exactly 4,353 tokens under this engine's
            EAGLE-type prefix-cache rule, cached = (floor((P-1)/4352) - 1) * 4352 (first-contrast DIAGNOSIS-fc1 s.2).
            Checks: windows of one reply never overlap; no stop id (154820/154827/154829) and no </think> (154842)
            inside any forced window or prefix part of the reasoning; o_start + 512 <= N_reasoning.
  Batches   replies ordered by alternating class (finished, capped, finished, ...), each class in its selection
            order.  Batches 1-5: replies (3b, 3b+1, 3b+2) contribute (early, deep, close), (early, deep, close) and
            (deep, close); batch 6: reply 16's three windows plus the five early windows left over from batches 1-5.
            Every batch is 8 windows (8 requests in flight); its warm set is one prefix per reply (the longest P it
            needs in that batch).  The plan is identical in every session (A, B, A').

usage: select_windows.py [--dry-sources]     (--dry-sources: counts only, no tokenization, nothing written)
"""
import gzip
import hashlib
import json
import os
import random
import sys

os.umask(0o077)
LR, REPO, OUT, PRIV = "/lr", "/repo", "/out", "/priv"
SNAP = "0490d2f708b12145f6516555ab066aaeb401cd21"
TEMPLATE = "/opt/glm53/templates/glm53-zai-a5b45eb.jinja"
TEMPLATE_SHA = "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5"
TOKJSON_SHA = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
SOURCES = [("d091", "d091/PRIVATE/raw/s1.raw.jsonl.gz", 70704), ("d091b", "d091b/PRIVATE/raw/s1.raw.jsonl.gz", 70705)]
SOURCE_SHA = {"d091": "4d3f631315e7d8219902ee344fcf00b499501c5ca841291f2c13129e800ca40f",
              "d091b": "a07fc7bdb87ec67af80f17a8885f447faa7105b33c8c13caddb734df2cabe07a"}
N_PER = 8
SEED = 20261023
W = 512
BLOCK = 4352
EARLY, DEEP = 16000, 96000
CAPPED_CLOSE_END = 131072          # amendment A2
FIN_MIN = 100000
STOP_IDS = (154820, 154827, 154829)
THINK_OPEN, THINK_CLOSE, ASSISTANT = 154841, 154842, 154828


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def aligned_near(prompt_len, target):
    """P = 1 (mod BLOCK) nearest to prompt_len + target (ties -> lower)."""
    t = prompt_len + target
    lo = ((t - 1) // BLOCK) * BLOCK + 1
    hi = lo + BLOCK
    return lo if (t - lo) <= (hi - t) else hi


def load_rows():
    rows = []
    for src, rel, seed in SOURCES:
        p = os.path.join(LR, rel)
        with gzip.open(p, "rt") as f:
            for line_no, line in enumerate(f):
                d = json.loads(line)
                ch = d["response"]["choices"][0]
                rows.append(dict(source=src, line=line_no, seed=d["seed"], doc_id=line_no,
                                 finish=ch["finish_reason"], completion_tokens=d["response"]["usage"]["completion_tokens"],
                                 prompt_tokens=d["response"]["usage"]["prompt_tokens"], _d=d))
    return rows


def classify(rows):
    fin = [r for r in rows if r["finish"] == "stop" and r["completion_tokens"] >= FIN_MIN]
    cap = [r for r in rows if r["finish"] == "length"]
    key = lambda r: (r["source"], r["doc_id"])  # noqa: E731
    fin.sort(key=key)
    cap.sort(key=key)
    rng = random.Random(SEED)
    pick = {}
    for name, lst in (("finished", fin), ("capped", cap)):
        if len(lst) > N_PER:
            idx = sorted(rng.sample(range(len(lst)), N_PER))
            pick[name] = [lst[i] for i in idx]
        else:
            pick[name] = list(lst)
    return fin, cap, pick


def batch_plan(replies):
    """replies: list of 16 reply ids in plan order -> list of 6 batches of (reply, kind)."""
    assert len(replies) == 16
    batches, leftover = [], []
    for b in range(5):
        r0, r1, r2 = replies[3 * b: 3 * b + 3]
        batches.append([(r0, "early"), (r0, "deep"), (r0, "close"), (r1, "early"), (r1, "deep"), (r1, "close"),
                        (r2, "deep"), (r2, "close")])
        leftover.append((r2, "early"))
    r15 = replies[15]
    batches.append([(r15, "early"), (r15, "deep"), (r15, "close")] + leftover)
    assert all(len(b) == 8 for b in batches)
    allw = [w for b in batches for w in b]
    assert len(allw) == 48 and len(set(allw)) == 48
    return batches


def main():
    dry = "--dry-sources" in sys.argv
    checks = []

    def chk(name, ok, detail=""):
        checks.append(dict(check=name, ok=bool(ok), detail=detail))
        if not ok:
            print("CHECK FAILED:", name, detail, flush=True)

    for src, rel, _ in SOURCES:
        chk(f"source {src} sha256", fsha(os.path.join(LR, rel)) == SOURCE_SHA[src])
    rows = load_rows()
    fin, cap, pick = classify(rows)
    counts = dict(rows=len(rows), finished_available=len(fin), capped_available=len(cap),
                  finished_selected=len(pick["finished"]), capped_selected=len(pick["capped"]),
                  shortfall={k: max(0, N_PER - len(v)) for k, v in pick.items()})
    print("candidates:", json.dumps(counts), flush=True)
    if dry:
        return 0
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    snapdir = f"{REPO}/snapshots/{SNAP}"
    chk("served chat template sha256", fsha(TEMPLATE) == TEMPLATE_SHA)
    chk("tokenizer.json sha256", fsha(f"{snapdir}/tokenizer.json") == TOKJSON_SHA)
    template = open(TEMPLATE).read()
    hf = AutoTokenizer.from_pretrained(snapdir)
    tk = Tokenizer.from_file(f"{snapdir}/tokenizer.json")
    # plan order: alternate classes
    order = []
    for i in range(N_PER):
        for name in ("finished", "capped"):
            if i < len(pick[name]):
                order.append((name, pick[name][i]))
    replies, priv_replies, pub_replies = [], {}, []
    for k, (cls, r) in enumerate(order):
        rid = f"R{k + 1:02d}"
        d = r["_d"]
        req, resp = d["request"], d["response"]
        msgs = req["messages"]
        kw = req.get("chat_template_kwargs") or {}
        text = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, chat_template=template, **kw)
        p_ids = tk.encode(text, add_special_tokens=False).ids
        p_hf = hf.encode(text, add_special_tokens=False)
        reasoning = resp["choices"][0]["message"].get("reasoning") or ""
        r_ids = tk.encode(reasoning, add_special_tokens=False).ids
        N = len(r_ids)
        P0 = len(p_ids)
        chk(f"{rid}: prompt ids equal the server's logged prompt_tokens", P0 == r["prompt_tokens"], f"{P0} vs {r['prompt_tokens']}")
        chk(f"{rid}: transformers and tokenizers agree on the prompt", list(p_hf) == list(p_ids))
        chk(f"{rid}: prompt ends <|assistant|><think>", list(p_ids[-2:]) == [ASSISTANT, THINK_OPEN])
        chk(f"{rid}: no stop id or </think> in the reasoning ids",
            not any(t in STOP_IDS or t == THINK_CLOSE for t in r_ids))
        wins = {}
        e = aligned_near(P0, EARLY) - P0
        dp = aligned_near(P0, DEEP) - P0
        cl = (N - W) if cls == "finished" else (CAPPED_CLOSE_END - W)
        for kind, o in (("early", e), ("deep", dp), ("close", cl)):
            wins[kind] = dict(o_start=int(o), P=int(P0 + o))
            chk(f"{rid} {kind}: window inside the reasoning", 0 < o and o + W <= N, f"o {o} N {N}")
        chk(f"{rid}: windows do not overlap",
            wins["early"]["o_start"] + W <= wins["deep"]["o_start"] and wins["deep"]["o_start"] + W <= wins["close"]["o_start"])
        chk(f"{rid}: early/deep P = 1 (mod {BLOCK})", wins["early"]["P"] % BLOCK == 1 and wins["deep"]["P"] % BLOCK == 1)
        full = list(p_ids) + list(r_ids)
        priv_replies[rid] = dict(cls=cls, source=r["source"], line=r["line"], seed=r["seed"], prompt_len=P0,
                                 n_reasoning=N, completion_tokens_logged=r["completion_tokens"], windows=wins,
                                 ids=full)
        pub_replies.append(dict(reply=rid, cls=cls, source=r["source"], prompt_ids_sha256=ids_sha(p_ids),
                                reasoning_ids_sha256=ids_sha(r_ids),
                                reasoning_text_sha256=hashlib.sha256(reasoning.encode()).hexdigest(),
                                messages_sha256=d.get("messages_sha256"),
                                window_ids_sha256={kd: ids_sha(r_ids[w["o_start"]:w["o_start"] + W]) for kd, w in wins.items()},
                                prefix_ids_sha256={kd: ids_sha(full[:w["P"]]) for kd, w in wins.items()}))
        replies.append(rid)
    batches = batch_plan(replies)
    plan = []
    for b, items in enumerate(batches, 1):
        wl = []
        for (rid, kind) in items:
            wid = f"{rid}-{kind}"
            wl.append(wid)
        warm = {}
        for (rid, kind) in items:
            warm[rid] = max(warm.get(rid, 0), priv_replies[rid]["windows"][kind]["P"])
        plan.append(dict(batch=b, windows=wl, warm=[dict(reply=k, P=v) for k, v in warm.items()]))
    # remainder per forced request under the EAGLE rule (cached = (floor((P-1)/4352) - 1) * 4352)
    for bp in plan:
        rem = []
        for wid in bp["windows"]:
            rid, kind = wid.split("-")
            P = priv_replies[rid]["windows"][kind]["P"]
            rem.append(P - max(0, ((P - 1) // BLOCK - 1) * BLOCK))
        bp["remainder_tokens"] = rem
        bp["warm_tokens"] = sum(w["P"] for w in bp["warm"])
    fixture = dict(note="PRIVATE: GPQA-derived token ids and positions (P4); never published", W=W, block=BLOCK,
                   replies=priv_replies, plan=plan, rule_doc=__doc__)
    path = os.path.join(PRIV, "windows.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists (never overwritten)")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(fixture, f)
    tot_warm = sum(bp["warm_tokens"] for bp in plan)
    tot_rem = sum(sum(bp["remainder_tokens"]) for bp in plan)
    pub = dict(what="P4 window manifest (public: hashes, classes, counts and token totals; no ids, text or positions)",
               rule="windows/select_windows.py docstring", counts=counts, n_replies=len(replies), n_windows=48,
               batches=[dict(batch=bp["batch"], windows=bp["windows"], n_warm_prefixes=len(bp["warm"])) for bp in plan],
               totals=dict(warm_prefill_tokens_per_session=tot_warm,
                           remainder_tokens_per_pass=tot_rem,
                           forced_positions_per_pass=48 * W),
               replies=pub_replies, checks=checks, private_fixture_sha256=fsha(path),
               all_checks_ok=all(c["ok"] for c in checks))
    json.dump(pub, open(os.path.join(OUT, "window-manifest.json"), "w"), indent=1)
    print(json.dumps(dict(all_checks_ok=pub["all_checks_ok"], totals=pub["totals"], counts=counts)), flush=True)
    return 0 if pub["all_checks_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
