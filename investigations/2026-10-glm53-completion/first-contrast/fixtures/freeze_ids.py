#!/usr/bin/env python3
"""Freeze the exact token ids of the 8 first-contrast fixtures (PROPOSAL.md s.3.2; first_contrast.json "ids" rule).
CPU only: runs in the 0.9.1 release image (tokenizer as fixture-inventory/scripts/fixture_scan.py) as a container with
--network none, no --gpus, CUDA_VISIBLE_DEVICES= and read-only mounts:
    /runs  = <data-runs> (ro)          /model = the K3.25 HF repo dir (ro)          /fx = first-contrast/fixtures (rw)
Reads the client-saved reasoning text of each fixture's source history and the rendered GPQA prompt of its doc, and
writes ids, lengths and hashes only (fixtures/ids/<fixture>.json.gz, fixtures/ids_manifest.json). No reasoning or
prompt TEXT is written.

Conventions (asserted; position_plan.py):
  prompt = rendered chat prompt (add_generation_prompt, enable_thinking; ends in <think> 154841), length prompt_tokens;
  reasoning = re-tokenized reasoning text (add_special_tokens=False); sha256(json.dumps(reasoning)) == history_sha256;
  P = prompt + s; prefix ids = prompt + reasoning[:s];
  forced list = reasoning[s : end + 1] (W + 1 ids) where reasoning[end] is the history's next token; for a history
  that closed at `end` (= len(reasoning)), the next token is </think> 154842 (appended).
Also written per fixture (for the client): the LP prompt ids end (P_L = P + 7168) and the extra ids used by the
different-build-length prefill reference (A22-3: P + 8192 <= P + W), all as slices of the same arrays.
"""
import glob
import gzip
import hashlib
import json
import os

SNAP = "/model/snapshots/0490d2f708b12145f6516555ab066aaeb401cd21"
RUNS = "/runs"
FX = "/fx"
FC = f"{RUNS}/investigations/2026-10-glm53-completion/hook-validation/fixtures/first_contrast.json"
INV = f"{RUNS}/investigations/2026-10-glm53-completion/fixture-inventory/inventory.json"
TT = f"{RUNS}/investigations/2026-10-glm53-completion/fixture-inventory/termination_tokens.json"
SAMPLES = glob.glob(f"{RUNS}/gpqa/glm53-flash-exl3-k4@tpurtell+v0.8.0__pass1/**/samples_*.jsonl", recursive=True)[0]
THINK_OPEN, THINK_CLOSE = 154841, 154842
STOPS = (154820, 154827, 154829)
LP_SHIFT, REF2_LEN = 7168, 8192


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def main():
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    tok = Tokenizer.from_file(f"{SNAP}/tokenizer.json")
    hf = AutoTokenizer.from_pretrained(SNAP)
    tmpl = open(f"{SNAP}/chat_template.jinja").read()
    fc = json.load(open(FC))
    inv = json.load(open(INV))
    tt = json.load(open(TT))
    rows = {r["id"]: r for r in inv["text_rows"]}
    prompts = {}
    need = {int(f["doc"]) for f in fc["fixtures"]}
    for ln in open(SAMPLES):
        r = json.loads(ln)
        d = r["doc_id"]
        if d in prompts or d not in need:
            continue
        msgs = json.loads(r["arguments"]["gen_args_0"]["arg_0"][0])
        txt = hf.apply_chat_template(msgs, chat_template=tmpl, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True)
        prompts[d] = hf.encode(txt, add_special_tokens=False)
    os.makedirs(f"{FX}/ids", exist_ok=True)
    man = {"source": "freeze_ids.py", "tokenizer": f"{SNAP}/tokenizer.json",
           "tokenizer_sha256": hashlib.sha256(open(f"{SNAP}/tokenizer.json", "rb").read()).hexdigest(),
           "first_contrast_sha256": hashlib.sha256(open(FC, "rb").read()).hexdigest(),
           "fixtures": {}, "checks": []}
    chk = man["checks"]
    for f in fc["fixtures"]:
        name, doc = f["fixture"], int(f["doc"])
        row = [v for k, v in rows.items() if k.startswith(f["source_id"])]
        assert len(row) == 1, name
        row = row[0]
        text = open(os.path.join(RUNS, row["path"]), encoding="utf-8").read()
        reas = tok.encode(text, add_special_tokens=False).ids
        pr = prompts[doc]
        s, end = int(f["window_start_reasoning_index"]), int(f["window_end_reasoning_index"])
        P, W = int(f["prefix_len_P"]), int(f["W_forced"])
        closed = bool(row.get("closed"))
        nxt = reas[end] if end < len(reas) else (THINK_CLOSE if closed and end == len(reas) else None)
        prefix = pr + reas[:s]
        forced = reas[s:end] + [nxt]
        lp_forced = forced[LP_SHIFT:]
        c = [
            ("prompt len == prompt_tokens", len(pr) == int(f["prompt_tokens"])),
            ("prompt ends in <think>", pr[-1] == THINK_OPEN),
            ("prompt sha == termination_tokens.json rendered", ids_sha(pr) == tt["sha256_rendered_prompt_ids"][str(doc)]),
            ("reasoning ids sha == history_sha256", ids_sha(reas) == f["history_sha256"]),
            ("reasoning len == inventory n_reasoning_tokens", len(reas) == int(row["n_reasoning_tokens"])),
            ("P == prompt + s", P == len(pr) + s and len(prefix) == P),
            ("P % 8 == 0", P % 8 == 0),
            ("W == end - s", W == end - s),
            ("forced list len == W + 1", len(forced) == W + 1),
            ("next token determined", nxt is not None),
            ("near-close: forced list ends in </think> exactly at o = W",
             (forced[-1] == THINK_CLOSE) == (f["cls"] == "near_close")),
            ("no </think> or stop id inside the window", not any(t in (THINK_CLOSE,) + STOPS for t in forced[:-1])),
            ("no special ids in prefix reasoning", not any(t in (THINK_OPEN, THINK_CLOSE) + STOPS for t in reas[:s])),
            ("LP forced list = forced[7168:], len W - 7168 + 1", len(lp_forced) == W - LP_SHIFT + 1),
            ("reference-2 build P + 8192 within the history window", REF2_LEN <= W),
            ("P + W < 131072", P + W < 131072),
        ]
        for nm, ok in c:
            chk.append([name, nm, bool(ok)])
        rec = {"fixture": name, "doc": doc, "P": P, "W": W, "prefix": prefix, "forced": forced}
        raw = json.dumps(rec, separators=(",", ":")).encode()
        p = f"{FX}/ids/{name}.json.gz"
        with gzip.GzipFile(p, "wb", mtime=0) as g:      # mtime 0: byte-reproducible archive
            g.write(raw)
        man["fixtures"][name] = {
            "file": f"ids/{name}.json.gz", "file_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "P": P, "W": W, "forced_list_len": len(forced), "prompt_len": len(pr),
            "prefix_sha256": ids_sha(prefix), "forced_sha256": ids_sha(forced),
            "lp_prefix_sha256": ids_sha(prefix + forced[:LP_SHIFT]), "lp_forced_sha256": ids_sha(lp_forced),
            "ref2_prompt_sha256": ids_sha(prefix + forced[:REF2_LEN]),
            "next_token": nxt, "source_path": row["path"], "history_sha256": f["history_sha256"]}
    man["checks_pass"] = all(c[2] for c in chk)
    json.dump(man, open(f"{FX}/ids_manifest.json", "w"), indent=1)
    for c in chk:
        if not c[2]:
            print("FAIL", c)
    print("fixtures %d, checks %d, all pass %s" % (len(man["fixtures"]), len(chk), man["checks_pass"]))


if __name__ == "__main__":
    main()
