#!/usr/bin/env python3
"""[E copy of rep070/refreeze/checks/seed_scan.py (sha256 3d8e80e4...); only SKIP_DIRS differs]
Seed inventory for the re-freeze of the stock 0.7.0 replication (variant F needs three request seeds never used by any
GPQA pass on this machine). CPU only, standard library, read only. Writes ids, seeds and counts only: no question, answer
or reply text (the messages hash is used to recognise a GPQA prompt; no message content is read).

Sources scanned
  1. every request log (*.requests.jsonl and any *.jsonl whose rows carry both 'messages_sha256' and 'seed') under
     <data-runs> and the public repo: per file, request seed -> count, and how many of those requests carry one of the 198
     GPQA Diamond prompts (messages_sha256 in the investigation's prompt map) -> "GPQA request seeds" [M];
  2. every lm-eval results_*.json under <data-runs> and the public repo: task names, model_args seed (the request seed;
     lm-eval's local-chat-completions default is 1234 when absent [S api_models.py:126/174, openai_completions.py]),
     and the lm-eval process seed -> "GPQA pass seeds" [M from config; default 1234 inferred when absent];
  3. every published per-question results.jsonl under the public repo runs/: the 'seed' column [M];
  4. a broad text scan (superset): every integer that follows a seed token ('"seed": N', 'seed=N', '--seed N',
     'SEED=N', 'seed N', 'seeds N/N/N') in text files below 200 MB under <data-runs>, the public repo and <eval-tools>.
     This catches probes, screens, replays and analysis/bootstrap seeds; it over-collects by design.
A seed is eligible for variant F only if it appears in none of 1-4.

usage: seed_scan.py --out seed-inventory.json [--candidates 7101 7102 7103]
"""
import argparse, glob, json, os, re, sys
from collections import Counter, defaultdict

ROOTS = ["<data-runs>", "<this-repo>"]
TEXT_ROOTS = ROOTS + ["<eval-tools>"]
SKIP_DIRS = ("<runs>/long-reply/e-cross",)  # E copy: this package only (its own seed lists)
TOKMAP = "<runs>/d7/exposure/gpqa_prompt_tokens.json"
TEXT_EXT = (".json", ".jsonl", ".log", ".txt", ".sh", ".py", ".md", ".out", ".csv", ".yaml", ".yml", ".toml", ".err")
SEED_RX = re.compile(r'(?i)(?:"seed"\s*:\s*|\bseed\s*=\s*|--seed[ =]|\bSEED[A-Z_]*\s*=\s*|\bseeds?\s+)(\d+(?:\s*[/,]\s*\d+)*)')


def skip(p):
    return any(p.startswith(s) for s in SKIP_DIRS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", nargs="*", type=int, default=[])
    a = ap.parse_args()
    gpqa_msgs = {d["messages_sha256"] for d in json.load(open(TOKMAP))["docs"]}
    assert len(gpqa_msgs) == 198
    out = {"what": "seed inventory (ids, seeds and counts only)", "request_logs": {}, "results_json": [], "published_rows": {},
           "gpqa_request_seeds": Counter(), "gpqa_pass_seeds": Counter(), "any_request_seeds": Counter(), "text_scan_seeds": Counter()}
    # 1. request logs
    for root in ROOTS:
        for p in sorted(glob.glob(root + "/**/*.jsonl", recursive=True)):
            if skip(p) or os.path.getsize(p) > 200 << 20:
                continue
            seeds, gseeds, rows = Counter(), Counter(), 0
            try:
                with open(p, errors="replace") as f:
                    for i, line in enumerate(f):
                        if not line.strip():
                            continue
                        try:
                            r = json.loads(line)
                        except ValueError:
                            break
                        if not isinstance(r, dict) or "seed" not in r or "messages_sha256" not in r:
                            if i == 0:
                                break
                            continue
                        rows += 1
                        seeds[r["seed"]] += 1
                        if r["messages_sha256"] in gpqa_msgs:
                            gseeds[r["seed"]] += 1
            except OSError:
                continue
            if rows:
                out["request_logs"][p] = {"rows": rows, "seeds": dict(sorted(seeds.items(), key=lambda kv: str(kv[0]))),
                                          "gpqa_rows_by_seed": dict(sorted(gseeds.items(), key=lambda kv: str(kv[0])))}
                out["any_request_seeds"].update({str(k): v for k, v in seeds.items()})
                out["gpqa_request_seeds"].update({str(k): v for k, v in gseeds.items()})
    # 2. lm-eval results jsons
    for root in ROOTS:
        for p in sorted(glob.glob(root + "/**/results_*.json", recursive=True)):
            if skip(p):
                continue
            try:
                r = json.load(open(p))
            except (OSError, ValueError):
                continue
            cfg = r.get("config", {}) or {}
            ma = cfg.get("model_args") or {}
            if isinstance(ma, str):
                ma = dict(kv.split("=", 1) for kv in ma.split(",") if "=" in kv)
            tasks = sorted((r.get("results") or {}).keys())
            gp = any("gpqa" in t for t in tasks)
            seed_arg = ma.get("seed")
            req_seed = int(seed_arg) if seed_arg not in (None, "") else 1234
            rec = {"file": p, "tasks": tasks, "gpqa": gp, "model_args_seed": seed_arg,
                   "request_seed": req_seed, "request_seed_source": "model_args" if seed_arg not in (None, "") else "lm-eval default (inferred)",
                   "random_seed": cfg.get("random_seed"), "numpy_seed": cfg.get("numpy_seed"), "torch_seed": cfg.get("torch_seed"),
                   "fewshot_seed": cfg.get("fewshot_seed")}
            out["results_json"].append(rec)
            if gp:
                out["gpqa_pass_seeds"][str(req_seed)] += 1
    # 3. published rows
    for p in sorted(glob.glob("<this-repo>/runs/*/results.jsonl")):
        c = Counter()
        for line in open(p, errors="replace"):
            if line.strip():
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                if isinstance(x, dict) and "seed" in x:
                    c[str(x["seed"])] += 1
        if c:
            out["published_rows"][p] = dict(c)
            out["gpqa_pass_seeds"].update({k: 0 for k in c})   # ensure listed
            for k in c:
                out["gpqa_request_seeds"][k] += 0
    # 4. broad text scan
    nfiles = 0
    for root in TEXT_ROOTS:
        for dp, dn, fn in os.walk(root):
            if skip(dp) or "/.git" in dp or "__pycache__" in dp:
                continue
            for f in fn:
                p = os.path.join(dp, f)
                if not f.endswith(TEXT_EXT) or os.path.islink(p):
                    continue
                try:
                    if os.path.getsize(p) > 200 << 20:
                        continue
                    with open(p, errors="replace") as fh:
                        txt = fh.read()
                except OSError:
                    continue
                nfiles += 1
                for m in SEED_RX.finditer(txt):
                    for v in re.split(r"\s*[/,]\s*", m.group(1)):
                        if v.isdigit() and len(v) <= 12:
                            out["text_scan_seeds"][v] += 1
    out["text_scan_files"] = nfiles
    used = set(out["gpqa_request_seeds"]) | set(out["gpqa_pass_seeds"]) | set(out["any_request_seeds"]) | set(out["text_scan_seeds"])
    out["candidates"] = {str(c): {"in_gpqa_request_logs": str(c) in out["gpqa_request_seeds"], "in_gpqa_pass_configs": str(c) in out["gpqa_pass_seeds"],
                                  "in_any_request_log": str(c) in out["any_request_seeds"], "in_text_scan": str(c) in out["text_scan_seeds"],
                                  "eligible": str(c) not in used} for c in a.candidates}
    for k in ("gpqa_request_seeds", "gpqa_pass_seeds", "any_request_seeds", "text_scan_seeds"):
        out[k] = dict(sorted(out[k].items(), key=lambda kv: (len(kv[0]), kv[0])))
    out["summary"] = {"gpqa_seeds_used": sorted({int(k) for k in list(out["gpqa_request_seeds"]) + list(out["gpqa_pass_seeds"]) if str(k).isdigit()}),
                      "request_seeds_any_prompt": sorted({int(k) for k in out["any_request_seeds"] if str(k).isdigit()}),
                      "non_integer_request_seeds": sorted(k for k in out["any_request_seeds"] if not str(k).isdigit()),
                      "text_scan_distinct_values": len(out["text_scan_seeds"]),
                      "candidates_all_eligible": all(v["eligible"] for v in out["candidates"].values()) if a.candidates else None}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))
    print(json.dumps(out["candidates"], indent=1))
    return 0 if (not a.candidates or out["summary"]["candidates_all_eligible"]) else 3


if __name__ == "__main__":
    sys.exit(main())
