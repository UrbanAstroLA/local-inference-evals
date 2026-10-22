#!/usr/bin/env python3
"""[E copy of d091/checks/privacy_scan.py (diff in checks/privacy_scan.diff): several raw folders (d091 replies, B+
continuations), raw records of either shape, 64 id runs per prefix, and an answer-field key scan added.]
Privacy scan (host, stdlib, read only): no GPQA question, answer or reply text and no GPQA-derived token-id run may
appear in any file of WORK outside WORK/PRIVATE/. Prints counts and file paths only, never the text searched for.

Needles (held in memory only):
  - prompt text: 40-character windows of every one of the 198 GPQA prompts (lm-eval samples of the 0.9.1 record's pass 1;
    the choices are inside the prompt), at most 6 per question;
  - reply text: 40-character windows of the reasoning and final text of every raw record in RAW_DIR (3 + 1 per record);
  - token ids: 12-id runs of every prefix in WORK/PRIVATE/prefixes.json, if it exists (comma and comma-space joins), 64 per
    prefix spread over its length;
  - answer fields: JSON keys "target", "answer", "gold", "correct_answer" in .json / .jsonl / .md / .txt files.
Also checks WORK/PRIVATE is mode 0700 and every file in it 0600.
usage: privacy_scan_e.py WORK RAW_DIR [RAW_DIR ...]        exit 0 = no hit and modes correct
"""
import glob, gzip, json, os, re, stat, sys

KEY_RX = re.compile(r'"(target|answer|gold|correct_answer)"\s*:')

G = "<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass1"
TEXT_LIMIT = 200 << 20


def windows(t, k, w=40):
    t = t.strip()
    if len(t) < w + 20:
        return []
    return [t[int(i * (len(t) - w) / max(1, k - 1)):][:w] for i in range(k)] if k > 1 else [t[len(t) // 2:][:w]]


def main():
    work, rawds = sys.argv[1], sys.argv[2:]
    needles = set()
    sf = glob.glob(G + "/**/samples_*.jsonl", recursive=True)[0]
    nq = 0
    for line in open(sf):
        r = json.loads(line)
        if r["filter"] != "flexible-extract":
            continue
        nq += 1
        for m in json.loads(r["arguments"]["gen_args_0"]["arg_0"][0]):
            c = m.get("content") or ""
            needles.update(x for x in windows(c, 6) if len(set(x)) > 8)
    n_prompt = len(needles)
    nrec = 0
    for p in sorted(q for rawd in rawds for q in glob.glob(os.path.join(rawd, "*.raw.jsonl.gz"))):
        with gzip.open(p, "rt") as f:
            for line in f:
                d = json.loads(line)
                resp = d.get("response") or {}
                for ch in (resp.get("choices") or [])[:1]:
                    msg = ch.get("message") or {}
                    needles.update(x for x in windows(msg.get("reasoning") or "", 3) if len(set(x)) > 8)
                    needles.update(x for x in windows(msg.get("content") or "", 1) if len(set(x)) > 8)
                if "text" in d:
                    needles.update(x for x in windows(d.get("text") or "", 4) if len(set(x)) > 8)
                nrec += 1
    n_text = len(needles)
    idrun = set()
    pf = os.path.join(work, "PRIVATE", "prefixes.json")
    if os.path.exists(pf):
        for ids in json.load(open(pf))["prefixes"].values():
            for k in range(0, len(ids) - 12, max(1, (len(ids) - 12) // 64)):
                run = [str(i) for i in ids[k:k + 12]]
                idrun.add(",".join(run))
                idrun.add(", ".join(run))
    files = hits_t = hits_i = hits_k = 0
    hit_files = []
    for dp, dn, fn in os.walk(work):
        if os.path.abspath(dp).startswith(os.path.abspath(os.path.join(work, "PRIVATE"))) or "__pycache__" in dp:
            dn[:] = []
            continue
        for f in fn:
            p = os.path.join(dp, f)
            if os.path.islink(p) or os.path.getsize(p) > TEXT_LIMIT:
                continue
            try:
                raw = open(p, "rb").read()
            except OSError:
                continue
            if p.endswith(".gz"):
                try:
                    raw = gzip.decompress(raw)
                except OSError:
                    pass
            t = raw.decode("utf-8", "replace")
            files += 1
            ht = sum(1 for n in needles if n in t)
            hi = sum(1 for n in idrun if n in t)
            if ht or hi:
                hit_files.append(os.path.relpath(p, work))
            hits_t += ht
            hits_i += hi
            if p.endswith((".json", ".jsonl", ".md", ".txt")):
                hk = len(KEY_RX.findall(t))
                if hk:
                    hit_files.append(os.path.relpath(p, work) + " (answer key)")
                hits_k += hk
    priv = os.path.join(work, "PRIVATE")
    mode_ok = stat.S_IMODE(os.stat(priv).st_mode) == 0o700
    bad_modes = []
    for dp, dn, fn in os.walk(priv):
        for f in fn:
            m = stat.S_IMODE(os.stat(os.path.join(dp, f)).st_mode)
            if m != 0o600:
                bad_modes.append(f"{os.path.relpath(os.path.join(dp, f), work)} {oct(m)}")
        for d in dn:
            m = stat.S_IMODE(os.stat(os.path.join(dp, d)).st_mode)
            if m & 0o077:
                bad_modes.append(f"{os.path.relpath(os.path.join(dp, d), work)}/ {oct(m)}")
    print(f"privacy scan: {files} files outside PRIVATE/ searched; {n_prompt} prompt windows ({nq} questions), "
          f"{n_text - n_prompt} reply windows ({nrec} raw records), {len(idrun)} token-id runs; text hits {hits_t}, id-run hits {hits_i}; answer-key hits {hits_k}; "
          f"PRIVATE/ mode {'0700' if mode_ok else 'WRONG'}; files not 0600 or group/other-accessible dirs: {bad_modes or 'none'}")
    if hit_files:
        print("files with hits:", hit_files)
    return 0 if not (hits_t or hits_i or hits_k or bad_modes) and mode_ok else 1


if __name__ == "__main__":
    sys.exit(main())
