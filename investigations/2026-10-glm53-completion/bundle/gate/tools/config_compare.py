#!/usr/bin/env python3
"""Effective server configuration: gate run vs the historical clean 0.9.1 three-pass GPQA run
(first server block of its log).  Usage: config_compare.py GATE_SERVER_LOG HIST_SERVER_LOG"""
import ast
import re
import sys

gate_path, hist_path = sys.argv[1], sys.argv[2]


def first_block(path):
    lines = open(path, errors="replace").read().splitlines()
    starts = [i for i, l in enumerate(lines) if "non-default args:" in l]
    end = starts[1] if len(starts) > 1 else len(lines)
    return lines[:end]


def startup(lines):
    out = []
    for l in lines:
        if "Engine 000: Avg" in l:
            break
        out.append(l)
    return out


def strip(l):
    l = re.sub(r"^\([^)]*\)\s+", "", l)
    l = re.sub(r"^(INFO|WARNING|ERROR|DEBUG)\s+\d\d-\d\d \d\d:\d\d:\d\d\s+\[[^\]]*\]\s+", r"\1 ", l)
    return l


def nondefault(lines):
    for l in lines:
        if "non-default args:" in l:
            return ast.literal_eval(l.split("non-default args: ", 1)[1])
    return {}


def engine_cfg(lines):
    for l in lines:
        if "Initializing a V1 LLM engine" in l:
            return l.split("with config: ", 1)[1]
    return ""


def split_cfg(s):
    # top-level comma split respecting brackets/quotes
    out, depth, cur, q = [], 0, "", None
    for ch in s:
        if q:
            cur += ch
            if ch == q:
                q = None
            continue
        if ch in "'\"":
            q = ch
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    d = {}
    for kv in out:
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k.strip()] = v.strip()
    return d


G = startup(first_block(gate_path))
H = startup(first_block(hist_path))
print(f"startup lines: gate {len(G)}, historical {len(H)}")

nd_g, nd_h = nondefault(G), nondefault(H)
print("\n## non-default args (API server)")
for k in sorted(set(nd_g) | set(nd_h)):
    if nd_g.get(k) != nd_h.get(k):
        print(f"DIFF {k}: gate={nd_g.get(k)!r} hist={nd_h.get(k)!r}")
print(f"identical keys: {sum(1 for k in nd_g if nd_g.get(k) == nd_h.get(k))} of {len(set(nd_g) | set(nd_h))}")

cg, ch = split_cfg(engine_cfg(G)), split_cfg(engine_cfg(H))
print("\n## engine config line")
for k in sorted(set(cg) | set(ch)):
    if cg.get(k) != ch.get(k):
        print(f"DIFF {k}: gate={cg.get(k)!r}\n          hist={ch.get(k)!r}")
print(f"identical fields: {sum(1 for k in cg if cg.get(k) == ch.get(k))} of {len(set(cg) | set(ch))}")

env_g = sorted(set(re.findall(r"Unknown vLLM environment variable detected: (\S+)", "\n".join(G))))
env_h = sorted(set(re.findall(r"Unknown vLLM environment variable detected: (\S+)", "\n".join(H))))
print("\n## VLLM_* environment variables reported by vLLM")
print("only gate:", sorted(set(env_g) - set(env_h)))
print("only hist:", sorted(set(env_h) - set(env_g)))
print("common:", len(set(env_g) & set(env_h)))

KEYS = re.compile(r"cudagraph|Capturing|Breakable|B12x|B12X|kpool|owner|OWNER|split|NOPE|record|DCP|dcp|"
                  r"DFlash|dflash|prefix|Prefix|backend|all-reduce|MoE|EXL3|indexer|mHC|max_num|chunk",
                  re.I)


def norm(l):
    s = strip(l)
    s = re.sub(r"\d+(\.\d+)?", "N", s)
    s = re.sub(r"pid=N", "", s)
    return s.strip()


sg = {norm(l) for l in G if KEYS.search(l) and "it/s]" not in l and "s/it]" not in l}
sh = {norm(l) for l in H if KEYS.search(l) and "it/s]" not in l and "s/it]" not in l}
print("\n## configuration-bearing startup lines (numbers masked) present in only one log")
for s in sorted(sg - sh):
    print("only gate:", s[:300])
for s in sorted(sh - sg):
    print("only hist:", s[:300])

print("\n## key lines (gate | historical)")
pats = ["cudagraph_mode", "cudagraph_capture_sizes", "Breakable CUDA graph", "Using B12x fused paged",
        "mla_owned", "VLLM_GLM53_MLA_OWNERS", "NOPE", "kv_gmem_stride", "DSA indexer decode path",
        "decode_context_parallel", "enable_prefix_caching", "num_spec_tokens", "Capturing CUDA graphs (FULL)",
        "max_num_seqs", "max_num_batched_tokens"]
for p in pats:
    def find(L):
        r = []
        for l in L:
            if p in l:
                m = l
                if p in ("cudagraph_mode", "cudagraph_capture_sizes", "enable_prefix_caching",
                         "num_spec_tokens", "decode_context_parallel", "max_num_seqs",
                         "max_num_batched_tokens"):
                    mm = re.search(re.escape(p) + r"[^,]*(,[^,=]*)*?(?=, [a-z_]+=|, '[a-z_]+':|$)", l)
                    m = mm.group(0) if mm else l
                r.append(strip(m)[:240])
        return sorted(set(r))[:3]
    fg, fh = find(G), find(H)
    print(f"- {p}: {'SAME' if fg == fh else 'DIFFERENT'}")
    print(f"    gate: {fg}")
    if fg != fh:
        print(f"    hist: {fh}")

print("\n## graph captures (progress-bar totals)")
for name in ("Capturing CUDA graphs (PIECEWISE)", "Capturing CUDA graphs (FULL)", "Capturing dflash2 CUDA graphs (FULL)",
             "Profiling CUDA graph memory (PIECEWISE)", "Profiling CUDA graph memory (FULL)"):
    def tot(L):
        t = set()
        for l in L:
            for m in re.finditer(re.escape(name) + r":\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)", l):
                if m.group(1) == m.group(2):
                    t.add(int(m.group(2)))
        return sorted(t)
    print(f"- {name}: gate completed {tot(G)} | hist completed {tot(H)}")
