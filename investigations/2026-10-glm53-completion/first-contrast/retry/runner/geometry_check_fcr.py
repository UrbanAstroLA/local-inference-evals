#!/usr/bin/env python3
"""[R-1] Pre-traffic check that the served KV-cache geometry and cache configuration are the ones the F7 replay used
(replay/engine_replay.py: SERVED and the placement summary reproduced exactly from run-fc1). Host python.

usage: geometry_check_fcr.py SERVER_LOG CONTAINER_ENV MIN_BLOCKS   -> rc 0 iff every check holds
Checks: the 'glm53-placement kv-capacity' line holds the replayed 10 groups (spec, block size, page bytes, blocks per
max request) and num_blocks >= MIN_BLOCKS (the replay sweep keeps every lookup a hit down to 1,100 blocks; 1,150 is
the frozen floor); the engine's non-default args hold prefix caching on, mamba_cache_mode align,
max_num_batched_tokens 2048, max_num_seqs 16, DFlash with 3 speculative tokens; the container environment holds
GLM53_DFLASH_BOUNDARY_LOOKUP=0 and no VLLM_PREFIX_CACHE_RETENTION_INTERVAL (dense boundary-state retention).
"""
import json
import re
import sys

GROUPS = [("MLAAttentionSpec", 4352, 143616, 241), ("KpoolTailSpec", 8, 143616, 1)] + \
    [("MambaSpec", 4352, 2297856, 5)] * 7 + [("SlidingWindowSpec", 1088, 2297856, 7)]


def main(logp, envp, min_blocks):
    bad = []
    log = open(logp, errors="replace").read()
    m = re.search(r"glm53-placement kv-capacity (\{.*\})", log)
    if not m:
        bad.append("no glm53-placement kv-capacity line")
    else:
        d = json.loads(m.group(1))
        got = [(g["spec"], g["block_size"], g["page_bytes"], g["blocks_per_max_request"]) for g in d["groups"]]
        if got != GROUPS:
            bad.append("group geometry differs: %s" % got)
        if d["num_blocks"] < min_blocks:
            bad.append("num_blocks %d < %d" % (d["num_blocks"], min_blocks))
        print("num_blocks", d["num_blocks"], "groups", len(got))
    a = re.search(r"non-default args: (\{.*\})", log)
    args = a.group(1) if a else ""
    for k, v in (("'enable_prefix_caching': True", None), ("'mamba_cache_mode': 'align'", None),
                 ("'max_num_batched_tokens': 2048", None), ("'max_num_seqs': 16", None),
                 ("'method': 'dflash'", None), ("'num_speculative_tokens': 3", None)):
        if k not in args:
            bad.append("engine args lack %s" % k)
    env = dict(x.split("=", 1) for x in open(envp).read().splitlines() if "=" in x)
    if env.get("GLM53_DFLASH_BOUNDARY_LOOKUP") != "0":
        bad.append("GLM53_DFLASH_BOUNDARY_LOOKUP=%s" % env.get("GLM53_DFLASH_BOUNDARY_LOOKUP"))
    if "VLLM_PREFIX_CACHE_RETENTION_INTERVAL" in env:
        bad.append("VLLM_PREFIX_CACHE_RETENTION_INTERVAL set (%s)" % env["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"])
    for b in bad:
        print("GEOMETRY-FAIL", b)
    print("geometry check: %d failures" % len(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
