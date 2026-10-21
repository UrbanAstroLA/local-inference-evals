#!/usr/bin/env python3
"""Prefix construction and verification for d091 (PROPOSAL.md section 7) through B+'s build_prefixes.py, UNCHANGED
(byte-identical copy in scripts/, sha256 checked below). CPU-only container, --network none, no GPU (scripts/run_pipeline.sh).
NOTHING IS CONTINUED IN THIS PACKAGE.

build_prefixes.py reads its request logs from a module constant, REQLOGS = {seed: request-log file name under /gpqa}, fixed
to rep070-F's three seeds. That constant is the only thing set here (the d091 request log at seed 70704); every other
input reaches the unchanged code through the same mount points B+ used:
  /work  this package (scripts/trajectory.py, out/selection.json; writes PRIVATE/prefixes.json, PRIVATE/prompts.json
         (mode 0600, token ids and prompt messages), out/prefix_manifest.json and checks/prefix-verify-<engine>.json:
         hashes, lengths, checks, covariates only)
  /raw   PRIVATE/raw (the d091 raw bodies, read-only)       /disc  this package's out/ (rep070F_index.json = the d091 index)
  /repo  the K3.25 weights repo (tokenizer, templates)      /gpqa  <gpqa-runs> (request logs)      /d7  d7's prompt tokens
Proof of identity: tests/prefix-repro: with REQLOGS left at B+'s value and B+'s inputs mounted, this wrapper rebuilds B+'s
12 prefixes and its out/prefix_manifest.json byte for byte.

usage (inside the container): prefix_build_d091.py build|verify [--seed S --reqlog NAME | --bplus-defaults]
"""
import hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BP_SHA = "1ebbec2ef0cdea409692a72b9ce0e576018131bcbce4498e1e2f3d2abd3598c0"
SEED = 70704
REQLOG = "glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091.s1.requests.jsonl"


def main():
    if hashlib.sha256(open(os.path.join(HERE, "build_prefixes.py"), "rb").read()).hexdigest() != BP_SHA:
        sys.exit("scripts/build_prefixes.py is not B+'s frozen file")
    args = sys.argv[1:]
    mode = args[0]
    sys.path.insert(0, HERE)
    import build_prefixes as BP
    if "--bplus-defaults" not in args:
        seed = int(args[args.index("--seed") + 1]) if "--seed" in args else SEED
        log = args[args.index("--reqlog") + 1] if "--reqlog" in args else REQLOG
        BP.REQLOGS = {seed: log}
    print(f"build_prefixes.py (sha256 {BP_SHA[:12]}) mode {mode}; REQLOGS {BP.REQLOGS}", flush=True)
    sys.argv = ["build_prefixes.py", mode]
    return BP.main()


if __name__ == "__main__":
    sys.exit(main())
