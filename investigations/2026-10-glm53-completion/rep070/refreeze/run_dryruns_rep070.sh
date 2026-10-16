#!/usr/bin/env bash
# Dry runs of the re-freeze candidate analysis on existing records only (no new data, no GPU). Not set -e: every exit code
# is recorded. Synthetic inputs (seed relabels, a samples file missing one question, a server log missing its ready line)
# are built in a scratch directory outside the package and removed on every exit path; only ids, hashes and numbers are
# written into dryruns/. Usage: run_dryruns_rep070.sh [SCRATCH_DIR]
H=$(cd "$(dirname "$0")" && pwd); G=<gpqa-runs>; R=<this-repo>/runs; OUT="$H/dryruns"
V070="$G/glm53-flash-exl3-k3.25@tpurtell"; V091="$G/glm53-flash-exl3-k3.25@tpurtell+v0.9.1"
S=${1:-$(mktemp -d)}; mkdir -p "$S" "$OUT" || exit 1
trap 'rm -rf "$S"' EXIT
A="$H/analysis_rep070.py"
summary() { echo "$1 rc=$2 $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));r=d.get('reading',{});print('valid', d['valid'], 'outcome', r.get('outcome'), 'empties', r.get('rep_empties'), 'p1', r.get('p1_one_sided_rep_fewer_than_v091'), 'p2', r.get('p2_one_sided_rep_more_than_old070'), 'failed', [c[0] for c in d['checks'] if not c[1]][:4])" "$OUT/$1.json")"; }

# ---- D1: old 0.7.0 record as rep, variant R (expect valid, O1; C2 zero by construction)
python3 "$A" --variant R --rep-label "old 0.7.0 record standing in as rep" \
  --rep-pass "$V070+ps__pass1" "$V070+ps__pass2" "$V070+ps__pass3" --rep-reqlog "$V070+ps.requests.jsonl" --rep-hedged 2:25 2:94 \
  --rep-serverlog "$V070+ps.server.log" --rep-published "$R/2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond/results.jsonl" \
  --require-rep-reqlog 0 --require-rep-serverlog 0 --require-seed-args 0 \
  --dry-run-label "D1 old 0.7.0 record as rep, variant R (expect valid, O1)" --out "$OUT/D1-rep-old070-R.json" > "$OUT/D1-rep-old070-R.md"
summary D1-rep-old070-R $?
# ---- D2: 0.9.1 record as rep, variant R (expect valid, O2)
python3 "$A" --variant R --rep-label "0.9.1 record standing in as rep" \
  --rep-pass "$V091+ps__pass1" "$V091+ps__pass2" "$V091+ps__pass3" --rep-reqlog "$V091.requests.jsonl" "$V091+ps.requests.jsonl" \
  --rep-serverlog "$V091.server.log" "$V091+ps.server.log" --rep-published "$R/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl" \
  --require-config-identity 0 --require-seed-args 0 \
  --dry-run-label "D2 0.9.1 record as rep, variant R (expect valid, O2)" --out "$OUT/D2-rep-v091-R.json" > "$OUT/D2-rep-v091-R.md"
summary D2-rep-v091-R $?

# ---- synthetic variant-F inputs: the 0.9.1 record relabelled to seeds 70701/70702/70703 (request logs and lm-eval model_args;
# samples files linked, not copied)
mk() { # name seed1 seed2 seed3 [drop_doc_pass3] [mode: taskhash | target:DOC]
  python3 - "$S/$1" "$V091" "$2" "$3" "$4" "${5:-}" "${6:-}" <<'EOF'
import glob, json, os, sys
d, base, s1, s2, s3, drop, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
os.makedirs(d, exist_ok=True)
new = {1: s1, 2: s2, 3: s3}
for p in (1, 2, 3):
    src = f"{base}+ps__pass{p}"
    pd = os.path.join(d, f"pass{p}", "model"); os.makedirs(pd, exist_ok=True)
    sf = glob.glob(src + "/**/samples_*.jsonl", recursive=True)[0]; rf = glob.glob(src + "/**/results_*.json", recursive=True)[0]
    if drop and p == 3:   # negative case: one question missing from pass 3 (scratch copy, removed on exit)
        with open(os.path.join(pd, os.path.basename(sf)), "w") as f:
            for line in open(sf):
                if json.loads(line)["doc_id"] != int(drop):
                    f.write(line)
    elif mode.startswith("target:") and p == 1:   # negative case: one question's target hash altered (scratch copy)
        doc = int(mode.split(":")[1])
        with open(os.path.join(pd, os.path.basename(sf)), "w") as f:
            for line in open(sf):
                x = json.loads(line)
                if x["doc_id"] == doc:
                    x["target_hash"] = "0" * 64
                f.write(json.dumps(x) + "\n")
    else:
        os.symlink(sf, os.path.join(pd, os.path.basename(sf)))
    r = json.load(open(rf))
    r["config"]["model_args"]["seed"] = new[p]
    if mode == "taskhash" and p == 2:              # negative case: a different lm-eval task hash on pass 2
        r["task_hashes"]["gpqa_diamond_thinking"] = "f" * 64
    json.dump(r, open(os.path.join(pd, os.path.basename(rf)), "w"))
old = {1234: s1, 1235: s2, 1236: s3}
for src, dst in ((f"{base}.requests.jsonl", "s1.requests.jsonl"), (f"{base}+ps.requests.jsonl", "s2.requests.jsonl")):
    with open(os.path.join(d, dst), "w") as f:
        for line in open(src):
            x = json.loads(line); x["seed"] = old[x["seed"]]; f.write(json.dumps(x) + "\n")
EOF
}
runF() { # tag variant scratchdir label [extra args...]
  local tag=$1 var=$2 sd=$3 lab=$4; shift 4
  python3 "$A" --variant $var --rep-label "0.9.1 record relabelled ($lab)" \
    --rep-pass "$S/$sd/pass1" "$S/$sd/pass2" "$S/$sd/pass3" --rep-reqlog "$S/$sd/s1.requests.jsonl" "$S/$sd/s2.requests.jsonl" \
    --dry-run-label "$tag $lab" --out "$OUT/$tag.json" "$@" > "$OUT/$tag.md"
  summary $tag $?
}
mk relabelF 70701 70702 70703
mk wrongseed 70701 1235 70703
mk missingq 70701 70702 70703 17
mk taskhash 70701 70702 70703 "" taskhash
mk target 70701 70702 70703 "" target:42
# server log with the ready line removed (negative warm-up case; scratch copy)
grep -v 'warmup complete; container is ready' "$V091+ps.server.log" > "$S/s2-noready.server.log"
# ---- D3: synthetic fresh-seed relabel, variant F (expect valid, O2 as D2)
runF D3-relabel-v091-F F relabelF "fresh seeds 70701/70702/70703, variant F; expect valid, O2" \
  --rep-serverlog "$V091.server.log" "$V091+ps.server.log" --require-config-identity 0
# ---- negative cases (each must be invalid, exit code 3)
runF N1-wrongseed-F F wrongseed "pass 2 carried seed 1235 instead of 70702; expect INVALID" \
  --rep-serverlog "$V091.server.log" "$V091+ps.server.log" --require-config-identity 0
runF N2-missingq-F F missingq "question 17 missing from pass 3; expect INVALID" \
  --rep-serverlog "$V091.server.log" "$V091+ps.server.log" --require-config-identity 0
runF N3-variantR-on-F-seeds R relabelF "fresh-seed run analysed as variant R; expect INVALID" \
  --rep-serverlog "$V091.server.log" "$V091+ps.server.log" --require-config-identity 0
runF N4-noready-F F relabelF "session-2 server log without its ready line; expect INVALID" \
  --rep-serverlog "$V091.server.log" "$S/s2-noready.server.log" --require-config-identity 0
runF N5-config-F F relabelF "0.9.1 server logs with configuration identity gating; expect INVALID" \
  --rep-serverlog "$V091.server.log" "$V091+ps.server.log"
runF N6-one-serverlog-F F relabelF "only one rep server log; expect INVALID" \
  --rep-serverlog "$V091+ps.server.log" --require-config-identity 0
runF N7-taskhash-F F taskhash "pass 2 with a different lm-eval task hash; expect INVALID" \
  --rep-serverlog "$V091.server.log" "$V091+ps.server.log" --require-config-identity 0
runF N8-target-F F target "question 42's target hash altered in pass 1; expect INVALID" \
  --rep-serverlog "$V091.server.log" "$V091+ps.server.log" --require-config-identity 0
# ---- self-test: variant F's per-pass loader reproduces the single-call loader on R seeds (same rows, same checks)
python3 - "$H" <<'EOF'
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1]); sys.argv = ["x"]
import analysis_rep070 as X
M = X.M
V = "<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1"
dirs = [f"{V}+ps__pass{p}" for p in (1, 2, 3)]; logs = [f"{V}.requests.jsonl", f"{V}+ps.requests.jsonl"]
a = M.load_arm("L", dirs, logs, [], None)
rows, checks = {}, []
old = M.SEED_BASE
for p, s in enumerate((1234, 1235, 1236), 1):
    M.SEED_BASE = s - p
    r, c, i = M.load_arm("L", dirs, logs, [], None)
    rows.update({k: v for k, v in r.items() if k[1] == p})
    checks += [x for x in c if x[0].startswith(f"L pass {p}:") or x[0].startswith(f"L pass {p} q")]
M.SEED_BASE = old
ok = rows == a[0] and sorted(map(str, checks)) == sorted(map(str, a[1])) and M.SEED_BASE == 1233
print("SELFTEST per-pass loader equals single-call loader on seeds 1234-1236:", ok, f"({len(rows)} rows, {len(checks)} checks)")
EOF
echo "dry runs done"
