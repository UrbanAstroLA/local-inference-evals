#!/usr/bin/env bash
set -u
# Dry runs of analysis_d091.py on an existing pass used as a stand-in (no new data, no GPU). d091b: the stand-in is option
# D's real 2026-10-21 pass (stock 0.9.1, seed 70704; <gpqa-runs>/...+d091__pass1, its proxy request log +d091.s1 and
# its own session log d091/run/server-s1.log) relabelled to request seed 70705 (request log rows and lm-eval model_args);
# N1 sends D's own seed 70704 (a used seed). D's package used the 0.9.1 record's pass 1 relabelled to 70704. Synthetic
# inputs are built under PRIVATE/scratch-dryruns (umask 077, mode 0700; samples files are symlinked, and the two negative
# cases that need an altered samples file write it there) and removed on every exit path; only ids, hashes and numbers
# are written to tests/dryruns/. Not set -e: every exit code is recorded. Usage: tests/run_analysis_dryruns.sh
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H")
case "$PKG" in <runs>/long-reply/d091b) ;; *) echo "ABORT: unexpected package path $PKG"; exit 1 ;; esac
S="$PKG/PRIVATE/scratch-dryruns"; OUT="$H/dryruns"
[ -n "$S" ] && case "$S" in <runs>/long-reply/d091b/PRIVATE/scratch-dryruns) ;; *) exit 1 ;; esac
[ -e "$S" ] && { echo "ABORT: $S exists"; exit 1; }
cleanup() {
  [ -n "$S" ] && [ -d "$S" ] && case "$S" in <runs>/long-reply/d091b/PRIVATE/scratch-dryruns)
    find "$S" -type f -exec shred -u {} + 2>/dev/null; rm -rf -- "$S" ;; esac
}
trap cleanup EXIT
( umask 077; mkdir -p "$S" ) || exit 1
mkdir -p "$OUT" || exit 1
V091=<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1
DP="$V091+d091__pass1"; DREQ="$V091+d091.s1.requests.jsonl"
DSRV=<runs>/long-reply/d091/run/server-s1.log
F070=<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+rep070-F
A="$PKG/analysis_d091.py"
mk() { # name reqseed argseed [mode]   modes: missingq | taskhash | target | resend | http | procseed | noargseed
  ( umask 077; python3 -B - "$S/$1" "$DP" "$2" "$3" "${4:-}" "$DREQ" <<'EOF'
import glob, json, os, sys
d, src, rs, as_, mode, reqf = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
pd = os.path.join(d, "pass1", "model"); os.makedirs(pd, exist_ok=True)
sf = glob.glob(src + "/**/samples_*.jsonl", recursive=True)[0]; rf = glob.glob(src + "/**/results_*.json", recursive=True)[0]
if mode in ("missingq", "target"):
    with open(os.path.join(pd, os.path.basename(sf)), "w") as f:
        for line in open(sf):
            x = json.loads(line)
            if mode == "missingq" and x["doc_id"] == 17:
                continue
            if mode == "target" and x["doc_id"] == 42:
                x["target_hash"] = "0" * 64
                line = json.dumps(x) + "\n"
            f.write(line)
else:
    os.symlink(sf, os.path.join(pd, os.path.basename(sf)))
r = json.load(open(rf))
if mode != "noargseed":
    r["config"]["model_args"]["seed"] = as_
if mode == "taskhash":
    r["task_hashes"]["gpqa_diamond_thinking"] = "f" * 64
if mode == "procseed":
    r["config"]["random_seed"] = 1236
json.dump(r, open(os.path.join(pd, os.path.basename(rf)), "w"))
rows = [json.loads(l) for l in open(reqf)]
with open(os.path.join(d, "requests.jsonl"), "w") as f:
    for i, x in enumerate(rows):
        x["seed"] = rs
        if mode == "http" and i == 5:
            x["http_status"] = 500
        f.write(json.dumps(x) + "\n")
        if mode == "resend" and i == 7:
            f.write(json.dumps(x) + "\n")
EOF
  )
}
srvlog() { # name mode: noready | nondefault | kvsmall | kvother
  ( umask 077; python3 -B - "$DSRV" "$S/$1.server.log" "$2" <<'EOF'
import sys
src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
with open(dst, "w") as f:
    for line in open(src, errors="replace"):
        if mode == "noready" and "warmup complete; container is ready" in line:
            continue
        if mode == "nondefault" and "non-default args:" in line:
            line = line.replace("'max_num_seqs': 16", "'max_num_seqs': 15")
        if "GPU KV cache size: 4,707,515 tokens" in line:
            if mode == "kvsmall":
                line = line.replace("4,707,515", "2,000,000")
            if mode == "kvother":
                line = line.replace("4,707,515", "4,698,112")
        f.write(line)
EOF
  )
}
summary() { echo "$1 rc=$2 $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));p=d.get('pass') or {};print('valid', d['valid'], 'empty', p.get('empty'), 'raw', p.get('raw_correct_flexible'), 'failed', [c[0][:110] for c in d['checks'] if not c[1] and not c[0].startswith('stated adjudicator')][:4])" "$OUT/$1.json")"; }
run() { # tag scratch serverlog label [extra]
  local tag=$1 sd=$2 sl=$3 lab=$4; shift 4
  python3 -B "$A" --pass-dir "$S/$sd/pass1" --reqlog "$S/$sd/requests.jsonl" --serverlog "$sl" --dry-run-label "$tag $lab" \
    --out "$OUT/$tag.json" "$@" > "$OUT/$tag.md" 2> "$OUT/$tag.err"
  summary "$tag" $?
}
mk relabel 70705 70705
mk wrongseed 70704 70704
mk noargseed 70705 0 noargseed
mk missingq 70705 70705 missingq
mk taskhash 70705 70705 taskhash
mk target 70705 70705 target
mk resend 70705 70705 resend
mk http 70705 70705 http
mk procseed 70705 70705 procseed
srvlog noready noready; srvlog nondefault nondefault; srvlog kvsmall kvsmall; srvlog kvother kvother
{
echo "# analysis dry runs $(date -Is) (CPU only; zero GPU); stand-in = option D's 2026-10-21 pass (seed 70704) relabelled to seed 70705"
run D1-standin relabel "$DSRV" "option D's 2026-10-21 pass relabelled to 70705 with its own session log; expect VALID" --rows-out "$OUT/D1-standin.rows.jsonl"
run D2-kv-other-size relabel "$S/kvother.server.log" "KV pool 4,698,112 (not a record size, above the floor); expect VALID (pool reported, not identity-gated)"
run N1-wrongseed wrongseed "$DSRV" "request seed 70704 (option D's, used) instead of 70705; expect INVALID"
run N2-missingq missingq "$DSRV" "question 17 missing from the samples; expect INVALID"
run N3-taskhash taskhash "$DSRV" "different lm-eval task hash; expect INVALID"
run N4-noready relabel "$S/noready.server.log" "server log without its ready line; expect INVALID"
run N5-other-engine relabel "$F070.s1.requests.jsonl" "a request log given as the server log (no engine start); expect INVALID"
run N6-resend resend "$DSRV" "one request re-sent; expect INVALID"
run N7-target target "$DSRV" "question 42's target hash altered; expect INVALID"
run N8-kv-below-floor relabel "$S/kvsmall.server.log" "KV pool 2,000,000 below the 8 x cap floor; expect INVALID"
run N9-procseed procseed "$DSRV" "lm-eval process seed 1236 instead of 1235; expect INVALID"
run N10-http500 http "$DSRV" "one request HTTP 500; expect INVALID"
run N11-nondefault relabel "$S/nondefault.server.log" "non-default args differ (max_num_seqs); expect INVALID"
run N12-noargseed noargseed "$DSRV" "seed not in lm-eval model_args; expect INVALID"
run N13-rep070F-serverlog relabel "<runs>/rep070/run-F/server-s1.log" "the 0.7.0 replication's server log; expect INVALID"
python3 -B "$A" --pass-dir "$S/relabel/pass1" --reqlog "$S/relabel/absent.requests.jsonl" --serverlog "$DSRV" \
  --dry-run-label "N14-no-reqlog proxy request log absent; expect INVALID" --out "$OUT/N14-no-reqlog.json" > "$OUT/N14-no-reqlog.md" 2> "$OUT/N14-no-reqlog.err"
summary N14-no-reqlog $?
python3 -B "$A" --pass-dir "$S/relabel/pass1" --reqlog "$S/relabel/requests.jsonl" --serverlog "$S/absent.server.log" \
  --dry-run-label "N15-no-serverlog server log absent; expect INVALID" --out "$OUT/N15-no-serverlog.json" > "$OUT/N15-no-serverlog.md" 2> "$OUT/N15-no-serverlog.err"
summary N15-no-serverlog $?
} | tee "$OUT/run_dryruns.out"
python3 - "$OUT/run_dryruns.out" <<'EOF'
import sys
L = [l for l in open(sys.argv[1]) if " rc=" in l]
ok = 0
for l in L:
    tag = l.split()[0]
    exp_valid = tag.startswith("D")
    good = (" rc=0 valid True" in l) if exp_valid else (" rc=3 valid False" in l)
    ok += good
    print(("PASS " if good else "FAIL ") + tag)
print(f"analysis dry runs: {ok} of {len(L)} as expected")
sys.exit(0 if ok == len(L) == 17 else 1)
EOF
