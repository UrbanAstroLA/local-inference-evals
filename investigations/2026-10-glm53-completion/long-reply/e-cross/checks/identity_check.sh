#!/usr/bin/env bash
set -u
# E byte-identity proof (CHECKS.md): every reused file is compared by sha256 with its source in B+ or d091; every
# modified copy is listed with its diff file (regenerated here) so the only differences can be read line by line; the
# request-defining fields of out/plan.json are compared with B+'s. Read only except checks/*.diff. Hashes and names only.
# [22ah: D's sources now compared through ref/ and PRIVATE/src-d091/; D2's sources added.]
# usage: checks/identity_check.sh > checks/identity.txt
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H")
EC=<runs>/long-reply/e-cross
case "$PKG" in "$EC") ;; *) echo "ABORT: unexpected package path $PKG"; exit 2 ;; esac
L=<runs>/long-reply
BP=$L/bplus; D9=$L/d091; D2=$L/d091b; R7=<runs>/rep070/refreeze
cd "$PKG" || exit 1
PASS=0; FAIL=0
same() { # e-cross path, source path
  local a b; a=$(sha256sum "$1" 2>/dev/null | cut -c1-64); b=$(sha256sum "$2" 2>/dev/null | cut -c1-64)
  if [ -n "$a" ] && [ "$a" = "$b" ]; then PASS=$((PASS+1)); echo "IDENTICAL $1 = ${2#<runs>/} (${a:0:16})"
  else FAIL=$((FAIL+1)); echo "DIFFERENT $1 (${a:0:16}) vs $2 (${b:0:16})"; fi
}
echo "# byte-identical copies"
for f in trajectory.py analysis_col511.py bplus_stats.py analysis_bplus_22v.py build_prefixes.py client_bplus.py d7_prefill_exposure.py; do same "scripts/$f" "$BP/scripts/$f"; done
same tests/fake_server.py "$BP/tests/fake_server.py"
same checks/snapshot-0490d2f7.txt "$BP/checks/snapshot-0490d2f7.txt"
same ref/bplus-result-22v.json "$BP/run/result-22v.json"
same ref/bplus-prefix_manifest.json "$BP/out/prefix_manifest.json"
same ref/bplus-plan.json "$BP/out/plan.json"
same ref/bplus-power.json "$BP/out/power.json"
same scripts/prefix_build_d091.py "$D9/scripts/prefix_build_d091.py"
same ref/d091-selection.json "$D9/out/selection.json"
same ref/d091-prefix_manifest.json "$D9/out/prefix_manifest.json"
same PRIVATE/src-d091/prefixes.json "$D9/PRIVATE/prefixes.json"
same PRIVATE/src-d091/prompts.json "$D9/PRIVATE/prompts.json"
same checks/prefix-verify-v091.json "$D9/checks/prefix-verify-v091.json"
# amendment 22ah: D2's sources (the composed 13-prefix files are checked entry by entry in checks/compose-prefixes.out)
same ref/d091b-selection.json "$D2/out/selection.json"
same ref/d091b-prefix_manifest.json "$D2/out/prefix_manifest.json"
same PRIVATE/src-d091b/prefixes.json "$D2/PRIVATE/prefixes.json"
same PRIVATE/src-d091b/prompts.json "$D2/PRIVATE/prompts.json"
same checks/prefix-verify-d2-v091.json "$D2/checks/prefix-verify-v091.json"
echo "# modified copies (diff files regenerated; every hunk is listed in PROPOSAL.md section 9)"
dif() { # e-cross path, source path, diff file
  diff "$2" "$1" > "$3"; local n; n=$(grep -c '^[<>]' "$3")
  echo "MODIFIED $1 <- ${2#<runs>/}: $n changed lines ($3, sha256 $(sha256sum "$3" | cut -c1-16))"
}
dif scripts/plan.py "$BP/scripts/plan.py" checks/plan.diff
dif scripts/run_cpu.sh "$BP/scripts/run_cpu.sh" checks/run_cpu.diff
dif runner/run_e.sh "$BP/runner/run_bplus.sh" checks/runner.diff
dif scripts/source_check.sh "$BP/scripts/source_check.sh" checks/source_check.diff
dif tests/make_synthetic_run_e.py "$BP/tests/make_synthetic_run.py" checks/make_synthetic_run.diff
dif tests/run_analysis_dryruns.sh "$BP/tests/run_analysis_dryruns.sh" checks/run_analysis_dryruns.diff
dif tests/test_runner_refusals.sh "$BP/tests/test_runner_refusals.sh" checks/test_runner_refusals.diff
dif tests/session_sim.sh "$BP/tests/session_sim.sh" checks/session_sim.diff
dif make_sums.sh "$BP/make_sums.sh" checks/make_sums.diff
dif checks/seed_scan_e.py "$R7/checks/seed_scan.py" checks/seed_scan_e.diff
dif checks/raw_seed_scan_e.py "$D9/checks/raw_seed_scan.py" checks/raw_seed_scan_e.diff
dif checks/privacy_scan_e.py "$D9/checks/privacy_scan.py" checks/privacy_scan.diff
echo "# request definition: out/plan.json vs B+'s out/plan.json"
python3 - out/plan.json "$BP/out/plan.json" <<'EOF'
import json, sys
e, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ok = True
for k in ("model", "params", "params_sha256", "order_seed"):
    same = e[k] == b[k]
    ok &= same
    print(f"{'IDENTICAL' if same else 'DIFFERENT'} plan.{k}: {str(e[k])[:80]}")
print(f"SWAPPED plan.seed_base: {b['seed_base']} -> {e['seed_base']}; engines {[x['engine'] for x in b['engines']]} -> {[x['engine'] for x in e['engines']]}; "
      f"requests per engine {len(b['order'])} -> {len(e['order'])}; prefixes {len({r['prefix_id'] for r in b['order']})} -> {len({r['prefix_id'] for r in e['order']})}")
sys.exit(0 if ok else 1)
EOF
[ $? = 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
echo "# B+'s source reading of the two images, rerun: checks/source-check.txt blocks vs bplus/checks/source-check.txt"
python3 - "$BP/checks/source-check.txt" checks/source-check.txt <<'EOF'
import sys
def blocks(p):
    out, cur = {}, None
    for l in open(p):
        if l.startswith("################ "):
            cur = l.split()[1]
            out[cur] = []
        elif cur:
            out[cur].append(l)
    return out
a, b = blocks(sys.argv[1]), blocks(sys.argv[2])
bad = [k for k in b if a.get(k) != b[k]]
for k in b:
    print(f"{'IDENTICAL' if k not in bad else 'DIFFERENT'} source-check block {k}")
sys.exit(1 if bad or not b else 0)
EOF
[ $? = 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
echo "identity: $PASS pass, $FAIL fail"
[ "$FAIL" = 0 ]
