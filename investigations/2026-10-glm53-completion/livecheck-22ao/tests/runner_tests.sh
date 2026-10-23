#!/bin/bash
# CPU tests of runner/run_22ao.sh on a scratch copy of the package (ROOT_OVERRIDE). No GPU, no docker server:
# refusals and DRY_RUN read only image metadata; MOCK=1 sessions run tests/mock_server_22ao.py on 127.0.0.1:18001.
# Results: tests/out/runner-tests.txt (one PASS/FAIL line per case). No `set -e`.
set -u
PKG=$(cd "$(dirname "$0")/.." && pwd)
SCR=<scratch>/22ao-runner-tests
OUT=$PKG/tests/out/runner-tests.txt
[ -n "$PKG" ] && [ -n "$SCR" ] || exit 2
rm -rf "${SCR:?}"; mkdir -p "$SCR"; : > "$OUT"
copy() {  # fresh scratch copy of everything the runner touches
  rm -rf "${SCR:?}/pkg"; mkdir -p "$SCR/pkg"
  cp -r "$PKG/runner" "$PKG/analysis" "$PKG/tests" "$SCR/pkg/"
  umask 077; mkdir -p "$SCR/pkg/PRIVATE"; cp "$PKG/PRIVATE/panel_fixture.LOCAL.json" "$SCR/pkg/PRIVATE/"; umask 022
  rm -rf "$SCR/pkg/tests/out"; mkdir -p "$SCR/pkg/tests/out"
  ( cd "$SCR/pkg" && find runner analysis -type f | sort | xargs sha256sum > SHA256SUMS )
}
R() { ROOT_OVERRIDE="$SCR/pkg" bash "$SCR/pkg/runner/run_22ao.sh" "$@"; }
check() {  # name, condition-result (0 = pass), detail
  if [ "$2" = 0 ]; then echo "PASS $1  $3" | tee -a "$OUT"; else echo "FAIL $1  $3" | tee -a "$OUT"; fi
}
# --- refusals
copy
o=$(R 2>&1); echo "$o" | grep -q "REFUSED: set APPROVED_GPU_H"; check R1_no_budget $? "$(echo "$o" | tail -n1)"
o=$(APPROVED_GPU_H=2.5 R 2>&1); echo "$o" | grep -q "REFUSED: APPROVED_GPU_H=2.5 outside"; check R2_over_ceiling $? "$(echo "$o" | tail -n1)"
o=$(APPROVED_GPU_H=0 R 2>&1); echo "$o" | grep -q "REFUSED: APPROVED_GPU_H=0 outside"; check R3_zero $? "$(echo "$o" | tail -n1)"
echo '{"item":"livecheck-22ao","gpu_hours":0.1}' > "$SCR/pkg/costs.jsonl"
o=$(APPROVED_GPU_H=2.0 R 2>&1); echo "$o" | grep -q "REFUSED: 1 cost lines already recorded"; check R4_second_attempt $? "$(echo "$o" | tail -n1)"
rm -f "$SCR/pkg/costs.jsonl"; mkdir -p "$SCR/pkg/run-1"
o=$(APPROVED_GPU_H=2.0 R 2>&1); echo "$o" | grep -q "ABORT: .*run-1 exists"; check R5_run_dir_exists $? "$(echo "$o" | tail -n1)"
rm -rf "${SCR:?}/pkg/run-1"; echo "# tamper" >> "$SCR/pkg/runner/client_22ao.py"
o=$(APPROVED_GPU_H=2.0 DRY_RUN=1 R 2>&1); echo "$o" | grep -q "ABORT: frozen hashes differ"; check R6_hash_mismatch $? "$(echo "$o" | head -n1)"
copy; sed -i 's/sha256:dd51/sha256:0000/' "$SCR/pkg/runner/images.txt"; ( cd "$SCR/pkg" && find runner analysis -type f | sort | xargs sha256sum > SHA256SUMS )
o=$(APPROVED_GPU_H=2.0 DRY_RUN=1 R 2>&1); echo "$o" | grep -q "ABORT: local/glm53-v091-22ao-a:local is"; check R7_image_pin $? "$(echo "$o" | tail -n1)"
copy; mv "$SCR/pkg/PRIVATE/panel_fixture.LOCAL.json" "$SCR/pkg/PRIVATE/x.json"
o=$(APPROVED_GPU_H=2.0 DRY_RUN=1 R 2>&1); echo "$o" | grep -q "ABORT: panel fixture missing"; check R8_fixture_missing $? "$(echo "$o" | tail -n1)"
# --- real DRY_RUN on the scratch copy (gates pass, nothing started)
copy
o=$(APPROVED_GPU_H=2.0 DRY_RUN=1 R 2>&1); echo "$o" | grep -q "^DRY_RUN: budget gate"; check D1_dry_run $? "$(echo "$o" | tail -n1)"
[ ! -e "$SCR/pkg/run-1" ]; check D1b_dry_run_wrote_nothing $? ""
# --- MOCK session: both arms complete; 8 outstanding; 24 seeds; two cost lines; analysis runs
copy
o=$(APPROVED_GPU_H=2.0 MOCK=1 MOCK_TOK_S=8000 R 2>&1); M=$SCR/pkg/tests/out/mock-run
for a in a c; do
  python3 - "$M/arm-$a/mock-stats.json" "$M/arm-$a/client.jsonl" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); c = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
req = [x for x in c if x.get("kind") == "request"]
ok = s["peak"] == 8 and s["served"] == 24 and sorted(s["seeds"]) == list(range(72201, 72225)) and len(req) == 24 \
     and not any(x.get("error") for x in req) and max(x["inflight_at_send"] for x in req) == 8
print("peak", s["peak"], "served", s["served"], "requests", len(req)); sys.exit(0 if ok else 1)
EOF
  check "M1_mock_arm_$a" $? "$(python3 -c "import json;s=json.load(open('$M/arm-$a/mock-stats.json'));print('peak',s['peak'],'served',s['served'])")"
done
n=$(grep -c '"item":"livecheck-22ao"' "$SCR/pkg/tests/out/mock-costs.jsonl"); [ "$n" = 2 ]; check M2_two_cost_lines $? "lines $n"
[ -s "$M/result.json" ] && grep -q "no guard records" "$M/result.json"; check M3_analysis_runs_incomplete_without_guard $? ""
# --- MOCK watchdog: a slow server and a short cap; the watchdog stops it, the cost line is still written
copy
o=$(APPROVED_GPU_H=2.0 MOCK=1 MOCK_TOK_S=12 WALL_CAP=400 MIN_WALL=100 RESERVE=40 R 2>&1)   # watchdog at 325 s fires before the client deadline (360 s + 15 s); long requests take ~512 s
echo "$o" | grep -q "WALL CAP reached arm a"; check W1_watchdog_fired $? ""
n=$(grep -c '"item":"livecheck-22ao"' "$SCR/pkg/tests/out/mock-costs.jsonl"); [ "$n" = 2 ]; check W2_cost_lines_after_watchdog $? "lines $n"
ps aux | grep -q '[m]ock_server_22ao'; [ $? != 0 ]; check W3_no_mock_server_left $? ""
rm -rf "${SCR:?}"
echo "done $(date -u +%FT%TZ)" | tee -a "$OUT"
grep -c '^FAIL' "$OUT" | xargs -I{} echo "failures: {}"
