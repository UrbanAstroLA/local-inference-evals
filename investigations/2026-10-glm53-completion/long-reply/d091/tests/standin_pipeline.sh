#!/usr/bin/env bash
set -u
# Dry run of the whole long-reply pipeline (scripts/run_pipeline.sh, unchanged) on a stand-in pass with real raw bodies:
# rep070-F pass 1 (stock 0.7.0, seed 70701; rep070/run-F/raw/s1.raw.jsonl.gz, read-only). No 0.9.1 pass with raw bodies
# exists yet; this exercises every step (index, trajectory, periods, side-by-side tables, prefix selection, prefix build in
# the 0.7.0 image, verify in the 0.9.1 and B1 images, privacy scan) on the code path the d091 run will take. The P(finish)
# table is fed the analysis dry run D1 (the 0.9.1 record's pass 1 relabelled), so that section mixes stand-ins by design.
# Extra checks: the stand-in's per-reply, per-window and post hoc rows equal discovery's rows for the same reply ids (same
# code, same text); prefix hashes of any reply B+ also selected equal B+'s manifest. Work dir under PRIVATE/ (shredded on
# exit); ids, metrics, hashes and tables are copied to tests/standin/. CPU only; zero GPU.
H=$(cd "$(dirname "$0")" && pwd); PKG=$(dirname "$H")
case "$PKG" in <runs>/long-reply/d091) ;; *) echo "ABORT: unexpected package path"; exit 1 ;; esac
B=<runs>
W="$PKG/PRIVATE/standin-work"; KEEP="$H/standin"
[ -n "$W" ] && case "$W" in <runs>/long-reply/d091/PRIVATE/standin-work) ;; *) exit 1 ;; esac
[ -n "$KEEP" ] && case "$KEEP" in <runs>/long-reply/d091/tests/standin) ;; *) exit 1 ;; esac
[ -e "$W" ] && { echo "ABORT: $W exists"; exit 1; }
[ -e "$KEEP" ] && { echo "ABORT: $KEEP exists"; exit 1; }
cleanup() { [ -n "$W" ] && [ -d "$W" ] && case "$W" in "$PKG"/PRIVATE/standin-work) find "$W" -type f -exec shred -u {} + 2>/dev/null; rm -rf -- "$W" ;; esac; }
trap cleanup EXIT
( umask 077; mkdir -p "$W/scripts" "$W/out" "$W/PRIVATE" "$W/checks" ) || exit 1
chmod 700 "$W/PRIVATE"
for f in trajectory.py periods.py build_prefixes.py prefix_build_d091.py; do cp -p "$PKG/scripts/$f" "$W/scripts/$f" || exit 1; done
cp -p "$PKG/PRESPEC.md" "$W/PRESPEC.md" || exit 1
mkdir -p "$KEEP" || exit 1
PIPE_LABEL="STAND-IN: rep070-F pass 1 (stock 0.7.0, seed 70701) run through the d091 pipeline" \
  bash "$PKG/scripts/run_pipeline.sh" "$W" "$B/rep070/run-F/raw" "glm53-flash-exl3-k3.25@tpurtell+rep070-F.s1.requests.jsonl" 70701 \
  "$H/dryruns/D1-standin.json" > "$KEEP/run_pipeline.out" 2>&1
echo "run_pipeline rc=$?" >> "$KEEP/run_pipeline.out"
python3 -B - "$W/out" "$B/long-reply/discovery/out" "$B/long-reply/bplus/out/prefix_manifest.json" > "$KEEP/standin-checks.txt" 2>&1 <<'EOF'
import json, sys
w, d, bpm = sys.argv[1:4]
ok_all = True
def chk(name, ok, det=""):
    global ok_all
    ok_all &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name + (f" ({det})" if det else ""))
A = {r["id"]: r for r in json.load(open(f"{w}/replies_all.json"))}
D = {r["id"]: r for r in json.load(open(f"{d}/replies_all.json"))}
chk("stand-in replies >= 50k are exactly discovery's pass-1 replies >= 50k", set(A) == {i for i in D if i.endswith("p1")}, f"{len(A)} replies")
chk("per-reply trajectory rows identical to discovery's for the same ids", all(A[i] == D[i] for i in A))
def wins(p):
    o = {}
    for l in open(p):
        x = json.loads(l); o.setdefault(x["id"], []).append(x)
    return o
WA, WD = wins(f"{w}/windows_all.jsonl"), wins(f"{d}/windows_all.jsonl")
chk("per-window rows identical to discovery's for the same ids", all(WA[i] == WD[i] for i in WA), f"{sum(len(v) for v in WA.values())} windows")
PA = {r["id"]: r for r in json.load(open(f"{w}/periods_all.json"))}
PD = {r["id"]: r for r in json.load(open(f"{d}/periods_all.json"))}
chk("post hoc periods rows identical to discovery's for the same ids", all(PA[i] == PD[i] for i in PA))
sel = json.load(open(f"{w}/selection.json"))
chk("prefix selection ran (B+ rule); selected ids listed", sel["summary"]["selected"] >= 1, json.dumps(sel["summary"]))
print("selected:", [(s["prefix_id"], s["id"], s["stratum"]) for s in sel["selected"]])
man = {m["id"]: m for m in json.load(open(f"{w}/prefix_manifest.json"))["prefixes"]}
bman = {m["id"]: m for m in json.load(open(bpm))["prefixes"]}
common = sorted(set(man) & set(bman))
chk("prefix hashes of replies B+ also selected equal B+'s manifest", common and all(man[i]["prefix_ids_sha256"] == bman[i]["prefix_ids_sha256"] and man[i]["prompt_ids_sha256"] == bman[i]["prompt_ids_sha256"] for i in common), f"common ids {common}")
mchk = json.load(open(f"{w}/prefix_manifest.json"))["checks"]
chk("build checks all pass", all(c["ok"] for c in mchk), f"{sum(c['ok'] for c in mchk)}/{len(mchk)}")
for e in ("v091", "b1"):
    v = json.load(open(f"{w}/../checks/prefix-verify-{e}.json"))
    chk(f"verify in the {e} image: all checks pass (0.9.1 template render identical)", all(c["ok"] for c in v["checks"]), f"{sum(c['ok'] for c in v['checks'])}/{len(v['checks'])}")
print("stand-in pipeline checks:", "ALL PASS" if ok_all else "FAILURES")
sys.exit(0 if ok_all else 1)
EOF
echo "standin checks rc=$?" >> "$KEEP/standin-checks.txt"
for f in tables_d091.md tables_d091.json selection.json prefix_manifest.json pipeline.log privacy-scan.txt index_checks_d091.json index.log run_all.log run_periods.log select.log build.log verify-v091.log verify-b1.log compare.log; do
  [ -f "$W/out/$f" ] && cp -p "$W/out/$f" "$KEEP/$f"
done
for f in "$W/checks/"prefix-verify-*.json; do [ -f "$f" ] && cp -p "$f" "$KEEP/"; done
chmod 644 "$KEEP"/* 2>/dev/null
cat "$KEEP/run_pipeline.out" "$KEEP/standin-checks.txt"
grep -q "ALL PASS" "$KEEP/standin-checks.txt" && grep -q "run_pipeline rc=0" "$KEEP/run_pipeline.out" && ! grep -q "rc=[1-9]" "$KEEP/pipeline.log"
