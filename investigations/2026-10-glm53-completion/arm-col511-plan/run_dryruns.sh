#!/usr/bin/env bash
# Dry runs of the frozen analysis on existing records (no arm data). Not set -e: each run's exit code is reported.
H=$(cd "$(dirname "$0")" && pwd); G=<gpqa-runs>; B=<benchmark-runs>; R=<this-repo>/runs
V091="$G/glm53-flash-exl3-k3.25@tpurtell+v0.9.1"; V070="$G/glm53-flash-exl3-k3.25@tpurtell"
CMP=(--cmp-label "0.9.1 K3.25 DFlash2 x3 (historical 3-pass record)"
     --cmp-pass "$V091+ps__pass1" "$V091+ps__pass2" "$V091+ps__pass3"
     --cmp-reqlog "$V091.requests.jsonl" "$V091+ps.requests.jsonl"
     --cmp-serverlog "$V091.server.log" "$V091+ps.server.log"
     --cmp-chainlog "$B/2026-10-08-glm53-flash-exl3-k3.25@tpurtell+v0.9.1/chain.log" "$B/2026-10-09-glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps/chain.log"
     --cmp-published "$R/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl")
python3 "$H/analysis_col511.py" \
  --arm-label "0.7.0 K3.25 DFlash2 x5 (existing record, stands in for the arm)" \
  --arm-pass "$V070+ps__pass1" "$V070+ps__pass2" "$V070+ps__pass3" \
  --arm-reqlog "$V070+ps.requests.jsonl" --arm-serverlog "$V070+ps.server.log" \
  --arm-chainlog "$B/2026-09-29-glm53-flash-exl3-k3.25@tpurtell/chain.log" "$B/2026-10-10-glm53-flash-exl3-k3.25@tpurtell+ps/chain.log" \
  --arm-hedged 2:25 2:94 \
  --arm-published "$R/2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond/results.jsonl" \
  "${CMP[@]}" --dry-run-label "0.7.0 record as the arm, 0.9.1 record as comparator (proves the pipeline; different engines, draft depth, layout)" \
  --out "$H/dryrun-070-vs-091.json" > "$H/dryrun-070-vs-091.md"
echo "dry run 1 rc=$?"
python3 "$H/analysis_col511.py" \
  --arm-label "0.9.1 K3.25 DFlash2 x3 (same record as comparator)" \
  --arm-pass "$V091+ps__pass1" "$V091+ps__pass2" "$V091+ps__pass3" \
  --arm-reqlog "$V091.requests.jsonl" "$V091+ps.requests.jsonl" \
  --arm-published "$R/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl" \
  "${CMP[@]}" --dry-run-label "0.9.1 record against itself (sanity: zero effect expected)" \
  --out "$H/dryrun-091-vs-091.json" > "$H/dryrun-091-vs-091.md"
echo "dry run 2 rc=$?"
