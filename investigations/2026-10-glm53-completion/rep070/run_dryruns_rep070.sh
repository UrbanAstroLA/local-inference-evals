#!/usr/bin/env bash
# Dry runs of the DRAFT analysis on existing records (no new data). Not set -e.
H=$(cd "$(dirname "$0")" && pwd); G=<gpqa-runs>; R=<this-repo>/runs
V070="$G/glm53-flash-exl3-k3.25@tpurtell"; V091="$G/glm53-flash-exl3-k3.25@tpurtell+v0.9.1"
python3 "$H/analysis_rep070_draft.py" --rep-label "old 0.7.0 record standing in as rep" \
  --rep-pass "$V070+ps__pass1" "$V070+ps__pass2" "$V070+ps__pass3" --rep-reqlog "$V070+ps.requests.jsonl" --rep-hedged 2:25 2:94 \
  --rep-published "$R/2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond/results.jsonl" --require-rep-reqlog 0 \
  --dry-run-label "old 0.7.0 record as rep (expect O1; C2 zero by construction)" --out "$H/dryrun-rep-old070.json" > "$H/dryrun-rep-old070.md"
echo "dry run 1 rc=$?"
python3 "$H/analysis_rep070_draft.py" --rep-label "0.9.1 record standing in as rep" \
  --rep-pass "$V091+ps__pass1" "$V091+ps__pass2" "$V091+ps__pass3" --rep-reqlog "$V091.requests.jsonl" "$V091+ps.requests.jsonl" \
  --rep-published "$R/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl" \
  --dry-run-label "0.9.1 record as rep (expect O2; C1 zero by construction)" --out "$H/dryrun-rep-v091.json" > "$H/dryrun-rep-v091.md"
echo "dry run 2 rc=$?"
