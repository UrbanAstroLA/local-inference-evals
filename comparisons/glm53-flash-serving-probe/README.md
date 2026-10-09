# Serving probe: tpurtell 0.8.0 with and without the kpool fixes, with and without speculation

Five configurations on 3.25bpw: `tpurtell 0.8.0` with DFlash2 ×3 and with no speculation, and
`tpurtell 0.8.0 + kpool fixes ≈ 0.9.0` with DFlash2 ×3, DFlash2 ×5 and no speculation. One run each.

Fixes: no consistent change. With the same speculation setting, per-request decode speed moves by -3.5% to +1.7%,
aggregate throughput at 8 concurrent by +2.7% (DFlash2 ×3) and -7.4% (no speculation), acceptance by at most 0.004.
DFlash2 ×5 vs ×3 (both with the fixes): faster at concurrency 1 (163 vs 146 tok/s per request), slower at 8
(289 vs 374 tok/s aggregate).
**Single runs, no noise floor:** each configuration ran once, so speed and acceptance differences of a few percent
cannot be told from run-to-run variation.

Greedy agreement: the same configuration run twice, one request at a time, diverges after a median ~318 characters at
temperature 0, so the engine is not bitwise reproducible even without batching, and greedy parity cannot certify
speculative exactness here. **Receipts:** the per-pair prefix lengths in `parity.json` are derived from the greedy output
text, which is model output and is not published; each run publishes the outputs' SHA-256 hashes (`output_sha256`), and
the probe can be rerun with `tools/clients/serving_probe.py`. These lengths are the one serving figure without a
row-level receipt.
