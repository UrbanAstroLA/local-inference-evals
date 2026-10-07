# Serving probe: tpurtell 0.8.0 with and without the kpool fixes, with and without speculation

Five configurations on 3.25bpw: `tpurtell 0.8.0` with 3 drafts and without speculation, and
`tpurtell 0.8.0 + kpool fixes ≈ 0.9.0` with 3 drafts, 5 drafts and without speculation. One run each.

Fixes: no consistent change. With the same speculation setting, per-request decode speed moves by -3.5% to +1.7%,
aggregate throughput at 8 concurrent by +2.7% (3 drafts) and -7.4% (no speculation), acceptance by at most 0.004.
Five drafts vs three (both with the fixes): faster at concurrency 1 (163 vs 146 tok/s per request), slower at 8
(289 vs 374 tok/s aggregate).
Greedy agreement: identical configs diverge after a median ~318 characters at temperature 0 (the engine is not
bitwise reproducible), so greedy parity cannot certify speculative exactness here. Per-pair prefix lengths are in
`parity.json` (derived locally from output text, which is not published; output hashes are in each run).
