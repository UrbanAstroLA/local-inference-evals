# Serving probe: v0.8.0, kpool fixes and speculation

Fixes: no measurable change in speed or acceptance. Five drafts: faster at concurrency 1, slower at 8.
Greedy agreement: identical configs diverge after a median ~318 characters at temperature 0 (the engine is not
bitwise reproducible), so greedy parity cannot certify speculative exactness here. Per-pair prefix lengths are in
`parity.json` (derived locally from output text, which is not published; output hashes are in each run).
