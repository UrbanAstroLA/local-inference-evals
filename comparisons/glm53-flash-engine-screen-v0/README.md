# Engine screen (2026-09-30): tpurtell 0.7.0 vs tpurtell 0.8.0

Same 3.25bpw weights (the two model revisions have identical weight shards). Everything in `serving` differs too
(draft depth, EP, DCP, vision, slot sharing), so this compares engine releases as shipped, not a single factor.
Same session, protocol v0. Result: `3.25bpw · tpurtell 0.7.0 · DFlash2 ×5` 12/40 failed (loop or exhaustion),
`3.25bpw · tpurtell 0.8.0 · DFlash2 ×3` 24/40. The questions were chosen from tpurtell 0.8.0's empty GPQA answers, so
this gap is an upper-end estimate; the full GPQA runs (`glm53-flash-gpqa-configs`) carry the engine finding.
