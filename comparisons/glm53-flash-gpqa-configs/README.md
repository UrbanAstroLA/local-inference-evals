# GPQA Diamond across GLM-5.3-Flash configurations

Five full 3-pass runs: `3.25bpw · tpurtell 0.7.0 · 5 drafts`, `3.25bpw · tpurtell 0.8.0 · 3 drafts`,
`3.25bpw · tpurtell 0.8.0 · 5 drafts, sharing off`, `4bpw · tpurtell 0.8.0 · 3 drafts`, `4bpw · tpurtell 0.9.0 · 3 drafts`.

All five accuracy intervals overlap (84.7-86.2%, 95% intervals roughly ±4 points; paired sign tests over all 594
answers p >= 0.37): accuracy does not separate these configurations. Empty answers point to the engine: tpurtell 0.7.0
left 6 of 594; the four 0.8.0/0.9.0 configurations left 15-23. On the same 3.25bpw weights the difference is clear
(6 vs 20, Fisher exact p = 0.009; 6 vs 17 with 0.7.0's draft depth, p = 0.033); across weights it is 6 vs 15
(p = 0.08) and 6 vs 23 (p = 0.002). Recompute with `tools/analyze.py gpqa-pairs`.
