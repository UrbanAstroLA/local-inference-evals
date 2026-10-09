# Fixed-seed control: one request seed on every repeat vs distinct seeds

`3.25bpw · tpurtell 0.9.1 · DFlash2 ×3`, GPQA Diamond doc 88 × 12, 12 concurrent requests, fresh server, same client and
classifier. Only the request seed differs: 5001-5012 (arm B) vs 1234 on every repeat (arm S1234, added by a dated
amendment after the first arms were seen; `investigations/2026-10-glm53-looping/component-screen`).

| Arm | Request seeds | Failed to finish / 12 |
|---|---|---|
| B | 5001-5012 | 3 |
| S1234 | 1234 on every repeat | 11 |

Fisher p = 0.003. The amendment's reading rule (9 or more failures: the fixed seed explains most of the difference)
applies. The engine draws its sampling noise from the request seed. Repeats that share a seed still vary, because
concurrent batching changes the arithmetic and their texts diverge, but they draw on the same sampler noise, so that
variation is not a statistically meaningful sample of the configuration. S1234 is a control for the seed, not a
measurement of the configuration. It is why the earlier screens, which sent seed 1234 on every repeat, were withdrawn
as rate measures (notice: `investigations/2026-10-glm53-looping`). Recompute with `tools/analyze.py screen-v2`.
