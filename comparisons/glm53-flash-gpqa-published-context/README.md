# Published GPQA Diamond numbers, for context only

**Question.** Where do the local GPQA Diamond scores sit next to published model-card numbers? Context only, not a
comparison: `comparison.json` marks it `"comparable": false`.

**Grade.** Not graded; context only ([`FINDINGS.md`](../../FINDINGS.md#2b-pass-1-of-every-configuration), item 12).

## Result

| Source | Weights | Score | Stated protocol |
|---|---|---|---|
| NVIDIA model card | BF16 | 92.17 | temp 1.0, top_p 0.95, 327,680 max new tokens |
| NVIDIA model card | NVFP4 | 92.11 | same |
| Red Hat model card | NVFP4 | 90.57 | lm-eval / lighteval forks, vLLM, 3 seeds averaged |
| This repo (gpqa-diamond/v1), three-pass records (mean of three request seeds) | EXL3 3.25bpw / 4bpw TR3 (Brandon) | 88.2, 86.9, 85.7 (95% intervals 81.5-91.6) | see protocol |
| This repo (gpqa-diamond/v1), ten configurations, pass 1 each | EXL3 3.25bpw / 4bpw TR3 (Brandon) | 84.3-87.9 (95% intervals 78.8-91.9) | see protocol |

Each run's interval here covers all its passes (three for the three-pass records, one otherwise).

- **Raw score.** NVIDIA's values lie above the 95% interval of every local run, and Red Hat's lies inside the intervals
  of three (`3.25bpw · tpurtell 0.7.0 · DFlash2 ×5`, three passes; `3.25bpw · tpurtell 0.8.0 · DFlash2 ×5, sharing off`
  and `3.25bpw · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · DFlash2 ×5`, one pass each) and above the others.
- **Audited stated-answer score.** Red Hat's 90.6 lies inside the intervals of nine of the ten runs and NVIDIA's 92.1
  inside the same three.
- **Noise.** Passes of one configuration differ by 0.5 to 3.5 points
  ([`glm53-flash-gpqa-records`](../glm53-flash-gpqa-records)).

## Caveats

- No higher-precision version of the model (BF16 or NVFP4) was run on this hardware, so whether the EXL3 quants cost
  accuracy is not measured here.
- The published numbers used other weights and harnesses whose details (NVIDIA) or answer extraction (both) are not
  published, so the gap between them and these runs cannot be split into quantization, harness, scoring and runtime
  effects.

## Recompute

```bash
python3 tools/analyze.py gpqa-table     # local pass-1 scores and intervals
python3 tools/analyze.py gpqa-passes    # local three-pass records
```
