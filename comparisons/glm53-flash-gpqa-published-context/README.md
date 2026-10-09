# Published GPQA Diamond numbers, for context only

| Source | Weights | Score | Stated protocol |
|---|---|---|---|
| NVIDIA model card | BF16 | 92.17 | temp 1.0, top_p 0.95, 327,680 max new tokens |
| NVIDIA model card | NVFP4 | 92.11 | same |
| Red Hat model card | NVFP4 | 90.57 | lm-eval / lighteval forks, vLLM, 3 seeds averaged |
| This repo (gpqa-diamond/v1), nine configurations, pass 1 each | EXL3 3.25bpw / 4bpw TR3 (Brandon) | 84.3-87.9 (95% intervals 78.8-91.9) | see protocol |

NVIDIA's values lie above the 95% interval of every local run; Red Hat's lies inside the interval of one
(`3.25bpw · tpurtell 0.7.0 · DFlash2 ×5`, 82.8-91.9) and above the others. The raw local scores also run 0.5 to 2.5
points low per pass (the protocol's known limitation). Weights, harness and scoring differ, so these runs cannot
separate quantization, harness and runtime effects.
