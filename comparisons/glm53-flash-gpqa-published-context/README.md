# Published GPQA Diamond numbers, for context only

| Source | Weights | Score | Stated protocol |
|---|---|---|---|
| NVIDIA model card | BF16 | 92.17 | temp 1.0, top_p 0.95, 327,680 max new tokens |
| NVIDIA model card | NVFP4 | 92.11 | same |
| Red Hat model card | NVFP4 | 90.57 | lm-eval / lighteval forks, vLLM, 3 seeds averaged |
| This repo (gpqa-diamond/v1) | EXL3 K3.25 | 85.5 (95% CI ~81-89) | see protocol |

Both published values lie above the 95% interval of the local runs. The gap mixes quantization, harness and
runtime effects, which these runs cannot separate.
