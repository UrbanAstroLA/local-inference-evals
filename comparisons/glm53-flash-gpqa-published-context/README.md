# Published GPQA Diamond numbers, for context only

| Source | Weights | Score | Stated protocol |
|---|---|---|---|
| NVIDIA model card | BF16 | 92.17 | temp 1.0, top_p 0.95, 327,680 max new tokens |
| NVIDIA model card | NVFP4 | 92.11 | same |
| Red Hat model card | NVFP4 | 90.57 | lm-eval / lighteval forks, vLLM, 3 seeds averaged |
| This repo (gpqa-diamond/v1), five full runs | EXL3 3.25bpw / 4bpw TR3 (Brandon) | 84.7-86.2 (95% intervals 80.5-89.9) | see protocol |

Both published values lie above the 95% interval of every local full run. Scoring only answered questions would put
the local runs at 86.4-88.7%, still below both. The rest of the gap mixes quantization, harness and other runtime
effects, which these runs cannot separate.
