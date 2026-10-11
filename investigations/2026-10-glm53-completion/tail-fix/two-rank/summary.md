Decode-completed pools whose written index-K differs from the kernel's compression of the request's
committed keys. Real kernels of each image under TRITON_INTERPRET=1; synthetic scheduler; 3 seeds per
cell; prompts U[64,400] tokens, 160 generated tokens per request, arrivals every 2 steps.

| engine config | tail mapping | draft acceptance | requests | pools | wrong | % wrong | per seed % | per rank % | boundary pools wrong |
|---|---|---|---|---|---|---|---|---|---|
| 0.7.0: DCP 2, ring 4, K=5 | as released (no positions) | 1.0 | 1 | 121 | 121 | 100.0 | 100/100/100 | 100 / 100 | 1/1 |
| 0.7.0: DCP 2, ring 4, K=5 | as released (no positions) | 1.0 | 2 | 242 | 242 | 100.0 | 100/100/100 | 100 / 100 | 2/2 |
| 0.7.0: DCP 2, ring 4, K=5 | as released (no positions) | 1.0 | 8 | 970 | 970 | 100.0 | 100/100/100 | 100 / 100 | 18/18 |
| 0.7.0: DCP 2, ring 4, K=5 | positions passed | 1.0 | 1 | 121 | 1 | 0.8 | 0/2/0 | 0 / 2 | 1/1 |
| 0.7.0: DCP 2, ring 4, K=5 | positions passed | 1.0 | 2 | 242 | 2 | 0.8 | 1/1/0 | 1 / 1 | 2/2 |
| 0.7.0: DCP 2, ring 4, K=5 | positions passed | 1.0 | 8 | 970 | 18 | 1.9 | 2/2/2 | 2 / 2 | 18/18 |
| 0.7.0: DCP 2, ring 4, K=5 | as released (no positions) | 0.85 | 1 | 120 | 120 | 100.0 | 100/100/100 | 100 / 100 | 1/1 |
| 0.7.0: DCP 2, ring 4, K=5 | as released (no positions) | 0.85 | 2 | 244 | 244 | 100.0 | 100/100/100 | 100 / 100 | 2/2 |
| 0.7.0: DCP 2, ring 4, K=5 | as released (no positions) | 0.85 | 8 | 969 | 969 | 100.0 | 100/100/100 | 100 / 100 | 18/18 |
| 0.7.0: DCP 2, ring 4, K=5 | positions passed | 0.85 | 1 | 120 | 39 | 32.5 | 28/45/25 | 27 / 38 | 1/1 |
| 0.7.0: DCP 2, ring 4, K=5 | positions passed | 0.85 | 2 | 244 | 66 | 27.0 | 32/29/20 | 31 / 23 | 2/2 |
| 0.7.0: DCP 2, ring 4, K=5 | positions passed | 0.85 | 8 | 969 | 251 | 25.9 | 28/24/26 | 30 / 22 | 18/18 |
| 0.7.0: DCP 2, ring 4, K=5 | as released (no positions) | 0.6 | 1 | 120 | 120 | 100.0 | 100/100/100 | 100 / 100 | 1/1 |
| 0.7.0: DCP 2, ring 4, K=5 | as released (no positions) | 0.6 | 2 | 240 | 240 | 100.0 | 100/100/100 | 100 / 100 | 2/2 |
| 0.7.0: DCP 2, ring 4, K=5 | as released (no positions) | 0.6 | 8 | 963 | 963 | 100.0 | 100/100/100 | 100 / 100 | 18/18 |
| 0.7.0: DCP 2, ring 4, K=5 | positions passed | 0.6 | 1 | 120 | 83 | 69.2 | 72/72/62 | 72 / 67 | 1/1 |
| 0.7.0: DCP 2, ring 4, K=5 | positions passed | 0.6 | 2 | 240 | 159 | 66.2 | 62/66/70 | 62 / 70 | 2/2 |
| 0.7.0: DCP 2, ring 4, K=5 | positions passed | 0.6 | 8 | 963 | 654 | 67.9 | 68/64/72 | 67 / 69 | 18/18 |
| 0.9.1: DCP 1, ring 8, K=3 | as released (no positions) | 0.85 | 1 | 120 | 0 | 0.0 | 0/0/0 | 0 | 0/1 |
| 0.9.1: DCP 1, ring 8, K=3 | as released (no positions) | 0.85 | 2 | 242 | 114 | 47.1 | 33/59/49 | 47 | 1/2 |
| 0.9.1: DCP 1, ring 8, K=3 | as released (no positions) | 0.85 | 8 | 965 | 748 | 77.5 | 73/86/74 | 78 | 16/18 |
| 0.9.1: DCP 1, ring 8, K=3 | positions passed | 0.85 | 1 | 120 | 0 | 0.0 | 0/0/0 | 0 | 0/1 |
| 0.9.1: DCP 1, ring 8, K=3 | positions passed | 0.85 | 2 | 242 | 0 | 0.0 | 0/0/0 | 0 | 0/2 |
| 0.9.1: DCP 1, ring 8, K=3 | positions passed | 0.85 | 8 | 965 | 0 | 0.0 | 0/0/0 | 0 | 0/18 |
| 0.9.1: DCP 1, ring 8, K=3 | as released (no positions) | 0.6 | 1 | 121 | 0 | 0.0 | 0/0/0 | 0 | 0/1 |
| 0.9.1: DCP 1, ring 8, K=3 | as released (no positions) | 0.6 | 2 | 241 | 138 | 57.3 | 44/59/69 | 57 | 0/2 |
| 0.9.1: DCP 1, ring 8, K=3 | as released (no positions) | 0.6 | 8 | 967 | 868 | 89.8 | 91/88/90 | 90 | 17/18 |
| 0.9.1: DCP 1, ring 8, K=3 | positions passed | 0.6 | 1 | 121 | 0 | 0.0 | 0/0/0 | 0 | 0/1 |
| 0.9.1: DCP 1, ring 8, K=3 | positions passed | 0.6 | 2 | 241 | 0 | 0.0 | 0/0/0 | 0 | 0/2 |
| 0.9.1: DCP 1, ring 8, K=3 | positions passed | 0.6 | 8 | 967 | 0 | 0.0 | 0/0/0 | 0 | 0/18 |

Shadow classification (Python transcription of the ring addressing) agreed with the real-kernel outcome
for every pool in every cell. Classes of the wrong pools, pooled over seeds and acceptance rates:

| engine config | tail mapping | requests | foreign key read | never-written slot read | same request, other position | own rejected draft read |
|---|---|---|---|---|---|---|
| 0.7.0 | as released | 1 | 0% | 50% | 100% | 29% |
| 0.7.0 | as released | 2 | 74% | 51% | 50% | 2% |
| 0.7.0 | as released | 8 | 96% | 1% | 24% | 0% |
| 0.7.0 | positions passed | 1 | 0% | 2% | 0% | 98% |
| 0.7.0 | positions passed | 2 | 0% | 2% | 0% | 98% |
| 0.7.0 | positions passed | 8 | 0% | 5% | 0% | 95% |
| 0.9.1 | as released | 2 | 100% | 0% | 0% | 0% |
| 0.9.1 | as released | 8 | 100% | 0% | 0% | 0% |
