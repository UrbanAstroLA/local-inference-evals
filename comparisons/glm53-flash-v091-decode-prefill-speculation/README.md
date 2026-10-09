# Decode vs prefill on tpurtell 0.9.1: speculation off (two runs) and on (one run)

`3.25bpw · tpurtell 0.9.1 · no speculation, prefix cache off` (two runs) vs
`3.25bpw · tpurtell 0.9.1 · DFlash2 ×3, prefix cache off` (one run). Same 12 GPQA prompts and request seeds, one request at
a time, fresh server per run. With speculation on, generated tokens pass through the draft-and-verify path, and their
decode distributions are compared with prefill re-scoring of the same tokens.

| Positions | Speculation off, run 1 | Speculation off, run 2 | Speculation on |
|---|---|---|---|
| i < 2,044 | 0.0072 | 0.0077 | 0.0105 |
| i ≥ 2,048 | 0.0053 | 0.0098 | 0.0115 |

Descriptive only: speculation on sits somewhat above both speculation-off runs, by less than the spread between single
prompts across repeated runs (`glm53-flash-v091-decode-prefill-repeat`). One run with speculation on does not separate a
difference of this size from run-to-run variation. `tools/analyze.py decode-prefill`.
