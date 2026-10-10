# Comparisons

Each comparison is a directory with `comparison.json` and a `README.md` stating the question it answers.
Every README follows one template: **Question**, **Grade** (with the [`FINDINGS.md`](../FINDINGS.md) item it
supports), **Configurations**, **Result**, **Caveats**, **Recompute**.

## Index

| Comparison | Protocol | What it compares | Grade | Findings |
|---|---|---|---|---|
| [`glm53-flash-gpqa-records`](glm53-flash-gpqa-records) | gpqa-diamond/v1 | Three passes each: 3.25bpw on tpurtell 0.7.0 and 0.9.1, 4bpw TR3 (Brandon) on 0.9.1 | Supported (0.7.0 vs 0.9.1 empty answers); descriptive otherwise | 5-8 |
| [`glm53-flash-gpqa-configs`](glm53-flash-gpqa-configs) | gpqa-diamond/v1 | Pass 1 of all ten configurations, paired by question | Descriptive | 9-10 |
| [`glm53-flash-k4-v080-v090-gpqa`](glm53-flash-k4-v080-v090-gpqa) | gpqa-diamond/v1 | 4bpw TR3 (Brandon), tpurtell 0.8.0 vs 0.9.0 (kpool fixes), pass 1 | Descriptive | 9 |
| [`glm53-flash-k4-v090-v091-gpqa`](glm53-flash-k4-v090-v091-gpqa) | gpqa-diamond/v1 | 4bpw TR3 (Brandon), tpurtell 0.9.0 vs 0.9.1 (DCP1 tail fix), pass 1 | Descriptive | 4 |
| [`glm53-flash-gpqa-published-context`](glm53-flash-gpqa-published-context) | (not comparable) | Local scores next to published model-card numbers | Context only | 12 |
| [`glm53-flash-v090-tailfix-decode-prefill`](glm53-flash-v090-tailfix-decode-prefill) | decode-prefill-consistency/v1 | tpurtell 0.9.0 with and without the DCP1 tail fix | Supported (below 2,044 tokens) | 3 |
| [`glm53-flash-v091-decode-prefill-repeat`](glm53-flash-v091-decode-prefill-repeat) | decode-prefill-consistency/v1 | tpurtell 0.9.1 run twice: the noise floor | Supported (not bitwise reproducible) | 3, 19 |
| [`glm53-flash-v091-decode-prefill-speculation`](glm53-flash-v091-decode-prefill-speculation) | decode-prefill-consistency/v1 | tpurtell 0.9.1, speculation off vs on | Descriptive | 3 |
| [`glm53-flash-v090-tailfix-tool-eval`](glm53-flash-v090-tailfix-tool-eval) | tool-eval-bench/v1 | tpurtell 0.9.0 with and without the DCP1 tail fix | Descriptive | 4 |
| [`glm53-flash-serving-probe`](glm53-flash-serving-probe) | serving-probe/v1 | tpurtell 0.8.0 with and without the kpool fixes and speculation | Descriptive (speed); supported (not bitwise reproducible) | 18-19 |
| [`glm53-flash-v091-component-screen-doc88`](glm53-flash-v091-component-screen-doc88) | hard-prompt-screen/v2 | Question 88: three layout switches on tpurtell 0.9.1 | Descriptive | 13-14 |
| [`glm53-flash-doc79-image-screen`](glm53-flash-doc79-image-screen) | hard-prompt-screen/v2 | Question 79: tpurtell 0.9.1 vs the 0.7.0 image at three draft tokens | Descriptive | 13-14 |
| [`glm53-flash-fixed-seed-control`](glm53-flash-fixed-seed-control) | hard-prompt-screen/v2 | Question 88: seed 1234 on every repeat vs distinct seeds | Supported | 15 |

## Format and rules

```json
{
  "id": "glm53-flash-k4-v080-v090-gpqa",
  "question": "Does 4bpw TR3 (Brandon) score or finish differently on tpurtell 0.9.0 than on 0.8.0?",
  "protocol": "gpqa-diamond/v1",
  "runs": ["<run-id>", "..."],
  "varies": ["engine.version", "engine.image"],
  "comparable": true
}
```

`tools/verify.py` checks that every listed run uses `protocol` and the same hardware, and that their configs differ
only in the fields named in `varies`. Anything else fails. Use `"comparable": false` plus `"reason"` for context-only
tables (for example, a local score next to a published one measured with an unknown harness).

Runs under `hard-prompt-screen/v0` and `v1` are single draws per question (every repeat sent one request seed) and
cannot be compared; `tools/verify.py` rejects such a comparison. A `hard-prompt-screen/v2` fixed-seed control may appear
only in a comparison that says "fixed-seed control".
