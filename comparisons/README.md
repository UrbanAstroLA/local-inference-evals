# Comparisons

Each comparison is a directory with `comparison.json` and a `README.md` stating the question it answers.

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
