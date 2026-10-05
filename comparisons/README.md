# Comparisons

Each comparison is a directory with `comparison.json` and a `README.md` stating the question it answers.

```json
{
  "id": "glm53-flash-kpool-screen",
  "question": "Do the kpool tail fixes reduce non-completion on hard GPQA prompts?",
  "protocol": "hard-prompt-screen/v1",
  "runs": ["<run-id>", "..."],
  "varies": ["engine.patches", "serving.speculative.tokens"],
  "comparable": true
}
```

`tools/verify.py` checks that every listed run uses `protocol` and the same hardware, and that their configs differ
only in the fields named in `varies`. Anything else fails. Use `"comparable": false` plus `"reason"` for context-only
tables (for example, a local score next to a published one measured with an unknown harness).
