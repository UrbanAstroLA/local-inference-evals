# CPU-only reconnaissance, 2026-10-22 (after the close-out; operator: "Proceed with a revised CPU-only reconnaissance plan")

Nothing here used a GPU, downloaded anything or ran sudo, and nothing was pushed or posted. GPQA question, prompt,
reply and answer text appear in no output; outputs carry ids, flags, counts and token counts only.

| Step | Folder | Status |
|---|---|---|
| 1. Input manifest | `f1/manifest.json` (+ `f1/result.md` "Inputs") | done |
| 2. Original / fresh / pooled stock comparisons | `f1/` (`stock_compare.py`, `result.json`, `result.md`, `rows.jsonl`) | done; post hoc, descriptive |
| 3. Dispatch table | `dispatch/` (`DISPATCH.md`, `gather.py`, `observed.json`, `SOURCE-SHA256.txt`, `src/` image copies) | done |
| 4. Draft-slot CPU reproduction | `draft-slot/` (`harness.py`, `run.sh`, `out/`, `README.md`) | done: 0 violations in 6 reachable-state scenarios; 3 of 4 positive controls robust, lookahead control weak |
| 5. CPU EXL3 decoder | — | deferred, as instructed: an approximate negative engagement count cannot replace Stage B on the real kernels |
| 6. Coverage map (replaces validation recalibration) | `coverage/COVERAGE.md` | done |
| 7. One causal intervention | `INTERVENTION-PROPOSAL.md`, `NEXT-STEPS.md` (local only, not published) | proposal only; not authorized |

**Inputs touched, all read-only:**
- public `runs/` and investigation files;
- `<gpqa-runs>/*` request logs and lm-eval samples (rep070, through the frozen loader in-process);
- server logs;
- recipe env files;
- the two stock images, by digest:
  - `docker create` / `docker cp` to copy `vllm` and `b12x`, never run;
  - a CPU container of the stock 0.9.1 image for the draft-slot reproduction (`--network none`, no GPU flag, default
    runtime `runc`).

**Disk:** `dispatch/src` holds about 1.6 GB of source copied from the images, which are the durable reference. It can
be deleted and re-copied by digest.
