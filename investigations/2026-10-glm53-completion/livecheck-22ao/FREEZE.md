# 22ao live check: freeze record (2026-10-23)

Frozen before any data; nothing has run on the GPUs. `SHA256SUMS` covers every file the runner and the analysis use,
the images' build inputs and the CPU test code and outputs. The runner refuses to start if any listed file differs, or
if an image id differs from `runner/images.txt`.

- **Proposal:** `PROPOSAL.md` (pre-registration; amendment 22ao).
- **Images:**
  - arm A `local/glm53-v091-22ao-a:local` = `sha256:dd51ed7315ae6896602f9724cfc512382f5e97b38af9f37f70420c05163c420e`;
  - arm C `local/glm53-v091-22ao-c:local` = `sha256:5617e0df0c0a5a793754bc705784e538fc0a3639cbdf1bef9dc8950ac542686e`;
  - both built from `sha256:a9e985c09885710601b32ac86e81256aa1279b542caa345233ce331a816d7e07` with `--network none`.
- **Guard:** `guard/mapguard_hooks.py` `e2dcd964…`, from the frozen B1 guard `a8e59fa5…` via `guard/make_guard_22ao.py`.
- **Launch (after authorization only):** `APPROVED_GPU_H=2.0 runner/run_22ao.sh`. One attempt; arm A then arm C;
  a 1,740 s wall cap per arm.
- **Not published:** `PRIVATE/` (the GPQA panel fixture). Its hash is in `SHA256SUMS`.
