#!/usr/bin/env python3
"""GPU-idle preflight for replay-topk (PROTOCOL.md section 8).  Host python3, stdlib only.

usage: preflight_gpu.py [--samples N] [--interval S]
Reads `nvidia-smi` (the full process table, which lists graphics clients: type G and C+G; the compute-apps query
misses them) and the per-GPU memory / utilisation query.  Prints a report and a last line "PASS ..." or "FAIL <why>".

GPU 0 must have, in every sample until one passes (up to N samples):
  * no compute process (type C or C+G);
  * no graphics client using more than GRAPHICS_MAX_MIB (a desktop compositor using a few MiB is reported, allowed);
  * memory used <= MEM_MAX_MIB and utilisation <= UTIL_MAX_PCT.
Processes on GPU 1 are reported, never a refusal (the replay uses GPU 0 only).
NVSMI env var: path of the nvidia-smi binary (tests put a fake one on PATH).
"""
import os
import re
import subprocess
import sys
import time

MEM_MAX_MIB = 1024
UTIL_MAX_PCT = 5
GRAPHICS_MAX_MIB = 64
GPU = "0"


def smi(args):
    exe = os.environ.get("NVSMI", "nvidia-smi")
    r = subprocess.run([exe] + args, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"nvidia-smi {' '.join(args)} rc={r.returncode}: {r.stderr.strip()[:200]}")
    return r.stdout


def processes(table):
    """Rows of the 'Processes' section: (gpu, pid, type, name, mib)."""
    out = []
    in_proc = False
    for line in table.splitlines():
        if "Processes:" in line:
            in_proc = True
            continue
        if not in_proc:
            continue
        m = re.match(r"^\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+(C\+G|G|C)\s+(.*?)\s+(\d+)MiB\s*\|$", line)
        if m:
            out.append(dict(gpu=m.group(1), pid=int(m.group(2)), type=m.group(3), name=_short(m.group(4)),
                            mib=int(m.group(5))))
    return out


def _short(name):
    """Program name only (no directories or arguments): the report is kept with run outputs."""
    first = name.strip().split()[0] if name.strip() else "?"
    return os.path.basename(first.rstrip("/")) or first


def sample():
    q = smi(["--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    gpus = {}
    for line in q.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise RuntimeError(f"cannot parse GPU query line {line!r}")
        gpus[parts[0]] = dict(mem=int(parts[1]), util=int(parts[2]))
    return gpus, processes(smi([]))


def main():
    n = 12
    interval = 5.0
    a = sys.argv[1:]
    if "--samples" in a:
        n = int(a[a.index("--samples") + 1])
    if "--interval" in a:
        interval = float(a[a.index("--interval") + 1])
    why = "no sample"
    for i in range(1, n + 1):
        try:
            gpus, procs = sample()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL nvidia-smi unreadable: {exc}")
            return 1
        if GPU not in gpus:
            print(f"FAIL GPU {GPU} not listed")
            return 1
        g0 = gpus[GPU]
        print(f"sample {i}: " + "; ".join(f"GPU{k} mem {v['mem']} MiB util {v['util']}%" for k, v in sorted(gpus.items())))
        for p in procs:
            print(f"  process GPU{p['gpu']} pid {p['pid']} type {p['type']} {p['mib']} MiB {p['name'][:60]}")
        comp = [p for p in procs if p["gpu"] == GPU and p["type"] in ("C", "C+G")]
        gfx_big = [p for p in procs if p["gpu"] == GPU and p["type"] == "G" and p["mib"] > GRAPHICS_MAX_MIB]
        gfx = [p for p in procs if p["type"] in ("G", "C+G")]
        if comp:
            print(f"FAIL compute process on GPU {GPU}: " + ", ".join(f"{p['pid']} {p['name'][:40]}" for p in comp))
            return 1
        if gfx_big:
            print(f"FAIL graphics client on GPU {GPU} above {GRAPHICS_MAX_MIB} MiB: "
                  + ", ".join(f"{p['pid']} {p['name'][:40]} {p['mib']} MiB" for p in gfx_big))
            return 1
        if g0["mem"] > MEM_MAX_MIB:
            print(f"FAIL GPU {GPU} memory {g0['mem']} MiB > {MEM_MAX_MIB} MiB")
            return 1
        if g0["util"] <= UTIL_MAX_PCT:
            print(f"PASS GPU {GPU} idle (mem {g0['mem']} MiB, util {g0['util']}%); graphics clients reported: "
                  f"{len(gfx)} ({sum(1 for p in gfx if p['gpu'] == GPU)} on GPU {GPU})")
            return 0
        why = f"GPU {GPU} utilisation {g0['util']}% > {UTIL_MAX_PCT}%"
        if i < n:
            time.sleep(interval)
    print(f"FAIL {why} after {n} samples")
    return 1


if __name__ == "__main__":
    sys.exit(main())
