#!/usr/bin/env python3
"""Compile-cache inventory for the replication (review section 6). Read only: never clears, rewrites or retunes the cache.
  cache_inventory.py snap CACHE_DIR OUT.json         path, size, mtime_ns, sha256 (or 'unreadable') of every file
  cache_inventory.py diff BEFORE.json AFTER.json OUT.json   added / removed / changed files (new compilation activity)
Standard library only; ids, sizes and hashes only."""
import hashlib, json, os, sys


def snap(root, out):
    files = {}
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            p = os.path.join(dp, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            rec = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
            try:
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for b in iter(lambda: fh.read(1 << 20), b""):
                        h.update(b)
                rec["sha256"] = h.hexdigest()
            except OSError:
                rec["sha256"] = "unreadable"
            files[os.path.relpath(p, root)] = rec
    meta = hashlib.sha256("\n".join(f"{k} {v['size']} {v['mtime_ns']} {v['sha256']}" for k, v in sorted(files.items())).encode()).hexdigest()
    json.dump({"root": root, "files": files, "n_files": len(files), "n_unreadable": sum(v["sha256"] == "unreadable" for v in files.values()),
               "total_bytes": sum(v["size"] for v in files.values()), "inventory_sha256": meta}, open(out, "w"), indent=0, sort_keys=True)
    print(f"cache inventory: {len(files)} files, {sum(v['sha256'] == 'unreadable' for v in files.values())} unreadable, sha256 {meta[:16]}")


def diff(a, b, out):
    A, B = json.load(open(a))["files"], json.load(open(b))["files"]
    added = sorted(set(B) - set(A)); removed = sorted(set(A) - set(B))
    changed = sorted(k for k in set(A) & set(B) if A[k] != B[k])
    tops = {}
    for k in added + changed:
        t = "/".join(k.split("/")[:2]); tops[t] = tops.get(t, 0) + 1
    res = {"before": os.path.basename(a), "after": os.path.basename(b), "added": added, "removed": removed, "changed": changed,
           "added_bytes": sum(B[k]["size"] for k in added), "by_top_directory": tops}
    json.dump(res, open(out, "w"), indent=1)
    print(f"cache diff {os.path.basename(a)} -> {os.path.basename(b)}: {len(added)} added, {len(removed)} removed, {len(changed)} changed")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "snap":
        snap(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "diff":
        diff(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
