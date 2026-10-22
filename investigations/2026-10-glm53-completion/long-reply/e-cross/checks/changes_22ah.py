#!/usr/bin/env python3
"""Amendment 22ah change record: writes CHANGES-22ah.json at the package root. Host, standard library, read only otherwise.

For every file F with a first-freeze copy F.pre22ah: before = sha256(F.pre22ah), after = sha256(F), or "removed" when F no
longer exists. New files = files present now (make_sums.sh's scope: not run/, PRIVATE/scratch-*, __pycache__, *.pyc,
FROZEN-VERIFY.txt) that the first freeze's SHA256SUMS (SHA256SUMS.pre22ah) did not list. SHA256SUMS and FROZEN-VERIFY.txt
are rewritten after this record; their new hashes are in FROZEN-VERIFY.txt. Hashes and paths only (PRIVATE files by hash).
usage: checks/changes_22ah.py
"""
import hashlib
import json
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EC = "<runs>/long-reply/e-cross"
LATE = {"SHA256SUMS", "FROZEN-VERIFY.txt", "CHANGES-22ah.json"}


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def tree():
    out = []
    for dp, dn, fn in os.walk(PKG):
        rel = os.path.relpath(dp, PKG)
        if rel == "run" or rel.startswith("run/") or rel.startswith("PRIVATE/scratch-") or "__pycache__" in rel:
            dn[:] = []
            continue
        for f in fn:
            r = os.path.normpath(os.path.join(rel, f))
            if not f.endswith(".pyc"):
                out.append(r)
    return sorted(out)


def main():
    if PKG != EC:
        sys.exit(f"unexpected package path {PKG}")
    old = {}
    for l in open(f"{PKG}/SHA256SUMS.pre22ah"):
        h, p = l.rstrip("\n").split("  ", 1)
        old[p[2:] if p.startswith("./") else p] = h
    files = tree()
    changed, unchanged_backups = [], []
    for r in files:
        if not r.endswith(".pre22ah"):
            continue
        f = r[:-len(".pre22ah")]
        before = fsha(f"{PKG}/{r}")
        if f in LATE:
            after = "rewritten after this record (see FROZEN-VERIFY.txt)"
        elif os.path.exists(f"{PKG}/{f}"):
            after = fsha(f"{PKG}/{f}")
        else:
            after = "removed"
        if before == after:
            unchanged_backups.append(f)
        changed.append({"path": f, "before_sha256": before, "after_sha256": after, "first_freeze_copy": r,
                        "first_freeze_listed_hash_matches_copy": old.get(f) == before if f in old else None})
    new = [{"path": r, "sha256": fsha(f"{PKG}/{r}")} for r in files
           if not r.endswith(".pre22ah") and r not in old and r not in LATE and r != "FROZEN-VERIFY.txt"]
    gone = sorted(p for p in old if not os.path.exists(f"{PKG}/{p}"))
    rec = {"what": "amendment 22ah re-freeze of e-cross: every file changed since the first freeze (22ac, SHA256SUMS 0be97e4c...)",
           "first_freeze_sha256sums_sha256": fsha(f"{PKG}/SHA256SUMS.pre22ah"),
           "changed": changed, "new": new, "removed_without_copy": [p for p in gone if not os.path.exists(f"{PKG}/{p}.pre22ah")],
           "backups_identical_to_current": unchanged_backups,
           "summary": {"changed": sum(c["after_sha256"] != c["before_sha256"] for c in changed), "new": len(new),
                       "removed": sum(c["after_sha256"] == "removed" for c in changed)}}
    json.dump(rec, open(f"{PKG}/CHANGES-22ah.json", "w"), indent=1)
    print(json.dumps(rec["summary"]), "identical backups:", unchanged_backups, "removed without copy:", rec["removed_without_copy"])
    bad = [c["path"] for c in changed if c["first_freeze_listed_hash_matches_copy"] is False]
    if bad:
        print("WARNING: first-freeze copy differs from the first SHA256SUMS:", bad)
    return 1 if bad or rec["removed_without_copy"] else 0


if __name__ == "__main__":
    sys.exit(main())
