#!/usr/bin/env python3
"""The built image's installed files equal the reviewed hook copies (hook/hooked-b1-mapguard + the module)."""
import hashlib
import os
P = "/usr/local/lib/python3.12/dist-packages/vllm"
H = "/hv/hook"
ok = True
pairs = [("forcewin_hooks.py", os.path.join(H, "forcewin_hooks.py"))]
for root, _, fs in os.walk(os.path.join(H, "hooked-b1-mapguard")):
    for f in fs:
        if f.endswith(".py"):
            rel = os.path.relpath(os.path.join(root, f), os.path.join(H, "hooked-b1-mapguard"))
            pairs.append((rel, os.path.join(root, f)))
for rel, src in pairs:
    a = hashlib.sha256(open(os.path.join(P, rel), "rb").read()).hexdigest()
    b = hashlib.sha256(open(src, "rb").read()).hexdigest()
    ok &= a == b
    print("%s %s %s" % ("PASS" if a == b else "FAIL", rel, a[:16]))
print("%s IMG1 installed files == reviewed hook copies (%d files)" % ("PASS" if ok else "FAIL", len(pairs)))
