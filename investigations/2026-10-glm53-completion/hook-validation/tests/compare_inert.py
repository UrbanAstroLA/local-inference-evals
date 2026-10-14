import re, sys, os
out = sys.argv[1]
hs = {}
for n in ("inert_orig", "inert_hooked_off", "inert_hooked_on_unkeyed", "image_inert_off"):
    p = os.path.join(out, n + ".log")
    if not os.path.exists(p):
        continue
    t = open(p).read()
    m = re.findall(r"^HASH (\w+)", t, re.M)
    hs[n] = m[-1] if m else None
ok = len(set(hs.values())) == 1 and None not in hs.values()
for k, v in hs.items():
    print(k, v)
print("%s I1 identical outputs: original B1 files / hooked env unset / hooked env on, no keyed request / built image env unset" % ("PASS" if ok else "FAIL"))
