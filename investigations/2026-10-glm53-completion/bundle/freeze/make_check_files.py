#!/usr/bin/env python3
"""Build the observe-only verification files for a bundle-check image (PLAN.md amendment 14 (3)).
Additions only, over the bundle's own files:
  sparse_attn_indexer_kpool.py = bundle D1 file (9104be37) + col511 checker additions (gate/freeze, = 7a486159)
                                 + tail-guard hooks (tailguard/make_patch.py anchors, as in the GPU-run U-guard image)
  gate_linear.py               = bundle gate_linear.py + the verify13 guarded install block (verify13/build/gate_linear.diff)
                                 + the d4limit guarded install block (D4 reachability override, verification only)
usage: make_check_files.py <c511chk_kpool.py> <tailguard_make_patch.py> <bundle_gate_linear.py> <out_dir>"""
import difflib, hashlib, sys

c511_path, tg_make, gl_path, out = sys.argv[1:5]
src = open(c511_path).read()
assert hashlib.sha256(src.encode()).hexdigest().startswith("7a486159bc38a0da"), "not the frozen checker file"
ns = {}
code = open(tg_make).read().split("for a in (IMPORT_ANCHOR")[0]   # constants only
code = code.replace("src = open(src_path).read()", "").replace("src_path, out_dir = sys.argv[1], sys.argv[2]", "")
exec(code, ns)
for a in (ns["IMPORT_ANCHOR"], ns["SEED_ANCHOR"], ns["STASH_ANCHOR"]):
    assert src.count(a) == 1, a
k = src.replace(ns["IMPORT_ANCHOR"], ns["IMPORT_ANCHOR"] + ns["IMPORT_ADD"])
k = k.replace(ns["SEED_ANCHOR"], ns["SEED_ADD"] + ns["SEED_ANCHOR"])
k = k.replace(ns["STASH_ANCHOR"], ns["STASH_ADD"] + ns["STASH_ANCHOR"])

V13_BLOCK = (
    "\n\n# --- verify13 (observe-only capture; inert unless GLM53_VERIFY13=1) ---\n"
    "import os as _v13_os  # noqa: E402\n\n"
    "if _v13_os.environ.get(\"GLM53_VERIFY13\", \"\").strip() == \"1\":\n"
    "    from vllm import verify13_capture as _v13  # noqa: E402\n\n"
    "    _v13.install(GateLinear)\n"
    "\n"
    "# --- d4limit (verification image only; inert unless GLM53_D4_LIMIT_OVERRIDE or GLM53_D4_INSPECT is set) ---\n"
    "if (_v13_os.environ.get(\"GLM53_D4_LIMIT_OVERRIDE\", \"\").strip()\n"
    "        or _v13_os.environ.get(\"GLM53_D4_INSPECT\", \"\").strip() == \"1\"):\n"
    "    from vllm import d4limit as _d4l  # noqa: E402\n\n"
    "    _d4l.install_on_gate(GateLinear)\n"
)
gl = open(gl_path).read()
assert gl.endswith("\n") and "_v13" not in gl
g = gl.rstrip("\n") + "\n" + V13_BLOCK

def additions_only(a, b, name):
    d = list(difflib.unified_diff(a.splitlines(True), b.splitlines(True), "a/" + name, "b/" + name))
    rem = [x for x in d if x.startswith("-") and not x.startswith("---")]
    add = [x for x in d if x.startswith("+") and not x.startswith("+++")]
    assert not rem, rem
    return d, len(add)

# tail-guard additions alone (checker file -> check file), as test_tailguard.py expects it next to the file
d_tg, _ = None, None
for name, a, b in (("sparse_attn_indexer_kpool.py", src, k), ("gate_linear.py", gl, g)):
    open(f"{out}/{name}", "w").write(b)
    d, n = additions_only(a, b, name)
    open(f"{out}/{name}.observer-additions.diff", "w").writelines(d)
    print(f"{name}: +{n} lines, -0 (additions only over its input); sha256 {hashlib.sha256(b.encode()).hexdigest()}")
d_tg = list(difflib.unified_diff(src.splitlines(True), k.splitlines(True), "a/sparse_attn_indexer_kpool.py", "b/sparse_attn_indexer_kpool.py"))
assert not [x for x in d_tg if x.startswith("-") and not x.startswith("---")]
open(f"{out}/tailguard.diff", "w").writelines(d_tg)
print("tailguard.diff (checker file -> check file): +%d lines, -0" % len([x for x in d_tg if x.startswith("+") and not x.startswith("+++")]))
