#!/usr/bin/env python3
"""Switch off (GLM53_P4=on GLM53_P4_S0=on, GLM53_CAP0 unset): cap0_hooks is never imported, the p4 module's
cap0 handle is None, and every cap0 call site is skipped (CPU container)."""
import sys
import vllm.p4_hooks as p4
ok = p4._cap0 is None and p4._CAP0_ERR is None and "vllm.cap0_hooks" not in sys.modules and \
    "vllm.cap0_hooks_layout" not in sys.modules and p4.S0 and p4.ENABLED
print(("PASS" if ok else "FAIL") + " S1 GLM53_CAP0 unset: cap0_hooks not imported; p4._cap0 is None; stage-0 recording on")
sys.exit(0 if ok else 1)
