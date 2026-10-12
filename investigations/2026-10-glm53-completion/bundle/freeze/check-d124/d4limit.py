# SPDX-License-Identifier: Apache-2.0
"""d4limit: VERIFICATION-IMAGE-ONLY override of the routed-expert swiglu_limit that the EXL3
runtime planners read (D4 real-weight reachability check, PLAN.md amendment 14 (3)).

Inert unless GLM53_D4_LIMIT_OVERRIDE or GLM53_D4_INSPECT is set at process start:
  GLM53_D4_LIMIT_OVERRIDE="1" / "1.0" / any positive finite float -> the planners see that limit
  GLM53_D4_LIMIT_OVERRIDE="none"                                   -> the planners see no limit (release behaviour)
  GLM53_D4_INSPECT=1  -> no override; once per new runtime, a bounded host-side walk of the runtime's plan objects
                         logs every attribute named swiglu_limit / has_swiglu_limit with its owner class
                         (the compiled kernels carry the limit as a compile-time constant: kernel.py:6281-6282)
Mechanism: wraps Exl3MoEMethod._rank_sliced_runtime (projection-mixed, the K3.25 path) and
_mixed_rank_sliced_runtime (expert-mixed): for the duration of each call the layer's swiglu_limit
attribute is the override, then the original attribute is restored. With the D4 patch the planners read
getattr(layer, "swiglu_limit", None) inside these two methods only (exl3.py:2754 and :2972 of the bundle),
so the B12x plans and the compiled kernels carry the override; no engine file is changed, nothing runs
inside a captured graph (host attribute writes only, in eager Python). Installed from a guarded block
appended to gate_linear.py in the verification image; at the first GateLinear construction, before any
MoE forward plans a runtime. One JSON line per process goes to $GLM53_D4_LIMIT_DIR/d4limit_p<pid>.jsonl.
"""
from __future__ import annotations

import json
import math
import os

_RAW = os.environ.get("GLM53_D4_LIMIT_OVERRIDE", "").strip()
OVERRIDE = _RAW != ""
INSPECT = os.environ.get("GLM53_D4_INSPECT", "").strip() == "1"
ENABLED = OVERRIDE or INSPECT
OUT_DIR = os.environ.get("GLM53_D4_LIMIT_DIR", "/tmp/d4limit")
_ABSENT = object()


def parse(raw: str):
    if raw.lower() == "none":
        return None
    v = float(raw)
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"GLM53_D4_LIMIT_OVERRIDE must be 'none' or a positive finite number, got {raw!r}")
    return v


VALUE = parse(_RAW) if OVERRIDE else None
STATE = {"installed": False, "calls": {}, "seen_original": set(), "logged": set(), "inspected": set()}
_NAMES = ("swiglu_limit", "has_swiglu_limit")


def walk_limits(root, max_depth: int = 7, max_nodes: int = 20000) -> list:
    """Bounded BFS over host objects (attributes, dicts, lists, tuples); tensors are not touched.
    Returns [(path, owner_class, name, value)] for every attribute named in _NAMES."""
    import collections
    found, seen, q, n = [], set(), collections.deque([(root, "rt", 0)]), 0
    while q and n < max_nodes:
        obj, path, d = q.popleft()
        if id(obj) in seen or obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            continue
        seen.add(id(obj))
        n += 1
        tname = type(obj).__module__ + "." + type(obj).__qualname__
        if tname.startswith("torch.") or "Tensor" in tname or "Parameter" in tname:
            continue
        for nm in _NAMES:
            try:
                if nm in getattr(obj, "__dict__", {}) or (hasattr(type(obj), "__slots__") and nm in getattr(type(obj), "__slots__", ())):
                    found.append((path, type(obj).__name__, nm, getattr(obj, nm)))
            except Exception:  # noqa: BLE001
                pass
        if d >= max_depth:
            continue
        kids = []
        if isinstance(obj, dict):
            kids = [(v, f"{path}[{k!r}]") for k, v in list(obj.items())[:256]]
        elif isinstance(obj, (list, tuple, set, frozenset)):
            kids = [(v, f"{path}[{i}]") for i, v in enumerate(list(obj)[:256])]
        else:
            attrs = dict(getattr(obj, "__dict__", {}) or {})
            for sl in getattr(type(obj), "__slots__", ()) or ():
                try:
                    attrs[sl] = getattr(obj, sl)
                except Exception:  # noqa: BLE001
                    pass
            kids = [(v, f"{path}.{k}") for k, v in list(attrs.items())[:256]]
        for v, pth in kids:
            q.append((v, pth, d + 1))
    return found


def _log(rec: dict) -> None:
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, f"d4limit_p{os.getpid()}.jsonl"), "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _wrap(cls, name: str) -> None:
    orig = cls.__dict__[name]

    def wrapped(self, layer, *a, **kw):
        old = getattr(layer, "swiglu_limit", _ABSENT)
        key = (name, repr(old))
        if key not in STATE["logged"]:
            STATE["logged"].add(key)
            _log(dict(kind="override_applied", method=name, original=None if old is _ABSENT else old,
                      original_absent=old is _ABSENT, override=VALUE, pid=os.getpid()))
        STATE["calls"][name] = STATE["calls"].get(name, 0) + 1
        if OVERRIDE:
            layer.swiglu_limit = VALUE
        try:
            rt = orig(self, layer, *a, **kw)
            if INSPECT and id(rt) not in STATE["inspected"]:
                STATE["inspected"].add(id(rt))
                try:
                    hits = walk_limits(rt)
                    _log(dict(kind="plan_limits", method=name, runtime_id=id(rt), layer_limit=None if old is _ABSENT else old,
                              n_hits=len(hits), values=sorted({repr(h[3]) for h in hits}),
                              owners=sorted({f"{h[1]}.{h[2]}={h[3]!r}" for h in hits})[:64], pid=os.getpid()))
                except Exception as exc:  # noqa: BLE001
                    _log(dict(kind="plan_limits_error", error=repr(exc), pid=os.getpid()))
            return rt
        finally:
            if OVERRIDE:
                if old is _ABSENT:
                    try:
                        delattr(layer, "swiglu_limit")
                    except AttributeError:
                        pass
                else:
                    layer.swiglu_limit = old

    wrapped.__name__ = name
    wrapped.__qualname__ = f"{cls.__name__}.{name}"
    wrapped._d4limit_orig = orig
    setattr(cls, name, wrapped)


def install(Exl3MoEMethod=None) -> None:
    if not ENABLED or STATE["installed"]:
        return
    if Exl3MoEMethod is None:
        from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod  # noqa: PLC0415
    for name in ("_rank_sliced_runtime", "_mixed_rank_sliced_runtime"):
        if name in Exl3MoEMethod.__dict__:
            _wrap(Exl3MoEMethod, name)
    STATE["installed"] = True
    _log(dict(kind="installed", override_on=OVERRIDE, override=VALUE, raw=_RAW, inspect=INSPECT, pid=os.getpid()))


def install_on_gate(GateLinear) -> None:
    """Called from the guarded block appended to gate_linear.py: defer install() to the first GateLinear
    construction (model build, worker process, before any MoE forward)."""
    if not ENABLED or getattr(GateLinear, "_d4limit", False):
        return
    orig_init = GateLinear.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        try:
            install()
        except Exception as exc:  # noqa: BLE001
            _log(dict(kind="install_error", error=repr(exc), pid=os.getpid()))
            raise

    GateLinear.__init__ = __init__
    GateLinear._d4limit = True
