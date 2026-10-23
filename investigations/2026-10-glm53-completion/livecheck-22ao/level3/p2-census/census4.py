#!/usr/bin/env python3
"""P2 writer census v2 (import/class-resolved taint propagation).

Carrier = any expression that holds the raw KV allocation or a view/pointer of it.
Seeds: names kv_cache, kv_caches, kv_cache_raw_tensors, runner_kv_caches, kv_caches_dict,
kv_raw_tensor, packed_backing; attributes .kv_cache, .kv_caches (any object).
Propagation: assignments / unpacking / for / with / walrus; subscripts; view-like methods
(VIEW_METHODS) on a carrier; torch view functions; calls to *resolved* in-scope defs whose
return is a carrier; `self.X = carrier` taints self.X within that class; carrier call args
taint the parameters of the *resolved* callee (fixpoint).
Callee resolution: same-file def, `from m import f [as g]`, `import m as a; a.f`,
`self.f` / `cls.f` in the enclosing class or its in-scope bases, ClassName(...) -> __init__,
kernel[grid](...) -> kernel; other `obj.f(...)` resolve by name only if the name has <=2
defs in scope and is not generic; otherwise the sink is kept and marked unresolved.
Sinks: every call with a carrier argument, every in-place method on a carrier, every
subscript store / aug-assign on a carrier.
"""
import ast, os, re, sys, json, collections

ROOT = "<census-work>/src/py"
OUT = "<census-work>/out"
os.makedirs(OUT, exist_ok=True)

def glob_re(p):
    r = re.escape(p).replace(r"\*\*", "\0").replace(r"\*", "[^/]*").replace("\0", ".*")
    return re.compile("^" + r + "$")

INCLUDE = [glob_re(p) for p in [
    "vllm/v1/worker/gpu/**", "vllm/v1/worker/*.py",
    "vllm/v1/attention/*.py",
    "vllm/v1/attention/backends/utils.py", "vllm/v1/attention/backends/registry.py",
    "vllm/v1/attention/backends/flash_attn.py", "vllm/v1/attention/backends/fa_utils.py",
    "vllm/v1/attention/backends/gdn_attn.py", "vllm/v1/attention/backends/mamba_attn.py",
    "vllm/v1/attention/backends/mla/b12x_mla_sparse.py",
    "vllm/v1/attention/backends/mla/b12x_dcp_topk.py",
    "vllm/v1/attention/backends/mla/indexer.py",
    "vllm/v1/attention/backends/mla/sparse_utils.py",
    "vllm/model_executor/layers/attention/**",
    "vllm/model_executor/layers/attention_layer_base.py",
    "vllm/model_executor/layers/mla.py", "vllm/model_executor/layers/mla_cache_format.py",
    "vllm/model_executor/layers/sparse_attn_indexer_kpool.py",
    "vllm/model_executor/layers/sparse_attn_indexer.py",
    "vllm/model_executor/layers/b12x_preparation.py",
    "vllm/model_executor/layers/mamba/abstract.py",
    "vllm/model_executor/layers/mamba/mamba_utils.py",
    "vllm/model_executor/layers/mamba/gdn/**", "vllm/model_executor/layers/mamba/ops/*.py",
    "vllm/model_executor/warmup/**",
    "vllm/models/glm5next/**", "vllm/models/common/**",
    "vllm/model_executor/models/qwen3_dflash.py", "vllm/model_executor/models/qwen3_dflash2.py",
    "vllm/model_executor/models/qwen3.py",
    "vllm/third_party/flash_linear_attention/**",
    "vllm/forward_context.py", "vllm/compilation/**", "vllm/_custom_ops.py",
    "vllm/vllm_flash_attn/*.py",
    "b12x/**",
]]
EXCLUDE = [(glob_re(p), why) for p, why in [
    ("vllm/v1/worker/gpu_model_runner.py", "V1 runner (served: V2, server.log:61)"),
    ("vllm/v1/worker/cpu_*", "CPU platform"), ("vllm/v1/worker/xpu_*", "XPU platform"),
    ("vllm/v1/worker/tpu_*", "TPU platform"),
    ("vllm/v1/worker/gpu/spec_decode/autoregressive/**", "speculator not selected"),
    ("vllm/v1/worker/gpu/spec_decode/dspark/**", "speculator not selected"),
    ("vllm/v1/worker/gpu/spec_decode/eagle/**", "speculator not selected"),
    ("vllm/v1/worker/gpu/spec_decode/gemma4/**", "speculator not selected"),
    ("vllm/v1/worker/gpu/spec_decode/mtp/**", "speculator not selected"),
    ("vllm/v1/worker/gpu/spec_decode/multi_module_mtp/**", "speculator not selected"),
]]

def in_scope(rel):
    if any(r.match(rel) for r, _ in EXCLUDE):
        return False
    return any(r.match(rel) for r in INCLUDE)

SEED_NAMES = {"kv_cache", "kv_caches", "kv_cache_raw_tensors", "runner_kv_caches",
              "kv_caches_dict", "kv_raw_tensor", "packed_backing"}
SEED_ATTRS = {"kv_cache", "kv_caches"}
NONTENSOR_ATTRS = {"shape", "dtype", "device", "ndim", "is_cuda", "nbytes", "layout",
                   "requires_grad", "itemsize"}
NONTENSOR_METHODS = {"size", "stride", "numel", "dim", "element_size", "is_contiguous",
                     "storage_offset", "get_device", "nelement", "type", "keys", "__len__",
                     "is_shared", "is_pinned", "tolist", "item", "any", "all", "sum",
                     "isfinite", "isnan", "eq", "ne", "equal", "count_nonzero", "cpu",
                     "clone", "float", "to", "detach", "nbytes", "mean", "abs", "max", "min"}
VIEW_METHODS = {"view", "reshape", "permute", "transpose", "select", "narrow", "flatten",
                "unflatten", "squeeze", "unsqueeze", "contiguous", "as_strided", "chunk",
                "split", "unbind", "view_as", "unfold", "expand", "movedim", "t",
                "untyped_storage", "data_ptr", "values", "items", "get", "__getitem__",
                "mT", "diagonal", "split_with_sizes", "tensor_split", "swapaxes", "swapdims"}
TORCH_VIEW_FUNCS = {"as_strided", "narrow", "select", "split", "chunk", "unbind", "flatten",
                    "reshape", "transpose", "permute", "squeeze", "unsqueeze", "movedim",
                    "tuple", "list", "zip", "enumerate", "reversed", "iter", "next", "cast",
                    "from_dlpack", "as_tensor", "tensor", "asarray", "array", "int", "sorted",
                    "set", "dict", "frozenset", "chain", "from_iterable", "iprod", "product",
                    "MappingProxyType"}
INPLACE = {"copy_", "fill_", "zero_", "index_put_", "index_copy_", "index_fill_", "scatter_",
           "scatter_add_", "scatter_reduce_", "masked_fill_", "masked_scatter_", "put_",
           "add_", "sub_", "mul_", "div_", "clamp_", "clamp_max_", "clamp_min_", "set_",
           "as_strided_", "random_", "uniform_", "normal_", "__setitem__", "nan_to_num_",
           "resize_", "lerp_", "addcmul_", "addcdiv_", "squeeze_", "unsqueeze_"}
CONTAINER_ADD = {"set_", "append", "extend", "insert", "add", "update", "setdefault", "__setitem__", "appendleft"}
GENERIC = {"__init__", "__call__", "forward", "run", "apply", "compile", "launch", "build",
           "get", "update", "append", "extend", "call", "execute", "kernel", "_kernel",
           "wrapper", "fn", "func", "main", "step", "load", "store", "prepare"}

files = []
for dp, dn, fn in os.walk(ROOT):
    for f in fn:
        if f.endswith(".py"):
            files.append(os.path.relpath(os.path.join(dp, f), ROOT))
files.sort()
scope = [f for f in files if in_scope(f)]
scope_set = set(scope)

def modname(rel):
    m = rel[:-3].replace("/", ".")
    return m[:-9] if m.endswith(".__init__") else m
mod2rel = {modname(r): r for r in scope}

trees, srcs = {}, {}
for rel in scope:
    s = open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace").read()
    srcs[rel] = s
    try:
        trees[rel] = ast.parse(s)
    except SyntaxError as e:
        print("PARSE-FAIL", rel, e, file=sys.stderr)

# ---- indices ----
top_defs = collections.defaultdict(dict)     # rel -> name -> node
classes = collections.defaultdict(dict)      # rel -> cls -> ClassDef
class_by_name = collections.defaultdict(list)  # cls -> [(rel, ClassDef)]
defs_by_name = collections.defaultdict(list)   # name -> [(rel, node)]
parent_class = {}                              # id(funcnode) -> (rel, ClassDef)
imports = collections.defaultdict(dict)        # rel -> alias -> ("mod", modname) | ("sym", modname, name)

for rel, t in trees.items():
    pkg = modname(rel).rsplit(".", 1)[0] if not rel.endswith("__init__.py") else modname(rel)
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_defs[rel][n.name] = n
    for n in ast.walk(t):
        if isinstance(n, ast.ClassDef):
            classes[rel][n.name] = n
            class_by_name[n.name].append((rel, n))
            for m in n.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    parent_class[id(m)] = (rel, n)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs_by_name[n.name].append((rel, n))
        if isinstance(n, ast.Import):
            for a in n.names:
                imports[rel][a.asname or a.name.split(".")[0]] = ("mod", a.name if a.asname else a.name.split(".")[0])
        if isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                parts = pkg.split(".")
                parts = parts[: len(parts) - (n.level - 1)] if n.level > 1 else parts
                base = ".".join(parts + ([n.module] if n.module else []))
            for a in n.names:
                full = base + "." + a.name
                if full in mod2rel:
                    imports[rel][a.asname or a.name] = ("mod", full)
                else:
                    imports[rel][a.asname or a.name] = ("sym", base, a.name)

def resolve_sym(mod, name, depth=0):
    rel = mod2rel.get(mod)
    if rel is None or depth > 4:
        return []
    if name in top_defs[rel]:
        return [(rel, top_defs[rel][name])]
    if name in classes[rel]:
        return class_init(rel, classes[rel][name])
    imp = imports[rel].get(name)
    if imp and imp[0] == "sym":
        return resolve_sym(imp[1], imp[2], depth + 1)
    return []

def class_methods(rel, cdef, name, depth=0):
    for m in cdef.body:
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == name:
            return [(rel, m)]
    if depth > 6:
        return []
    out = []
    for b in cdef.bases:
        bn = b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else None)
        if bn is None:
            continue
        for r2, c2 in class_by_name.get(bn, []):
            out += class_methods(r2, c2, name, depth + 1)
        if out:
            break
    return out

def class_init(rel, cdef):
    return class_methods(rel, cdef, "__init__")

def resolve_call(rel, func, encl):
    """Return (list of (rel,def), resolved_flag)."""
    if isinstance(func, ast.Subscript):
        return resolve_call(rel, func.value, encl)
    if isinstance(func, ast.Name):
        nm = func.id
        if nm in top_defs[rel]:
            return [(rel, top_defs[rel][nm])], True
        if nm in classes[rel]:
            return class_init(rel, classes[rel][nm]), True
        imp = imports[rel].get(nm)
        if imp and imp[0] == "sym":
            r = resolve_sym(imp[1], imp[2])
            return r, bool(r)
        # nested defs in the enclosing function
        if encl is not None and not isinstance(encl, ast.Module):
            for n in ast.walk(encl):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == nm and n is not encl:
                    return [(rel, n)], True
        return [], False
    if isinstance(func, ast.Attribute):
        v = func.value
        if isinstance(v, ast.Name) and v.id in ("self", "cls") and encl is not None and id(encl) in parent_class:
            r2, c = parent_class[id(encl)]
            r = class_methods(r2, c, func.attr)
            if r:
                return r, True
        if isinstance(v, ast.Name):
            imp = imports[rel].get(v.id)
            if imp and imp[0] == "mod":
                r = resolve_sym(imp[1], func.attr)
                return r, bool(r)
        cands = defs_by_name.get(func.attr, [])
        if func.attr not in GENERIC and 0 < len(cands) <= 2:
            return cands, True
        return [], False
    if isinstance(func, ast.Call):
        return resolve_call(rel, func.func, encl)
    return [], False

def callee_name(func):
    if isinstance(func, ast.Name): return func.id
    if isinstance(func, ast.Attribute): return func.attr
    if isinstance(func, (ast.Subscript,)): return callee_name(func.value)
    if isinstance(func, ast.Call): return callee_name(func.func)
    return "?"

def key(rel, node):
    return (rel, getattr(node, "lineno", 0), getattr(node, "name", "<module>"))

param_taint = collections.defaultdict(set)
param_origin = {}
cls_attr_taint = collections.defaultdict(set)   # (rel, clsname) -> attrs
cls_attr_origin = {}
ret_taint = set()
GLOBAL_ATTRS = set()
GENERIC_ATTRS = {'data','value','values','state','buf','out','output','tensor','cache','args','x','t','_meta','meta'}

class Scan:
    def __init__(self, rel, node):
        self.rel, self.node = rel, node
        self.t = set(param_taint.get(key(rel, node), ()))
        self.cls = parent_class.get(id(node))
        self.changed = False

    def cattrs(self):
        if self.cls is None:
            return set()
        r, c = self.cls
        out = set(cls_attr_taint.get((r, c.name), ()))
        return out

    def tainted(self, e):
        if e is None: return False
        if isinstance(e, ast.Name):
            return e.id in self.t or e.id in SEED_NAMES
        if isinstance(e, ast.Attribute):
            if e.attr in NONTENSOR_ATTRS: return False
            if e.attr in SEED_ATTRS: return True
            if isinstance(e.value, ast.Name) and e.value.id == "self" and e.attr in self.cattrs():
                return True
            if e.attr in GLOBAL_ATTRS:
                return True
            return e.attr in VIEW_METHODS | {"T", "data", "mT"} and self.tainted(e.value)
        if isinstance(e, (ast.Subscript, ast.Starred)):
            return self.tainted(e.value)
        if isinstance(e, (ast.Tuple, ast.List, ast.Set)):
            return any(self.tainted(x) for x in e.elts)
        if isinstance(e, ast.Dict):
            return any(self.tainted(v) for v in e.values)
        if isinstance(e, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            return self.tainted(e.elt) or any(self.tainted(g.iter) for g in e.generators)
        if isinstance(e, ast.DictComp):
            return self.tainted(e.value) or any(self.tainted(g.iter) for g in e.generators)
        if isinstance(e, ast.IfExp):
            return self.tainted(e.body) or self.tainted(e.orelse)
        if isinstance(e, ast.BoolOp):
            return any(self.tainted(v) for v in e.values)
        if isinstance(e, ast.NamedExpr):
            return self.tainted(e.value)
        if isinstance(e, ast.BinOp):
            return self.tainted(e.left) or self.tainted(e.right)
        if isinstance(e, ast.Call):
            f = e.func
            if isinstance(f, ast.Attribute):
                if f.attr in NONTENSOR_METHODS: return False
                if f.attr in VIEW_METHODS and self.tainted(f.value): return True
            cn = callee_name(f)
            anyarg = any(self.tainted(a) for a in e.args) or any(self.tainted(k.value) for k in e.keywords)
            if cn in TORCH_VIEW_FUNCS and anyarg: return True
            r, ok = resolve_call(self.rel, f, self.node)
            if ok and any(key(r2, d) in ret_taint for r2, d in r):
                return True
            return False
        return False

    def bind(self, tgt, v):
        if not v: return
        if isinstance(tgt, ast.Name):
            self.t.add(tgt.id)
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            for x in tgt.elts: self.bind(x, True)
        elif isinstance(tgt, ast.Starred):
            self.bind(tgt.value, True)
        elif isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self" and self.cls:
            r, c = self.cls
            k2 = (r, c.name)
            if tgt.attr not in cls_attr_taint[k2] and tgt.attr not in NONTENSOR_ATTRS:
                cls_attr_taint[k2].add(tgt.attr)
                cls_attr_origin[(r, c.name, tgt.attr)] = f"{self.rel}:{tgt.lineno}"
                self.changed = True

    def own_nodes(self):
        out = []
        stack = list(ast.iter_child_nodes(self.node))
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(n, ast.ClassDef):
                if isinstance(self.node, ast.Module):
                    stack.extend(x for x in n.body if not isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)))
                continue
            out.append(n)
            stack.extend(ast.iter_child_nodes(n))
        return out

    def run(self):
        nodes = self.own_nodes()
        for _ in range(8):
            b = len(self.t)
            for n in nodes:
                if isinstance(n, ast.Assign):
                    v = self.tainted(n.value)
                    for tg in n.targets: self.bind(tg, v)
                elif isinstance(n, ast.AnnAssign) and n.value is not None:
                    self.bind(n.target, self.tainted(n.value))
                elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
                    self.bind(n.target, self.tainted(n.iter))
                elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                    self.bind(n.optional_vars, self.tainted(n.context_expr))
                elif isinstance(n, ast.NamedExpr):
                    self.bind(n.target, self.tainted(n.value))
                if isinstance(n, ast.Assign) and self.tainted(n.value):
                    for tg in n.targets:
                        if isinstance(tg, ast.Subscript):
                            root = tg.value
                            while isinstance(root, ast.Subscript): root = root.value
                            self.bind(root, True)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in CONTAINER_ADD:
                    if any(self.tainted(a) for a in n.args) or any(self.tainted(k.value) for k in n.keywords):
                        root = n.func.value
                        while isinstance(root, ast.Subscript): root = root.value
                        self.bind(root, True)
            if len(self.t) == b: break
        k = key(self.rel, self.node)
        if isinstance(self.node, (ast.FunctionDef, ast.AsyncFunctionDef)) and k not in ret_taint:
            for n in nodes:
                if isinstance(n, ast.Return) and self.tainted(n.value):
                    ret_taint.add(k); self.changed = True; break
        self.calls = []
        for n in nodes:
            if not isinstance(n, ast.Call): continue
            targs = [i for i, a in enumerate(n.args) if self.tainted(a)]
            tkws = [kw.arg for kw in n.keywords if kw.arg and self.tainted(kw.value)]
            if not targs and not tkws: continue
            r, ok = resolve_call(self.rel, n.func, self.node)
            self.calls.append((n, r, ok))
            is_attr = isinstance(n.func, ast.Attribute) or (isinstance(n.func, ast.Name) and any(d.name == "__init__" for _, d in r))
            for r2, d in r:
                params = [a.arg for a in d.args.posonlyargs + d.args.args]
                off = 1 if (params and params[0] in ("self", "cls") and is_attr) else 0
                kk = key(r2, d)
                for i in targs:
                    if any(isinstance(a, ast.Starred) for a in n.args[: i + 1]):
                        p = d.args.vararg.arg if d.args.vararg else None
                    elif i + off < len(params):
                        p = params[i + off]
                    else:
                        p = d.args.vararg.arg if d.args.vararg else None
                    if p and p not in param_taint[kk]:
                        param_taint[kk].add(p); param_origin[(kk, p)] = f"{self.rel}:{n.lineno}"; self.changed = True
                allp = set(params) | {a.arg for a in d.args.kwonlyargs}
                for kw in tkws:
                    p = kw if kw in allp else (d.args.kwarg.arg if d.args.kwarg else None)
                    if p and p not in param_taint[kk]:
                        param_taint[kk].add(p); param_origin[(kk, p)] = f"{self.rel}:{n.lineno}"; self.changed = True
        return self.changed

def all_units():
    for rel, t in trees.items():
        yield rel, t
        for n in ast.walk(t):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield rel, n

for it in range(60):
    ch = False
    for v in list(cls_attr_taint.values()):
        for a in v:
            if a not in GENERIC_ATTRS and a not in GLOBAL_ATTRS:
                GLOBAL_ATTRS.add(a); ch = True
    for rel, node in all_units():
        if Scan(rel, node).run(): ch = True
    if not ch: break
print("fixpoint iterations", it + 1, file=sys.stderr)

def seg(rel, n):
    try: return (ast.get_source_segment(srcs[rel], n) or "").replace("\n", " ")
    except Exception: return ""

sinks = []
for rel, node in all_units():
    s = Scan(rel, node); s.run()
    fn = getattr(node, "name", "<module>")
    for n in s.own_nodes():
        if isinstance(n, ast.Call):
            f = n.func; cn = callee_name(f)
            recv = isinstance(f, ast.Attribute) and s.tainted(f.value)
            targ = [i for i, a in enumerate(n.args) if s.tainted(a)]
            tkw = [kw.arg or "**" for kw in n.keywords if s.tainted(kw.value)]
            if not targ and not tkw:
                if recv and cn in INPLACE: kind = "inplace"
                else: continue
            else:
                if cn in TORCH_VIEW_FUNCS | VIEW_METHODS | NONTENSOR_METHODS | {"isinstance", "len", "id", "hasattr", "getattr", "print", "type", "repr", "str", "format", "debug", "info", "warning", "info_once", "warning_once", "debug_once", "assert_", "setdefault"}:
                    continue
                kind = "inplace" if (recv and cn in INPLACE) else "arg"
            r, ok = resolve_call(rel, f, node)
            sinks.append(dict(file=rel, line=n.lineno, fn=fn, callee=cn, kind=kind,
                              resolved=[f"{r2}:{d.lineno}" for r2, d in r][:3] if ok else [],
                              args=[seg(rel, n.args[i])[:50] for i in targ] + [k + "=" for k in tkw],
                              text=seg(rel, n)[:150]))
        elif isinstance(n, (ast.Assign, ast.AugAssign)):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in tg:
                hit = next((x for x in ast.walk(t) if isinstance(x, ast.Subscript) and s.tainted(x.value)), None)
                if hit is not None:
                    sinks.append(dict(file=rel, line=n.lineno, fn=fn, callee="[]=", kind="store",
                                      resolved=[], args=[seg(rel, hit)[:50]], text=seg(rel, n)[:150]))
                    break
u = {}
for x in sinks: u[(x["file"], x["line"], x["callee"], x["kind"])] = x
sinks = sorted(u.values(), key=lambda x: (x["file"], x["line"]))
json.dump(dict(global_attrs=sorted(GLOBAL_ATTRS), n_files_total=len(files), scope=scope, sinks=sinks, ret_taint=sorted(map(str, ret_taint)),
               cls_attr_taint={f"{k[0]}:{k[1]}": sorted(v) for k, v in cls_attr_taint.items() if v},
               param_taint={f"{k[0]}:{k[1]}:{k[2]}": sorted(v) for k, v in param_taint.items() if v},
               param_origin={f"{k[0][0]}:{k[0][1]}:{k[0][2]}|{k[1]}": v for k, v in param_origin.items()}),
          open(os.path.join(OUT, "census4.json"), "w"), indent=1)
print("files_total", len(files), "scope", len(scope), "sinks", len(sinks),
      "tainted_funcs", sum(1 for v in param_taint.values() if v), file=sys.stderr)
