import ast,re,collections,sys
ROOT="<census-work>/src/py/b12x/"
STORE_FN={'st_global_u64','st_global_u8','st_global_u16','st_global_f32','st_global_v2_f32','st_global_v2_u32','st_global_v4_f32','st_global_u32','st_global_v4_u32','atomic_add_global_i32','red_add_global_i32','red_add_global_f32','atomic_add_global_u64','red_add_global_release_i32','red_add_global_bf16x2','red_max_global_f32_nonnegative','atomic_cas_global_i32','st_global_i32','st_global_release_i32','scatter_add_bf16x2','store_bf16x2','store_v4_bf16x2','scatter_add_bf16','scatter_add_v4_bf16x2'}
files=sys.argv[1:]
def root(e):
    while True:
        if isinstance(e,ast.Call):
            if isinstance(e.func,ast.Name) and e.func.id in('get_ptr_as_int64','Int64','Int32') and e.args: e=e.args[0]; continue
            if isinstance(e.func,ast.Attribute) and e.func.attr in ('iterator','toint') : e=e.func.value; continue
            return ast.unparse(e)[:40]
        if isinstance(e,ast.BinOp): e=e.left; continue
        if isinstance(e,ast.Subscript): e=e.value; continue
        if isinstance(e,ast.Attribute): return ast.unparse(e)
        if isinstance(e,ast.Name): return e.id
        return ast.unparse(e)[:40]
c=collections.Counter(); sites=collections.defaultdict(list)
for f in files:
    s=open(ROOT+f).read(); t=ast.parse(s)
    for fn in ast.walk(t):
        if not isinstance(fn,(ast.FunctionDef,)): continue
        local=set()
        for n in ast.walk(fn):
            if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call):
                src=ast.unparse(n.value.func)
                if re.search(r'make_fragment|make_rmem|make_tensor|zeros|full|rmem|fragment|alloc|recast|allocate',src):
                    for tg in n.targets:
                        if isinstance(tg,ast.Name): local.add(tg.id)
        for n in ast.walk(fn):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in STORE_FN and n.args:
                r=root(n.args[0]); c[r]+=1; sites[r].append(f"{f}:{n.lineno}")
            if isinstance(n,ast.Assign):
                for tg in n.targets:
                    if isinstance(tg,ast.Subscript):
                        r=root(tg)
                        if r in local: continue
                        c['[]='+r]+=1; sites['[]='+r].append(f"{f}:{n.lineno}")
for k,v in sorted(c.items(), key=lambda x:-x[1]):
    print(v,k,' '.join(sites[k][:4]))
