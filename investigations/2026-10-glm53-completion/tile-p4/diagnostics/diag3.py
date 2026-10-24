import sys
sys.path.insert(0, "/p4/analysis")
import analyze_p4 as A
S = A.Session("A", A.pick_attempt("/p4", "A"))
x, y = A.WinData(S, "R01-close", 1), A.WinData(S, "R01-close", 2)
s1, s2 = int(x.m[0, x.c["step"]]), int(y.m[0, y.c["step"]])
print("first decode-measured step rep1", s1, "rep2", s2)
def show(lo, hi):
    out=[]
    for s in range(lo, hi + 1):
        h = S.steps.get(s)
        if h: out.append((s, h["num_reqs"], h["num_tokens"], h["num_tokens_padded"], h["num_draft_tokens"], h["n_prefilling"], h["graph_mode"]))
    return out
r1 = show(s1 - 16, s1); r2 = show(s2 - 16, s2)
for a, b in zip(r1, r2):
    print(a, "|", b, "" if a[1:] == b[1:] else "  <-- differs")
