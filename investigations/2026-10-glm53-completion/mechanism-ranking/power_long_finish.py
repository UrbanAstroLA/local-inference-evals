"""Rough power of a stock-0.7.0 replication for the long-reply finishing contrast (exploratory; inferred, pure Python).
Model: replies reaching 100k in a 3-pass replication ~ Poisson(21) (0.7.0 record: 7.0 per pass, amendment 19f);
each finishes with probability p_true, independently (no question clustering, so optimistic).
Comparator: the stock 0.9.1 record as observed, 10 of 25 finished (amendment 20c). Test: one-sided Fisher exact, 0.05."""
import math, random
def fisher_greater(a, b, c, d):            # P(X >= a) under the hypergeometric with fixed margins
    n1, n2, k = a + b, c + d, a + c
    tot = math.comb(n1 + n2, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / tot
def poisson(lam, r):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= r.random()
        if p <= L: return k
        k += 1
r = random.Random(20261015)
for p in (0.71, 0.55, 0.40):
    N, hits = 4000, 0
    for _ in range(N):
        n = poisson(21, r); kk = sum(r.random() < p for _ in range(n))
        if n and fisher_greater(kk, n - kk, 10, 15) < 0.05: hits += 1
    print(f"p_true {p:.2f}: P(one-sided Fisher p < 0.05 against 0.9.1's 10/25) = {hits / N:.2f}")
