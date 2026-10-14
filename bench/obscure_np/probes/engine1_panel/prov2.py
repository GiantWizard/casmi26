import pickle, sys, numpy as np, pandas as pd
res = pickle.load(open(sys.argv[1], 'rb')); src = pd.read_parquet(sys.argv[2], columns=['src']).src.values
def first_rank(keys, score, key):
    seen = set()
    for j in np.argsort(-score, kind='stable'):
        k = keys[j]
        if k in seen: continue
        seen.add(k)
        if k == key: return len(seen)
    return 99
rk, n3, feats = [], [], []
names = None
for key, r in res.items():
    if 'Xall' not in r: continue
    names = r['names']; X = r['Xall']; pid = r['cpid']; ks = r['ckeys']
    s = np.where(pid >= 0, src[np.maximum(pid, 0)], 0)
    fz = X[:, names.index('fz')]; fzn = (fz - fz.mean()) / (fz.std() + 1e-9)
    rk.append(first_rank(ks, (s == 3) * 100 + fzn, key)); n3.append(int((s == 3).sum()))
    t = ks.index(key)
    for j in range(len(ks)):
        if j == t: continue
        feats.append(np.r_[s[j], X[j]])
rk = np.array(rk); mrr = np.where(rk <= 25, 1 / rk, 0).mean()
print(f'src3-first + fz: top1 {(rk == 1).mean():.3f} MRR {mrr:.3f} | src-3 candidates per window median {np.median(n3)}')
F = pd.DataFrame(feats, columns=['src'] + names)
a, b = F[F.src == 3], F[F.src == 2]
sep = {}
for c in names:
    va, vb = a[c].values, b[c].values
    sd = np.sqrt((va.var() + vb.var()) / 2) + 1e-9
    sep[c] = (va.mean() - vb.mean()) / sd
sep = pd.Series(sep).sort_values(key=abs, ascending=False)
print('non-truth candidates: features separating train∩COCONUT (src 3) from COCONUT-only (src 2), standardized mean diff:')
print(sep.head(20).round(2).to_string())
