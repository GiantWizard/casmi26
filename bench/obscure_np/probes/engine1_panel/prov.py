import pickle, sys, numpy as np, pandas as pd
res = pickle.load(open(sys.argv[1], 'rb'))
pm = pd.read_parquet(sys.argv[2], columns=['src'])
src = pm.src.values
def first_rank(keys, score, key):
    seen = set()
    for j in np.argsort(-score, kind='stable'):
        k = keys[j]
        if k in seen: continue
        seen.add(k)
        if k == key: return len(seen)
    return 99
rows = []
for key, r in res.items():
    if 'Xall' not in r: continue
    names = r['names']; X = r['Xall']; pid = r['cpid']; ks = r['ckeys']
    s = np.where(pid >= 0, src[np.maximum(pid, 0)], 0)          # 0 generated, 1 train-only, 2 COCONUT-only, 3 both
    is_tr = np.isin(s, [1, 3]).astype(float)
    fz = X[:, names.index('fz')]; fzn = (fz - fz.mean()) / (fz.std() + 1e-9)
    t = ks.index(key)
    rk_raw = first_rank(ks, r['raw_score'], key)
    top = int(np.argmax(r['raw_score']))
    rows.append(dict(key=key, n=len(ks), n_train=int(is_tr.sum()), n_coco=int((s == 2).sum()), n_gen=int((s == 0).sum()),
                     truth_src=int(s[t]), top_src=int(s[top]),
                     rk_ranker=rk_raw, rk_fz=first_rank(ks, fzn, key), rk_trainfirst=first_rank(ks, is_tr * 100 + fzn, key)))
d = pd.DataFrame(rows)
mrr = lambda r: np.where(r <= 25, 1 / r, 0).mean()
print('molecules', len(d), '| candidates per window: median', d.n.median(), 'train', d.n_train.median(), 'COCONUT-only', d.n_coco.median(), 'generated', d.n_gen.median())
print('truth src', d.truth_src.value_counts().to_dict(), '| ranker top-1 src', d.top_src.value_counts().to_dict())
for c in ['rk_ranker', 'rk_fz', 'rk_trainfirst']:
    print(f'{c:<15} top1 {(d[c] == 1).mean():.3f}  MRR {mrr(d[c]):.3f}')
