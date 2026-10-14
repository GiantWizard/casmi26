import pickle, sys, numpy as np, pandas as pd
res = pickle.load(open(sys.argv[1], 'rb'))
feats = ['raw_score', 'fz', 'fz_single', 'fz_merged', 'model_views__fzD_z', 'model_views__fzABD_rank', 'fp_ll']
rows = []
for key, r in res.items():
    if 'Xall' not in r: continue
    names = r['names']; X = r['Xall']; t = [i for i, k in enumerate(r['ckeys']) if k == key]
    if not t: continue
    row = dict(key=key, n=len(names and r['ckeys']))
    for f in feats:
        v = r['raw_score'] if f == 'raw_score' else X[:, names.index(f)] if f in names else None
        if v is None: continue
        if f.endswith('_rank'): v = -v      # rank_norm: 0 = best
        order = np.argsort(-v, kind='stable'); seen, rk = set(), None
        for j in order:
            k = r['ckeys'][j]
            if k in seen: continue
            seen.add(k)
            if k == key: rk = len(seen); break
        row[f] = rk
    rows.append(row)
d = pd.DataFrame(rows)
print(len(d), 'molecules; columns available:', [c for c in feats if c in d])
for f in feats:
    if f in d:
        r = d[f]; print(f'{f:<28} top1 {(r == 1).mean():.3f}  MRR@25 {np.where(r <= 25, 1 / r, 0).mean():.3f}')
