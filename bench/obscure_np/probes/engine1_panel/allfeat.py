import pickle, sys, numpy as np, pandas as pd, lightgbm as lgb
res = pickle.load(open(sys.argv[1], 'rb'))
RK = pickle.load(open(sys.argv[2], 'rb')); B = [lgb.Booster(model_str=s) for s in RK['boosters']]
imp = sum(pd.Series(b.feature_importance('gain'), index=RK['features']) for b in B); imp = imp / imp.sum()
names = next(r['names'] for r in res.values() if 'names' in r)
acc = {}
for f in RK['features']:
    j = names.index(f); hits = []
    for key, r in res.items():
        if 'Xall' not in r: continue
        v = r['Xall'][:, j]; ks = r['ckeys']
        best = v.max(); top = [k for k, x in zip(ks, v) if x == best]
        hits.append((key in top) / len(set(top)))          # expected top-1 with random tie-break (both directions below)
    hi = np.mean(hits)
    hits2 = []
    for key, r in res.items():
        if 'Xall' not in r: continue
        v = r['Xall'][:, j]; ks = r['ckeys']; best = v.min(); top = [k for k, x in zip(ks, v) if x == best]
        hits2.append((key in top) / len(set(top)))
    acc[f] = (max(hi, np.mean(hits2)), 'max' if hi >= np.mean(hits2) else 'min', imp[f])
d = pd.DataFrame(acc, index=['top1', 'dir', 'gain']).T.sort_values('top1', ascending=False)
print(d.head(25).to_string())
