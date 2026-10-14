import pickle, sys, numpy as np, pandas as pd, lightgbm as lgb
res = pickle.load(open(sys.argv[1], 'rb')); RK = pickle.load(open(sys.argv[2], 'rb'))
B = [lgb.Booster(model_str=s) for s in RK['boosters']]; FE = list(RK['features'])
def first_rank(keys, score, key):
    seen = set()
    for j in np.argsort(-score, kind='stable'):
        k = keys[j]
        if k in seen: continue
        seen.add(k)
        if k == key: return len(seen)
    return 99
mols = [(k, r) for k, r in res.items() if 'Xall' in r]
names = mols[0][1]['names']; pos = [names.index(f) for f in FE]
fam = lambda f: f.split('__')[0] if '__' in f else ('lib' if f.startswith('lib') else 'fz/fp' if f.startswith(('fz', 'fp')) else 'base:' + f.split('_')[0])
groups = pd.Series({f: fam(f) for f in FE})
def evaluate(neutral=()):
    rk = []
    for key, r in mols:
        X = r['Xall'][:, pos].copy()
        for j, f in enumerate(FE):
            if f in neutral: X[:, j] = X[:, j].mean()
        s = np.mean([b.predict(X, num_threads=4) for b in B], 0)
        rk.append(first_rank(r['ckeys'], s, key))
    rk = np.array(rk); return (rk == 1).mean(), np.where(rk <= 25, 1 / rk, 0).mean()
print('full ranker', evaluate())
fams = groups.value_counts()
coarse = {'fz/fp': 'fz/fp', 'lib': 'lib', 'derivation': 'derivation', 'frag2': 'frag2', 'analog_struct': 'analog_struct',
          'model_views': 'model_views', 'fragnet': 'fragnet'}
G = groups.map(lambda g: coarse.get(g, 'base-other'))
for g in G.unique():
    t1, m = evaluate(set(G.index[G == g]))
    print(f'flatten {g:<15} ({(G == g).sum():>3} feats): top1 {t1:.3f} MRR {m:.3f}')
print('base-other features:', list(G.index[G == 'base-other']))
