import pickle, sys, numpy as np, pandas as pd, multiprocessing as mp
sys.path.insert(0, sys.argv[1] + '/code')
from casmi import derive, chem
pm = pd.read_parquet(sys.argv[2] + '/pool_meta.parquet', columns=['key', 'smiles', 'mass', 'train_sid'])
W = pickle.load(open(sys.argv[3], 'rb'))
k2i = pd.Series(np.arange(len(pm)), index=pm.key); k2i = k2i[~k2i.index.duplicated()]
fpa = np.load(sys.argv[2] + '/pool_fp.npy')
POPC = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint16)
cnt = POPC[fpa].sum(1).astype(np.float32)
jobs = []
for key, r in W.items():
    if 'ckeys' not in r or key in set(r['ckeys']) or key not in k2i: continue
    i = int(k2i[key]); inter = POPC[np.bitwise_and(fpa, fpa[i])].sum(1).astype(np.float32)
    tan = inter / (cnt + cnt[i] - inter + 1e-9); tan[i] = 0
    for j in np.argsort(-tan)[:10]:
        if tan[j] < 0.6: break
        d = float(pm.mass.iloc[i] - pm.mass.iloc[j])
        if derive.matching_transforms(d, 0.004, 2):
            jobs.append((key, pm.smiles.iloc[j], d, float(tan[j]), int(pm.train_sid.iloc[j]) >= 0))
del fpa
def run(job):
    key, psmi, d, tan, haslib = job
    out = {}
    for cap, budget_note in ((200, 'shipped caps'), (100000, 'no caps')):
        if cap == 100000:
            old = derive.derive.__defaults__
        prods = derive.derive(psmi, d, tol=0.004, max_steps=2, max_per_analog=cap)
        out[budget_note] = any(chem.score_key(chem.standardize_smiles(p) or '') == key for p in prods)
    return key, tan, haslib, out['shipped caps'], out['no caps']
with mp.Pool(4) as p:
    res = p.map(run, jobs)
d = pd.DataFrame(res, columns=['key', 'tan', 'haslib', 'hit_cap200', 'hit_nocap'])
g = d.groupby('key').agg(hit200=('hit_cap200', 'any'), hitno=('hit_nocap', 'any'), haslib=('haslib', 'any'))
print('missed answers with reachable-mass relatives:', len(g), '| answer derivable from a relative with per-parent cap 200:',
      int(g.hit200.sum()), '| with cap 100000:', int(g.hitno.sum()), '| of derivable, relative has spectra:', int(g[g.hitno].haslib.sum()))
