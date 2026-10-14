import pickle, sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1] + '/code')
from casmi import derive
pm = pd.read_parquet(sys.argv[2] + '/pool_meta.parquet', columns=['key', 'src', 'mass'])
fp = np.load(sys.argv[2] + '/pool_fp.npy', mmap_mode='r')
w = pickle.load(open(sys.argv[3], 'rb'))
k2i = pd.Series(np.arange(len(pm)), index=pm.key); k2i = k2i[~k2i.index.duplicated()]
POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint16)
fpa = np.asarray(fp)                                   # load once (~0.9 GB)
cnt = POP[fpa].sum(1).astype(np.float32)
rows = []
miss = [k for k, r in w.items() if 'ckeys' in r and k not in set(r['ckeys'])]
for key in miss:
    if key not in k2i: continue
    i = int(k2i[key]); q = fpa[i]
    inter = POP[np.bitwise_and(fpa, q)].sum(1).astype(np.float32)
    tan = inter / (cnt + cnt[i] - inter + 1e-9); tan[i] = 0
    for j in np.argsort(-tan)[:20]:
        if tan[j] < 0.6: break
        delta = float(pm.mass.iloc[i] - pm.mass.iloc[j])
        ok = len(derive.matching_transforms(delta, 0.004, 2)) > 0
        rows.append(dict(key=key, tan=float(tan[j]), src=int(pm.src.iloc[j]), delta=round(delta, 3), transform=ok))
d = pd.DataFrame(rows); d.to_csv(sys.argv[3] + '.poolparent.csv', index=False)
print('missed by wide generation:', len(miss))
if len(d):
    best = d[d['transform']].groupby('key').agg(tan=('tan', 'max'), any_nolib=('src', lambda s: (s == 2).any()), any_lib=('src', lambda s: s.isin([1, 3]).any()))
    print('missed answers with a transform-reachable pool relative (Tanimoto>=0.6):', len(best),
          '| of those, reachable only via a COCONUT-only (no spectra) parent:', int((best.any_nolib & ~best.any_lib).sum()),
          '| via a library structure too:', int(best.any_lib.sum()))
