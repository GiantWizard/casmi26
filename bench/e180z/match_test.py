"""Do hidden test spectra match Enveda-180 Zenodo spectra of structures missing from train.parquet?

For each test spectrum: reference spectra with the same adduct and polarity, precursor within PPM, cosine on
sqrt-intensities with greedy peak matching at TOL Da. Reports per-molecule best cosine against (a) Zenodo structures
absent from train.parquet ("new") and (b) Enveda-180 structures already in train (calibration: ~ class-1 share).
usage: match_test.py ZENODO.parquet TRAIN.parquet TEST.parquet OUT.csv
"""
import sys
import numpy as np, pandas as pd

zp, trp, tep, out = sys.argv[1:5]
PPM, TOL = 10.0, 0.01


def cosine(mz1, it1, mz2, it2):
    a = np.sqrt(np.clip(it1, 0, None)); b = np.sqrt(np.clip(it2, 0, None))
    if not len(a) or not len(b):
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    pairs = []
    j0 = 0
    for i, m in enumerate(mz1):
        while j0 < len(mz2) and mz2[j0] < m - TOL:
            j0 += 1
        j = j0
        while j < len(mz2) and mz2[j] <= m + TOL:
            pairs.append((a[i] * b[j], i, j)); j += 1
    pairs.sort(reverse=True)
    ui, uj, s = set(), set(), 0.0
    for v, i, j in pairs:
        if i not in ui and j not in uj:
            ui.add(i); uj.add(j); s += v
    return s / (na * nb)


def prep(df, mzc, itc):
    o = []
    for x, y in zip(df[mzc], df[itc]):
        x = np.asarray(x, float); y = np.asarray(y, float); k = np.argsort(x)
        o.append((x[k], y[k]))
    return o


Z = pd.read_parquet(zp)
print('zenodo columns', Z.columns.tolist(), len(Z), flush=True)
need = dict(add='adduct', pmz='precursor_mz', mz='ms2_mzs', it='ms2_normalized_intensities')
Z['adduct'] = Z.adduct.replace({'[M+FA-H]-': '[M+CH2O2-H]-'})
Z['ik14'] = Z.inchikey14
print('zenodo structures', Z.ik14.nunique(), '| absent from train', Z[Z.new].ik14.nunique(), '| spectra of new', int(Z.new.sum()), flush=True)
Z = Z.sort_values(need['pmz']).reset_index(drop=True)
zm = Z[need['pmz']].values.astype(float); zs = prep(Z, need['mz'], need['it'])
T = pd.read_parquet(tep)
ts = prep(T, 'ms2_mzs', 'ms2_normalized_intensities')
rows = []
for k, r in enumerate(T.itertuples()):
    p = float(r.precursor_mz); tol = p * PPM * 1e-6 + 0.0006
    lo, hi = np.searchsorted(zm, p - tol), np.searchsorted(zm, p + tol)
    best = {True: (0.0, None), False: (0.0, None)}
    for j in range(lo, hi):
        if str(Z.at[j, need['add']]) != r.adduct:
            continue
        c = cosine(*ts[k], *zs[j]); nw = bool(Z.at[j, 'new'])
        if c > best[nw][0]:
            best[nw] = (c, Z.at[j, 'ik14'])
    rows.append(dict(molecule_id=r.molecule_id, adduct=r.adduct, prec=p, cos_new=best[True][0], ik_new=best[True][1],
                     cos_old=best[False][0], ik_old=best[False][1]))
R = pd.DataFrame(rows); R.to_csv(out, index=False)
M = R.groupby('molecule_id')[['cos_new', 'cos_old']].max()
print(f'test molecules {len(M)}', flush=True)
for th in (0.95, 0.9, 0.8, 0.7):
    print(f'best cosine >= {th}: new-structure match {int((M.cos_new >= th).sum())} | train-structure match {int((M.cos_old >= th).sum())} '
          f'| new beats old {int(((M.cos_new >= th) & (M.cos_new > M.cos_old)).sum())}', flush=True)
