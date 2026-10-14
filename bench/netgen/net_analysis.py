"""Idea 2 groundwork: within one test-like set, do spectrally related molecules turn out to be structural relatives?

For every pair of molecules in a set: best modified cosine over their spectra (same polarity; peaks matched directly
or shifted by the precursor mass difference), precursor mass difference, and Tanimoto (Morgan r2, 2048) of the true
structures. Reports how many molecules have a relative at all, and P(relative | spectral link) per link threshold.

usage: net_analysis.py SOLUTION.csv TEST_DRAW.parquet
"""
import sys, itertools
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')

sol = pd.read_csv(sys.argv[1]); te = pd.read_parquet(sys.argv[2])
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = {r.molecule_id: gen.GetFingerprint(Chem.MolFromSmiles(r.smiles)) for r in sol.itertuples() if Chem.MolFromSmiles(r.smiles)}
TOL = 0.01


def prep(mz, it, prec):
    m = np.asarray(mz, float); i = np.asarray(it, float)
    keep = m < prec - 0.5
    m, i = m[keep], i[keep]
    if len(m) == 0:
        return m, i
    o = np.argsort(-i)[:60]; m, i = m[o], np.sqrt(i[o]); i = i / (np.linalg.norm(i) + 1e-12)
    return m, i


def modcos(a, b, dm):
    (m1, i1), (m2, i2) = a, b
    if len(m1) == 0 or len(m2) == 0:
        return 0.0
    pairs = []
    for shift in (0.0, dm):
        d = np.abs(m1[:, None] - (m2[None, :] + shift))
        for x, y in zip(*np.where(d < TOL)):
            pairs.append((i1[x] * i2[y], x, y))
    pairs.sort(reverse=True); u1, u2, s = set(), set(), 0.0
    for v, x, y in pairs:
        if x not in u1 and y not in u2:
            u1.add(x); u2.add(y); s += v
    return s


spec = {}
for r in te.itertuples():
    spec.setdefault(r.molecule_id, []).append((r.ionization_mode, r.precursor_mz, prep(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)))
mids = [m for m in sol.molecule_id if m in fp and m in spec]
rows = []
for a, b in itertools.combinations(mids, 2):
    best, dmb = 0.0, None
    for ma, pa, sa in spec[a]:
        for mb, pb, sb in spec[b]:
            if ma != mb:
                continue
            dm = pa - pb
            s = modcos(sa, sb, dm)
            if s > best:
                best, dmb = s, dm
    if best <= 0:
        continue
    rows.append((a, b, best, dmb, DataStructs.TanimotoSimilarity(fp[a], fp[b])))
P = pd.DataFrame(rows, columns=['a', 'b', 'modcos', 'dm', 'tani'])
n = len(mids)
rel = P[P.tani >= 0.6]
has_rel = set(rel.a) | set(rel.b)
print(f'molecules {n} | pairs with any shared peak {len(P)} | molecules with a structural relative (Tanimoto >= 0.6) in the set: '
      f'{len(has_rel)} ({len(has_rel) / n:.1%})')
for th in (0.5, 0.6, 0.7, 0.8):
    L = P[P.modcos >= th]
    lm = set(L.a) | set(L.b)
    print(f'modcos >= {th}: links {len(L)}, molecules linked {len(lm)} ({len(lm) / n:.1%}), '
          f'P(Tanimoto >= 0.6 | link) {np.mean(L.tani >= 0.6) if len(L) else float("nan"):.2f}, median Tanimoto {L.tani.median():.2f}')
P.to_parquet(sys.argv[1].replace('solution.csv', 'pairs.parquet'))
