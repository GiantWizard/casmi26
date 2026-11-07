"""Refinement: penalise seen candidates whose own training spectra do not match the query.
Best cosine (sqrt intensity, 0.01 Da, greedy one-to-one) between the query's MSG spectra and the candidate's own
train.parquet spectra (positive mode), ignoring near-duplicates of a query spectrum (cos >= 0.995: the same record copied
into train). New score = z - gamma * seen * (cos < c). Reports MRR@25 on U (unseen answers) and S (seen answers)."""
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
D = pd.read_parquet('/Users/b/casmi26/seenbias_rows.parquet')
m = pd.read_csv('/Users/b/casmi26/ext/msg/MassSpecGym.tsv', sep='\t'); m['ik'] = m.inchikey.str[:14]
need = sorted(set(D[D.seen].key))
t = pq.read_table('/Users/b/casmi26/data/train.parquet', columns=['inchikey14', 'ionization_mode', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'])
t = t.filter(pc.and_(pc.is_in(t['inchikey14'], pa.array(need)), pc.equal(t['ionization_mode'], 'positive'))).to_pandas()
def prep(mz, it):
    mz = np.asarray(mz, float); it = np.asarray(it, float); o = np.argsort(mz); mz, it = mz[o], it[o]; k = it > 0
    return mz[k], np.sqrt(it[k])
def cos(q, r, tol=0.01):
    qm, qa = q; rm, ra = r
    if not len(qm) or not len(rm): return 0.0
    j0 = np.searchsorted(rm, qm - tol); j1 = np.searchsorted(rm, qm + tol, 'right')
    pairs = sorted(((qa[i] * ra[j], i, j) for i in range(len(qm)) for j in range(j0[i], j1[i])), reverse=True)
    ui, uj, s = set(), set(), 0.0
    for p, i, j in pairs:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s += p
    return s / (np.linalg.norm(qa) * np.linalg.norm(ra) + 1e-12)
lib = {}
for r in t.itertuples(): lib.setdefault(r.inchikey14, []).append((r.precursor_mz, prep(r.ms2_mzs, r.ms2_normalized_intensities)))
lib = {k: v[:40] for k, v in lib.items()}
Q = {ik: [(float(r.precursor_mz), prep([float(x) for x in r.mzs.split(',')], [float(x) for x in r.intensities.split(',')])) for r in g.head(4).itertuples()]
     for ik, g in m[m.ik.isin(set(D.ik))].groupby('ik')}
best = []
for r in D.itertuples():
    if not r.seen: best.append(np.nan); continue
    b = np.nan
    for qp, q in Q[r.ik]:
        for sp, s in lib.get(r.key, []):
            if abs(sp - qp) > qp * 1e-5 + 0.002: continue      # comparable only: same precursor m/z (same adduct)
            c = cos(q, s)
            if c < 0.995: b = c if np.isnan(b) else max(b, c)
    best.append(b)
D['libcos'] = best
D.to_parquet('/Users/b/casmi26/seenbias_rows_cos.parquet')
print('comparable share: seen truths %.2f, seen decoys %.2f' % (D[D.truth & D.seen].libcos.notna().mean(), D[~D.truth & D.seen].libcos.notna().mean()))
print('seen truth libcos (S, duplicates removed): median %.3f' % D[D.truth & D.seen].libcos.median(),
      '| seen decoy libcos: median %.3f' % D[~D.truth & D.seen].libcos.median())
def mrr(d, g, c):
    out = []
    for ik, x in d.groupby('ik'):
        pen = x.seen & x.libcos.notna() & (x.libcos < c)
        zz = x.z - g * pen; tv = zz[x.truth].iloc[0]; k = 1 + int((zz[~x.truth] > tv).sum()); out.append(1 / k if k <= 25 else 0)
    return np.mean(out)
for c in (0.3, 0.5, 0.7):
    for g in (0.5, 1.0, 2.0):
        print(f'c={c} gamma={g}:  U {mrr(D[D.grp=="U"], g, c):.3f} (base {mrr(D[D.grp=="U"], 0, c):.3f})   S {mrr(D[D.grp=="S"], g, c):.3f} (base {mrr(D[D.grp=="S"], 0, c):.3f})')
