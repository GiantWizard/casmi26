"""Calibrate a library-match rank-1 rule on Enveda timsTOF training molecules (no leaderboard).

Queries: Enveda timsTOF spectra (enveda-180, enveda-np-examples) of sampled structures that also have spectra from other
libraries. References: all non-Enveda training spectra in each query's precursor window (10 ppm + 0.002 Da, same ion mode).
Cosine: square-root intensities, 0.01 Da, greedy one-to-one (same as heliosli's public overlay).
Reports, per (TAU, K): with the truth among the references, how often the best hit clears the bar and is right / wrong;
with the truth's spectra removed, how often a (necessarily false) hit clears the bar.
"""
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc

TRAIN = '/Users/b/casmi26/data/train.parquet'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
COLS = ['ingest_lib', 'inchikey14', 'ionization_mode', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']
ENV = {'enveda-180', 'enveda-np-examples'}

meta = pq.read_table(TRAIN, columns=['ingest_lib', 'inchikey14']).to_pandas()
env_ik = set(meta[meta.ingest_lib.isin(ENV)].inchikey14)
lib_ik = set(meta[~meta.ingest_lib.isin(ENV)].inchikey14)
both = sorted(env_ik & lib_ik)
rng = np.random.default_rng(0)
pick = set(rng.choice(both, size=min(N, len(both)), replace=False))
print('enveda structs', len(env_ik), 'with other-library spectra', len(both), 'sampled', len(pick), flush=True)

t = pq.read_table(TRAIN, columns=COLS)
isenv = pc.is_in(t['ingest_lib'], pa_set := __import__('pyarrow').array(sorted(ENV)))
Q = t.filter(pc.and_(isenv, pc.is_in(t['inchikey14'], __import__('pyarrow').array(sorted(pick))))).to_pandas()
Q = Q.groupby('inchikey14').head(4)                       # up to 4 spectra per query molecule, like a test molecule
R = t.filter(pc.invert(isenv)).select(['inchikey14', 'ionization_mode', 'precursor_mz']).to_pandas()
R['ridx'] = np.arange(len(R))
mode = lambda s: np.where(pd.Series(s).str.lower().str.startswith('pos'), 1, -1)
R['m'] = mode(R.ionization_mode); Q['m'] = mode(Q.ionization_mode)
R = R.sort_values('precursor_mz'); P = R.precursor_mz.values
need = set()
win = {}
for i, r in enumerate(Q.itertuples()):
    tol = r.precursor_mz * 1e-5 + 0.002
    a, b = np.searchsorted(P, r.precursor_mz - tol), np.searchsorted(P, r.precursor_mz + tol, 'right')
    sub = R.iloc[a:b]; sub = sub[sub.m == r.m]
    win[i] = sub.ridx.values; need.update(sub.ridx.values.tolist())
print('query spectra', len(Q), 'reference spectra needed', len(need), flush=True)
nonenv = t.filter(pc.invert(isenv))
idx = np.array(sorted(need))
RS = nonenv.take(idx).select(['inchikey14', 'ms2_mzs', 'ms2_normalized_intensities']).to_pandas()
pos = {r: k for k, r in enumerate(idx)}


def prep(m, i, prec=None):
    m = np.asarray(m, float); i = np.asarray(i, float); o = np.argsort(m); m, i = m[o], i[o]
    k = i > 0
    if prec is not None: k &= m <= prec + 0.5
    return m[k], np.sqrt(i[k])


def cosine(qm, qa, rm, ra, tol=0.01):
    if not len(qm) or not len(rm): return 0.0, 0
    j0 = np.searchsorted(rm, qm - tol); j1 = np.searchsorted(rm, qm + tol, 'right')
    pairs = sorted(((qa[i] * ra[j], i, j) for i in range(len(qm)) for j in range(j0[i], j1[i])), reverse=True)
    ui, uj, s, n = set(), set(), 0.0, 0
    for p, i, j in pairs:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s += p; n += 1
    return s / (np.linalg.norm(qa) * np.linalg.norm(ra) + 1e-12), n


RP = [prep(m, i) for m, i in zip(RS.ms2_mzs, RS.ms2_normalized_intensities)]
rows = []                                                   # per query spectrum: best true hit, best false hit
for i, r in enumerate(Q.itertuples()):
    qm, qa = prep(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
    bt, bf = (0.0, 0), (0.0, 0)
    for ridx in win[i]:
        k = pos[ridx]; c, n = cosine(qm, qa, *RP[k])
        if RS.inchikey14.iat[k] == r.inchikey14:
            bt = max(bt, (c, n))
        else:
            bf = max(bf, (c, n))
    rows.append((r.inchikey14, bt[0], bt[1], bf[0], bf[1], len(win[i])))
D = pd.DataFrame(rows, columns=['ik', 'tc', 'tn', 'fc', 'fn', 'nref'])
D.to_parquet('/Users/b/casmi26/bench_extlib_calib.parquet')
print(D.describe().round(3).to_string())
print('\nmolecule-level (best over its spectra); a hit needs cosine>=TAU and >=K matched peaks')
for K in (3, 4, 6):
    for TAU in (0.6, 0.7, 0.8, 0.85, 0.9):
        g = D.assign(t_ok=(D.tc >= TAU) & (D.tn >= K), f_ok=(D.fc >= TAU) & (D.fn >= K),
                     tsc=np.where((D.tn >= K), D.tc, 0), fsc=np.where((D.fn >= K), D.fc, 0)).groupby('ik').agg(
            t=('tsc', 'max'), f=('fsc', 'max'))
        right = ((g.t >= TAU) & (g.t >= g.f)).mean(); wrong = ((g.f >= TAU) & (g.f > g.t)).mean(); absent = (g.f >= TAU).mean()
        print(f'K={K} TAU={TAU:.2f}  truth present: right {right:.3f} wrong {wrong:.3f} | truth absent: false hit {absent:.3f}')
