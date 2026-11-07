"""timsTOF validation of the spectral-mismatch penalty (no leaderboard).

Queries: Enveda timsTOF spectra (enveda-180 / enveda-np-examples) of structures that also have non-Enveda library spectra.
Class-1 case: the truth's library spectra are its NON-Enveda spectra only (cross-instrument, as for a hidden class-1 answer).
Class-2 case: the truth has no library spectra (cannot be penalised).
Candidates: truth + up to N same-formula pool structures; a seen decoy's library = all its train spectra.
Base order: public spectrum scorer z-score (per list). Penalty: -gamma (z units) for a seen candidate whose comparable
(same mode, precursor within 10 ppm + 0.002 Da) library spectra all match the query below C. Bootstrap CI over molecules.
"""
import sys, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
KIT = '/Users/b/casmi26/ext/kitv1/casmi_spectrum_model_v1'; sys.path.insert(0, KIT)
from inference import Scorer
NMOL, N = int(sys.argv[1]) if len(sys.argv) > 1 else 400, 100
ENV = ['enveda-180', 'enveda-np-examples']
TRAIN = '/Users/b/casmi26/data/train.parquet'
meta = pq.read_table(TRAIN, columns=['ingest_lib', 'inchikey14', 'normalized_smiles', 'molecular_formula']).to_pandas()
env_ik = set(meta[meta.ingest_lib.isin(ENV)].inchikey14); lib_ik = set(meta[~meta.ingest_lib.isin(ENV)].inchikey14)
seen_all = set(meta.inchikey14)
both = sorted(env_ik & lib_ik); rng = np.random.default_rng(1)
pick = list(rng.choice(both, size=min(NMOL, len(both)), replace=False))
first = meta.drop_duplicates('inchikey14').set_index('inchikey14')
pool = pq.read_table('/Users/b/casmi26/inputs/v2-pool/pool_meta.parquet', columns=['key', 'smiles', 'formula']).to_pandas()
byf = {f: g for f, g in pool.groupby('formula')}
lists = {}
for ik in pick:
    f = first.loc[ik, 'molecular_formula']; c = byf.get(f)
    if c is None: continue
    c = c[c.key != ik]
    if len(c) < 5: continue
    c = c.sample(min(N, len(c)), random_state=0)
    lists[ik] = [(ik, first.loc[ik, 'normalized_smiles'])] + list(zip(c.key, c.smiles))
need = sorted({k for L in lists.values() for k, _ in L if k in seen_all})
cols = ['ingest_lib', 'inchikey14', 'ionization_mode', 'adduct', 'precursor_mz', 'collision_energy_ev', 'ms2_mzs', 'ms2_normalized_intensities']
T = pq.read_table(TRAIN, columns=cols, filters=[('inchikey14', 'in', need)]).to_pandas()
print('molecules', len(lists), 'seen keys', len(need), 'spectra loaded', len(T), flush=True)
pol = lambda v: 1 if str(v).lower().startswith('pos') else -1
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
LIB = {}
for r in T.itertuples(index=False):
    LIB.setdefault(r.inchikey14, []).append((r.ingest_lib in ENV, pol(r.ionization_mode), float(r.precursor_mz), prep(r.ms2_mzs, r.ms2_normalized_intensities)))
sc = Scorer(KIT, device='cpu')
rows = []
for n, (ik, L) in enumerate(lists.items()):
    q = T[(T.inchikey14 == ik) & T.ingest_lib.isin(ENV)].head(4)
    spectra = [dict(mzs=list(r.ms2_mzs), intensities=list(r.ms2_normalized_intensities), precursor_mz=float(r.precursor_mz), adduct=r.adduct,
                    ionization_mode=r.ionization_mode, collision_energy=list(r.collision_energy_ev) if r.collision_energy_ev is not None else [])
               for r in q.itertuples()]
    Q = [(pol(r.ionization_mode), float(r.precursor_mz), prep(r.ms2_mzs, r.ms2_normalized_intensities)) for r in q.itertuples()]
    try: s = sc.score_candidates(spectra, [x[1] for x in L])
    except Exception: continue
    ok = np.isfinite(s)
    if not ok[0] or ok.sum() < 5: continue
    z = (s - s[ok].mean()) / (s[ok].std() + 1e-9)
    for i, (k, smi) in enumerate(L):
        if not ok[i]: continue
        best = None
        for is_env, lm, lp, sp in LIB.get(k, [])[:80]:
            if i == 0 and is_env: continue                       # truth: only cross-instrument (non-Enveda) spectra
            for qm, qp, qq in Q:
                if lm != qm or abs(lp - qp) > qp * 1e-5 + 0.002: continue
                c = cos(qq, sp); best = c if best is None else max(best, c)
        rows.append((ik, i == 0, k in seen_all, z[i], best))
    if n % 50 == 0: print(n, flush=True)
D = pd.DataFrame(rows, columns=['ik', 'truth', 'seen', 'z', 'best'])
D.to_parquet('/Users/b/casmi26/timstof_val_rows.parquet')
def rr(x, C, g, cls):
    pen = x.seen & x.best.notna() & (x.best < C)
    if cls == 2: pen = pen & ~x.truth                            # class 2: truth has no library spectra
    zz = x.z - g * pen; t = zz[x.truth].iloc[0]; k = 1 + int((zz[~x.truth] > t).sum()); return 1 / k if k <= 25 else 0.0
tr = D[D.truth]
print(f'\ntruth comparable cross-instrument spectra: {tr.best.notna().mean():.2f}; of those below 0.5: {(tr.best.dropna() < 0.5).mean():.3f}, below 0.3: {(tr.best.dropna() < 0.3).mean():.3f}')
dec = D[~D.truth & D.seen]; print(f'seen decoys comparable: {dec.best.notna().mean():.2f}; of those below 0.5: {(dec.best.dropna() < 0.5).mean():.3f}')
ids = D.ik.unique(); G = {k: g for k, g in D.groupby('ik')}
for cls in (1, 2):
    base = np.array([rr(G[k], 0.5, 0.0, cls) for k in ids])
    print(f'\nclass {cls} sim: base MRR {base.mean():.3f} (n {len(ids)})')
    for C in (0.3, 0.5):
        for g in (0.5, 1.0, 2.0):
            v = np.array([rr(G[k], C, g, cls) for k in ids]); d = v - base
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            print(f'  C={C} gamma={g}: {v.mean():.3f}  diff {d.mean():+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]')
