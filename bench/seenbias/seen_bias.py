"""Seen-structure bias test for the public spectrum scorer (sanpier casmi-spectrum-model-v1, trained on train.parquet).

U (unseen answers): MassSpecGym structures whose InChIKey14 is not in train.parquet, with their MSG spectra.
S (seen control):  MassSpecGym structures that are in train.parquet, same instruments, random sample.
Candidates: the truth plus up to N same-formula structures from the 711k pool; each marked seen = in train.parquet.
Reports, for U and S: MRR@25; the per-list z-score of seen vs unseen decoys within Tanimoto-to-truth bins; and MRR after
subtracting beta * seen from the per-list z-score.
"""
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
KIT = '/Users/b/casmi26/ext/kitv1/casmi_spectrum_model_v1'; sys.path.insert(0, KIT)
from inference import Scorer
N, NS = int(sys.argv[1]) if len(sys.argv) > 1 else 100, 300
rng = np.random.default_rng(0)
tr = set(pq.read_table('/Users/b/casmi26/data/train.parquet', columns=['inchikey14']).to_pandas().inchikey14)
m = pd.read_csv('/Users/b/casmi26/ext/msg/MassSpecGym.tsv', sep='\t')
m['ik'] = m.inchikey.str[:14]; m['seen'] = m.ik.isin(tr)
pool = pq.read_table('/Users/b/casmi26/inputs/v2-pool/pool_meta.parquet', columns=['key', 'smiles', 'formula']).to_pandas()
pool['seen'] = pool.key.isin(tr); byf = {f: g for f, g in pool.groupby('formula')}
U = m[~m.seen].ik.unique().tolist()
S = rng.choice(m[m.seen].ik.unique(), NS, replace=False).tolist()
sc = Scorer(KIT, device='cpu')
mg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fpc = {}
def fp(s):
    if s not in fpc:
        x = Chem.MolFromSmiles(s); fpc[s] = mg.GetFingerprint(x) if x is not None else None
    return fpc[s]
rows = []
for grp, iks in (('U', U), ('S', S)):
    for ik in iks:
        g = m[m.ik == ik]; r0 = g.iloc[0]
        cand = byf.get(r0.formula)
        if cand is None: continue
        cand = cand[cand.key != ik]
        if len(cand) < 5: continue
        cand = cand.sample(min(N, len(cand)), random_state=0)
        smis = [r0.smiles] + cand.smiles.tolist(); seen = [bool(r0.seen)] + cand.seen.tolist(); keys = [ik] + cand.key.tolist()
        spectra = [dict(mzs=[float(v) for v in r.mzs.split(',')], intensities=[float(v) for v in r.intensities.split(',')],
                        precursor_mz=float(r.precursor_mz), adduct=r.adduct, ionization_mode='positive',
                        collision_energy=[float(r.collision_energy)] if pd.notna(r.collision_energy) else []) for r in g.head(4).itertuples()]
        try: s = sc.score_candidates(spectra, smis)
        except Exception as e: continue
        ok = np.isfinite(s)
        if not ok[0] or ok.sum() < 5: continue
        z = (s - s[ok].mean()) / (s[ok].std() + 1e-9)
        ft = fp(smis[0])
        for i in range(len(smis)):
            if not ok[i]: continue
            f = fp(smis[i]); t = DataStructs.TanimotoSimilarity(ft, f) if (ft is not None and f is not None) else np.nan
            rows.append((grp, ik, i == 0, seen[i], z[i], t, keys[i]))
D = pd.DataFrame(rows, columns=['grp', 'ik', 'truth', 'seen', 'z', 'tan', 'key'])
D.to_parquet('/Users/b/casmi26/seenbias_rows.parquet')
def mrr(d, beta=0.0):
    out = []
    for ik, g in d.groupby('ik'):
        zz = g.z - beta * g.seen; t = zz[g.truth].iloc[0]; k = 1 + int((zz[~g.truth] > t).sum())
        out.append(1 / k if k <= 25 else 0)
    return np.mean(out), len(out)
for grp in ('U', 'S'):
    d = D[D.grp == grp]; dec = d[~d.truth]
    print(f'\n== {grp}: molecules {d.ik.nunique()}, truth seen {d[d.truth].seen.mean():.2f}, decoys {len(dec)} (seen share {dec.seen.mean():.2f})')
    print('   MRR@25 by beta:', {b: round(mrr(d, b)[0], 3) for b in (0, 0.25, 0.5, 0.75, 1.0, 1.5)})
    dec = dec.assign(bin=pd.cut(dec.tan, [0, .2, .3, .4, .5, .7, 1.0]))
    t = dec.groupby(['bin', 'seen'], observed=True).z.agg(['mean', 'size']).unstack()
    print('   decoy z by Tanimoto-to-truth bin (seen False / True):'); print(t.round(3).to_string())
    # within-list paired gap: mean z(seen decoys) - mean z(unseen decoys), lists with both
    gaps = [g[g.seen].z.mean() - g[~g.seen].z.mean() for _, g in dec.groupby('ik') if g.seen.any() and (~g.seen).any()]
    print(f'   within-list seen-unseen decoy gap: mean {np.mean(gaps):.3f} SD, median {np.median(gaps):.3f}, n lists {len(gaps)}, '
          f'95% CI ±{1.96*np.std(gaps)/np.sqrt(len(gaps)):.3f}')
