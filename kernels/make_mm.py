"""Generate kernels/mm/<tag>: jc-pop481 (exact copy of popwenman's 0.481) + spectral-mismatch penalty on ranks 2-25.

A candidate that is a training structure has library spectra. If one of them shares the query's ion mode and precursor m/z
(10 ppm + 0.002 Da, i.e. the same adduct) and the best cosine (sqrt intensity, 0.01 Da, greedy) between the molecule's
query spectra and those comparable spectra is below C, the candidate gets -PEN in the ranks 2-25 sort (rank 1 untouched).
Offline (bench/seenbias): +0.011 MRR on unseen answers, +0.037 on seen; true answers with comparable spectra fall below
0.5 about 1.4% of the time (Enveda timsTOF vs other instruments). Any error leaves the order unchanged.
"""
import json, os, sys
C, PEN, TAG = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
R1 = len(sys.argv) > 4 and sys.argv[4] == 'r1'
SRC = '/Users/b/casmi26-repo/kernels/jc-pop481'
nb0 = json.load(open(f'{SRC}/casmi26-jc-pop481.ipynb')); meta0 = json.load(open(f'{SRC}/kernel-metadata.json'))
CELL = r'''# ---- spectral-mismatch penalty (kernels/make_mm.py): seen candidates whose comparable library spectra do not match ----
MM_PEN, MM_REPORT = {}, dict(status='not run')
def _mm_build(cand_path, test_path, train_path, C=%(c)s, PEN=%(pen)s, budget=2400):
    import time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*'); t0 = time.monotonic()
    nf = pd.read_parquet(cand_path, columns=['molecule_id', 'smiles']).drop_duplicates()
    te = pd.read_parquet(test_path, columns=['molecule_id', 'ionization_mode', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'])
    tenum = rdMolStandardize.TautomerEnumerator(); kmap = {}
    def key(s):
        if s not in kmap:
            try:
                m = Chem.MolFromSmiles(s); Chem.RemoveStereochemistry(m); m = tenum.Canonicalize(m); kmap[s] = Chem.MolToInchiKey(m)[:14] or None
            except Exception:
                kmap[s] = None
        return kmap[s]
    nf['k'] = [key(s) for s in nf.smiles]
    keys = sorted(set(k for k in nf.k if k))
    t = pq.read_table(train_path, columns=['inchikey14', 'ionization_mode', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'],
                      filters=[('inchikey14', 'in', keys)]).to_pandas()
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
    pol = lambda v: 1 if str(v).lower().startswith('pos') else -1
    lib = {}
    for r in t.itertuples(index=False):
        lib.setdefault(r.inchikey14, []).append((pol(r.ionization_mode), float(r.precursor_mz), prep(r.ms2_mzs, r.ms2_normalized_intensities)))
    Q = {}
    for r in te.itertuples(index=False):
        Q.setdefault(str(r.molecule_id), []).append((pol(r.ionization_mode), float(r.precursor_mz), prep(r.ms2_mzs, r.ms2_normalized_intensities)))
    rep = dict(candidates=len(nf), seen=0, comparable=0, penalised=0, molecules_touched=0, timed_out=False, C=C, PEN=PEN)
    out = {}
    for mid, g in nf.groupby('molecule_id', sort=False):
        if time.monotonic() - t0 > budget: rep['timed_out'] = True; break
        mid = str(mid); qs = Q.get(mid, [])[:6]; d = {}
        for s, k in zip(g.smiles, g.k):
            L = lib.get(k)
            if not L: continue
            rep['seen'] += 1; best = None
            for qm_, qp, q in qs:
                for lm, lp, sp in L[:60]:
                    if lm != qm_ or abs(lp - qp) > qp * 1e-5 + 0.002: continue
                    c = cos(q, sp); best = c if best is None else max(best, c)
            if best is None: continue
            rep['comparable'] += 1
            if best < C: d[s] = -PEN; rep['penalised'] += 1
        if d: out[mid] = d; rep['molecules_touched'] += 1
    rep['seconds'] = round(time.monotonic() - t0, 1)
    return out, rep
try:
    if globals().get('PIPELINE_DONE') and (ROOT / 'neural_candidate_features.parquet').is_file():
        MM_PEN, MM_REPORT = _mm_build(ROOT / 'neural_candidate_features.parquet', COMP / 'test.parquet', COMP / 'train.parquet')
    else:
        MM_REPORT = dict(status='skipped: pipeline did not run (fast commit)')
except Exception as _e:
    import traceback; traceback.print_exc(); MM_PEN, MM_REPORT = {}, dict(status='failed: ' + repr(_e)[:300])
print('[mm]', MM_REPORT, flush=True)
'''
OLD = "rest.sort(key=lambda _s: _score_cand(_s) + _v8_pen.get(_s, 0.0), reverse=True)"
NEW = "rest.sort(key=lambda _s: _score_cand(_s) + _v8_pen.get(_s, 0.0) + MM_PEN.get(str(mid), {}).get(_s, 0.0), reverse=True)"
OLD2 = "        rows.append((mid, [leader] + rest))"
NEW2 = '''        _mm = MM_PEN.get(str(mid), {})                     # mm rank 1: a penalised leader yields to the best unpenalised candidate
        if MM_R1 and leader in _mm:
            _alt = next((x for x in rest if x not in _mm), None)
            if _alt is not None:
                rest = [x for x in rest if x != _alt] + [leader]
                rest.sort(key=lambda _s: (_score_cand(_s) if (rest and cinfo) else 0.0) + _mm.get(_s, 0.0), reverse=True) if cinfo else None
                leader = _alt; counts['mm_leader_swaps'] = counts.get('mm_leader_swaps', 0) + 1
        rows.append((mid, [leader] + rest))'''
nb = json.loads(json.dumps(nb0)); cells = nb['cells']
i = next(k for k, c in enumerate(cells) if c['cell_type'] == 'code' and 'def pxm_merge' in ''.join(c['source']))
s = ''.join(cells[i]['source']); assert s.count(OLD) == 1 and s.count(OLD2) == 1; s = s.replace(OLD, NEW)
if R1: s = s.replace(OLD2, NEW2)
cells[i]['source'] = s
cells.insert(i, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': CELL % dict(c=C, pen=PEN) + f'MM_R1 = {R1}\n'})
cells[0] = dict(cells[0], source=f'# jc-pop481-mm-{TAG}: exact copy of popwenman 0.481 + spectral-mismatch penalty on ranks 2-25 (C {C}, PEN {PEN}, rank-1 rule {R1}). '
                'Built by kernels/make_mm.py.\n\n' + ''.join(cells[0]['source']))
slug = f'casmi26-jc-pop481-mm-{TAG}'; d = f'/Users/b/casmi26-repo/kernels/mm/{TAG}'; os.makedirs(d, exist_ok=True)
json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
json.dump(dict(meta0, id=f'giantwizard/{slug}', title=slug, code_file=f'{slug}.ipynb'), open(f'{d}/kernel-metadata.json', 'w'), indent=1)
print('wrote', d, 'cell at', i)
