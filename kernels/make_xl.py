"""Generate kernels/xl/<tag>: jc-p2s481v22 (our 0.478 copy of the public 0.481, unchanged) + a library-match rank-1 rule.

References: public library spectra of structures absent from train.parquet (giantwizard/casmi26-extlib; first version:
MassSpecGym records whose InChIKey14 is not in train.parquet, 1,013 spectra of 232 structures). The host allows newer
releases of train.parquet's sources (discussion 750684). A test molecule hits a reference when one of its spectra and one
reference spectrum share the ion mode, the precursor m/z (10 ppm + 0.002 Da), a square-root-intensity cosine >= TAU
(0.01 Da, greedy one-to-one) and >= K matched peaks. On a hit the structure goes to rank 1 (moved up if listed, else
inserted; 25 kept) unless the list already has a strong training-library match (library_max >= LIB_TAU).
TAU and K come from bench/extlib/calibrate.py (Enveda timsTOF training molecules vs other-library spectra), fixed before
any leaderboard result. Any error leaves the submission unchanged.
"""
import json, os, sys

TAU, K, LIB_TAU = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
TAG = sys.argv[4]
SRC = '/Users/b/casmi26-repo/kernels/jc-p2s481v22'
nb0 = json.load(open(f'{SRC}/casmi26-jc-p2s481v22.ipynb')); meta0 = json.load(open(f'{SRC}/kernel-metadata.json'))
CELL = r'''# ---- library-match rank 1 (kernels/make_xl.py): public spectra of structures absent from train.parquet ----
def _xl_overlay(TAU=%(tau)s, K=%(k)s, LIB_TAU=%(lib)s):
    import glob, json, shutil, time
    import numpy as np, pandas as pd
    t0 = time.monotonic()
    hits = sorted(glob.glob('/kaggle/input/**/casmi26-extlib/**/refs*.parquet', recursive=True))
    if not hits:
        print('[xl] skipped: casmi26-extlib not attached'); return
    D = pd.concat([pd.read_parquet(h) for h in hits], ignore_index=True)
    refs = []
    for r in D.itertuples(index=False):
        m = np.asarray(r.mz, float); i = np.asarray(r.it, float); o = np.argsort(m); m, i = m[o], i[o]; k = i > 0
        refs.append((int(r.ionmode), float(r.prec), m[k], np.sqrt(i[k]), r.key, r.smiles_canonical))
    order = np.argsort([x[1] for x in refs], kind='stable'); P = np.array([refs[j][1] for j in order])

    def cosine(qm, qa, rm, ra, tol=0.01):
        if not len(qm) or not len(rm): return 0.0, 0
        j0 = np.searchsorted(rm, qm - tol); j1 = np.searchsorted(rm, qm + tol, 'right')
        pairs = sorted(((qa[i] * ra[j], i, j) for i in range(len(qm)) for j in range(j0[i], j1[i])), reverse=True)
        ui, uj, s, n = set(), set(), 0.0, 0
        for p, i, j in pairs:
            if i in ui or j in uj: continue
            ui.add(i); uj.add(j); s += p; n += 1
        return s / (np.linalg.norm(qa) * np.linalg.norm(ra) + 1e-12), n

    te = pd.read_parquet(COMP / 'test.parquet'); spectra = {}
    for r in te.itertuples(index=False):
        m = np.asarray(r.ms2_mzs, float); i = np.asarray(r.ms2_normalized_intensities, float); o = np.argsort(m); m, i = m[o], i[o]
        k = (m <= float(r.precursor_mz) + 0.5) & (i > 0)
        mode = 1 if str(getattr(r, 'ionization_mode', 'positive')).lower().startswith('pos') else -1
        spectra.setdefault(str(r.molecule_id), []).append((mode, float(r.precursor_mz), m[k], np.sqrt(i[k])))
    lib = dict(globals().get('LIB_MAX') or {})
    sub_path = ROOT / 'submission.csv'
    sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    shutil.copy(sub_path, ROOT / 'submission_pre_xl.csv')
    out, audit = [], dict(refs=len(refs), structures=int(D.key.nunique()), hits=0, moved_up=0, inserted=0, already_top=0,
                          skipped_library=0, failed=0, cosines=[])
    for mid, smiles in zip(sub.molecule_id.astype(str), sub.smiles.astype(str)):
        smis = [s for s in smiles.split(';') if s]; new = smis
        try:
            best = (0.0, 0, None)
            for mode, prec, qm, qa in spectra.get(mid, []):
                tol = prec * 1e-5 + 0.002
                for t in order[np.searchsorted(P, prec - tol):np.searchsorted(P, prec + tol, 'right')]:
                    r = refs[t]
                    if r[0] != mode: continue
                    c, n = cosine(qm, qa, r[2], r[3])
                    if n >= K and c > best[0]: best = (c, n, r[5])
            if best[2] is not None and best[0] >= TAU:
                if lib.get(mid, 0.0) >= LIB_TAU:
                    audit['skipped_library'] += 1
                else:
                    audit['hits'] += 1; audit['cosines'].append(round(best[0], 3)); smi = best[2]
                    if smis and smis[0] == smi: audit['already_top'] += 1
                    elif smi in smis: new = [smi] + [s for s in smis if s != smi]; audit['moved_up'] += 1
                    else: new = ([smi] + smis)[:25]; audit['inserted'] += 1
        except Exception as e:
            audit['failed'] += 1
            if audit['failed'] <= 3: print('[xl] molecule', mid, 'failed:', repr(e)[:200])
        out.append(';'.join(new) if new else smiles)
    res = sub.copy(); res['smiles'] = out
    assert len(res) == len(sub) and list(res.molecule_id) == list(sub.molecule_id)
    assert res.smiles.str.split(';').map(len).max() <= 25
    res.to_csv(ROOT / 'submission.tmp.csv', index=False); os.replace(ROOT / 'submission.tmp.csv', sub_path)
    audit['seconds'] = round(time.monotonic() - t0, 1)
    json.dump(audit, open(ROOT / 'xl_audit.json', 'w'), indent=1)
    print('[xl] TAU', TAU, 'K', K, 'LIB_TAU', LIB_TAU, 'done:', json.dumps(audit), flush=True)


try:
    _xl_overlay()
except Exception:
    import traceback; traceback.print_exc(); print('[xl] overlay failed; submission unchanged')
'''
nb = json.loads(json.dumps(nb0)); cells = nb['cells']
i = next(k for k, c in enumerate(cells) if c['cell_type'] == 'code' and ''.join(c['source']).startswith('LIB_MAX = {}'))
cells.insert(i + 1, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                     'source': CELL % dict(tau=TAU, k=K, lib=LIB_TAU)})
cells[0] = dict(cells[0], source=f'# jc-p2s481v22-xl-{TAG}: jc-p2s481v22 (0.478 copy of public 0.481) + library-match rank 1 '
                f'(TAU {TAU}, K {K}, LIB_TAU {LIB_TAU}). Built by kernels/make_xl.py.\n\n' + ''.join(cells[0]['source']))
slug = f'casmi26-jc-p2s481v22-xl-{TAG}'
d = f'/Users/b/casmi26-repo/kernels/xl/{TAG}'; os.makedirs(d, exist_ok=True)
json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
json.dump(dict(meta0, id=f'giantwizard/{slug}', title=slug, code_file=f'{slug}.ipynb',
               dataset_sources=meta0['dataset_sources'] + ['giantwizard/casmi26-extlib']), open(f'{d}/kernel-metadata.json', 'w'), indent=1)
print('wrote', d)
