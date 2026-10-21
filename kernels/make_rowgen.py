"""Generate kernels/rowgen/s{0..3}: the v465 pipeline run on simulated test molecules made from held-out training
structures, saving every stage's candidates and scores as stacked-reranker training rows.

Queries: the NP evaluation half of our harness (np1_evalhalf, NP-like fold-0 structures our engine-1 ranker never trained
on) plus the Enveda NP panel (fold 'np').
  class 2: structures also in COCONUT (so both engines' pools hold them without training spectra); every spectrum of the
           structure and of its metric-key twins is removed from the library;
  class 1: other evalhalf structures with >= 4 spectra; only the query spectra are removed, the rest stay searchable.
Every model the pipeline fits on training structures is swapped for one that never saw fold 0 / the panel:
  engine 1: fold-safe FPNet bank, no fe_A/B/D views, our ranker (trained without evalhalf and the panel);
  engine 2: the two FPNets retrained with the fold-safe split withheld (giantwizard/casmi26-e2fp-f0);
  PubChem channel: the fold-safe FPNet.
Engine 2's two rankers (trained on simulated rows of all training structures) and ICEBERG/GLACIER (MassSpecGym) are kept;
rows on MassSpecGym training structures are flagged later from msg_overlap.json.
Four shards of about 400 molecules each, so ICEBERG/GLACIER budgets per molecule match a scoring run.
"""
import json, os
import numpy as np, pandas as pd

SRC = '/Users/b/casmi26-repo/kernels/jc-v465'; OUT = '/Users/b/casmi26-repo/kernels/rowgen'; NSH = 4
nb0 = json.load(open(f'{SRC}/casmi26-jc-v465.ipynb')); meta0 = json.load(open(f'{SRC}/kernel-metadata.json'))

# ---- queries ----
pm = pd.read_parquet('/Users/b/casmi26/inputs/v2-pool/pool_meta.parquet', columns=['train_sid', 'src'])
src = pm[pm.train_sid >= 0].set_index('train_sid').src
ts = pd.read_parquet('/Users/b/casmi26/inputs/v2-pool/train_structs.parquet', columns=['sid', 'n_spec']).set_index('sid').n_spec
sp = pd.read_parquet('/Users/b/casmi26/inputs/foldsafe/split.parquet')
ev = pd.read_parquet('/Users/b/casmi26/rk/np1_evalhalf.parquet').sid.values
panel = sp[sp.fold == 'np'].sid.values
c2 = [int(s) for s in np.concatenate([ev, panel]) if src.get(s) == 3]
rng = np.random.default_rng(26)
pool1 = [int(s) for s in ev if src.get(s) != 3 and ts.get(s, 0) >= 4]
c1 = sorted(rng.choice(pool1, min(300, len(pool1)), replace=False).tolist())
Q = pd.DataFrame({'sid': c2 + c1, 'cls': [2] * len(c2) + [1] * len(c1)})
Q['panel'] = Q.sid.isin(panel)
Q = Q.sample(frac=1, random_state=26).reset_index(drop=True)
Q['shard'] = np.arange(len(Q)) % NSH
print('queries', len(Q), Q.groupby(['cls', 'panel']).size().to_dict())


def esc(x):                                        # code inside cell 13/15's embedded-file strings is escaped
    return x.replace("'", "\\'").replace('\n', '\\n')


def rep(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:80])
    return s.replace(old, new)


FAKE = r'''# ---- ROWGEN: simulated test molecules from held-out training structures (kernels/make_rowgen.py) ----
import os, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
_Q = pd.DataFrame(%(Q)s)
_REAL = COMP
_FAKE = '/kaggle/working/fake_comp'; os.makedirs(_FAKE, exist_ok=True); os.makedirs('/kaggle/working/rowgen', exist_ok=True)
_st = pd.read_parquet(find('train_structs.parquet'), columns=['sid', 'inchikey14', 'key', 'smiles'])
_q = _Q.merge(_st, on='sid', how='left'); assert _q.inchikey14.notna().all()
_m = pq.read_table(os.path.join(_REAL, 'train.parquet'), columns=['inchikey14', 'instrument_type', 'adduct']).to_pandas()
_m['row'] = np.arange(len(_m))
_rng = np.random.default_rng(%(seed)d)
_MAIN = {'[M+H]+': 0, '[M-H]-': 1, '[M+CH2O2-H]-': 2, '[M+Na]+': 3}
_qrows, _drop, _mid = [], [], []
_by = _m.groupby('inchikey14').row.apply(np.asarray).to_dict()
for r in _q.itertuples():
    rows = _by[r.inchikey14]
    d = _m.loc[rows]
    pri = np.lexsort((_rng.random(len(d)), d.adduct.map(_MAIN).fillna(9).values, (d.instrument_type != 'timsTOF').values))
    k = int(_rng.choice([1, 2, 3, 3, 4, 4, 5, 6]))
    k = min(k, len(d) if r.cls == 2 else len(d) - 1)
    qr = d.row.values[pri[:k]]
    _qrows += qr.tolist(); _mid += ['q%%d' %% r.sid] * len(qr)
    if r.cls == 2:
        twins = set(_st.inchikey14[_st.key == r.key])
        for t in twins | {r.inchikey14}:
            _drop += _by.get(t, np.zeros(0, int)).tolist()
    else:
        _drop += qr.tolist()
_drop = np.unique(np.asarray(_drop + _qrows, np.int64))
_pf = pq.ParquetFile(os.path.join(_REAL, 'train.parquet')); _w = None; _base = 0
for _rg in range(_pf.metadata.num_row_groups):
    _t = _pf.read_row_group(_rg); _n = _t.num_rows
    _keep = ~np.isin(np.arange(_base, _base + _n), _drop)
    _t = _t.filter(pa.array(_keep))
    _w = _w or pq.ParquetWriter(os.path.join(_FAKE, 'train.parquet'), _t.schema)
    _w.write_table(_t); _base += _n
_w.close()
_qr = pd.DataFrame({'row': _qrows, 'molecule_id': _mid})
_tcols = ['ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct', 'ionization_mode', 'instrument_type',
          'precursor_mz', 'collision_energy_orig', 'collision_energy_ev', 'collision_energy_orig_units']
_parts, _base = [], 0
for _rg in range(_pf.metadata.num_row_groups):
    _n = _pf.metadata.row_group(_rg).num_rows
    _sel = _qr[(_qr.row >= _base) & (_qr.row < _base + _n)]
    if len(_sel):
        _d = _pf.read_row_group(_rg, columns=_tcols).to_pandas().iloc[_sel.row.values - _base].reset_index(drop=True)
        _d.insert(0, 'spectrum_id', ['qs_%%d' %% x for x in _sel.row.values]); _d.insert(0, 'molecule_id', _sel.molecule_id.values)
        _parts.append(_d)
    _base += _n
_te = pd.concat(_parts, ignore_index=True)
_te.to_parquet(os.path.join(_FAKE, 'test.parquet'))
_ss = pd.read_csv(os.path.join(_REAL, 'sample_submission.csv'), nrows=1)
pd.DataFrame({'molecule_id': sorted(_te.molecule_id.unique()), _ss.columns[1]: _ss.iloc[0, 1]}).to_csv(os.path.join(_FAKE, 'sample_submission.csv'), index=False)
_q.assign(molecule_id=['q%%d' %% s for s in _q.sid]).to_json('/kaggle/working/rowgen/truth.json', orient='records')
COMP = _FAKE
print('ROWGEN shard %(shard)d:', _te.molecule_id.nunique(), 'molecules,', len(_te), 'spectra;', len(_drop), 'library spectra removed', flush=True)
'''

DUMP = r'''# ---- ROWGEN: save every stage's candidates and scores (%(when)s) ----
import os, pickle, shutil
_R = '/kaggle/working/rowgen'; os.makedirs(_R, exist_ok=True)
_objs = {}
for _n in ['BASE', 'BASE_RAW', 'ENG', 'ICE_SCORES', 'GL_SCORES', 'PREFUSION_LISTS', 'FUSION_OVERFLOW', 'FRAG_GAP', 'lib_max_of']:
    if _n in globals():
        _objs[_n] = globals()[_n]
try:
    _objs['GL_BY_MID'] = {str(m): gl_of(m) for m in BASE}
except Exception as _e:
    print('gl_of dump failed', repr(_e))
try:
    _objs['POOL_POP_BASE'] = {m: [float(POOL_POP[p]) if p >= 0 else None for p in v[3]] for m, v in BASE_RAW.items()}
except Exception as _e:
    print('pop dump failed', repr(_e))
for _n, _v in _objs.items():
    try:
        pickle.dump(_v, open(f'{_R}/{_n}_%(when)s.pkl', 'wb'))
    except Exception as _e:
        print('could not save', _n, repr(_e))
for _f in ['submission.csv', 'pc_lists.json', 'eng_lists.json', 'ice_work/ice_out.json', 'gl_work/gl_out.json',
           'e004_candidate_inputs.json', 'candidate_audit.json', 'e2r2_stats.json']:
    _p = os.path.join('/kaggle/working', _f)
    if os.path.exists(_p):
        shutil.copy(_p, f'{_R}/%(when)s_' + _f.replace('/', '_'))
print('ROWGEN dump %(when)s:', sorted(os.listdir(_R)), flush=True)
'''

for sh in range(NSH):
    nb = json.loads(json.dumps(nb0))
    cells = nb['cells']
    # engine 2: retrained FPNets, and keep the blend scores of the listed candidates
    s = ''.join(cells[13]['source'])
    s = rep(s, esc("if 'casmi26-fp-models-v2' in p)"), esc("if 'casmi26-e2fp-f0' in p)\n    assert len(fp_models) == 2, fp_models"))
    s = rep(s, esc("""        smis, keys, seen = [], [], set()
        for c in r['cand'][order]:
            s = E.POOL['smiles'][c]; k = E.canon_key(s)
            if k is None or k in seen:
                continue
            seen.add(k); smis.append(s); keys.append(k)
            if len(smis) >= 40:
                break
        res[str(mid)] = dict(smiles=smis, keys=keys)"""), esc("""        smis, keys, seen, sc, spv, sours = [], [], set(), [], [], []
        for i in order:
            c = r['cand'][i]
            s = E.POOL['smiles'][c]; k = E.canon_key(s)
            if k is None or k in seen:
                continue
            seen.add(k); smis.append(s); keys.append(k); sc.append(float(bl[mid][i]))
            spv.append(float(S['pv'][mid][i])); sours.append(float(S['ours'][mid][i]))
            if len(smis) >= 40:
                break
        res[str(mid)] = dict(smiles=smis, keys=keys, blend=sc, pv=spv, ours=sours)"""))
    cells[13]['source'] = s
    # PubChem channel: fold-safe FPNet
    s = ''.join(cells[15]['source'])
    s = rep(s, esc("""models = sorted(glob.glob(os.path.join(V3MOD, 'fpnet_*.pt')))
    assert len(models) == 2, models"""), esc("""models = sorted(glob.glob('/kaggle/input/**/casmi26-fold-safe-fpnet/**/fpnet_fold0.pt', recursive=True))
    assert len(models) == 1, models"""))
    cells[15]['source'] = s
    # engine 1: fold-safe bank, no fe views, our ranker (same swaps as kernels/make_own.py)
    s = ''.join(cells[18]['source'])
    s = rep(s, """_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)   # v4m
bank = fpnet.ModelBank([_FULL[0] if _FULL else os.path.join(V4MOD, 'fpnet_0.pt')], device=dev)
print('engine FP bank:', _FULL[0] if _FULL else 'fpnet_0 (A) FALLBACK', flush=True)""",
            """_F0 = glob.glob('/kaggle/input/**/casmi26-fold-safe-fpnet/**/fpnet_fold0.pt', recursive=True)
assert _F0, 'fold-safe FPNet not attached'
bank = fpnet.ModelBank([_F0[0]], device=dev)
print('engine FP bank (fold-safe):', _F0[0], flush=True)""")
    s = rep(s, """V = v1engine.V1FE(E, [os.path.join(V4MOD, m['file']) for m in MAN['fe_models']], MAN['families'], device=dev)""",
            """OWN_FAMILIES = [f for f in MAN['families'] if f not in ('model_views', 'fragnet')]
V = v1engine.V1FE(E, [], OWN_FAMILIES, device=dev)
def _no_views(prep, _nb=P.nbits):
    z = np.zeros((0, 0, _nb), np.float32); e = np.zeros((0, 0, 10), np.float32)
    return z, e, z, e, np.zeros(0, np.float32)
V.model_views = _no_views
print('engine-1 families (fold-safe):', OWN_FAMILIES, flush=True)""")
    s = rep(s, """RK = pickle.load(open(os.path.join(V4MOD, MAN['ranker']['file']), 'rb'))""",
            """_RKF = glob.glob('/kaggle/input/**/casmi26-jc-own-ranker/**/ranker_ours.pkl', recursive=True)
assert _RKF, 'our ranker not attached'
RK = pickle.load(open(_RKF[0], 'rb'))""")
    cells[18]['source'] = s
    # engine 1 lists before the popularity re-sort
    s = ''.join(cells[20]['source'])
    s = rep(s, "BASE, n_err = {}, 0\n", "BASE, n_err = {}, 0\nBASE_RAW = {}\n")
    s = rep(s, "    if POOLPOP_MU > 0 and len(scs) > 1:", "    BASE_RAW[mid] = (list(smis), list(keys), list(scs), list(pids_), lib_max)\n    if POOLPOP_MU > 0 and len(scs) > 1:")
    cells[20]['source'] = s
    q = Q[Q.shard == sh][['sid', 'cls', 'panel']]
    fake = FAKE % dict(Q=repr({c: q[c].tolist() for c in q.columns}), seed=1000 + sh, shard=sh)
    code = lambda x: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': x}
    new = cells[:9] + [code(fake)] + cells[9:27] + [code(DUMP % dict(when='fusion'))] + cells[27:] + [code(DUMP % dict(when='final'))]
    new[0] = dict(new[0], source='# ROWGEN shard %d: the v465 pipeline (fold-safe models) on simulated test molecules from held-out '
                  'training structures; saves every stage for stacked-reranker rows. Not a submission. Built by kernels/make_rowgen.py.\n\n' % sh
                  + ''.join(new[0]['source']))
    nb['cells'] = new
    slug = f'casmi26-jc-rowgen-s{sh}'
    ds = [d for d in meta0['dataset_sources'] if d not in ('ahmedberatozer/casmi26-fpnet-full1', 'prvsiyan/casmi26-fp-models-v2')]
    ds += ['giantwizard/casmi26-jc-own-ranker', 'dmitriigluzdov/casmi26-fold-safe-fpnet', 'giantwizard/casmi26-e2fp-f0']
    d = f'{OUT}/s{sh}'; os.makedirs(d, exist_ok=True)
    json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
    json.dump(dict(meta0, id=f'giantwizard/{slug}', title=slug, code_file=f'{slug}.ipynb', dataset_sources=ds),
              open(f'{d}/kernel-metadata.json', 'w'), indent=1)
Q.to_parquet(f'{OUT}/queries.parquet')
print('wrote', NSH, 'shards to', OUT)
