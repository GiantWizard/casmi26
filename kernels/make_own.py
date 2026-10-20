"""Generate kernels/jc-v7c-own: the public 0.443 notebook (jc-v7c-e2r2) with a COMPLIANT engine 1.
  FP bank      : fpnet_full1 (no published training code)   -> fold-safe fold0 (dmitriigluzdov, training code published)
  model views  : fe_A/B/D weights (no published code)       -> not loaded; model_views + fragnet families dropped
  ranker       : ranker_v4b (no published code)             -> ours (giantwizard/casmi26-jc-own-ranker, compliance_ranker.py)
Everything else (engine 2, PubChem channel, GLACIER/ICEBERG, fusion, e2rank2) unchanged, so one LB submission measures
the cost of the engine-1 swap alone. No silent fallbacks: missing replacement files fail the run.
"""
import json, os
SRC = '/Users/b/casmi26-repo/kernels/jc-v7c-e2r2'; OUT = '/Users/b/casmi26-repo/kernels/jc-v7c-own'
nb = json.load(open(f'{SRC}/casmi26-jc-v7c-e2r2.ipynb')); meta = json.load(open(f'{SRC}/kernel-metadata.json'))
c = nb['cells'][18]; s = ''.join(c['source'])
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
rep("""_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)   # v4m
bank = fpnet.ModelBank([_FULL[0] if _FULL else os.path.join(V4MOD, 'fpnet_0.pt')], device=dev)
print('engine FP bank:', _FULL[0] if _FULL else 'fpnet_0 (A) FALLBACK', flush=True)""",
"""_F0 = glob.glob('/kaggle/input/**/casmi26-fold-safe-fpnet/**/fpnet_fold0.pt', recursive=True)   # jc-own: compliant bank
assert _F0, 'fold-safe FPNet (dmitriigluzdov/casmi26-fold-safe-fpnet) not attached'
bank = fpnet.ModelBank([_F0[0]], device=dev)
print('engine FP bank (compliant):', _F0[0], flush=True)""")
rep("""V = v1engine.V1FE(E, [os.path.join(V4MOD, m['file']) for m in MAN['fe_models']], MAN['families'], device=dev)""",
"""OWN_FAMILIES = [f for f in MAN['families'] if f not in ('model_views', 'fragnet')]   # jc-own: no fe_A/B/D weights
V = v1engine.V1FE(E, [], OWN_FAMILIES, device=dev)
def _no_views(prep, _nb=P.nbits):                     # model views are only read by the dropped model_views family
    z = np.zeros((0, 0, _nb), np.float32); e = np.zeros((0, 0, 10), np.float32)
    return z, e, z, e, np.zeros(0, np.float32)
V.model_views = _no_views
print('engine-1 families (compliant):', OWN_FAMILIES, flush=True)""")
rep("""RK = pickle.load(open(os.path.join(V4MOD, MAN['ranker']['file']), 'rb'))""",
"""_RKF = glob.glob('/kaggle/input/**/casmi26-jc-own-ranker/**/ranker_ours.pkl', recursive=True)   # jc-own: our ranker
assert _RKF, 'our ranker (giantwizard/casmi26-jc-own-ranker) not attached'
RK = pickle.load(open(_RKF[0], 'rb'))""")
c['source'] = s
nb['cells'][0]['source'] = ('# jc-v7c-own = jc-v7c-e2r2 (public 0.443) with a compliant engine 1: fold-safe FPNet bank (published training '
                            'code), no fe_A/B/D model views, our own ranker (giantwizard/casmi26-jc-own-ranker). Built by '
                            'kernels/make_own.py.\n' + ''.join(nb['cells'][0]['source']))
ds = [d for d in meta['dataset_sources'] if d != 'ahmedberatozer/casmi26-fpnet-full1']
ds += ['giantwizard/casmi26-jc-own-ranker', 'dmitriigluzdov/casmi26-fold-safe-fpnet']
meta.update(id='giantwizard/casmi26-jc-v7c-own', title='casmi26-jc-v7c-own', code_file='casmi26-jc-v7c-own.ipynb', dataset_sources=ds)
os.makedirs(OUT, exist_ok=True)
json.dump(nb, open(f'{OUT}/casmi26-jc-v7c-own.ipynb', 'w'), indent=1); json.dump(meta, open(f'{OUT}/kernel-metadata.json', 'w'), indent=1)
print('wrote', OUT, '| datasets', len(ds))
