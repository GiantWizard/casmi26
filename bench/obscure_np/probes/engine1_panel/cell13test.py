"""Main-channel development harness on the enveda-np-examples panel (conditional proxy, NOT clean CV: the frozen
ranker and feature resources may have seen these molecules).

Scenarios (the engine's own simulation switches):
  lib     hide only the panel's NP-example spectra; spectra of the same molecule from other libraries stay   (class 1)
  nospec  hide every reference spectrum of the molecule; its structure stays in the pool                    (class 2, in pool)
  dbdrop  nospec + remove the structure from the pool: only generation or the PubChem channel can find it   (class 2 PubChem-only / class 3)

usage: python main_tables.py <v4b_dir> <pool_dir> <train.parquet> <pop_dir> <work_dir> <out_dir> <scenario> [limit]
Writes <out_dir>/<scenario>.pkl: per molecule the full ranked candidate table (keys, pids, scores, formulas, flags).
"""
import json, os, pickle, sys, time
import numpy as np
import pandas as pd

V4, POOL, TRAIN, POP, WORK, OUT, SCEN = sys.argv[1:8]
LIMIT = int(sys.argv[8]) if len(sys.argv) > 8 else 0
V4CODE = os.path.join(V4, 'code')
os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(WORK.rstrip('/'))))
wd = os.path.basename(WORK.rstrip('/'))
for f in ['pool_meta.parquet', 'pool_fp.npy', 'pool_frag_off.npy', 'pool_frag_mass.npy', 'fp_bits.npy', 'train_structs.parquet', 'train_fp_sel.npy']:
    dst = os.path.join(wd, f)
    if not os.path.exists(dst):
        os.symlink(os.path.join(POOL, f), dst)
sys.path.insert(0, V4CODE)
T0 = time.time()


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


from casmi.build import build_spec_cache
if not os.path.exists(os.path.join(wd, 'spec_meta.parquet')):
    build_spec_cache(TRAIN, wd)
sys.path.insert(1, os.path.join(V4CODE, 'fe_v4'))
import torch, lightgbm as lgb
import pyarrow.parquet as pq
from casmi.library import Library
from casmi.pool import Pool
from casmi.engine import Engine, EngineCfg, FEATURES
from casmi import fpnet, chem
import v1engine

MAN = json.load(open(os.path.join(V4, 'MANIFEST.json')))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
L = Library(wd); P = Pool(wd)
tfp = np.load(os.path.join(wd, 'train_fp_sel.npy'))
bank = fpnet.ModelBank([os.path.join(V4, 'fpnet_0.pt')], device=dev)
_cfg = EngineCfg(generate=True)
for _k, _v in json.loads(os.environ.get('GEN_CFG', '{}')).items():      # ADDED: generation-config overrides
    setattr(_cfg, _k, type(getattr(_cfg, _k))(_v))
print('EngineCfg', _cfg, flush=True)
E = Engine(L, P, tfp, _cfg, bank)
V = v1engine.V1FE(E, [os.path.join(V4, m['file']) for m in MAN['fe_models']], MAN['families'], device=dev)
RK = pickle.load(open(os.path.join(V4, MAN['ranker']['file']), 'rb'))
RK_FEATS = list(RK['features']); RK_B = [lgb.Booster(model_str=s) for s in RK['boosters']]


def rank_score(X, names):
    pos = {n: i for i, n in enumerate(names)}
    Xs = np.ascontiguousarray(X[:, [pos[f] for f in RK_FEATS]], dtype=np.float32)
    return np.mean([b.predict(Xs, num_threads=4) for b in RK_B], axis=0)


POOL_POP = np.load(os.path.join(POP, 'pool_lsid.npy')).astype(np.float32) + np.load(os.path.join(POP, 'pool_lpmid.npy')).astype(np.float32)
ts = pd.read_parquet(os.path.join(wd, 'train_structs.parquet'))
NPBIT = 9
np_keys = sorted(set(ts.loc[(ts.libmask.astype('int64') & (1 << NPBIT)) != 0, 'key']))
alias = ts[ts.key.isin(np_keys)]
raw2key = dict(zip(alias.inchikey14, alias.key))
sids_of = alias.groupby('key').sid.apply(lambda s: sorted(set(int(x) for x in s))).to_dict()
pool_pid = {k: i for i, k in enumerate(P.key)}
cols = ['ingest_lib', 'inchikey14', 'ionization_mode', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities', 'collision_energy_ev']
pf = pq.ParquetFile(TRAIN); parts = []
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=cols).to_pandas()
    parts.append(t[t.ingest_lib == 'enveda-np-examples'])
q = pd.concat(parts, ignore_index=True)
q['key'] = q.inchikey14.map(raw2key)
q['nm'] = chem.neutral_mass(q.precursor_mz.values.astype(np.float64), q.adduct.values)
log('engine ready', f'{time.time()-T0:.0f}s', 'queries', len(q), 'molecules', q.key.nunique(), 'device', dev, 'scenario', SCEN)


import json as _json
nb = _json.load(open('/home/user/casmi26/kernels/jc-gen/casmi26-jc-gen.ipynb'))
c13 = ''.join(nb['cells'][13]['source']); c17 = ''.join(nb['cells'][17]['source'])
# a 6-molecule "test.parquet" from panel spectra (3 normal + 3 whose structure we can't drop here; smoke only)
keys6 = list(q.key.dropna().unique()[:6])
te_df = q[q.key.isin(keys6)].copy(); te_df['molecule_id'] = te_df.key.map({k: f'm{i}' for i, k in enumerate(keys6)})
te_df['spectrum_id'] = [f's{i}' for i in range(len(te_df))]
COMP = '/tmp/cell13_comp'; os.makedirs(COMP, exist_ok=True)
te_df.drop(columns=['key', 'nm']).to_parquet(os.path.join(COMP, 'test.parquet'))
TOPN = 60
def find(n):
    return os.path.join(POP, n)
g = dict(globals()); g.update(V=V, E=E, rank_score=rank_score, FEATURES=FEATURES, chem=chem, P=P, np=np, pd=pd, os=os, time=time,
                              T0=time.time(), COMP=COMP, TOPN=TOPN, find=find)
exec(c13, g)
print('BASE molecules', len(g['BASE']), '| GEN_TOP_SCORE', g['GEN_TOP_SCORE'], '| n_err', g['n_err'])
for mid, (smis, keys, lib_max, scs, forms) in g['BASE'].items():
    print(mid, len(smis), 'gen-tagged', sum(f.endswith('|gen') for f in forms), 'lib_max %.2f' % lib_max)
