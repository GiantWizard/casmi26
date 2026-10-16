"""Leak-free ranker training rows: run the public v1 engine (+ fe_v4 families) on held-out training structures as if
they were test molecules, and label every candidate.

Structures come from the fold-safe split's 'hold' fold (55,102 structures the fold-safe FPNet never saw), minus the
v4n fold0_np validation molecules and the isomer-benchmark panels. The engine uses that fold-safe FPNet, so fingerprint
features on these molecules behave like the full FPNet's do on unseen test molecules.

Each structure is simulated as one of three test classes:
  class 1 (15%): its spectra from the query library are hidden; spectra from other libraries stay searchable
  class 2 (60%): every spectrum of the structure (and its metric-key twins) is hidden; the structure stays in the pool
  class 3 (25%): as class 2, and the structure is also removed from the pool (only generation can recover it)

usage: gen_rows.py --v4code DIR --v4mod DIR --work DIR --train PARQUET --fpnet PT --split PARQUET --exclude FILE...
                   --out DIR [--n 8000] [--shard 0 --nshards 1] [--seed 0] [--max-keep 300] [--no-selftest] [--cpu]
Writes OUT/rows_<shard>_<chunk>.npz every --chunk molecules; finished molecules are skipped on restart.
"""
import argparse, glob, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('--v4code', required=True); ap.add_argument('--v4mod', required=True); ap.add_argument('--work', required=True)
ap.add_argument('--train', required=True); ap.add_argument('--fpnet', required=True, nargs='+'); ap.add_argument('--split', required=True)
ap.add_argument('--exclude', nargs='*', default=[]); ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=8000); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max-keep', type=int, default=300)
ap.add_argument('--max-spec', type=int, default=6); ap.add_argument('--chunk', type=int, default=50)
ap.add_argument('--no-selftest', action='store_true'); ap.add_argument('--cpu', action='store_true')
ap.add_argument('--prep-raw', default=None); ap.add_argument('--raw', default=None)
ap.add_argument('--sids-file', default=None, help='parquet/npy of train sids to simulate (e.g. NP-like held-out structures)')
ap.add_argument('--fold', default='hold', help="split fold to simulate: 'hold' (55k) or 'np' (Enveda NP panel)"); ap.add_argument('--np-only', default=None, help='pool_np.npy: keep only natural-product structures (COCONUT/LOTUS/NPAtlas)')
a = ap.parse_args()

import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
_np_load = np.load
def _mmap_load(f, *x, **k):                       # big read-only engine arrays: map instead of copy, so parallel
    if isinstance(f, str) and f.endswith('.npy') and 'mmap_mode' not in k and os.path.getsize(f) > 50_000_000:
        k['mmap_mode'] = 'r'                       # processes share one copy in the page cache
    return _np_load(f, *x, **k)
np.load = _mmap_load
sys.path.insert(0, a.v4code); sys.path.insert(1, os.path.join(a.v4code, 'fe_v4'))
os.chdir(os.path.dirname(os.path.abspath(a.work)))
WORK = os.path.basename(os.path.abspath(a.work))
from casmi.library import Library
from casmi.pool import Pool
from casmi.engine import Engine, EngineCfg, FEATURES
from casmi import fpnet
import v1engine

TEST_ADDUCTS = ['[M+H]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+', '[M+Na]+', '[M+K]+',
                '[M-H]-', '[M-H2O-H]-', '[M+CH2O2-H]-', '[M+Cl]-']
os.makedirs(a.out, exist_ok=True)
dev = 'cpu' if a.cpu or not torch.cuda.is_available() else 'cuda'
T0 = time.time()
log = lambda *x: print(f'[{time.time() - T0:7.0f}s]', *x, flush=True)

# ---------------------------------------------------------------- engine (as in the v4n notebook, fold-safe FPNet)
man = json.load(open(os.path.join(a.v4mod, 'MANIFEST.json')))
L = Library(WORK); P = Pool(WORK)
tfp = np.load(os.path.join(WORK, 'train_fp_sel.npy'))
bank = fpnet.ModelBank(list(a.fpnet), device=dev)   # several checkpoints = averaged logits (ensemble)
E = Engine(L, P, tfp, EngineCfg(generate=True), bank)
V = v1engine.V1FE(E, [os.path.join(a.v4mod, m['file']) for m in man['fe_models']], man['families'], device=dev)
if 'derivation' in man['families'] and not a.no_selftest:
    bad = V.mods['derivation'].self_test()          # rows and the new ranker share one RDKit build, so a mismatch with
    print('derivation self-test mismatches:', bad, flush=True)   # the author's build is logged, not fatal
log('engine ready on', dev, '| families', man['families'])

# ---------------------------------------------------------------- structures to simulate
st = pd.read_parquet(os.path.join(WORK, 'train_structs.parquet'), columns=['sid', 'key', 'inchikey14'])
split = pd.read_parquet(a.split)
excl = set()
for f in a.exclude:
    if f.endswith('.csv'):
        excl |= set(pd.read_csv(f).inchikey14.astype(str))
    else:
        excl |= set(open(f).read().split())
hold = set(split.loc[split.fold == a.fold, 'inchikey14']) - excl
cand_sids = st.sid.values[st.inchikey14.isin(hold).values]
key_of = st.key.values
twins = pd.Series(st.sid.values).groupby(st.key.values).apply(list).to_dict()
pid_of_key = {k: i for i, k in enumerate(P.key)}
meta = L.meta
spec_ok = meta.adduct.isin(TEST_ADDUCTS).values
by_sid = pd.Series(np.flatnonzero(spec_ok)).groupby(L.sid[spec_ok]).apply(np.asarray).to_dict()
cand_sids = np.array([s for s in cand_sids if s in by_sid and key_of[s] in pid_of_key])
if a.np_only:                                      # natural-product structures only (the test's chemistry)
    _npf = np.load(a.np_only)
    cand_sids = np.array([s for s in cand_sids if _npf[pid_of_key[key_of[s]]] > 0])
if a.sids_file:                                    # explicit query list (same list + seed = identical simulation across banks)
    _keep = set((pd.read_parquet(a.sids_file).sid if a.sids_file.endswith('.parquet') else np.load(a.sids_file)).tolist())
    cand_sids = np.array([s for s in cand_sids if int(s) in _keep])
rng = np.random.default_rng(a.seed)
chosen = rng.permutation(cand_sids)[:a.n]
mine = chosen[a.shard::a.nshards]
cls_draw = np.random.default_rng(a.seed + 1).random(len(chosen))
cls_of = {int(s): (1 if u < 0.15 else (3 if u > 0.75 else 2)) for s, u in zip(chosen, cls_draw)}
log(f'hold structures {len(hold)} | usable {len(cand_sids)} | chosen {len(chosen)} | this shard {len(mine)}')

# raw peaks of the selected rows (the spec cache holds cleaned peaks; queries use raw peaks like the test file)
assert len(meta) == pq.ParquetFile(a.train).metadata.num_rows, 'spec cache rows must align with train.parquet rows'
if a.prep_raw:                                     # one pass over train.parquet for every shard's molecules
    mine = chosen
need = np.unique(np.concatenate([by_sid[int(s)] for s in mine]))
if a.raw:
    raw = pd.read_parquet(a.raw); raw = raw.loc[raw.index.isin(need)]
else:
  pf = pq.ParquetFile(a.train)
  rg_end = np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
  rg_start = rg_end - np.array([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
  parts = []
  for i in np.unique(np.searchsorted(rg_end, need, side='right')):   # read only the row groups that hold needed rows
      loc = need[(need >= rg_start[i]) & (need < rg_end[i])]
      t_ = pf.read_row_group(int(i), columns=['ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct',
                                              'ionization_mode', 'collision_energy_ev'])
      d_ = t_.take(loc - rg_start[i]).to_pandas(); d_.index = loc
      parts.append(d_)
  raw = pd.concat(parts)
  del parts

if a.prep_raw:
    raw.to_parquet(a.prep_raw); log('wrote', a.prep_raw, len(raw), 'spectra for', len(mine), 'molecules'); sys.exit(0)
done = set()
for f in glob.glob(os.path.join(a.out, f'rows_{a.shard}_*.npz')):
    done |= set(np.load(f)['sids'].tolist())
log('already done', len(done))


def spectra_of(rows):
    out = []
    for i, r in rows.iterrows():
        ce = r.collision_energy_ev
        ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
        out.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                        mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct, prec=float(r.precursor_mz),
                        ce=float(ce_arr.mean()) if len(ce_arr) else np.nan, ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
    return out


fz_col = list(FEATURES).index('fz_rank')
buf, names, chunk_id = [], None, len(glob.glob(os.path.join(a.out, f'rows_{a.shard}_*.npz')))


def flush():
    global buf, chunk_id
    if not buf:
        return
    np.savez_compressed(os.path.join(a.out, f'rows_{a.shard}_{chunk_id:04d}.npz'),
                        X=np.concatenate([b['X'] for b in buf]).astype(np.float16),
                        y=np.concatenate([b['y'] for b in buf]).astype(np.int8),
                        g=np.concatenate([np.full(len(b['y']), b['sid'], np.int64) for b in buf]),
                        cls=np.array([b['cls'] for b in buf], np.int8), sids=np.array([b['sid'] for b in buf], np.int64),
                        n_cand=np.array([b['n_cand'] for b in buf], np.int32), pos_in=np.array([b['pos_in'] for b in buf], np.int8),
                        names=np.array(list(FEATURES) + list(names or [])))
    chunk_id += 1; buf = []


n_ok = n_err = n_pos = 0
for t, s in enumerate(mine):
    s = int(s)
    if s in done:
        continue
    try:
        cls = cls_of[s]
        rows = by_sid[s]
        r_rng = np.random.default_rng(a.seed * 1000003 + s)
        if cls == 1:                                           # one library's spectra are the query, others stay
            qlib = int(r_rng.choice(L.lib[rows]))
            rows = rows[L.lib[rows] == qlib]
        rows = r_rng.permutation(rows)[:a.max_spec]
        spectra = spectra_of(raw.loc[rows])
        target = float(np.median(meta.nm.values[rows]))
        tw = np.array(twins[key_of[s]], np.int64)
        if cls == 1:
            exclude = np.isin(L.sid, tw) & (L.lib == qlib)
            sim = dict(exclude=exclude, exclude_sid=s, exclude_lib=qlib)
        else:
            exclude = np.isin(L.sid, tw)
            sim = dict(exclude=exclude, exclude_sid=s)
            if cls == 3:
                sim['drop_pid'] = pid_of_key[key_of[s]]
        C, X, F, nm, info = V.run(spectra, target, **sim)
        if names is None and nm:
            names = nm
        if len(C.pid) == 0:
            continue
        y = np.array([k == key_of[s] for k in C.key], np.int8)
        Xa = np.concatenate([X, F], 1) if F.shape[1] else X
        keep = np.flatnonzero(y)
        if len(C.pid) > a.max_keep:                            # keep every positive + the fingerprint's hardest negatives
            order = np.argsort(X[:, fz_col], kind='stable')
            keep = np.unique(np.concatenate([keep, order[:a.max_keep]]))
        else:
            keep = np.arange(len(C.pid))
        buf.append(dict(X=Xa[keep], y=y[keep], sid=s, cls=cls, n_cand=len(C.pid), pos_in=int(y.any())))
        n_ok += 1; n_pos += int(y.any())
    except Exception as e:
        n_err += 1
        print('ERROR', s, repr(e)[:200], flush=True)
    if len(buf) >= a.chunk:
        flush()
    if (t + 1) % 25 == 0:
        log(f'{t + 1}/{len(mine)} | ok {n_ok} | truth among candidates {n_pos} | errors {n_err}')
flush()
log(f'finished | ok {n_ok} | truth among candidates {n_pos} | errors {n_err}')
