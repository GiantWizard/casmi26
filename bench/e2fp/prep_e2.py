"""Build the inputs of engine 2's FPNet trainers (third_party/train_fp3.py, train_fp4.py) with our holdout withheld.

Same layout as the legacy recipe (cubres/enveda-casmi-2026, reproduction/legacy-v2/public_source: build_fps, build_dataset,
build_pool), with two changes:
  * is_val = the structure is in the fold-safe split (identity fold 0 + NP panel), so the trainers never fit on it;
  * the 6,930-bit selection is NOT recomputed: we use the fp_bits.npy shipped with prvsiyan/coconut-casmi26-candidates,
    the one engine 2 uses at inference, so the new checkpoints are drop-ins for fp_single_s2 / fp_merged_m1.
The COCONUT part of the pool is that dataset's coco_fp/coco_mass/coco_meta (same bits), so no COCONUT CSV is needed.

usage: prep_e2.py TRAIN_PARQUET COCO_DIR SPLIT_PARQUET OUT_DIR [--workers N]
writes OUT_DIR/ds/spec.npz, OUT_DIR/up_train/fp_meta.pkl, OUT_DIR/pool_{fp,mass}.npy, OUT_DIR/pool_meta.pkl
"""
import argparse, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party'))
import fpmodel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('train'); ap.add_argument('coco'); ap.add_argument('split'); ap.add_argument('out')
ap.add_argument('--workers', type=int, default=os.cpu_count())
a = ap.parse_args()
t0 = time.time()
log = lambda m: print(f'[{time.time() - t0:6.0f}s] {m}', flush=True)
BITS = np.load(os.path.join(a.coco, 'fp_bits.npy'))
assert len(BITS) == 6930, len(BITS)
_g = {}


def fp_and_mass(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    from rdkit.Chem.Descriptors import ExactMolWt
    if not _g:
        RDLogger.DisableLog('rdApp.*')
        _g['m2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
        _g['m3'] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
        _g['rk'] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        fp = np.concatenate([_g['m2'].GetFingerprintAsNumPy(m).astype(np.uint8), _g['m3'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g['rk'].GetFingerprintAsNumPy(m).astype(np.uint8), np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])
        return np.packbits(fp[BITS]), float(ExactMolWt(m))
    except Exception:
        return None


def prep_chunk(args):
    return [fpmodel.prep_peaks(mz, it, p) for mz, it, p in zip(*args)]


if __name__ == '__main__':
    # ---- spectra (speclib.load_flat: both peak arrays rounded to float32 first) ----
    cols = ['inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz', 'instrument_type', 'ionization_mode']
    t = pq.read_table(a.train, columns=cols + ['ms2_mzs', 'ms2_normalized_intensities', 'precursor_error_ppm', 'collision_energy_ev'])
    mzc = t.column('ms2_mzs').combine_chunks(); itc = t.column('ms2_normalized_intensities').combine_chunks()
    off = mzc.offsets.to_numpy().astype(np.int64)
    AMZ = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    AIN = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    meta = {c: (t.column(c).to_numpy(zero_copy_only=False) if pa.types.is_floating(t.column(c).type)
                else np.asarray(t.column(c).cast(pa.string()).to_pylist(), dtype=object)) for c in cols}
    ppm = t.column('precursor_error_ppm').to_numpy(zero_copy_only=False).astype(np.float64)
    ce = np.array([float(np.mean(v)) if v is not None and len(v) > 0 else -1.0 for v in t.column('collision_energy_ev').to_pylist()], np.float32)
    del t, mzc, itc
    n = len(off) - 1
    log(f'{n:,} spectra, {len(AMZ):,} peaks')

    # ---- structures: training structures with valid fingerprints ----
    ik, smi = meta['inchikey14'], meta['normalized_smiles']
    first = pd.DataFrame({'k': ik, 's': smi}).dropna()
    first = first[(first.k != '') & (first.s != '')].drop_duplicates('k').sort_values('k')
    with Pool(a.workers) as p:
        res = p.map(fp_and_mass, list(first.s), chunksize=500)
    ok = [i for i, r in enumerate(res) if r is not None]
    tr_keys = list(first.k.values[ok]); tr_smi = list(first.s.values[ok])
    tr_fp = np.stack([res[i][0] for i in ok]); tr_mass = np.array([res[i][1] for i in ok]); del res
    log(f'training structures with fingerprints: {len(tr_keys):,}')
    os.makedirs(os.path.join(a.out, 'up_train'), exist_ok=True)
    pickle.dump({'keys': tr_keys, 'smiles': tr_smi, 'nbits': 6930}, open(os.path.join(a.out, 'up_train', 'fp_meta.pkl'), 'wb'), protocol=4)

    # ---- pool = training structures + COCONUT (legacy build_pool, COCONUT rows taken prebuilt) ----
    cm = pickle.load(open(os.path.join(a.coco, 'coco_meta.pkl'), 'rb'))
    co_keys = np.asarray(cm['keys'], dtype=object); co_smi = np.asarray(cm['smiles'], dtype=object)
    co_fp = np.load(os.path.join(a.coco, 'coco_fp.npy')); co_mass = np.load(os.path.join(a.coco, 'coco_mass.npy'))
    # bit-layout check: our fingerprints of a few COCONUT SMILES must equal the shipped rows
    chk = np.random.default_rng(0).choice(len(co_keys), 200, replace=False)
    same = [np.array_equal(fp_and_mass(co_smi[i])[0], co_fp[i, :tr_fp.shape[1]]) for i in chk if fp_and_mass(co_smi[i]) is not None]
    log(f'bit-layout check vs shipped COCONUT fingerprints: {np.mean(same):.3f} identical ({len(same)} checked)')
    assert np.mean(same) > 0.95, 'fingerprint layout differs from engine 2'
    new = ~pd.Series(co_keys).isin(set(tr_keys)).values
    FP = np.concatenate([tr_fp, co_fp[new][:, :tr_fp.shape[1]]]); MASS = np.concatenate([tr_mass, co_mass[new]])
    KEYS = np.concatenate([np.array(tr_keys, dtype=object), co_keys[new]]); SMI = np.concatenate([np.array(tr_smi, dtype=object), co_smi[new]])
    SRC = np.concatenate([np.zeros(len(tr_keys), np.int8), np.ones(int(new.sum()), np.int8)])
    good = np.isfinite(MASS) & (MASS >= 80) & (MASS <= 1400)
    o = np.flatnonzero(good)[np.argsort(MASS[good], kind='mergesort')]
    np.save(os.path.join(a.out, 'pool_fp.npy'), FP[o]); np.save(os.path.join(a.out, 'pool_mass.npy'), MASS[o])
    pickle.dump({'keys': KEYS[o], 'smiles': SMI[o], 'src': SRC[o], 'npl': np.full(len(o), np.nan), 'nbits': 6930},
                open(os.path.join(a.out, 'pool_meta.pkl'), 'wb'), protocol=4)
    log(f'pool {len(o):,} structures ({int(new.sum()):,} from COCONUT)')
    del FP, co_fp

    # ---- usable spectra, holdout flag, peak preprocessing (legacy build_dataset) ----
    key2ix = {k: i for i, k in enumerate(tr_keys)}
    six = np.array([key2ix.get(k, -1) for k in ik], np.int32)
    npk = np.diff(off)
    use = np.isfinite(meta['precursor_mz']) & (npk >= 3) & ~(np.isfinite(ppm) & (np.abs(ppm) > 20)) & (six >= 0)
    idx = np.flatnonzero(use)
    hold = set(pd.read_parquet(a.split).inchikey14)
    is_val = np.array([ik[i] in hold for i in idx])
    log(f'usable spectra {len(idx):,}; withheld {is_val.sum():,} spectra of {len({ik[i] for i in idx[is_val]}):,} '
        f'structures (split lists {len(hold):,}); training {(~is_val).sum():,}')
    CH = 20000
    chunks = [([AMZ[off[i]:off[i + 1]] for i in idx[s:s + CH]], [AIN[off[i]:off[i + 1]] for i in idx[s:s + CH]],
               [meta['precursor_mz'][i] for i in idx[s:s + CH]]) for s in range(0, len(idx), CH)]
    del AMZ, AIN
    res = []
    with Pool(a.workers) as p:
        for j, r in enumerate(p.imap(prep_chunk, chunks)):
            res.extend(r)
            if j % 20 == 0:
                log(f'  peaks chunk {j}/{len(chunks)}')
    lens = np.array([len(x) for x, _ in res], np.int64)
    noff = np.zeros(len(res) + 1, np.int64); np.cumsum(lens, out=noff[1:])
    ad = np.array([fpmodel.ADDUCT_IX.get(x, fpmodel.ADDUCT_IX['<unk>']) for x in meta['adduct'][idx]], np.int16)
    ins = np.array([fpmodel.instr_family(s) for s in meta['instrument_type'][idx]], np.int8)
    mode = np.array([1.0 if str(m) == 'positive' else -1.0 for m in meta['ionization_mode'][idx]], np.float32)
    os.makedirs(os.path.join(a.out, 'ds'), exist_ok=True)
    np.savez(os.path.join(a.out, 'ds', 'spec.npz'), off=noff, mz=np.concatenate([x for x, _ in res]).astype(np.float32),
             it=np.concatenate([y for _, y in res]).astype(np.float32), prec=meta['precursor_mz'][idx].astype(np.float32), ad=ad,
             ins=ins, ce=ce[idx], mode=mode, six=six[idx], is_val=is_val, src=idx.astype(np.int64))
    log('saved ' + a.out)
