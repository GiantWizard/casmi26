"""MIST (coleygroup/MIST, MassSpecGym augmented checkpoint; the real-only one predicts a constant fingerprint) as a same-formula re-ordering signal for a finished submission.

For each molecule and each molecular formula shared by two or more of its listed structures, MIST predicts a 4,096-bit
Morgan fingerprint from every positive-mode spectrum, given that formula (peak formulas assigned within 20 ppm, the 10
most intense assigned peaks plus the precursor token, as in the MSG training caches). A candidate's score is the cosine
between its own fingerprint and the prediction, averaged over spectra. Members of the formula group are re-ordered
within the positions they already hold by z(-position) + LAM * z(MIST); everything else keeps its place.
"""
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from mist.data.featurizers import assign_subformulae
from mist.models.scaled_mist import ScaledMIST
from mist.utils import chem_utils

RDLogger.DisableLog('rdApp.*')
ION = {'[M+H]+': '[M+H]+', '[M+Na]+': '[M+Na]+', '[M+K]+': '[M+K]+', '[M+NH4]+': '[M+H3N+H]+', '[M-H2O+H]+': '[M-H2O+H]+'}
MAX_PEAKS, PPM = 10, 20.0
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)


def load(ckpt, device):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    m = ScaledMIST(**ck['config']['model'])
    m.load_state_dict(ck['ema_model'], strict=True)
    return m.to(device).eval()


def formula_of(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        f = CalcMolFormula(m) if m is not None else None
        if f and all(ch not in f for ch in '+-'):
            chem_utils.formula_to_dense(f)               # raises on elements MIST cannot represent
            return f
    except Exception:
        pass
    return None


def fingerprint(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else _MORGAN.GetFingerprintAsNumPy(m).astype(np.float32)


def featurize(mz, it, prec, formula, adduct):
    """One spectrum + one candidate formula -> MIST input arrays, or None when no peak is assignable."""
    mz = np.asarray(mz, float); it = np.asarray(it, float)
    keep = (mz <= prec + 1) & (it > 0)
    mz, it = mz[keep], it[keep]
    if not len(mz):
        return None
    it = np.sqrt(it / it.max())
    o = np.argsort(-it)[:60]
    mz, it = mz[o], it[o]
    mz, it = mz[it > 0.001], it[it > 0.001]
    tree = assign_subformulae({'cand_form': formula, 'cand_ion': adduct,
                               'output_tbl': {'mz': mz.tolist(), 'ms2_inten': it.tolist()}}, PPM)
    tab = tree['output_tbl']
    if tab is None:
        return None
    o = np.argsort(-np.asarray(tab['ms2_inten']))[:MAX_PEAKS]
    forms = [chem_utils.formula_to_dense(tab['formula'][i]) for i in o] + [chem_utils.formula_to_dense(formula)]
    ions = [chem_utils.get_ion_idx(tab['ions'][i]) for i in o] + [chem_utils.get_ion_idx(adduct)]
    intens = [tab['ms2_inten'][i] for i in o] + [1.0]
    types = [0] * len(o) + [3]
    W, n = MAX_PEAKS + 1, len(forms)
    fv = np.zeros((W, len(chem_utils.VALID_ELEMENTS)), np.float32); fv[:n] = np.asarray(forms)
    iv = np.zeros(W, np.int64); iv[:n] = ions
    tv = np.zeros(W, np.int64); tv[:n] = types
    nv = np.zeros(W, np.float32); nv[:n] = intens
    return dict(form_vec=fv, ion_vec=iv, intens=nv, types=tv, num_peaks=n, instruments=chem_utils.get_instr_idx('Unknown'))


@torch.inference_mode()
def predict(model, items, device, bs=256):
    out = []
    for s in range(0, len(items), bs):
        b = items[s:s + bs]
        batch = {k: torch.as_tensor(np.stack([x[k] for x in b])) for k in ('form_vec', 'ion_vec', 'intens', 'types')}
        batch['num_peaks'] = torch.as_tensor([x['num_peaks'] for x in b], dtype=torch.long)
        batch['instruments'] = torch.as_tensor([x['instruments'] for x in b], dtype=torch.long)
        p, _ = model({k: v.to(device) for k, v in batch.items()})
        out.append(p.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 4096), np.float32)


def cos(p, f):
    return float(p @ f / (np.linalg.norm(p) * np.linalg.norm(f) + 1e-9))


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9) if len(v) > 1 else np.zeros(len(v))


def rerank(sub, test, model, device, lam=1.0):
    """sub: DataFrame molecule_id, smiles (';'-joined); test: test.parquet rows. Returns (new sub, stats)."""
    pos = test[test.adduct.isin(list(ION))]
    by = {m: g for m, g in pos.groupby('molecule_id')}
    jobs, keys = [], []                                   # (molecule, formula, spectrum) prediction jobs
    plan = {}
    for mid, smis in zip(sub.molecule_id, sub.smiles):
        g = by.get(mid)
        toks = str(smis).split(';')
        if g is None or len(toks) < 2:
            continue
        forms = [formula_of(s) for s in toks]
        groups = {}
        for i, f in enumerate(forms):
            if f:
                groups.setdefault(f, []).append(i)
        groups = {f: ix for f, ix in groups.items() if len(ix) >= 2}
        if not groups:
            continue
        plan[mid] = (toks, groups)
        for f in groups:
            for r in g.itertuples():
                x = featurize(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz), f, ION[r.adduct])
                if x is not None:
                    jobs.append(x); keys.append((mid, f))
    P = predict(model, jobs, device)
    pred = {}
    for k, p in zip(keys, P):
        pred.setdefault(k, []).append(p)
    out, st = [], dict(molecules=0, groups=0, changed_top1=0, changed=0, jobs=len(jobs))
    for mid, smis in zip(sub.molecule_id, sub.smiles):
        if mid not in plan:
            out.append(smis); continue
        toks, groups = plan[mid]
        new = list(toks); touched = False
        for f, ix in groups.items():
            ps = pred.get((mid, f))
            if not ps:
                continue
            fps = [fingerprint(toks[i]) for i in ix]
            if any(x is None for x in fps):
                continue
            sc = np.array([np.mean([cos(p, fp) for p in ps]) for fp in fps])
            comb = z(-np.asarray(ix, float)) + lam * z(sc)
            order = [ix[j] for j in np.argsort(-comb, kind='stable')]
            for slot, src in zip(ix, order):
                new[slot] = toks[src]
            st['groups'] += 1; touched = True
        st['molecules'] += touched
        st['changed'] += int(new != toks); st['changed_top1'] += int(new[0] != toks[0])
        out.append(';'.join(new))
    res = sub.copy(); res['smiles'] = out
    return res, st


def selftest(train_path, model, device, n=60, seed=0):
    """Bug check on Enveda panel [M+H]+ spectra: rank of the true structure among same-formula training structures by
    MIST cosine (MassSpecGym overlaps the training file, so this checks wiring, not accuracy)."""
    import pandas as pd, pyarrow.parquet as pq
    cols = ['ingest_lib', 'normalized_smiles', 'inchikey14', 'molecular_formula', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']
    meta = pq.read_table(train_path, columns=['molecular_formula', 'normalized_smiles', 'inchikey14']).to_pandas().drop_duplicates('inchikey14')
    pf = pq.ParquetFile(train_path); rows = []
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i, columns=cols).to_pandas()
        rows.append(t[(t.ingest_lib == 'enveda-np-examples') & (t.adduct == '[M+H]+')])
    d = pd.concat(rows).drop_duplicates('inchikey14').sample(frac=1, random_state=seed)
    ranks = []
    for r in d.itertuples():
        f = formula_of(r.normalized_smiles)
        iso = meta[(meta.molecular_formula == r.molecular_formula) & (meta.inchikey14 != r.inchikey14)].normalized_smiles.head(30).tolist()
        x = featurize(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz), f, '[M+H]+') if f else None
        if x is None or len(iso) < 5:
            continue
        p = predict(model, [x], device)[0]
        s = [cos(p, fingerprint(s)) for s in [r.normalized_smiles] + iso if fingerprint(s) is not None]
        ranks.append((1 + sum(v > s[0] for v in s[1:]), len(s)))
        if len(ranks) >= n:
            break
    r = np.array(ranks)
    return dict(n=len(r), top1=float((r[:, 0] == 1).mean()), mean_rank=float(r[:, 0].mean()), mean_group=float(r[:, 1].mean()),
                random_mean_rank=float(((r[:, 1] + 1) / 2).mean())) if len(r) else dict(n=0)
