"""MIST (FRIGID's MassSpecGym encoder, trained with ICEBERG-simulated PubChem spectra) as a same-formula re-scorer.

For every query spectrum with a positive-mode adduct MIST knows, and every candidate formula, the spectrum's peaks are
annotated with subformulae of that formula (MIST's own assign_subforms, 20 ppm), MIST predicts a 4096-bit Morgan
(r=2) fingerprint, and each candidate of that formula scores cosine(predicted probabilities, its fingerprint).
A candidate's score is the mean over the molecule's usable spectra. Output: {mid: {smiles: score | None}}, the same
shape as the GLACIER/ICEBERG score dicts, so gl_fuse.rerank_multi and lr_core take it unchanged.

usage: mist_score.py ITEMS_JSON OUT_JSON --ckpt mist_msg.pt [--pkg DIR] [--device cpu]
ITEMS_JSON: list of {mid, spectra: [{mz, it, prec, adduct, ...}], cands: [smiles]} (gl_work/gl_in.json works).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('items'); ap.add_argument('out')
ap.add_argument('--ckpt', required=True)
ap.add_argument('--pkg', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pkg'))
ap.add_argument('--device', default='cpu')
ap.add_argument('--ppm', type=float, default=20.0)
ap.add_argument('--batch', type=int, default=64)
a = ap.parse_args()
sys.path.insert(0, a.pkg)
from mist import utils  # noqa: E402
from mist.models.spectra_encoder import SpectraEncoderGrowing  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

RDLogger.DisableLog('rdApp.*')
ADDUCT = {'[M+H]+': '[M+H]+', '[M+Na]+': '[M+Na]+', '[M+K]+': '[M+K]+', '[M+NH4]+': '[M+H3N+H]+',
          '[M+H-H2O]+': '[M-H2O+H]+', '[M-H2O+H]+': '[M-H2O+H]+', '[M]+': '[M]+'}
MSG_CFG = dict(form_embedder='pos-cos', output_size=4096, hidden_size=640, spectra_dropout=0.1, peak_attn_layers=2,
               num_heads=8, set_pooling='cls', refine_layers=4, pairwise_featurization=True, embed_instrument=False,
               inten_transform='float', magma_modulo=2048, inten_prob=0.1, remove_prob=0.5, cls_type='ms1',
               spec_features='peakformula', mol_features='fingerprint', top_layers=1)


def load_encoder(path, device):
    cfg = dict(MSG_CFG)
    if 'canopus' in os.path.basename(path):          # FRIGID's NPLIB1 MIST is the small model
        cfg.update(hidden_size=512, magma_modulo=512)
    enc = SpectraEncoderGrowing(**cfg)
    ck = torch.load(path, map_location='cpu', weights_only=True)
    sd = ck.get('state_dict', ck) if isinstance(ck, dict) else ck
    sub = {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')}
    sd = sub or sd
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    print(f'MIST weights: {len(sd)} tensors, missing {len(missing)}, unexpected {len(unexpected)}', flush=True)
    if missing:
        print('  missing e.g.', missing[:5], flush=True)
    return enc.to(device).eval()


def proc_spec(mz, it, prec):
    """MIST's process_spec_file for one spectrum: drop peaks above precursor + 1, max-normalise, sqrt, top 60, > 0.001."""
    s = np.array([mz, it], dtype=np.float64).T
    if not len(s):
        return None
    s = s[s[:, 0] <= prec + 1]
    if not len(s) or s[:, 1].max() <= 0:
        return None
    s[:, 1] = np.sqrt(s[:, 1] / s[:, 1].max())
    return utils.max_inten_spec(s, max_num_inten=60, inten_thresh=0.001)


def featurize(form, ion, spec):
    """PeakFormula._featurize (inference mode, cls_type 'ms1', float intensities) on MIST's subformula table."""
    tree = utils.assign_subforms(form, spec, ion, mass_diff_thresh=a.ppm)
    tbl = tree['output_tbl']
    frags, intens, ions = (tbl['formula'], tbl['ms2_inten'], tbl['ions']) if tbl else ([], [], [])
    forms_vec = [utils.formula_to_dense(f) for f in frags]
    root_vec = utils.formula_to_dense(form)
    inten_vec = list(intens) + [1.0]
    type_vec = [0] * len(forms_vec) + [3]
    ion_vec = [utils.get_ion_idx(i) for i in ions] + [utils.get_ion_idx(ion)]
    forms_vec.append(root_vec)
    return dict(types=np.array(type_vec), form_vec=np.array(forms_vec, dtype=np.float32),
                ion_vec=np.array(ion_vec, dtype=np.float32), intens=np.array(inten_vec, dtype=np.float32),
                n_annot=len(frags))


def collate(fs, device):
    L = max(len(f['types']) for f in fs)
    pad = lambda x, w: np.pad(x, [(0, L - len(x))] + [(0, 0)] * (x.ndim - 1)) if w else x
    return dict(types=torch.tensor(np.stack([pad(f['types'], 1) for f in fs])).long().to(device),
                form_vec=torch.tensor(np.stack([pad(f['form_vec'], 1) for f in fs])).float().to(device),
                ion_vec=torch.tensor(np.stack([pad(f['ion_vec'], 1) for f in fs])).float().to(device),
                intens=torch.tensor(np.stack([pad(f['intens'], 1) for f in fs])).float().to(device),
                num_peaks=torch.tensor([len(f['types']) for f in fs]).long().to(device),
                instruments=torch.zeros(len(fs)).float().to(device), names=[str(i) for i in range(len(fs))])


_fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)


def fp_and_form(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    f = CalcMolFormula(m)
    if '+' in f or '-' in f or not utils.has_valid_els(f):
        return None, f
    return _fpg.GetFingerprintAsNumPy(m).astype(np.float32), f


t0 = time.time()
enc = load_encoder(a.ckpt, a.device)
items = json.load(open(a.items))
out, n_sp, n_pairs, n_annot = {}, 0, 0, []
for it_i, item in enumerate(items):
    mid, cands = str(item['mid']), list(dict.fromkeys(item.get('cands') or []))
    fps, forms = {}, {}
    for s in cands:
        fps[s], forms[s] = fp_and_form(s)
    specs = []
    for sp in item.get('spectra') or []:
        ion = ADDUCT.get(str(sp.get('adduct', '')).replace(' ', ''))
        if ion is None:
            continue
        ps = proc_spec(sp['mz'], sp['it'], float(sp.get('prec') or 1e6))
        if ps is not None and len(ps):
            specs.append((ion, ps))
    scores = {s: [] for s in cands}
    by_form = {}
    for s in cands:
        if fps[s] is not None:
            by_form.setdefault(forms[s], []).append(s)
    jobs = [(f, ion, ps) for f in by_form for ion, ps in specs]
    for b0 in range(0, len(jobs), a.batch):
        chunk = jobs[b0:b0 + a.batch]
        feats = []
        for f, ion, ps in chunk:
            try:
                feats.append(featurize(f, ion, ps))
            except Exception:
                feats.append(None)
        ok = [i for i, x in enumerate(feats) if x is not None]
        if not ok:
            continue
        with torch.no_grad():
            pred = enc(collate([feats[i] for i in ok], a.device))[0].float().cpu().numpy()
        for j, i in enumerate(ok):
            f = chunk[i][0]
            p = pred[j]; pn = np.linalg.norm(p) + 1e-9
            n_annot.append(feats[i]['n_annot']); n_pairs += 1
            for s in by_form[f]:
                v = fps[s]
                scores[s].append(float(p @ v / (pn * (np.linalg.norm(v) + 1e-9))))
    n_sp += len(specs)
    out[mid] = {s: (float(np.mean(v)) if v else None) for s, v in scores.items()}
    if (it_i + 1) % 50 == 0:
        print(f'{it_i + 1}/{len(items)} molecules, {time.time() - t0:.0f}s', flush=True)
json.dump(out, open(a.out, 'w'))
print(f'done: {len(out)} molecules, {n_sp} usable spectra, {n_pairs} spectrum-formula pairs, '
      f'median annotated peaks {np.median(n_annot) if n_annot else 0}, {time.time() - t0:.0f}s', flush=True)
