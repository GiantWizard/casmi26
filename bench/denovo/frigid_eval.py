"""De novo generation for class-3-like molecules with FRIGID (MIST MassSpecGym encoder + FRIGID-base DLM).

For each non-conjugate out-of-pool (cls 3) molecule of an ONP set: MIST predicts a 4096-bit Morgan fingerprint from
the molecule's positive-mode spectra (subformula-annotated against the formula), FRIGID generates formula-matched
structures (MassSpecGym benchmark settings: up to N attempts in batches of 32, temperature 1.0, randomness 0.1), ranked
by generation frequency then Tanimoto to the predicted fingerprint. Formula source: 'truth' (ceiling) or 'list'
(formula of our pipeline's own top-1 candidate, available at deployment).

usage: frigid_eval.py SOLUTION.csv TEST.parquet FINAL_LIST.csv OUT.json [--formula truth|list] [--attempts 96]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch, joblib

ap = argparse.ArgumentParser()
ap.add_argument('solution'); ap.add_argument('test'); ap.add_argument('final'); ap.add_argument('out')
ap.add_argument('--formula', default='list')
ap.add_argument('--attempts', type=int, default=96)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--th', type=float, default=0.187)
ap.add_argument('--device', default='cpu')
ap.add_argument('--max_heavy', type=int, default=50)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--fp_keep', type=float, default=1.0)  # oracle degradation: fraction of true on-bits kept
ap.add_argument('--fp_add', type=float, default=0.0)  # false bits added, as a multiple of the true on-bit count
ap.add_argument('--fp', default='mist')  # mist | truth (oracle ceiling)
a = ap.parse_args()
sys.path.insert(0, '/Users/b/casmi26/frigid/src'); sys.path.insert(0, '/Users/b/casmi26/mist/pkg')
from mist import utils as MU  # noqa: E402
from mist.models.spectra_encoder import SpectraEncoderGrowing  # noqa: E402
from dlm.sampler import Sampler  # noqa: E402
from dlm.utils.benchmark_utils import generate_with_formula_filter  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402
from rdkit.Chem.inchi import MolToInchiKey  # noqa: E402
RDLogger.DisableLog('rdApp.*')
dev = a.device

# ---- MIST encoder (FRIGID's MassSpecGym checkpoint) --------------------------------------------------------------
MSG_CFG = dict(form_embedder='pos-cos', output_size=4096, hidden_size=640, spectra_dropout=0.1, peak_attn_layers=2,
               num_heads=8, set_pooling='cls', refine_layers=4, pairwise_featurization=True, embed_instrument=False,
               inten_transform='float', magma_modulo=2048, inten_prob=0.1, remove_prob=0.5, cls_type='ms1',
               spec_features='peakformula', mol_features='fingerprint', top_layers=1)
enc = SpectraEncoderGrowing(**MSG_CFG)
enc.load_state_dict(torch.load('/Users/b/casmi26/mist/ckpt/mist_msg.pt', map_location='cpu', weights_only=True), strict=False)
enc = enc.eval()
ADDUCT = {'[M+H]+': '[M+H]+', '[M+Na]+': '[M+Na]+', '[M+K]+': '[M+K]+', '[M+NH4]+': '[M+H3N+H]+',
          '[M+H-H2O]+': '[M-H2O+H]+', '[M-H2O+H]+': '[M-H2O+H]+', '[M]+': '[M]+'}


def mist_fp(form, spectra):
    feats = []
    for mz, it, prec, ad in spectra:
        ion = ADDUCT.get(ad)
        if ion is None:
            continue
        s = np.array([mz, it], float).T
        s = s[s[:, 0] <= prec + 1]
        if not len(s) or s[:, 1].max() <= 0:
            continue
        s[:, 1] = np.sqrt(s[:, 1] / s[:, 1].max())
        s = MU.max_inten_spec(s, max_num_inten=60, inten_thresh=0.001)
        tree = MU.assign_subforms(form, s, ion, mass_diff_thresh=20)
        tbl = tree['output_tbl']
        fr, inten, ions = (tbl['formula'], tbl['ms2_inten'], tbl['ions']) if tbl else ([], [], [])
        fv = [MU.formula_to_dense(f) for f in fr] + [MU.formula_to_dense(form)]
        feats.append(dict(types=[0] * len(fr) + [3], form_vec=np.array(fv, np.float32), ion_vec=[MU.get_ion_idx(i) for i in ions] + [MU.get_ion_idx(ion)],
                          intens=list(inten) + [1.0]))
    if not feats:
        return None
    probs = []
    for f in feats:
        n = len(f['types'])
        b = dict(types=torch.tensor([f['types']]).long(), form_vec=torch.tensor(f['form_vec'][None]).float(),
                 ion_vec=torch.tensor([f['ion_vec']]).float(), intens=torch.tensor([f['intens']]).float(),
                 num_peaks=torch.tensor([n]).long(), instruments=torch.zeros(1), names=['x'])
        with torch.no_grad():
            probs.append(enc(b)[0][0].numpy())
    return np.mean(probs, axis=0)


# ---- FRIGID ------------------------------------------------------------------------------------------------------
S = Sampler('/Users/b/casmi26/mist/ckpt/DLM.ckpt')
S.model = S.model.to(dev); S.mdlm.to_device(torch.device(dev))
TM = joblib.load('/Users/b/casmi26/frigid/gb_token_full_model.joblib')
TF = json.load(open('/Users/b/casmi26/frigid/gb_token_full_model_features.json'))
aa = Chem.MolFromSmarts('[NX3][CX4]C(=O)[OX2H1,OX1-]')
sol = pd.read_csv(a.solution); te = pd.read_parquet(a.test)
fin = {str(m): [x for x in str(s).split(';') if x and x != 'CCO'] for m, s in pd.read_csv(a.final).values}
ik = lambda s: (lambda m: MolToInchiKey(m)[:14] if m else None)(Chem.MolFromSmiles(s))
todo = []
for r in sol.itertuples():
    m = Chem.MolFromSmiles(r.smiles)
    if r.cls == 3 and m is not None and not m.HasSubstructMatch(aa) and m.GetNumHeavyAtoms() <= a.max_heavy:
        todo.append(r)
if a.limit:
    todo = todo[:a.limit]
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
t0 = time.time()
for i, r in enumerate(todo):
    mid = str(r.molecule_id)
    if mid in res:
        continue
    if a.formula == 'truth':
        form = CalcMolFormula(Chem.MolFromSmiles(r.smiles))
    else:
        lst = fin.get(mid, [])
        if not lst or Chem.MolFromSmiles(lst[0]) is None:
            continue
        form = CalcMolFormula(Chem.MolFromSmiles(lst[0]))
    g = te[te.molecule_id == r.molecule_id]
    spectra = [(np.asarray(x.ms2_mzs, float), np.asarray(x.ms2_normalized_intensities, float), float(x.precursor_mz), x.adduct)
               for x in g.itertuples()]
    if a.fp == 'list':  # deployment: fingerprint of our pipeline's own #1 candidate (formula from --formula)
        lst = fin.get(mid, [])
        if not lst or Chem.MolFromSmiles(lst[0]) is None:
            continue
        p = np.array(list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(lst[0]), 2, 4096)), float)
    else:
        p = mist_fp(form, spectra) if a.fp == 'mist' else (lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, 4096))(Chem.MolFromSmiles(r.smiles))
    if a.fp == 'truth' and p is not None:
        p = np.array(list(p), float)
        rng = np.random.default_rng(int(mid.encode().hex(), 16) % 2**32)
        on = np.where(p > 0)[0]; off = np.where(p == 0)[0]
        p[rng.choice(on, int(round(len(on) * (1 - a.fp_keep))), replace=False)] = 0
        p[rng.choice(off, int(round(len(on) * a.fp_add)), replace=False)] = 1
    if p is None:
        res[mid] = dict(status='no positive-mode spectrum'); continue
    fpb = (p > a.th).astype(np.float32)
    out = generate_with_formula_filter(S, fpb, form, r.smiles, n_required=20, max_attempts=a.attempts, batch_size=a.batch,
                                       softmax_temp=1.0, randomness=0.1, token_model=TM, token_features=TF, is_ngboost=False)
    matched, sims, counter = out[0], out[1], out[5]
    # rank: frequency in this run, then similarity to the predicted fingerprint
    order = sorted(range(len(matched)), key=lambda j: (-counter.get(matched[j], 0), -sims[j]))
    ranked = [matched[j] for j in order]
    tk = ik(r.smiles)
    keys = [ik(x) for x in ranked]
    rank = keys.index(tk) + 1 if tk in keys else 0
    own_formula_ok = form == CalcMolFormula(Chem.MolFromSmiles(r.smiles))
    res[mid] = dict(rank=rank, n=len(ranked), formula_ok=own_formula_ok, gen=ranked[:25])
    json.dump(res, open(a.out, 'w'))
    if (i + 1) % 10 == 0:
        R = [v for v in res.values() if 'rank' in v]
        print(f'{i + 1}/{len(todo)} {time.time() - t0:.0f}s | top1 {np.mean([v["rank"] == 1 for v in R]):.3f} '
              f'top10 {np.mean([0 < v["rank"] <= 10 for v in R]):.3f} any {np.mean([v["rank"] > 0 for v in R]):.3f}', flush=True)
json.dump(res, open(a.out, 'w'))
R = [v for v in res.values() if 'rank' in v]
print(f'done {len(R)} molecules scored ({len(res) - len(R)} skipped) | top1 {np.mean([v["rank"] == 1 for v in R]):.3f} '
      f'top10 {np.mean([0 < v["rank"] <= 10 for v in R]):.3f} top25 {np.mean([v["rank"] > 0 for v in R]):.3f} '
      f'| formula correct {np.mean([v["formula_ok"] for v in R]):.3f}', flush=True)
