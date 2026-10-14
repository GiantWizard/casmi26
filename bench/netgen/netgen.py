"""Network-guided analog generation for a whole test set (CPU, minutes).

Test molecules whose spectra share fragments (modified cosine >= TH, peaks matched directly or shifted by the
precursor mass difference) are linked. For each molecule X, single biotransformation edits (edits.py) of its linked
neighbours' current top-1 structures whose exact mass matches X's neutral mass (10 ppm) become new candidates. They
are scored by an FPNet checkpoint against X's merged spectra plus 2 x link strength, and the best N_INS are inserted
at rank 1, or at rank 2 when X's own top-1 is a documented pool structure (popularity > POP_GATE).
Prior work: network re-ranking (NAP, PLOS Comp Biol 2018), biotransformation-edge propagation (NetID, Nat Methods
2021), modification localisation (ModiFinder, JASMS 2024).

run(test_df, lists, fpnet_model, fp_bits, chem, pop, device) -> (new lists, stats)
  lists: {molecule_id: [smiles, ...]} (current final lists); pop: {score_key: popularity}
"""
import itertools
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from edits import edit_products
RDLogger.DisableLog('rdApp.*')

TH, TOL, N_INS, POP_GATE, K_SEED = 0.6, 0.01, 2, 1.0, 1
ADD = {'[M+H]+': (0, 1.007276), '[M-H]-': (6, -1.007276), '[M+Na]+': (4, 22.989221), '[M+CH2O2-H]-': (8, 44.998203),
       '[M+NH4]+': (1, 18.033826), '[M+Cl]-': (9, 34.969401), '[M-H2O+H]+': (2, -17.003288), '[M+H-H2O]+': (2, -17.003288),
       '[M+K]+': (5, 38.963158), '[M]+': (12, -0.000549), '[M-H2O-H]-': (7, -19.01784)}


def _prep(mz, it, prec):
    m = np.asarray(mz, float); i = np.asarray(it, float)
    keep = m < prec - 0.5
    m, i = m[keep], i[keep]
    if len(m) == 0:
        return m, i
    o = np.argsort(-i)[:60]; m, i = m[o], np.sqrt(np.clip(i[o], 0, None))
    return m, i / (np.linalg.norm(i) + 1e-12)


def _modcos(a, b, dm):
    (m1, i1), (m2, i2) = a, b
    if len(m1) == 0 or len(m2) == 0:
        return 0.0
    pairs = []
    for shift in (0.0, dm):
        d = np.abs(m1[:, None] - (m2[None, :] + shift))
        for x, y in zip(*np.where(d < TOL)):
            pairs.append((i1[x] * i2[y], x, y))
    pairs.sort(reverse=True); u1, u2, s = set(), set(), 0.0
    for v, x, y in pairs:
        if x not in u1 and y not in u2:
            u1.add(x); u2.add(y); s += v
    return s


def links(test_df):
    spec = {}
    for r in test_df.itertuples():
        spec.setdefault(str(r.molecule_id), []).append((r.ionization_mode, float(r.precursor_mz),
                                                        _prep(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)))
    nb = {}
    for a, b in itertools.combinations(sorted(spec), 2):
        best = 0.0
        for ma, pa, sa in spec[a]:
            for mb, pb, sb in spec[b]:
                if ma == mb:
                    best = max(best, _modcos(sa, sb, pa - pb))
        if best >= TH:
            nb.setdefault(a, []).append((b, best)); nb.setdefault(b, []).append((a, best))
    return nb


def run(test_df, lists, model, fp_bits, chem, pop, device='cpu', merge_spectra=None, fpnet=None):
    import torch
    nb = links(test_df)
    kc = {}

    def key(s):
        if s not in kc:
            kc[s] = chem.score_key(s)
        return kc[s]
    fpc = {}

    def fpv(s):
        if s not in fpc:
            f = chem.raw_fingerprint(s); fpc[s] = None if f is None else f[fp_bits].astype(np.float32)
        return fpc[s]
    groups = {str(m): g for m, g in test_df.groupby('molecule_id')}
    out = {m: list(l) for m, l in lists.items()}
    st = dict(linked=len(nb), with_gen=0, at1=0, at2=0)
    for mid, nbs in nb.items():
        lst = lists.get(mid) or []
        g = groups[mid]; g = g[g.adduct.isin(ADD)]
        if not lst or not len(g):
            continue
        neutral = float(np.median([p - ADD[ad][1] for p, ad in zip(g.precursor_mz, g.adduct)]))
        own = {key(s) for s in lst[:25]}
        gen = {}
        for y, w in nbs:
            for seed in (lists.get(y) or [])[:K_SEED]:
                for smi in edit_products(seed):
                    m = Chem.MolFromSmiles(smi)
                    if m is None or abs(ExactMolWt(m) - neutral) > neutral * 1e-5:
                        continue
                    k = key(smi)
                    if k and k not in own and (k not in gen or gen[k][1] < w):
                        gen[k] = (smi, w)
        if not gen:
            continue
        gl = [v[0] for v in gen.values()]; gw = np.array([v[1] for v in gen.values()])
        F = [fpv(s) for s in gl]; ok = [i for i, f in enumerate(F) if f is not None]
        sc = np.full(len(gl), -9.0)
        if ok:
            Fm = torch.from_numpy(np.stack([F[i] for i in ok])).to(device); tot = np.zeros(len(ok))
            for mode, gm in g.groupby('ionization_mode'):
                specs = [(np.asarray(r.ms2_mzs, float), np.asarray(r.ms2_normalized_intensities, float)) for r in gm.itertuples()]
                mz, it = merge_spectra(specs) if len(specs) > 1 else specs[0]
                r0 = gm.iloc[0]
                pm, pi = fpnet.prep_peaks(np.asarray(mz, float), np.asarray(it, float), float(r0.precursor_mz))
                item = dict(mz=pm, it=pi, prec=float(r0.precursor_mz), adduct_ix=ADD[r0.adduct][0], ce=0.0, ce_known=0.0,
                            n_merged=min(len(specs), 8), mode=1.0 if mode == 'positive' else -1.0)
                with torch.no_grad():
                    z = model(*fpnet.collate([item], device))[0].float()[0]
                v = (Fm @ z).cpu().numpy(); tot += (v - v.mean()) / (v.std() + 1e-6) if len(v) > 1 else v
            sc[ok] = tot
        order = np.argsort(-(sc + 2.0 * gw))
        best = [gl[i] for i in order[:N_INS]]
        st['with_gen'] += 1
        if pop.get(key(lst[0]), 0.0) > POP_GATE:
            out[mid] = lst[:1] + best + lst[1:]; st['at2'] += 1
        else:
            out[mid] = best + lst; st['at1'] += 1
    return out, st
