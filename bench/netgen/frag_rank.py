"""Fragment-explanation score for generated analog candidates (MetFrag-style, using ms-pred's MAGMa engine).

For a candidate structure, MAGMa enumerates fragments (bond breaks, tree depth --depth, with hydrogen shifts); the
score is the share of the molecule's measured MS/MS intensity whose peaks match a fragment ion within --ppm.
Positional isomers from one edit differ exactly in which fragments carry the modification, so the peaks that
shift between a molecule and its neighbour discriminate them (the ModiFinder observation, JASMS 2024).

usage: frag_rank.py TEST.parquet GENSETS.json OUT.json [--depth 2] [--ppm 15]
"""
import argparse, json, sys, types
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('test'); ap.add_argument('gensets'); ap.add_argument('out')
ap.add_argument('--depth', type=int, default=2)
ap.add_argument('--ppm', type=float, default=15.0)
ap.add_argument('--pkg', default='/Users/b/casmi26/inputs/glacier')
ap.add_argument('--own', type=int, default=3, help='also score the first N own candidates')
a = ap.parse_args()
sys.path.insert(0, a.pkg)
import gl_runner as G  # noqa: E402
G.setup_env(types.SimpleNamespace(pkg=a.pkg, site='none', wheels=None, any_rdkit=True), lambda *x: None)
from ms_pred.magma import fragmentation as FR  # noqa: E402

ADD = {'[M+H]+': 1.007276, '[M+Na]+': 22.989221, '[M+NH4]+': 18.033826, '[M+K]+': 38.963158, '[M-H2O+H]+': 1.007276,
       '[M+H-H2O]+': 1.007276, '[M-H]-': -1.007276, '[M+Cl]-': -1.007276, '[M+CH2O2-H]-': -1.007276, '[M]+': -0.000549}
PROTON = 1.007276
te = pd.read_parquet(a.test)
G_ = json.load(open(a.gensets))
_cache = {}


def frag_masses(smi):
    if smi not in _cache:
        try:
            fe = FR.FragmentEngine(smi, max_tree_depth=a.depth, max_broken_bonds=6)
            fe.generate_fragments()
            _, _, _, masses, _ = fe.get_frag_masses()
            _cache[smi] = np.unique(np.round(masses, 5))
        except Exception:
            _cache[smi] = None
    return _cache[smi]


def explained(masses, spectra):
    """Mean over spectra of the intensity share of peaks matched by a fragment ion (neutral fragment mass + charge
    carrier: proton for protonated/positive, minus proton for negative; Na/K adduct fragments use the metal)."""
    vals = []
    for mz, it, ad, mode in spectra:
        if masses is None or len(mz) == 0:
            continue
        carrier = PROTON if mode == 'positive' else -PROTON
        ions = np.concatenate([masses + carrier] + ([masses + ADD[ad]] if ad in ('[M+Na]+', '[M+K]+') else []))
        ions.sort()
        idx = np.clip(np.searchsorted(ions, mz), 1, len(ions) - 1)
        d = np.minimum(np.abs(ions[idx] - mz), np.abs(ions[idx - 1] - mz))
        hit = d <= mz * a.ppm * 1e-6
        it = np.sqrt(np.clip(it, 0, None)); vals.append(float(it[hit].sum() / max(it.sum(), 1e-12)))
    return float(np.mean(vals)) if vals else np.nan


spec = {}
for r in te.itertuples():
    keep = r.ms2_mzs < r.precursor_mz - 1.0
    spec.setdefault(str(r.molecule_id), []).append((np.asarray(r.ms2_mzs)[keep], np.asarray(r.ms2_normalized_intensities)[keep],
                                                    r.adduct, r.ionization_mode))
out = {}
for i, (mid, d) in enumerate(G_.items()):
    sp = spec.get(mid, [])
    cands = d['smiles'] + d['own'][:a.own]
    out[mid] = {s: explained(frag_masses(s), sp) for s in cands}
    if (i + 1) % 50 == 0:
        print(i + 1, 'molecules', flush=True)
json.dump(out, open(a.out, 'w'))
print('done', len(out))
