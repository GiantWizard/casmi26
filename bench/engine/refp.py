"""Recompute engine 1's fingerprint-dependent features for a new FP bank from a gen_rows.py --cache run, in minutes
instead of a full engine pass. Everything the bank does not touch (library, analogs, fragments, subformula, priors,
generation) is reused as cached; the M-channel block and the fz interaction features are recomputed with the engine's
own formulas (casmi/engine.py). Family features that use fz (derivation__*fz*) cannot be recomputed here: drop them
in comparisons (compare_banks.py --no-fz-family). Fragment features were computed for the top 600 candidates by the
cache bank's fz; the 1% of molecules with more candidates keep that selection.

usage: refp.py CACHE_DIR OUT_DIR --ckpt a.pt [b.pt ...] [--oracle-beta B] [--device cpu|mps|cuda]
Self-test: --ckpt <the cache's own bank> must reproduce the cached columns (max abs diff reported).
"""
import argparse, glob, math, os, pickle, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('cache'); ap.add_argument('out'); ap.add_argument('--ckpt', nargs='+', required=True)
ap.add_argument('--only-sids', default='', help='parquet of sids: recompute only these molecules (others keep cached values)'); ap.add_argument('--xenc', default='', help='XEnc checkpoint: also fill the xs_* cross-encoder features (spectrum side = first --ckpt)'); ap.add_argument('--oracle-beta', type=float, default=0.0); ap.add_argument('--device', default='cpu')
ap.add_argument('--code', default='/Users/b/casmi26/inputs/v4b/code'); ap.add_argument('--work', default='/Users/b/casmi26/rk/work')
a = ap.parse_args()
a.cache, a.out = os.path.abspath(a.cache), os.path.abspath(a.out)
a.ckpt = [os.path.abspath(c) for c in a.ckpt]; a.xenc = os.path.abspath(a.xenc) if a.xenc else ''
sys.path.insert(0, a.code)
os.chdir(os.path.dirname(os.path.abspath(a.work)))
from casmi import chem, fpnet, frag as fragmod  # noqa: E402
from casmi.pool import Pool  # noqa: E402
from casmi.spectra import merge_spectra  # noqa: E402
from casmi.engine import FEATURES, _rank_norm, _z  # noqa: E402
from casmi import graphs as gmod  # noqa: E402

P = Pool(os.path.basename(os.path.abspath(a.work)))
bank = fpnet.ModelBank(list(a.ckpt), device=a.device)
XS = None
if a.xenc:
    from casmi import xenc as xmod
    XS = xmod.XScorer(a.ckpt[0], a.xenc, device=a.device)
    import torch as _t
    if (_t.load(a.xenc, map_location='cpu', weights_only=False).get('args') or {}).get('no_fz'):
        _fwd = XS.x.forward                      # trained with f.z zeroed: score the same way
        XS.x.forward = lambda sx, sp, at, sd, bo, ap_, fz: _fwd(sx, sp, at, sd, bo, ap_, _t.zeros_like(fz))
        print('XEnc trained without f.z input: scoring with f.z zeroed', flush=True)
_EMPTY_G = (np.zeros((0, gmod.N_ATOM_FEAT), np.uint8), np.zeros((0, 0), np.uint8), np.zeros((0, 0), np.uint8))
_gc = {}
def _graph(pid):
    if pid < 0:
        return _EMPTY_G
    g = _gc.get(pid)
    if g is None:
        g = gmod.mol_tokens(P.smiles[pid]) or _EMPTY_G; _gc[pid] = g
    return g
os.makedirs(a.out, exist_ok=True)
C_ = {n: i for i, n in enumerate(FEATURES)}
FP_COLS = ['fz', 'fz_z', 'fz_rank', 'fz_gap', 'fz_norm', 'fz_top', 'fz_softmax', 'fz_grp_ent', 'fz_grp_gap', 'fz_single',
           'fz_single_gap', 'fz_merged', 'fz_merged_gap', 'fp_cos', 'fp_tan', 'fp_ll', 'fp_ll_gap', 'el_c', 'el_n', 'el_o',
           'el_sum', 'el_rank', 'lib_x_fz', 'ap_x_fz', 'agree_lib', 'agree_ap', 'fr_x_fz', 'gen_rank_fz']


def items_of(spectra):
    items_s = []
    for s in spectra:
        pm, pi = fpnet.prep_peaks(s['mz'], s['it'], s['prec'])
        ce_ok = s['ce'] is not None and not np.isnan(s['ce'])
        items_s.append(dict(mz=pm, it=pi, prec=float(s['prec']), adduct_ix=chem.adduct_index(s['adduct']),
                            ce=float(s['ce']) if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0,
                            n_merged=min(int(s.get('ce_n', 1)), 8), mode=float(s['mode'])))
    items_m = []
    for md in (1, -1):
        grp = [s for s in spectra if s['mode'] == md]
        if not grp:
            continue
        mm, ii = merge_spectra([(s['mz'], s['it']) for s in grp])
        prec = float(np.median([s['prec'] for s in grp]))
        pm, pi = fpnet.prep_peaks(mm, ii, prec)
        ces = [s['ce'] for s in grp if s['ce'] is not None and not np.isnan(s['ce'])]
        adducts = [s['adduct'] for s in grp]
        ad = max(set(adducts), key=adducts.count)
        items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad),
                            ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                            n_merged=min(sum(max(1, int(s.get('ce_n', 1))) for s in grp), 8), mode=float(md)))
    return items_s, items_m


def fp_block(X, c, cfp, z_single, z_merged, el_pred):
    """Engine M channel + fz interactions, written into X (float32) in place."""
    nc = len(cfp); cbits = cfp.sum(1)
    fz_s = cfp @ z_single; fz_m = cfp @ z_merged; z_avg = 0.5 * (z_single + z_merged); fz_avg = cfp @ z_avg
    pz = 1.0 / (1.0 + np.exp(-np.clip(z_avg, -30, 30))); pz = np.clip(pz, 1e-6, 1 - 1e-6)
    ll = cfp @ np.log(pz) + (1 - cfp) @ np.log(1 - pz)
    X[:, C_['fp_ll']] = ll / max(1.0, P.nbits) * 100; X[:, C_['fp_ll_gap']] = (ll - ll.max()) / 100.0
    X[:, C_['fp_cos']] = (cfp @ pz) / (np.sqrt(cbits) * np.linalg.norm(pz) + 1e-9)
    pb = (pz > 0.5).astype(np.float32); inter = cfp @ pb
    X[:, C_['fp_tan']] = inter / (cbits + pb.sum() - inter + 1e-9)
    fzmax = float(fz_avg.max())
    sm = np.exp((fz_avg - fzmax) / math.sqrt(P.nbits) * 4.0); sm = sm / sm.sum()
    X[:, C_['fz']] = fz_avg / 100.0; X[:, C_['fz_z']] = _z(fz_avg); X[:, C_['fz_rank']] = _rank_norm(fz_avg)
    X[:, C_['fz_gap']] = (fz_avg - fzmax) / 100.0; X[:, C_['fz_norm']] = fz_avg / np.sqrt(np.maximum(cbits, 1)) / 10.0
    X[:, C_['fz_top']] = (fz_avg == fzmax).astype(np.float32); X[:, C_['fz_softmax']] = sm
    X[:, C_['fz_grp_ent']] = -(sm * np.log(sm + 1e-12)).sum()
    srt = np.sort(fz_avg)[::-1]
    X[:, C_['fz_grp_gap']] = (srt[0] - srt[1]) / 100.0 if nc > 1 else 0.0
    X[:, C_['fz_single']] = fz_s / 100.0; X[:, C_['fz_single_gap']] = (fz_s - fz_s.max()) / 100.0
    X[:, C_['fz_merged']] = fz_m / 100.0; X[:, C_['fz_merged_gap']] = (fz_m - fz_m.max()) / 100.0
    E = np.zeros((nc, 10), np.float32)
    for i, f in enumerate(c['formula']):
        fv = fragmod.formula_vec(f)
        if fv is not None:
            E[i] = np.log1p(fv)
    d = np.abs(E - el_pred[None, :])
    X[:, C_['el_c']] = d[:, 0]; X[:, C_['el_n']] = d[:, 2]; X[:, C_['el_o']] = d[:, 3]
    X[:, C_['el_sum']] = d.sum(1); X[:, C_['el_rank']] = _rank_norm(-d.sum(1))
    mr = _rank_norm(fz_avg)
    lib_max = X[:, C_['lib_max']]; apv = X[:, C_['ap']]
    X[:, C_['lib_x_fz']] = lib_max * (1.0 - mr); X[:, C_['ap_x_fz']] = apv * (1.0 - mr)
    X[:, C_['agree_lib']] = (1.0 - mr[int(np.argmax(lib_max))]) if lib_max.max() > 0 else 0.0
    X[:, C_['agree_ap']] = (1.0 - mr[int(np.argmax(apv))]) if apv.max() > 0 else 0.0
    X[:, C_['fr_x_fz']] = X[:, C_['fr_int']] * (1.0 - mr)
    gi = c['gen_idx']
    if len(gi):
        X[gi, C_['gen_rank_fz']] = _rank_norm(fz_avg[gi])


maxdiff = {}
ONLY = None
if a.only_sids:
    import pandas as pd
    ONLY = set(pd.read_parquet(a.only_sids).sid.tolist())
for cf in sorted(glob.glob(os.path.join(a.cache, 'cache_*.pkl'))):
    tag = os.path.basename(cf)[len('cache_'):-len('.pkl')]
    rz = np.load(os.path.join(a.cache, f'rows_{tag}.npz'))
    X = rz['X'].astype(np.float32); g = rz['g']; old = X.copy()
    for c in pickle.load(open(cf, 'rb')):
        if ONLY is not None and c['sid'] not in ONLY:
            continue
        sel = np.flatnonzero(g == c['sid'])
        pid = c['pid']; cfp = np.zeros((len(pid), P.nbits), np.float32)
        pool_i = np.flatnonzero(pid >= 0)
        cfp[pool_i] = P.fps(pid[pool_i]).astype(np.float32)
        if len(c['gen_idx']):
            cfp[c['gen_idx']] = np.unpackbits(c['gen_fp'], axis=1)[:, :P.nbits].astype(np.float32)
        items_s, items_m = items_of(c['spectra'])
        zs, els = bank.logits_el(items_s); zm, elm = bank.logits_el(items_m)
        z_single, z_merged = zs.mean(0), zm.mean(0)
        if a.oracle_beta:
            t = np.zeros(P.nbits, np.float32)
            y = rz['y'][sel]
            if y.any():
                t = cfp[int(np.flatnonzero(y)[0])]
            else:  # truth not among candidates (class 3): use the true structure's pool fingerprint if present
                t = None
            if t is not None:
                z_single = z_single + a.oracle_beta * (2 * t - 1); z_merged = z_merged + a.oracle_beta * (2 * t - 1)
        Xm = X[sel]
        fp_block(Xm, c, cfp, z_single, z_merged, 0.5 * (els.mean(0) + elm.mean(0)))
        if XS is not None:                         # engine X channel (casmi/engine.py), same formulas
            gitems = [_graph(int(p_)) for p_ in pid]
            xs_s = XS.score(items_s, gitems, cfp); xs_m = XS.score(items_m, gitems, cfp)
            for _a in (xs_s, xs_m):                # rare NaN (degenerate graph): lowest score of the molecule
                if np.isnan(_a).any():
                    _a[np.isnan(_a)] = np.nanmin(_a) if np.isfinite(_a).any() else 0.0
            xmax = xs_s.max(0); xmean = xs_s.mean(0); xmer = xs_m.mean(0); xall = 0.5 * (xmax + xmer)
            Xm[:, C_['xs_max']] = xmax; Xm[:, C_['xs_mean']] = xmean; Xm[:, C_['xs_merged']] = xmer
            Xm[:, C_['xs_gap']] = xall - xall.max(); Xm[:, C_['xs_rank']] = _rank_norm(xall); Xm[:, C_['xs_z']] = _z(xall)
            Xm[:, C_['xs_top']] = (xall >= xall.max() - 1e-6).astype(np.float32)
            xsrt = np.sort(xall)[::-1]
            Xm[:, C_['xs_grp_gap']] = (xsrt[0] - xsrt[1]) if len(xall) > 1 else 0.0
        X[sel] = Xm
    for n in FP_COLS:
        j = C_[n]; maxdiff[n] = max(maxdiff.get(n, 0.0), float(np.nanmax(np.abs(X[:, j] - old[:, j]))) if len(X) else 0.0)
    np.savez_compressed(os.path.join(a.out, f'rows_{tag}.npz'), X=X.astype(np.float16), y=rz['y'], g=g, cls=rz['cls'],
                        sids=rz['sids'], n_cand=rz['n_cand'], pos_in=rz['pos_in'], names=rz['names'])
    print('done', tag, flush=True)
print('max |new - cached| per recomputed column (self-test: ~0 with the cache bank):')
print({k: round(v, 4) for k, v in maxdiff.items()})
