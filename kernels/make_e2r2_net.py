"""Build kernels/jc-v7c-e2r2-net: jc-v7c-e2r2 (0.443) + transductive molecular-networking rerank as a last cell.

Test molecules whose spectra share fragments (modified cosine >= 0.7) are linked; inside each same-formula group of a
molecule's final list, candidates are reordered by position + 0.5 * resemblance to the linked neighbours' top-1
(confidence-weighted, 2 iterations). Same config as fp/net_rerank.py --th 0.7 --conf top1 --iters 2 --w 0.5 (pred).
Reorders only, never adds SMILES. Any failure keeps submission.csv unchanged. CPU only.

usage: make_e2r2_net.py            (writes the kernel)
       make_e2r2_net.py test SUB.csv TEST.parquet OUT.csv   (runs NET_CODE locally, for checking against net_rerank.py)
"""
import json, pathlib, sys

NET_CODE = r'''
import itertools as _it
import numpy as _np
from rdkit import Chem as _Ch, DataStructs as _DS, RDLogger as _RL
from rdkit.Chem import rdFingerprintGenerator as _rfg
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _cmf
_RL.DisableLog('rdApp.*')
NET_TH, NET_W, NET_ITERS, NET_TOL = 0.7, 0.5, 2, 0.01


def _net_prep(mz, it, prec):
    m = _np.asarray(mz, float); i = _np.asarray(it, float)
    keep = m < prec - 0.5
    m, i = m[keep], i[keep]
    if len(m) == 0:
        return m, i
    o = _np.argsort(-i)[:60]; m, i = m[o], _np.sqrt(i[o]); i = i / (_np.linalg.norm(i) + 1e-12)
    return m, i


def _net_modcos(a, b, dm):
    (m1, i1), (m2, i2) = a, b
    if len(m1) == 0 or len(m2) == 0:
        return 0.0
    pairs = []
    for shift in (0.0, dm):
        d = _np.abs(m1[:, None] - (m2[None, :] + shift))
        for x, y in zip(*_np.where(d < NET_TOL)):
            pairs.append((i1[x] * i2[y], x, y))
    pairs.sort(reverse=True); u1, u2, s = set(), set(), 0.0
    for v, x, y in pairs:
        if x not in u1 and y not in u2:
            u1.add(x); u2.add(y); s += v
    return s


def net_links(te):
    spec = {}
    for r in te.itertuples():
        spec.setdefault(str(r.molecule_id), []).append(
            (r.ionization_mode, r.precursor_mz, _net_prep(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)))
    nbr = {}
    for a, b in _it.combinations(sorted(spec), 2):
        best = 0.0
        for ma, pa, sa in spec[a]:
            for mb, pb, sb in spec[b]:
                if ma == mb:
                    best = max(best, _net_modcos(sa, sb, pa - pb))
        if best >= NET_TH:
            nbr.setdefault(a, []).append((b, best)); nbr.setdefault(b, []).append((a, best))
    return nbr


def net_rerank(sub, nbr):
    gen = _rfg.GetMorganGenerator(radius=2, fpSize=2048)
    _fp, _form = {}, {}

    def fp(s):
        if s not in _fp:
            m = _Ch.MolFromSmiles(s)
            _fp[s] = gen.GetFingerprint(m) if m else None
            _form[s] = _cmf(m) if m else s
        return _fp[s]

    def conf_of(lst):
        if len(lst) < 2 or fp(lst[0]) is None or fp(lst[1]) is None:
            return 1.0 if lst else 0.0
        return 1.0 - _DS.TanimotoSimilarity(fp(lst[0]), fp(lst[1]))

    def z(v):
        v = _np.asarray(v, float); s = v.std()
        return (v - v.mean()) / s if s > 0 else _np.zeros_like(v)

    cur, out = dict(sub), dict(sub)
    for _ in range(NET_ITERS):
        conf = {m: conf_of(l) for m, l in cur.items()}
        out = {}
        for m, lst in sub.items():
            if not lst or m not in nbr:
                out[m] = lst[:25]; continue
            ev = [(fp(cur[y][0]), conf.get(y, 1.0) * lw) for y, lw in nbr[m] if cur.get(y) and fp(cur[y][0]) is not None]
            net = [sum(c * _DS.TanimotoSimilarity(fp(s), f) for f, c in ev) if fp(s) is not None and ev else 0.0 for s in lst]
            order, groups = list(lst), {}
            for i, s in enumerate(lst):
                fp(s); groups.setdefault(_form[s], []).append(i)
            for idx in groups.values():
                if len(idx) < 2:
                    continue
                sc = z([-i for i in idx]) + NET_W * z([net[i] for i in idx])
                for slot, j in zip(idx, [idx[t] for t in _np.argsort(-sc, kind='stable')]):
                    order[slot] = lst[j]
            out[m] = order[:25]
        cur = out
    return out
'''

CELL = r"""
# ---- jc-net: molecular-networking rerank of the final lists (CPU, reorders only, never adds SMILES).
# Linked test molecules (modified cosine >= 0.7) pull same-formula isomers that resemble their neighbours' top-1 up.
# Offline (ONP conjugate-free slices): +0.020 set A (tuned), +0.006 set B (one-shot). Any failure keeps submission.csv.
import time as _t, json, shutil as _shn
_t0 = _t.time()
try:
__NET_CODE__
    _shn.copyfile('submission.csv', 'submission_pre_net.csv')
    _sub = pd.read_csv('submission_pre_net.csv', dtype=str, keep_default_na=False)
    _lists = {str(m): [x for x in str(s).split(';') if x and x != 'CCO'] for m, s in zip(_sub.molecule_id, _sub.smiles)}
    _te = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
    _te = _te[_te.molecule_id.astype(str).isin(_lists)]
    _nbr = net_links(_te)
    _new = net_rerank(_lists, _nbr)
    _rows = [(m, ';'.join(_new[str(m)]) if _new.get(str(m)) else s) for m, s in zip(_sub.molecule_id, _sub.smiles)]
    _fs = pd.DataFrame(_rows, columns=['molecule_id', 'smiles'])
    assert len(_fs) == len(_sub) and _fs.molecule_id.is_unique and _fs.smiles.map(lambda x: len(x.split(';'))).max() <= 25
    assert all(set(a.split(';')) - {'CCO'} == set(b.split(';')[:25]) - {'CCO'} for a, b in zip(_fs.smiles, _sub.smiles)), 'rerank changed a candidate set'
    _fs.to_csv('submission.csv', index=False)
    _st = dict(molecules=len(_lists), linked=len(_nbr), changed=int(sum(a != b for a, b in zip(_fs.smiles, _sub.smiles))),
               changed_top1=int(sum(a.split(';')[0] != b.split(';')[0] for a, b in zip(_fs.smiles, _sub.smiles))),
               seconds=round(_t.time() - _t0))
    json.dump(_st, open('net_stats.json', 'w'))
    print('jc-net:', _st, flush=True)
except Exception as _ex:
    import traceback; traceback.print_exc()
    if os.path.exists('submission_pre_net.csv'):
        _shn.copyfile('submission_pre_net.csv', 'submission.csv')
    json.dump({'failed': repr(_ex)[:500]}, open('net_stats.json', 'w'))
    print('jc-net FAILED -> submission.csv unchanged:', repr(_ex), flush=True)
""".replace("__NET_CODE__", "\n".join("    " + l if l else l for l in NET_CODE.strip("\n").splitlines()))

if len(sys.argv) > 1 and sys.argv[1] == "test":
    import pandas as pd
    sub_p, te_p, out_p = sys.argv[2:5]
    exec(NET_CODE)
    sub = {str(m): [x for x in str(s).split(';') if x and x != 'CCO'] for m, s in pd.read_csv(sub_p).values}
    te = pd.read_parquet(te_p)
    te = te[te.molecule_id.astype(str).isin(sub)]
    new = net_rerank(sub, net_links(te))
    pd.DataFrame([(m, ';'.join(l) if l else 'CCO') for m, l in new.items()], columns=['molecule_id', 'smiles']).to_csv(out_p, index=False)
    print('wrote', out_p)
    sys.exit()

ROOT = pathlib.Path(__file__).resolve().parents[1]
src = ROOT / "kernels/jc-v7c-e2r2/casmi26-jc-v7c-e2r2.ipynb"
nb = json.loads(src.read_text())
cells = nb["cells"]
assert "ENGINE INSERT APPLIED" in "".join(cells[-1]["source"]), "unexpected cell layout"
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CELL.lstrip("\n")})
cells[0]["source"] = ("# jc-v7c-e2r2-net = jc-v7c-e2r2 (public 0.443) + molecular-networking rerank as the last cell\n\n"
                      + "".join(cells[0]["source"]))
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None
out = ROOT / "kernels/jc-v7c-e2r2-net"
out.mkdir(exist_ok=True)
(out / "casmi26-jc-v7c-e2r2-net.ipynb").write_text(json.dumps(nb, indent=1))
meta = json.loads((ROOT / "kernels/jc-v7c-e2r2/kernel-metadata.json").read_text())
meta.update(id="giantwizard/casmi26-jc-v7c-e2r2-net", title="casmi26-jc-v7c-e2r2-net", code_file="casmi26-jc-v7c-e2r2-net.ipynb")
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", out)
