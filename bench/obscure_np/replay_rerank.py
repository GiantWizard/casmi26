"""Replay jc-c's cell-19 final fusion + same-formula rerank offline from a finished eval run's outputs.

Reproduces sub_{variant}.csv without a GPU, so rerank-only variants (weights, routing fixes) can be
tested in seconds. Needs fuse.py (casmi26-iceberg), gl_fuse.py (casmi26-glacier) and casmi/chem.py
(casmi26-v4b-models) on --lib.

    python replay_rerank.py --run OUT_DIR --lib LIB_DIR --out DIR
"""
import argparse
import json
import os
import sys

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="kernel output dir (eng_lists.json, *_work/, v4_submission.csv)")
ap.add_argument("--lib", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--v4", nargs="*", default=[], help="name=path: alternative pre-fusion main lists; each is replayed as c2fix")
args = ap.parse_args()
sys.path.insert(0, args.lib)
import fuse as ice_fuse  # noqa: E402
import gl_fuse  # noqa: E402
from casmi import chem  # noqa: E402
from rdkit import Chem as _Ch2  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF2  # noqa: E402

import functools  # noqa: E402
chem.score_key = functools.lru_cache(maxsize=None)(chem.score_key)

R = args.run
load = lambda p: json.load(open(os.path.join(R, p))) if os.path.exists(os.path.join(R, p)) else {}
ENG = load("eng_lists.json")
ICE_SCORES = load("ice_work/ice_out.json")
GL_SCORES = load("gl_work/gl_out.json")
GLFT_SCORES = load("glft_work/gl_out.json")
GLMIX_SCORES = load("glmix_work/gl_out.json")
ICE_LAM = GL_LAM = 1.0
GLFT_LAM = 1.0
ALPHA, KRR, FUSE_N = 0.6, 3.0, 80
CELL19_ICE = False


@functools.lru_cache(maxsize=None)
def _form2(s):
    try:
        return _CMF2(_Ch2.MolFromSmiles(s))
    except Exception:
        return s


def _nonempty(d, m):
    g = d.get(str(m), {}) if isinstance(d, dict) else {}
    return g if isinstance(g, dict) and any(v is not None for v in g.values()) else {}


gl_of = lambda m: _nonempty(GL_SCORES, m)
glft_of = lambda m: _nonempty(GLFT_SCORES, m)
glmix_of = lambda m: _nonempty(GLMIX_SCORES, m)


def fuse2(v_smis, e_smis, e_keys, n=FUSE_N):
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = chem.score_key(s) or s
        sc[k] = sc.get(k, 0.0) + 1.0 / (KRR + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + ALPHA / (KRR + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]


def run_variant(use_glft, glmix_lam, fix_ice_shell=False, glft_lam=GLFT_LAM, v4_path=None):
    """fix_ice_shell=False reproduces the LB notebooks exactly: ICEBERG's all-None shell dict for molecules it
    cannot score makes `if ice:` true, so the fine-tuned GLACIER branch is never reached."""
    v4 = pd.read_csv(v4_path or os.path.join(R, "v4_submission.csv"))
    out, n_glft = [], 0
    for mid, s in zip(v4.molecule_id, v4.smiles):
        vs = [x for x in s.split(";") if x and x != "CCO"]
        e = ENG.get(str(mid))
        if e and e["smiles"]:
            fsm, fk, fsc = fuse2(vs, e["smiles"], e["keys"])
            ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}
            if fix_ice_shell and not any(v is not None for v in ice.values()):
                ice = {}
            if ice:
                o = ice_fuse.rerank(fsm, fk, fsc, [_form2(x) for x in fsm],
                                    (ice if (CELL19_ICE or not gl_of(mid)) else {}), lam=ICE_LAM, top_n=len(fsm))
                _g = gl_of(mid)
                if _g:
                    o = gl_fuse.rerank_multi(fsm, fk, fsc, [_form2(x) for x in fsm],
                                             [(ice if CELL19_ICE else {}), _g, (glmix_of(mid) if glmix_lam > 0 else {})],
                                             [ICE_LAM, GL_LAM, glmix_lam], top_n=len(fsm))
                fsm = [fsm[i] for i in o]
            elif use_glft and glft_of(mid):
                o = ice_fuse.rerank(fsm, fk, fsc, [_form2(x) for x in fsm], glft_of(mid), lam=glft_lam, top_n=len(fsm))
                fsm = [fsm[i] for i in o]; n_glft += 1
            vs = fsm[:25]
        out.append((mid, ";".join(vs[:25]) if vs else "CCO"))
    return pd.DataFrame(out, columns=["molecule_id", "smiles"]), n_glft


os.makedirs(args.out, exist_ok=True)
VARIANTS = {
    "base2": dict(use_glft=False, glmix_lam=0.0),
    "c": dict(use_glft=True, glmix_lam=1.0),
    "c2": dict(use_glft=True, glmix_lam=0.0),
    "c2fix": dict(use_glft=True, glmix_lam=0.0, fix_ice_shell=True),
    "c2fix_l05": dict(use_glft=True, glmix_lam=0.0, fix_ice_shell=True, glft_lam=0.5),
    "c2fix_l2": dict(use_glft=True, glmix_lam=0.0, fix_ice_shell=True, glft_lam=2.0),
}
for spec in args.v4:
    _n, _, _p = spec.partition("=")
    VARIANTS[f"v4{_n}"] = dict(use_glft=True, glmix_lam=0.0, fix_ice_shell=True, glft_lam=0.5, v4_path=_p)
import time  # noqa: E402
for name, kw in VARIANTS.items():
    t0 = time.time()
    df, n = run_variant(**kw)
    df.to_csv(os.path.join(args.out, f"rp_{name}.csv"), index=False)
    print(f"{name:<10} fine-tuned GLACIER applied to {n} molecules ({time.time() - t0:.0f}s)", flush=True)
