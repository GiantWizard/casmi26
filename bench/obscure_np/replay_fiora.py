"""Replay jc-gate's final fusion (c2fix + gentle PubChem gate) offline with FIORA added to the same-formula rerank.

Needs a finished run of a *-fiora eval notebook (kernels/make_fiora_eval.py): eng_lists.json, pc_lists.json,
v4_submission.csv, ice_work/ice_out.json, gl_work/gl_out.json, glft_work/gl_out.json and fiora_work/fiora_out.json.
Writes one competition-format csv per FIORA weight, to be scored by mrr_eval.py:

    python replay_fiora.py --run OUT_DIR --lib LIB_DIR --out DIR --weights 0 0.25 0.5 1 2 [--mode gl|all]
    python mrr_eval.py --solution solution.csv --test test_draw0.parquet \
        --sub w0=DIR/rf_gl_w0.csv w0.5=DIR/rf_gl_w0.5.csv ... --baseline w0

Pipeline replayed (logic copied from gate_variants.py 'gentle' + replay_rerank.py 'c2fix', GLFT_LAM 0.5 = jc-gate):
  1. gentle gate: molecules whose PubChem list never reached v4 (lib_max >= 0.9, inferred) get the PubChem list
     interleaved at slots 4, 8, 12, 16, 20 (ranks 1-3 untouched).
  2. two-engine reciprocal-rank fusion (alpha 0.6, k 3, top 80).
  3. same-formula rerank: ICEBERG shell dicts dropped; where GLACIER covers the molecule
     rerank_multi([{}, GL, FIORA], [1, 1, w]); ICEBERG-only molecules: ICEBERG alone; else fine-tuned GLACIER (0.5).
     --mode all also adds FIORA (weight w) to the ICEBERG-only and fine-tuned-GLACIER branches, and lets FIORA alone
     rerank molecules no other simulator covers (e.g. [M-H]- only, no fine-tuned GLACIER scores).
w = 0 reproduces jc-gate's replay (rp_v4gentle.csv of replay_rerank.py --v4 gentle=...) exactly.
LIB_DIR holds fuse.py (casmi26-iceberg), gl_fuse.py (casmi26-glacier) and casmi/chem.py (casmi26-v4b-models).
"""
import argparse
import functools
import json
import os
import sys
import time

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="kernel output dir")
ap.add_argument("--lib", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
ap.add_argument("--mode", choices=["gl", "all"], default="gl")
ap.add_argument("--fiora", default="fiora_work/fiora_out.json", help="score file relative to --run "
                "(fiora_work/fiora_out_cos.json = sqrt-cosine variant)")
ap.add_argument("--gate", choices=["gentle", "none"], default="gentle")
ap.add_argument("--glft-lam", type=float, default=0.5)
ap.add_argument("--tag", default="", help="extra label in the output file names")
args = ap.parse_args()
sys.path.insert(0, args.lib)
import fuse as ice_fuse  # noqa: E402
import gl_fuse  # noqa: E402
from casmi import chem  # noqa: E402
from rdkit import Chem as _Ch2  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF2  # noqa: E402

chem.score_key = functools.lru_cache(maxsize=None)(chem.score_key)

R = args.run
load = lambda p: json.load(open(os.path.join(R, p))) if os.path.exists(os.path.join(R, p)) else {}
ENG = load("eng_lists.json")
PC = load("pc_lists.json")
ICE_SCORES = load("ice_work/ice_out.json")
GL_SCORES = load("gl_work/gl_out.json")
GLFT_SCORES = load("glft_work/gl_out.json")
FIORA_SCORES = load(args.fiora)
ICE_LAM = GL_LAM = 1.0
ALPHA, KRR, FUSE_N = 0.6, 3.0, 80
SLOTS_GENTLE = [4, 8, 12, 16, 20]
assert FIORA_SCORES or args.weights == [0.0], f"no FIORA scores at {os.path.join(R, args.fiora)}"


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
fiora_of = lambda m: _nonempty(FIORA_SCORES, m)


# ---- 1. gentle gate (= gate_variants.py 'gentle') ----------------------------------------------------------------
def merge(base, p, pk, slots, n=25):
    pcs = [s for s, k in zip(p, pk)]
    out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs):
            out.append(pcs[pj]); pj += 1
        elif bi < len(base):
            out.append(base[bi]); bi += 1
        elif pj < len(pcs):
            out.append(pcs[pj]); pj += 1
    return out


def gentle_v4(v4):
    rows, n_g = [], 0
    for mid, s in zip(v4.molecule_id, v4.smiles):
        lst = [x for x in s.split(";") if x]
        d = PC.get(mid) or {}
        if d.get("pc") and not (set(d["pc"]) & set(lst)):          # gated: lib_max >= 0.9 (inferred)
            n_g += 1
            new = merge([x for x in lst if x != "CCO"], d["pc"], d["pc_keys"], SLOTS_GENTLE)
            seen, ded = set(), []
            for x in new:
                if x not in seen:
                    seen.add(x); ded.append(x)
            lst = ded[:25] or ["CCO"]
        rows.append((mid, ";".join(lst)))
    print(f"gentle gate: {n_g} / {len(v4)} molecules gated", flush=True)
    return pd.DataFrame(rows, columns=["molecule_id", "smiles"])


# ---- 2. fusion + 3. rerank (= replay_rerank.py c2fix) --------------------------------------------------------------
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


def run(v4, w, mode):
    out, st = [], dict(gl=0, ice_only=0, glft=0, fiora_only=0, fiora_cov=0)
    for mid, s in zip(v4.molecule_id, v4.smiles):
        vs = [x for x in s.split(";") if x and x != "CCO"]
        e = ENG.get(str(mid))
        if e and e["smiles"]:
            fsm, fk, fsc = fuse2(vs, e["smiles"], e["keys"])
            forms = [_form2(x) for x in fsm]
            ice = _nonempty(ICE_SCORES, mid)                      # c2fix: all-None shell dict -> {}
            fi = fiora_of(mid) if w > 0 else {}
            st["fiora_cov"] += int(bool(fi))
            o = None
            if ice:
                _g = gl_of(mid)
                if _g:
                    o = gl_fuse.rerank_multi(fsm, fk, fsc, forms, [{}, _g, fi], [ICE_LAM, GL_LAM, w], top_n=len(fsm))
                    st["gl"] += 1
                elif mode == "all" and fi:
                    o = gl_fuse.rerank_multi(fsm, fk, fsc, forms, [ice, fi], [ICE_LAM, w], top_n=len(fsm))
                    st["ice_only"] += 1
                else:
                    o = ice_fuse.rerank(fsm, fk, fsc, forms, ice, lam=ICE_LAM, top_n=len(fsm))
                    st["ice_only"] += 1
            elif glft_of(mid):
                if mode == "all" and fi:
                    o = gl_fuse.rerank_multi(fsm, fk, fsc, forms, [glft_of(mid), fi], [args.glft_lam, w],
                                             top_n=len(fsm))
                else:
                    o = ice_fuse.rerank(fsm, fk, fsc, forms, glft_of(mid), lam=args.glft_lam, top_n=len(fsm))
                st["glft"] += 1
            elif mode == "all" and fi:
                o = gl_fuse.rerank_multi(fsm, fk, fsc, forms, [fi], [w], top_n=len(fsm))
                st["fiora_only"] += 1
            if o is not None:
                fsm = [fsm[i] for i in o]
            vs = fsm[:25]
        out.append((mid, ";".join(vs[:25]) if vs else "CCO"))
    return pd.DataFrame(out, columns=["molecule_id", "smiles"]), st


v4 = pd.read_csv(os.path.join(R, "v4_submission.csv"))
if args.gate == "gentle":
    v4 = gentle_v4(v4)
fm = FIORA_SCORES
print(f"FIORA file {args.fiora}: {len(fm)} molecules, {sum(1 for d in fm.values() if any(v is not None for v in d.values()))}"
      f" scored, {sum(sum(v is not None for v in d.values()) for d in fm.values())} candidate scores", flush=True)
os.makedirs(args.out, exist_ok=True)
ref = None
for w in args.weights:
    t0 = time.time()
    df, st = run(v4, w, args.mode)
    name = f"rf_{args.mode}{('_' + args.tag) if args.tag else ''}_w{w:g}.csv"
    df.to_csv(os.path.join(args.out, name), index=False)
    if ref is None:
        ref = df
        diff = ""
    else:
        a = ref.smiles.str.split(";"); b = df.smiles.str.split(";")
        diff = (f" | vs w={args.weights[0]:g}: top-1 changed {int(sum(x[:1] != y[:1] for x, y in zip(a, b)))},"
                f" top-25 order changed {int(sum(x != y for x, y in zip(a, b)))}")
    print(f"w={w:<5g} {name}  {st}{diff}  ({time.time() - t0:.0f}s)", flush=True)
