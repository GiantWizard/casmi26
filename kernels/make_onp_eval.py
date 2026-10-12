"""Generate casmi26-jc-onp-eval-d{N}: jc-c's pipeline run on an obscure-NP draw, emitting the
three LB variants from one GPU run.

jc-base2, jc-c and jc-c2 differ only in cell 19's final same-formula rerank, so this notebook
computes every simulator score once (jc-c computes the superset: GLACIER, ICEBERG, fine-tuned
GLACIER on uncovered molecules, mix-GLACIER on [M+H]+) and then replays cell 19 + the audit
pass (cell 22) per variant:
    base2: no fine-tuned GLACIER                    (LB 0.438)
    c:     fine-tune on uncovered + mix term λ=1     (LB 0.414)
    c2:    fine-tune on uncovered only               (LB 0.438)
Outputs final_{variant}.csv in competition format, scored offline by bench/obscure_np/mrr_eval.py.

    python kernels/make_onp_eval.py --draw 0 && kaggle kernels push -p kernels/jc-onp-eval-d0
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--draw", type=int, default=0)
ap.add_argument("--dataset", default="giantwizard/casmi26-jc-onp")
ap.add_argument("--tag", default="", help="slug suffix for a second eval set, e.g. 'b' -> casmi26-jc-onp-eval-b-d0")
ap.add_argument("--visible", action="store_true",
                help="Run on the competition's visible test.parquet (train-derived, i.e. class-1-like) in full, "
                     "instead of an ONP draw. Used to measure what gate changes cost when the answer is in the library.")
args = ap.parse_args()

src_dir = ROOT / "kernels" / "jc-c"
nb = json.loads((src_dir / "casmi26-jc-c.ipynb").read_text())
meta = json.loads((src_dir / "kernel-metadata.json").read_text())
cells = nb["cells"]
S = lambda i: "".join(cells[i]["source"])


def put(i, text):
    cells[i]["source"] = text
    cells[i]["outputs"] = []
    cells[i]["execution_count"] = None


def must_replace(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new)


# ---- cell 2: point COMP at the ONP draw -----------------------------------------------
c2 = S(2)
if args.visible:
    pass
else:
  c2 = must_replace(c2, "COMP = os.path.dirname(find('test.parquet'))", f"""COMP = os.path.dirname(find('test.parquet'))
# ---- ONP eval: swap the competition test for an obscure-NP draw (train.parquet stays the library) ----
ONP_DRAW = {args.draw}
_ONP = os.path.dirname(find(f'test_draw{{ONP_DRAW}}.parquet'))
_OC = '/kaggle/working/onp_comp'; os.makedirs(_OC, exist_ok=True)
import shutil as _shutil
_shutil.copy(os.path.join(_ONP, f'test_draw{{ONP_DRAW}}.parquet'), os.path.join(_OC, 'test.parquet'))
_shutil.copy(os.path.join(_ONP, 'sample_submission.csv'), os.path.join(_OC, 'sample_submission.csv'))
if not os.path.exists(os.path.join(_OC, 'train.parquet')):
    os.symlink(os.path.join(COMP, 'train.parquet'), os.path.join(_OC, 'train.parquet'))
COMP = _OC
print('ONP eval: draw', ONP_DRAW, 'from', _ONP)""")
put(2, c2)
if args.visible:  # force the full (non-smoke) path on the visible test
    c4 = S(4)
    c4 = must_replace(c4, "IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'",
                      "IS_RERUN = True   # onp-eval --visible: full run on the train-derived visible test")
    put(4, c4)

# ---- cell 8: PubChem channel uses every core (hard-coded 4 = Kaggle's count; Colab L4/A100 have more) ----
c8 = S(8)
c8 = must_replace(c8, "'/kaggle/working/pc_lists.json', '4']", "'/kaggle/working/pc_lists.json', str(os.cpu_count() or 4)]")
put(8, c8)

# ---- cell 19: run the final fusion per variant ------------------------------------------
c19 = S(19)
c19 = must_replace(c19, """n_glft = 0
FUSION_OVERFLOW = {}
if ENG:
    v4 = pd.read_csv('submission.csv')""", """import shutil
shutil.copy('submission.csv', 'v4_submission.csv')
# (fine-tune on uncovered, mix λ, drop ICEBERG's all-None shell dicts). The LB notebooks keep the shell, which makes
# `if ice:` true for molecules ICEBERG cannot score, so their fine-tuned-GLACIER branch never runs (c2 == base2).
VARIANTS = {'base2': (False, 0.0, False), 'c': (True, 1.0, False), 'c2': (True, 0.0, False), 'c2fix': (True, 0.0, True)}
FUSION_OVERFLOW_BY = {}
assert ENG, 'engine 2 produced nothing'
def run_variant(vname, use_glft, GLMIX_LAM, fix_shell):
    global FUSION_OVERFLOW
    n_glft = 0
    FUSION_OVERFLOW = {}
    v4 = pd.read_csv('v4_submission.csv')""")
c19 = must_replace(c19, "[(ice if CELL19_ICE else {}), _g, glmix_of(mid)]",
                   "[(ice if CELL19_ICE else {}), _g, (glmix_of(mid) if GLMIX_LAM > 0 else {})]")
c19 = must_replace(c19, "            elif glft_of(mid):", "            elif use_glft and glft_of(mid):")
_ICE_LINE = "            ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}"
c19 = must_replace(c19, _ICE_LINE, _ICE_LINE + "\n            if fix_shell and not any(v is not None for v in ice.values()):\n                ice = {}")
c19 = must_replace(c19, "    fs.to_csv('submission.csv', index=False)",
                   "    fs.to_csv(f'sub_{vname}.csv', index=False)\n    FUSION_OVERFLOW_BY[vname] = FUSION_OVERFLOW")
c19 = must_replace(c19, "    print('jc-c: fine-tuned GLACIER rerank applied to', n_glft, 'molecules ICEBERG and GLACIER cannot score')",
                   "    print(vname, ': fine-tuned GLACIER rerank applied to', n_glft, 'molecules ICEBERG and GLACIER cannot score')\n"
                   "for _v, (_u, _l, _f) in VARIANTS.items():\n    run_variant(_v, _u, _l, _f)")
put(19, c19)

# ---- cell 22: audit pass per variant ----------------------------------------------------
c22 = S(22)
# Engine 2 can miss a molecule or two on library-free sets; the eval keeps going (main list still covers it).
c22 = must_replace(c22, "assert _active <= set(map(str, ENG)), 'Engine 2 did not cover all active molecules'",
                   "print('engine 2 missing', len(_active - set(map(str, ENG))), 'molecules (eval: not fatal)')")
put(22, f"""_AUDIT_SRC = {c22!r}
import shutil
for _vname in VARIANTS:
    shutil.copy(f'sub_{{_vname}}.csv', 'submission.csv')
    FUSION_OVERFLOW = FUSION_OVERFLOW_BY[_vname]
    exec(_AUDIT_SRC, globals())
    shutil.copy('submission.csv', f'final_{{_vname}}.csv')
    shutil.copy('candidate_audit.json', f'audit_{{_vname}}.json')
    print('wrote final_' + _vname + '.csv')
# the commit's submission.csv is c2 (the current pick); nothing here is ever submitted
shutil.copy('final_c2.csv', 'submission.csv')""")

slug = "casmi26-jc-onp-eval-vis" if args.visible else f"casmi26-jc-onp-eval{'-' + args.tag if args.tag else ''}-d{args.draw}"
if args.visible:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": """
# onp-eval --visible: recover the visible test's answers from train.parquet (its rows are train-derived)
import pyarrow.parquet as pq
_tp = os.path.join(COMP, 'train.parquet')
_names = pq.ParquetFile(_tp).schema_arrow.names
print('train columns:', _names)
_want = [c for c in _names if c.lower() in ('spectrum_id', 'molecule_id', 'smiles', 'inchikey', 'inchikey14', 'canonical_smiles')]
_tr = pq.read_table(_tp, columns=_want).to_pandas()
_te = pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=['molecule_id', 'spectrum_id'])
_hit = _te.merge(_tr, on='spectrum_id', how='left', suffixes=('', '_train')) if 'spectrum_id' in _tr else _te
print('test spectra matched to train:', _hit.drop(columns=['molecule_id', 'spectrum_id']).notna().any(axis=1).mean())
_hit.to_csv('vis_truth.csv', index=False)
"""})

cells[0]["source"] = (f"# {slug}: jc-base2 / jc-c / jc-c2 on " + ("the visible test (class-1-like)" if args.visible else f"obscure-NP draw {args.draw}") + "\n"
                      "Generated by kernels/make_onp_eval.py from casmi26-jc-c. Offline evaluation only.\n\n") + S(0)
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

out = ROOT / "kernels" / slug.replace("casmi26-", "")
out.mkdir(parents=True, exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
meta.update(id=f"giantwizard/{slug}", title=slug, code_file=f"{slug}.ipynb")
meta.pop("id_no", None)
if not args.visible:
    meta["dataset_sources"] = list(meta["dataset_sources"]) + [args.dataset]
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", out)
