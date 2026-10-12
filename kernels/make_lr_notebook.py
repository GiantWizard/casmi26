"""Generate kernels/jc-lr: jc-gate + the learned final rerank (bench/obscure_np/lr_core.py, model in the
giantwizard/casmi26-jc-lr dataset) in cell 19.

Per molecule, after the two engines' lists are fused, the model scores every fused candidate from features the
notebook already has and reorders candidates inside each same-formula group ('slots' mode). It replaces the
hand-weighted ICEBERG/GLACIER rerank; any failure keeps that rerank's order for the molecule.

    python kernels/make_lr_notebook.py [--src jc-gate] [--out jc-lr]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="jc-gate")
ap.add_argument("--out", default="jc-lr2")  # "jc-lr" collides with the casmi26-jc-lr dataset slug on Kaggle
args = ap.parse_args()
src = ROOT / "kernels" / args.src
meta = json.loads((src / "kernel-metadata.json").read_text())
nb = json.loads((src / meta["code_file"]).read_text())
cells = nb["cells"]


def must_replace(text, old, new):
    assert text.count(old) == 1, (old[:70], text.count(old))
    return text.replace(old, new)


c19 = "".join(cells[19]["source"])
c19 = must_replace(c19, "n_glft = 0\nFUSION_OVERFLOW = {}", """n_glft = 0
# ---- jc-lr: learned final rerank (model trained offline on held-out eval sets; see bench/obscure_np) ----
LR_MODEL, LR, LR_POP, n_lr, n_lr_fail = None, None, {}, 0, 0
try:
    _lr_dir = os.path.dirname(find('lr_model.txt'))
    sys.path.insert(0, _lr_dir)
    import lr_core as LR
    LR_MODEL = lgb.Booster(model_file=os.path.join(_lr_dir, 'lr_model.txt'))
    assert list(LR_MODEL.feature_name()) == [f for f in LR.FEATS], 'model/feature mismatch'
    LR_POP = dict(zip(map(str, P.key), POOL_POP))
    print('jc-lr: model', _lr_dir, '| trees', LR_MODEL.num_trees(), '| pop keys', len(LR_POP))
except Exception as ex:
    LR_MODEL = None
    print('jc-lr: model unavailable -> simulator rerank kept:', repr(ex))
FUSION_OVERFLOW = {}""")
c19 = must_replace(c19, "            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])\n",
                   "            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])\n"
                   "            _f0 = (list(fsm), list(fk), list(fsc))\n")
c19 = must_replace(c19, "            changed += int(bool(vs) and fsm[:1] != vs[:1])\n", """            if LR_MODEL is not None:
                try:
                    _fs, _fk, _fc = _f0
                    _sims = {n: (d.get(str(mid), {}) if isinstance(d, dict) else {})
                             for n, d in (('ice', ICE_SCORES), ('gl', GL_SCORES), ('glft', GLFT_SCORES))}
                    _pcl = PC.get(mid) if isinstance(PC, dict) else None
                    _rows = LR.molecule_rows(mid, vs, [chem.score_key(x) or x for x in vs], e['keys'], _pcl,
                                             _fs, _fk, _fc, [_form2(x) for x in _fs], _sims, LR_POP)
                    _df = LR.add_group_feats(pd.DataFrame(_rows))
                    fsm = LR.slot_order(_df, LR_MODEL.predict(_df[LR.FEATS]))
                    n_lr += 1
                except Exception as ex:
                    n_lr_fail += 1
                    print('jc-lr rerank failed', mid, repr(ex))
            changed += int(bool(vs) and fsm[:1] != vs[:1])
""")
c19 = must_replace(c19, "    print('jc-c: fine-tuned GLACIER rerank applied to', n_glft, 'molecules ICEBERG and GLACIER cannot score')",
                   "    print('jc-c: fine-tuned GLACIER rerank applied to', n_glft, 'molecules ICEBERG and GLACIER cannot score')\n"
                   "    print('jc-lr: learned rerank applied to', n_lr, 'molecules;', n_lr_fail, 'fell back to the simulator rerank')")
cells[19]["source"] = c19
cells[0]["source"] = ("# jc-lr = jc-gate + learned final rerank (LightGBM, slots mode) replacing the hand-weighted "
                      "ICEBERG/GLACIER rerank\n\n" + "".join(cells[0]["source"]))
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
        compile("".join(c["source"]), "cell", "exec")
out = ROOT / "kernels" / args.out
out.mkdir(exist_ok=True)
slug = f"casmi26-{args.out}"
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
meta.update(id=f"giantwizard/{slug}", title=slug, code_file=f"{slug}.ipynb")
meta.pop("id_no", None)
meta["dataset_sources"] = list(dict.fromkeys(list(meta["dataset_sources"]) + ["giantwizard/casmi26-jc-lr"]))
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", out)
