"""Generate the stacked-reranker submission kernels: v465 + a final cell that reorders every molecule's candidates (the union
of all channels, as in bench/stack/build_rows.py) with the reranker trained by bench/stack/train_stack.py --save.
  stack-full: v465's own models (what the public pipeline runs); the reranker was trained on fold-safe features, so this
              tests whether it transfers to the stronger full models;
  stack-safe: the rowgen configuration (fold-safe models, compliant engine 1), the same features the reranker trained on.
The model and build_rows.py are read from the private dataset giantwizard/casmi26-jc-stack-model (Kaggle refuses a
notebook with the model embedded).
usage: make_stackinf.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_rowgen as RG  # noqa: E402

OUT = '/Users/b/casmi26-repo/kernels/stack'
STACK = r'''# ---- STACK: reorder each molecule's candidates with the stacked reranker (kernels/make_stackinf.py) ----
import glob, os, pickle, sys
import numpy as np, pandas as pd, lightgbm as lgb
from multiprocessing import Pool
_SD = os.path.dirname(glob.glob('/kaggle/input/**/casmi26-jc-stack-model/**/stack_model.pkl', recursive=True)[0])
sys.path.insert(0, _SD)
import build_rows as BR
_M = pickle.load(open(os.path.join(_SD, 'stack_model.pkl'), 'rb'))
_B = [lgb.Booster(model_str=x) for x in _M['boosters']]
_S = BR.load('/kaggle/working')
_sm = sorted(BR.collect_smiles(_S))
with Pool(os.cpu_count()) as _p:
    _ID = dict(zip(_sm, _p.map(BR.ident, _sm, chunksize=200)))
_sub = pd.read_csv('submission.csv')
_out, _n = [], 0
for _mid, _old in zip(_sub.molecule_id.astype(str), _sub.smiles):
    _df = BR.mol_frame(_S, _mid, _ID)
    if _df is None or not len(_df):
        _out.append(_old); continue
    _sc = np.mean([b.predict(_df[_M['features']]) for b in _B], 0)
    _out.append(';'.join(_df.smiles.values[np.argsort(-_sc, kind='stable')][:25])); _n += 1
_sub['smiles'] = _out
assert _sub.molecule_id.is_unique and _sub.smiles.map(lambda v: 1 <= len(v.split(';')) <= 25).all()
_sub.to_csv('submission.csv', index=False)
print('STACK reranked', _n, 'of', len(_sub), 'molecules', flush=True)
'''
code = lambda x: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': x}
for name, safe in (('full', False), ('safe', True)):
    nb = json.loads(json.dumps(RG.nb0))
    cells = RG.patch(nb['cells'], safe=safe)
    stack = STACK
    new = cells[:27] + [code(RG.DUMP % dict(when='fusion'))] + cells[27:] + [code(stack)]
    new[0] = dict(new[0], source=f'# jc-stack-{name}: v465 ({"fold-safe models" if safe else "its own models"}) + a stacked reranker over every '
                  'channel\'s candidates. Built by kernels/make_stackinf.py.\n\n' + ''.join(new[0]['source']))
    nb['cells'] = new
    slug = f'casmi26-jc-stack-{name}'
    ds = list(RG.meta0['dataset_sources']) + ['giantwizard/casmi26-jc-stack-model']
    if safe:
        ds = [d for d in ds if d not in ('ahmedberatozer/casmi26-fpnet-full1', 'prvsiyan/casmi26-fp-models-v2')]
        ds += ['giantwizard/casmi26-jc-own-ranker', 'dmitriigluzdov/casmi26-fold-safe-fpnet', 'giantwizard/casmi26-e2fp-f0']
    d = f'{OUT}/{name}'; os.makedirs(d, exist_ok=True)
    json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
    json.dump(dict(RG.meta0, id=f'giantwizard/{slug}', title=slug, code_file=f'{slug}.ipynb', dataset_sources=ds),
              open(f'{d}/kernel-metadata.json', 'w'), indent=1)
    print('wrote', d, f'{os.path.getsize(f"{d}/{slug}.ipynb") / 1e6:.1f} MB')
