"""Generate kernels/mist481/w{15,25}: the public 0.481 notebook (jc-p2s481, unchanged) with MIST as a third neural scorer.

Its ranks 2-25 (and outside candidates) are ordered by 0.65 z(public spectrum model) + 0.35 z(retrained model); rank 1
stays with the leader-safe rules. Here that blend becomes (1 - W) * blend + W * z(MIST cosine), z per molecule over the
candidates MIST can score (others get 0). MIST: coleygroup/MIST augmented MSG checkpoint (bench/mist/mist_rerank.py),
from the private dataset giantwizard/casmi26-mist. Right after the notebook's source/RDKit setup, MIST is loaded and its bug check printed, so
the fast commit run already proves the install works.
"""
import json, os

SRC = '/Users/b/casmi26-repo/kernels/jc-p2s481'
CODE = open('/Users/b/casmi26-repo/bench/mist/mist_rerank.py').read()
nb0 = json.load(open(f'{SRC}/casmi26-jc-p2s481.ipynb')); meta0 = json.load(open(f'{SRC}/kernel-metadata.json'))
LOAD = r'''# ---- MIST third scorer (kernels/make_mist481.py): load + bug check ----
import glob as _glob, os as _os, subprocess as _sp, sys as _sys, time as _time
import torch as _torch
_MD = _os.path.dirname(_glob.glob('/kaggle/input/**/casmi26-mist/**/mist_msg_aug.ckpt', recursive=True)[0])
_sp.check_call([_sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q', _glob.glob(_MD + '/mist-*.whl')[0]])
_os.makedirs('/kaggle/working/mistcode', exist_ok=True)
open('/kaggle/working/mistcode/mist_rerank.py', 'w').write(%(code)r)
_sys.path.insert(0, '/kaggle/working/mistcode')
import mist_rerank as MR
MIST_W = %(w)s
MIST_DEV = 'cuda' if _torch.cuda.is_available() else 'cpu'
_t = _time.time(); _m = MR.load(_os.path.join(_MD, 'mist_msg_aug.ckpt'), MIST_DEV)
print('MIST selftest:', MR.selftest(str(COMP / 'train.parquet'), _m, MIST_DEV), f'{_time.time() - _t:.0f}s', flush=True)
del _m; _torch.cuda.empty_cache() if MIST_DEV == 'cuda' else None


def mist_blend(frame, label):
    """frame: indexed by (molecule_id, smiles) with a blended neural_score; adds W * z(MIST) in place."""
    import numpy as _np, pandas as _pd
    _t = _time.time()
    model = MR.load(_os.path.join(_MD, 'mist_msg_aug.ckpt'), MIST_DEV)
    pairs = frame.index.to_frame(index=False)
    cosv = _pd.Series(MR.score_pairs(pairs, _pd.read_parquet(COMP / 'test.parquet'), model, MIST_DEV), index=frame.index)
    g = cosv.groupby(level=0)
    zm = ((cosv - g.transform('mean')) / g.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)).fillna(0.0)
    frame['neural_score'] = (1 - MIST_W) * frame['neural_score'] + MIST_W * zm
    del model
    if MIST_DEV == 'cuda':
        _torch.cuda.empty_cache()
    print(f'MIST blend ({label}): weight {MIST_W}, scored {int(cosv.notna().sum())} of {len(cosv)} candidates, '
          f'{frame.index.get_level_values(0).nunique()} molecules, {_time.time() - _t:.0f}s', flush=True)
'''
A_OLD = "        _a['neural_ok'] = _a['neural_ok'].astype(bool) & _b['neural_ok'].fillna(False).astype(bool)\n"
O_OLD = "            _oa['neural_ok'] = _oa['neural_ok'].astype(bool) & _ob['neural_ok'].fillna(False).astype(bool)\n"
for tag, w in (('w15', 0.15), ('w25', 0.25)):
    nb = json.loads(json.dumps(nb0))
    cells = nb['cells']
    i = next(k for k, c in enumerate(cells) if c['cell_type'] == 'code' and ''.join(c['source']).startswith('BUNDLE = materialise_sources()'))
    s = ''.join(cells[i]['source'])
    assert s.count(A_OLD) == 1 and s.count(O_OLD) == 1
    s = s.replace(A_OLD, A_OLD + "        mist_blend(_a, 'final lists')\n").replace(O_OLD, O_OLD + "            mist_blend(_oa, 'outside candidates')\n")
    first = 'BUNDLE = materialise_sources()\n'           # installs the official RDKit wheel that MIST needs
    assert s.startswith(first)
    cells[i]['source'] = first + LOAD % dict(code=CODE, w=w) + s[len(first):]
    cells[0] = dict(cells[0], source=f'# jc-p2s481-mist-{tag}: jc-p2s481 (public 0.481, unchanged) + MIST as a third neural scorer at weight {w}. '
                    'Built by kernels/make_mist481.py.\n\n' + ''.join(cells[0]['source']))
    slug = f'casmi26-jc-p2s481-mist-{tag}'
    d = f'/Users/b/casmi26-repo/kernels/mist481/{tag}'; os.makedirs(d, exist_ok=True)
    json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
    json.dump(dict(meta0, id=f'giantwizard/{slug}', title=slug, code_file=f'{slug}.ipynb',
                   dataset_sources=meta0['dataset_sources'] + ['giantwizard/casmi26-mist']), open(f'{d}/kernel-metadata.json', 'w'), indent=1)
    print('wrote', d)
