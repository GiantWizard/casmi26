"""Generate kernels/jc-v492-mist-l{05,10}: the public 0.464 notebook (jc-v492, unchanged) + a final cell that re-orders
same-formula candidates with MIST (bench/mist/mist_rerank.py) at weight LAM. The MIST checkpoint and code wheel come from
the private dataset giantwizard/casmi26-mist (Zenodo 10.5281/zenodo.23303289 + github.com/coleygroup/MIST, both MIT).
"""
import json, os

SRC = '/Users/b/casmi26-repo/kernels/jc-v492'
CODE = open('/Users/b/casmi26-repo/bench/mist/mist_rerank.py').read()
nb0 = json.load(open(f'{SRC}/casmi26-jc-v492.ipynb')); meta0 = json.load(open(f'{SRC}/kernel-metadata.json'))
CELL = r'''# ---- MIST same-formula re-ordering (kernels/make_mist.py), weight %(lam)s ----
import glob, os, subprocess, sys, time
import pandas as pd, torch
_MD = os.path.dirname(glob.glob('/kaggle/input/**/casmi26-mist/**/mist_msg_noaug.ckpt', recursive=True)[0])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q', glob.glob(_MD + '/mist-*.whl')[0]])
os.makedirs('/kaggle/working/mistcode', exist_ok=True)
open('/kaggle/working/mistcode/mist_rerank.py', 'w').write(%(code)r)
sys.path.insert(0, '/kaggle/working/mistcode')
import mist_rerank as MR
_t = time.time(); _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
_mm = MR.load(os.path.join(_MD, 'mist_msg_noaug.ckpt'), _dev)
print('MIST selftest (Enveda panel [M+H]+, truth vs same-formula training structures):',
      MR.selftest(os.path.join(COMP, 'train.parquet'), _mm, _dev), f'{time.time() - _t:.0f}s', flush=True)
_sub = pd.read_csv('submission.csv')
_te = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
_res, _st = MR.rerank(_sub, _te, _mm, _dev, lam=%(lam)s)
assert len(_res) == len(_sub) and _res.molecule_id.is_unique and _res.smiles.map(lambda v: 1 <= len(v.split(';')) <= 25).all()
assert (_res.smiles.str.split(';').map(sorted) == _sub.smiles.str.split(';').map(sorted)).all()
_res.to_csv('submission.csv', index=False)
print('MIST rerank', _st, f'{time.time() - _t:.0f}s', flush=True)
'''
for tag, lam in (('l05', 0.5), ('l10', 1.0)):
    nb = json.loads(json.dumps(nb0))
    nb['cells'].append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': CELL % dict(lam=lam, code=CODE)})
    nb['cells'][0] = dict(nb['cells'][0], source=f'# jc-v492-mist-{tag}: jc-v492 (public 0.464, unchanged) + MIST same-formula re-ordering at '
                          f'weight {lam}. Built by kernels/make_mist.py.\n\n' + ''.join(nb['cells'][0]['source']))
    slug = f'casmi26-jc-v492-mist-{tag}'
    d = f'/Users/b/casmi26-repo/kernels/mist/{tag}'; os.makedirs(d, exist_ok=True)
    json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
    json.dump(dict(meta0, id=f'giantwizard/{slug}', title=slug, code_file=f'{slug}.ipynb',
                   dataset_sources=meta0['dataset_sources'] + ['giantwizard/casmi26-mist']), open(f'{d}/kernel-metadata.json', 'w'), indent=1)
    print('wrote', d)
