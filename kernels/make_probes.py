"""LB probes on the public 0.443 notebook (jc-v7c-e2r2): a final cell rewrites submission.csv.
  top1     : keep only each molecule's first guess  -> score = rank-1 hit rate
  shufS    : the final top 25 in a random order (seed S), padded to exactly 25 with tiny decoy alkanes that are never an
             answer -> expected score = recall@25 * H25/25 (0.15264) for any rank distribution, so
             recall@25 = score / 0.15264 (average several seeds)
usage: make_probes.py  (writes kernels/probes/<name>/ with notebook + kernel-metadata.json)
"""
import copy, json, os
SRC = '/Users/b/casmi26-repo/kernels/jc-v7c-e2r2'
nb0 = json.load(open(f'{SRC}/casmi26-jc-v7c-e2r2.ipynb')); meta0 = json.load(open(f'{SRC}/kernel-metadata.json'))
CELL = r'''# ---- LB PROBE (not a real submission) mode=%(mode)r seed=%(seed)d ----
import hashlib as _hp, random as _rp
_MODE, _SEED = %(mode)r, %(seed)d
_DECOYS = ['C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC', 'CCCCCCC', 'CCCCCCCC', 'CCCCCCCCC', 'CCCCCCCCCC', 'CCCCCCCCCCC',
           'CCCCCCCCCCCC', 'CC(C)C', 'CCC(C)C', 'CC(C)(C)C', 'CCCC(C)C', 'CCC(C)CC', 'CC(C)C(C)C', 'CCC(C)(C)C',
           'C1CC1', 'C1CCC1', 'C1CCCC1', 'C1CCCCC1', 'CC1CC1', 'CC1CCC1']
_p = pd.read_csv('submission.csv')
_rows = []
for m, s in zip(_p.molecule_id, _p.smiles):
    toks = [t for t in str(s).split(';') if t and t != 'CCO']
    if _MODE == 'top1':
        out = toks[:1] or ['CCO']
    else:
        real = toks[:25]
        dec = [d for d in _DECOYS if d not in real][:25 - len(real)]
        out = real + dec
        r = _rp.Random(int(_hp.md5(f'{m}-{_SEED}'.encode()).hexdigest(), 16) %% 2**32)
        r.shuffle(out)
    _rows.append((m, ';'.join(out)))
_q = pd.DataFrame(_rows, columns=['molecule_id', 'smiles'])
assert len(_q) == len(_p) and _q.molecule_id.is_unique
_q.to_csv('submission.csv', index=False)
print('PROBE APPLIED', _MODE, _SEED, 'rows', len(_q), 'mean list len', _q.smiles.str.count(';').mean() + 1, flush=True)
'''
for name, mode, seed in (('top1', 'top1', 0), ('shuf1', 'shuf', 1), ('shuf2', 'shuf', 2), ('shuf3', 'shuf', 3)):
    nb = copy.deepcopy(nb0)
    nb['cells'].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                        "source": CELL % dict(mode=mode, seed=seed)})
    d = f'/Users/b/casmi26-repo/kernels/probes/{name}'; os.makedirs(d, exist_ok=True)
    slug = f'casmi26-jc-probe-{name}'
    json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
    m = dict(meta0, id=f'giantwizard/{slug}', title=slug, code_file=f'{slug}.ipynb')
    json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1)
    print('wrote', d)
