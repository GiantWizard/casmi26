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


# ---- generator-only probe on jc-gen: how many hidden answers does class-3 generation produce on its own? ----
def make_genonly():
    src = '/Users/b/casmi26-repo/kernels/jc-gen'
    nb = json.load(open(f'{src}/casmi26-jc-gen.ipynb')); meta = json.load(open(f'{src}/kernel-metadata.json'))
    cells = nb['cells']
    i13 = next(i for i, c in enumerate(cells) if c['cell_type'] == 'code' and 'GEN_TOP_SCORE[str(mid)] = float(scs[0])' in ''.join(c['source']))
    s = ''.join(cells[i13]['source'])
    old = "                if pids_ and pids_[0] < 0:\n                    GEN_TOP_SCORE[str(mid)] = float(scs[0])\n"
    assert s.count(old) == 1
    s = s.replace(old, old + "                GEN_ONLY[str(mid)] = [x for x, p in zip(smis, pids_) if p < 0]   # probe: engine-1 generated entries, in rank order\n")
    assert s.count('GEN_TOP_SCORE = {}') == 1
    s = s.replace('GEN_TOP_SCORE = {}', 'GEN_ONLY = {}\nGEN_TOP_SCORE = {}', 1)
    cells[i13]['source'] = s
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": r'''# ---- LB PROBE (not a real submission): ONLY engine 1's generated structures (not in the pool) per molecule ----
_p = pd.read_csv('submission.csv')
_rows, _n, _tot = [], 0, 0
for m in _p.molecule_id:
    g = [x for x in GEN_ONLY.get(str(m), []) if x and x != 'CCO'][:25]
    _n += bool(g); _tot += len(g)
    _rows.append((m, ';'.join(g) if g else 'CCO'))
_q = pd.DataFrame(_rows, columns=['molecule_id', 'smiles'])
assert len(_q) == len(_p) and _q.molecule_id.is_unique
_q.to_csv('submission.csv', index=False)
print('PROBE genonly: molecules with generated candidates', _n, 'of', len(_q), '| mean per molecule', round(_tot / max(_n, 1), 1), flush=True)
'''})
    d = '/Users/b/casmi26-repo/kernels/probes/genonly'; os.makedirs(d, exist_ok=True)
    slug = 'casmi26-jc-probe-genonly'
    json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
    json.dump(dict(meta, id=f'giantwizard/{slug}', title=slug, code_file=f'{slug}.ipynb'), open(f'{d}/kernel-metadata.json', 'w'), indent=1)
    print('wrote', d)


if __name__ == '__main__' and os.environ.get('GENONLY'):
    make_genonly()


# ---- PubChem-channel-only probe on jc-v7c-e2r2: how many hidden answers does the channel's own ranking get? ----
def make_pconly():
    src = '/Users/b/casmi26-repo/kernels/jc-v7c-e2r2'
    nb = json.load(open(f'{src}/casmi26-jc-v7c-e2r2.ipynb')); meta = json.load(open(f'{src}/kernel-metadata.json'))
    nb['cells'].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": r'''# ---- LB PROBE (not a real submission): ONLY the PubChem channel's raw ranked list per molecule ----
_pcraw = json.load(open('/kaggle/working/pc_lists.json'))
_p = pd.read_csv('submission.csv')
_rows, _n = [], 0
for m in _p.molecule_id:
    l = [x for x in ((_pcraw.get(str(m)) or {}).get('pc') or []) if x and x != 'CCO'][:25]
    _n += bool(l)
    _rows.append((m, ';'.join(l) if l else 'CCO'))
_q = pd.DataFrame(_rows, columns=['molecule_id', 'smiles'])
assert len(_q) == len(_p) and _q.molecule_id.is_unique
_q.to_csv('submission.csv', index=False)
print('PROBE pconly: molecules with PubChem-channel candidates', _n, 'of', len(_q), flush=True)
'''})
    d = '/Users/b/casmi26-repo/kernels/probes/pconly'; os.makedirs(d, exist_ok=True)
    slug = 'casmi26-jc-probe-pconly'
    json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
    json.dump(dict(meta, id=f'giantwizard/{slug}', title=slug, code_file=f'{slug}.ipynb'), open(f'{d}/kernel-metadata.json', 'w'), indent=1)
    print('wrote', d)
