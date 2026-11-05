"""Generate kernels/sc3/w{20,33}: jc-p2s481v22 (public 0.481, version 22, unchanged) with our own spectrum scorer as a
third model in its ranks 2-25 blend.

Version 22 orders ranks 2-25 by 0.5 z(public scorer) + 0.5 z(bgunshui's retrained scorer); rank 1 stays with its
leader-safe rules. Here that becomes (1 - W) * blend + W * z(ours3), where ours3 is the scorer trained in Colab with the
same public kit but a new seed, a wider/deeper network and more hard negatives (giantwizard/casmi26-scorer-ours3,
bench/scorer/make_colab_scorer.py). It is run through the notebook's own neural_scores.py on the same exported
candidates. If it fails, the notebook keeps the two-model blend and says so in the log.
"""
import json, os

SRC = '/Users/b/casmi26-repo/kernels/jc-p2s481v22'
nb0 = json.load(open(f'{SRC}/casmi26-jc-p2s481v22.ipynb')); meta0 = json.load(open(f'{SRC}/kernel-metadata.json'))
A_OLD = "        _a['neural_ok'] = _a['neural_ok'].astype(bool) & _b['neural_ok'].fillna(False).astype(bool)\n"
BLOCK = r'''        try:                                                       # ours3: third scorer (kernels/make_sc3.py)
            _sc3 = os.path.dirname(sorted(glob.glob('/kaggle/input/**/casmi26-scorer-ours3/**/final_model.pt', recursive=True))[0])
            run_stage([sys.executable, CODE / 'neural_scores.py', _sc3, 'final_model.pt', ROOT / 'frozen_candidate_evidence.parquet',
                       ROOT / 'effective_query_spectra.parquet', ROOT / 'neural_sc3.parquet', 'cuda'], ROOT / 'neural_sc3.log', 'Neural scorer · ours3')
            _c = _pd.read_parquet(ROOT / 'neural_sc3.parquet').set_index(['molecule_id', 'smiles']).reindex(_a.index)
            _a['neural_score'] = (1 - %(w)s) * _a['neural_score'] + %(w)s * _z(_c['neural_score'].fillna(0.0))
            print('SC3 blend: weight %(w)s, scored', int(_c['neural_score'].notna().sum()), 'of', len(_c), flush=True)
        except Exception as _e:
            print('SC3 FAILED, keeping the two-model blend:', repr(_e)[:400], flush=True)
'''
for tag, w in (('w20', 0.20), ('w33', 0.33)):
    nb = json.loads(json.dumps(nb0))
    cells = nb['cells']
    i = next(k for k, c in enumerate(cells) if c['cell_type'] == 'code' and A_OLD in ''.join(c['source']))
    s = ''.join(cells[i]['source'])
    assert s.count(A_OLD) == 1
    s = s.replace(A_OLD, A_OLD + BLOCK % dict(w=w))
    if 'import glob' not in s:
        s = 'import glob, os, sys\n' + s
    cells[i]['source'] = s
    cells[0] = dict(cells[0], source=f'# jc-p2s481v22-sc3-{tag}: jc-p2s481v22 (public 0.481, unchanged) + our own spectrum scorer as a third model '
                    f'at weight {w}. Built by kernels/make_sc3.py.\n\n' + ''.join(cells[0]['source']))
    slug = f'casmi26-jc-p2s481v22-sc3-{tag}'
    d = f'/Users/b/casmi26-repo/kernels/sc3/{tag}'; os.makedirs(d, exist_ok=True)
    json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
    json.dump(dict(meta0, id=f'giantwizard/{slug}', title=slug, code_file=f'{slug}.ipynb',
                   dataset_sources=meta0['dataset_sources'] + ['giantwizard/casmi26-scorer-ours3']), open(f'{d}/kernel-metadata.json', 'w'), indent=1)
    print('wrote', d)
