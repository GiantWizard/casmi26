"""Which of our withheld structures are in MassSpecGym (the training data of ICEBERG and GLACIER)?

ICEBERG/GLACIER scores on a withheld structure that MassSpecGym trained on are optimistic, so reranker rows for those
structures get a flag. Reads only the inchikey and fold columns of the MassSpecGym table on Hugging Face.
usage: msg_overlap.py SPLIT_PARQUET OUT_JSON
"""
import json, sys
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

split, out = sys.argv[1:3]
repo = 'roman-bushuiev/MassSpecGym'
tsv = sorted((f for f in list_repo_files(repo, repo_type='dataset') if f.endswith('MassSpecGym.tsv')), key=len)[0]
m = pd.read_csv(hf_hub_download(repo, tsv, repo_type='dataset'), sep='\t', usecols=['inchikey', 'fold'])
m['ik14'] = m.inchikey.str[:14]
sp = pd.read_parquet(split)
res = {'source': f'{repo}/{tsv}', 'folds': {}}
for f, g in m.groupby('fold'):
    keys = set(g.ik14)
    res['folds'][f] = sorted(keys)
    print(f'MassSpecGym {f}: {len(keys):,} structures; in our withheld split: '
          + ', '.join(f'{k} {sp[sp.fold == k].inchikey14.isin(keys).sum():,}/{(sp.fold == k).sum():,}' for k in sorted(sp.fold.unique())), flush=True)
json.dump(res, open(out, 'w'))
print('wrote', out)
