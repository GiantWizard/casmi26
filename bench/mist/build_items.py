"""Items for mist_score.py from a finished eval run: every spectrum of each molecule (test draw parquet) and every
candidate the cell-19 fusion can see (main list v4_submission.csv, optional extra main lists, and engine 2's list).

usage: build_items.py RUN_DIR TEST_DRAW.parquet OUT.json [--main extra_v4.csv ...]
"""
import argparse, json, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('test'); ap.add_argument('out')
ap.add_argument('--main', nargs='*', default=[])
a = ap.parse_args()
cands = {}
for p in [os.path.join(a.run, 'v4_submission.csv')] + a.main:
    for mid, s in pd.read_csv(p).values:
        cands.setdefault(str(mid), []).extend(x for x in str(s).split(';') if x and x != 'CCO')
eng = json.load(open(os.path.join(a.run, 'eng_lists.json')))
for mid, e in eng.items():
    cands.setdefault(str(mid), []).extend(e.get('smiles') or [])
te = pd.read_parquet(a.test)
items = []
for mid, g in te.groupby('molecule_id'):
    sp = [dict(mz=list(map(float, r.ms2_mzs)), it=list(map(float, r.ms2_normalized_intensities)),
               prec=float(r.precursor_mz), adduct=r.adduct, mode=r.ionization_mode, instrument=r.instrument_type)
          for r in g.itertuples()]
    items.append(dict(mid=str(mid), spectra=sp, cands=list(dict.fromkeys(cands.get(str(mid), [])))))
json.dump(items, open(a.out, 'w'))
print(len(items), 'molecules,', sum(len(i['cands']) for i in items), 'candidates,',
      sum(len(i['spectra']) for i in items), 'spectra')
