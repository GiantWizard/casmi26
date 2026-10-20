"""Public-notebook lineage for the Enveda CASMI 2026 competition: for every pulled notebook, the leaderboard scores it
states (with the sentence around each), the notebooks it references, and the Kaggle datasets it reads.
usage: extract.py NOTEBOOK_ROOT REFS_CSV OUT.csv
Scores are the authors' own claims, not verified submissions.
"""
import csv, glob, json, os, re, sys

root, refs_csv, out = sys.argv[1:4]
meta = {}
for line in open(refs_csv):
    p = next(csv.reader([line]))
    if len(p) >= 3 and '/' in p[0]:
        meta[p[0]] = dict(title=p[1], author=p[2], last_run=p[3] if len(p) > 3 else '', votes=p[4] if len(p) > 4 else '')
SCORE = re.compile(r'(?:public\s+)?(?:LB|leaderboard|public score|score)\D{0,12}(0\.4\d{2})', re.I)
REF = re.compile(r'kaggle\.com/code/([\w-]+/[\w-]+)|\b([a-z0-9]{3,}/casmi[\w-]+)\b', re.I)
DS = re.compile(r'/kaggle/input/(?:datasets/)?([\w-]+/[\w-]+|[\w-]+)')
rows = []
for d in sorted(glob.glob(os.path.join(root, '*'))):
    f = next(iter(glob.glob(os.path.join(d, '*.ipynb'))), None)
    if not f:
        continue
    try:
        nb = json.load(open(f))
    except Exception:
        continue
    cells = nb.get('cells', [])
    md = '\n'.join(''.join(c['source']) for c in cells if c.get('cell_type') == 'markdown')
    code = '\n'.join(''.join(c['source']) for c in cells if c.get('cell_type') == 'code')
    ref = os.path.basename(d).replace('_', '/', 1)
    title = meta.get(ref, {}).get('title', '')
    head = (md.strip().split('\n\n')[0] if md.strip() else code[:600]).replace('\n', ' ')[:900]
    claims = []
    for m in SCORE.finditer(title + '\n' + md[:20000] + '\n' + code[:4000]):
        a = max(0, m.start() - 90); ctx = (title + '\n' + md[:20000] + '\n' + code[:4000])[a:m.end() + 40].replace('\n', ' ')
        claims.append(f'{m.group(1)} :: {ctx.strip()}')
    refs = sorted({(x or y).lower() for x, y in REF.findall(md + code) if (x or y).lower() != ref.lower()})
    ds = sorted({x for x in DS.findall(code)})
    rows.append(dict(ref=ref, title=title, author=meta.get(ref, {}).get('author', ''), last_run=meta.get(ref, {}).get('last_run', ''),
                     votes=meta.get(ref, {}).get('votes', ''), n_cells=len(cells), head=head,
                     stated_scores=';'.join(sorted({c.split(' ::')[0] for c in claims})), claims=' || '.join(claims[:6]),
                     refs=';'.join(refs[:25]), datasets=';'.join(ds[:40])))
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('notebooks', len(rows), '| with a stated score', sum(bool(r['stated_scores']) for r in rows), '->', out)
