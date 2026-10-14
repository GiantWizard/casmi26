"""Pilot simulation list: every pool structure within +-10 ppm of each benchmark answer (hold_np_clean queries,
same selection as bench_np.py), then random pool structures, deduplicated. Writes pilot_pids.npy (windows first)."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/b/casmi26/inputs/v4b/code'); import os; os.chdir('/Users/b/casmi26/fp/run')
st = pd.read_parquet('work/split.parquet'); ts = pd.read_parquet('work/train_structs.parquet', columns=['sid', 'key'])
pool = pd.read_parquet('work/pool_meta.parquet', columns=['key', 'mass'])
k2p = dict(zip(pool.key.values, np.arange(len(pool)))); npf = np.load('/Users/b/casmi26/inputs/pop/pool_np.npy')
msg = set(json.load(open('/Users/b/casmi26/mist/runs/msg_train_keys.json')))
hold = st[st.fold == 'hold'].sid.values; key_of = dict(zip(ts.sid.values, ts.key.values))
ans = [k2p.get(key_of[s], -1) for s in hold]
keep = [(s, p) for s, p in zip(hold, ans) if p >= 0 and npf[p] > 0 and str(key_of[s])[:14] not in msg]
sel = np.random.default_rng(7).choice(len(keep), 600, replace=False)
keep = [keep[i] for i in sorted(sel)]
np.save('/Users/b/casmi26/fp/pilot_sids.npy', np.array([s for s, _ in keep]))
ans = np.array([p for _, p in keep])
pm = pool.mass.values
win = set()
for p in ans:
    m = pm[p]; lo = np.searchsorted(pm, m * (1 - 1e-5)); hi = np.searchsorted(pm, m * (1 + 1e-5), 'right')
    win.update(range(lo, hi))
rng = np.random.default_rng(0)
rest = np.setdiff1d(rng.choice(len(pm), 32000, replace=False), np.fromiter(win, int))
out = np.concatenate([np.array(sorted(win)), rest])
np.save('/Users/b/casmi26/fp/pilot_pids.npy', out)
print('answers', len(ans), 'window structures', len(win), 'random', len(rest), 'total', len(out))
