import pickle, sys, numpy as np
sys.path.insert(0, sys.argv[1]); from insert import combine
O = sys.argv[1] + '/out/'
a = pickle.load(open(O + 'dbdrop.pkl', 'rb'))
runs = {t: pickle.load(open(O + f'dbdrop_{t}.pkl', 'rb')) for t in sys.argv[2:]}
ks = [k for k in a if 'keys' in a[k] and all('keys' in r.get(k, {}) for r in runs.values())]
rr = lambda L, k: 1 / (L.index(k) + 1) if k in L[:25] else 0.0
base = np.array([rr(list(dict.fromkeys(a[k]['keys'])), k) for k in ks])
print(len(ks), 'molecules | default MRR %.3f reach %.3f' % (base.mean(), np.mean([k in set(a[k]['ckeys']) for k in ks])))
for t, r in runs.items():
    y = np.array([rr(combine(a[k], r[k]), k) for k in ks]); d = y - base
    ci = np.percentile([np.random.default_rng(i).choice(d, len(d)).mean() for i in range(2000)], [5, 95])
    print(f'  {t:<6} combined MRR {y.mean():.3f} (delta {d.mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}], better {(d>0).sum()} worse {(d<0).sum()}) '
          f'reach {np.mean([k in set(r[k]["ckeys"]) for k in ks]):.3f}  gen/mol median {np.median([(np.asarray(r[k]["cpid"])<0).sum() for k in ks]):.0f}')
