import pickle, sys, numpy as np
for p in sys.argv[1:]:
    res = pickle.load(open(p, 'rb')); rk, reach, ngen = [], [], []
    for key, r in res.items():
        if 'error' in r: continue
        t = r.get('truth_rank'); rk.append(t if t else 999)
        reach.append(key in set(r.get('ckeys', r.get('keys', []))))
        ngen.append(int((np.asarray(r.get('cpid', [])) < 0).sum()))
    rk = np.array(rk)
    print(f'{p.split("/")[-1]:<24} n {len(rk)} reach {np.mean(reach):.3f} top1 {(rk == 1).mean():.3f} top25 {(rk <= 25).mean():.3f} '
          f'MRR {np.where(rk <= 25, 1 / rk, 0).mean():.3f} | generated per molecule median {np.median(ngen):.0f}')
