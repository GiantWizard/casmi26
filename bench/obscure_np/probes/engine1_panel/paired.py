import pickle, sys, numpy as np
a=pickle.load(open(sys.argv[1],'rb')); b=pickle.load(open(sys.argv[2],'rb'))
rr=lambda r: (1/r['truth_rank'] if r.get('truth_rank') and r['truth_rank']<=25 else 0.0)
ks=[k for k in a if k in b and 'error' not in a[k] and 'error' not in b[k]]
x=np.array([rr(a[k]) for k in ks]); y=np.array([rr(b[k]) for k in ks]); d=y-x
ci=np.percentile([np.random.default_rng(i).choice(d,len(d)).mean() for i in range(2000)],[5,95])
print(sys.argv[2].split('/')[-1], len(ks),'molecules  base %.3f  variant %.3f  delta %+.4f [90%% CI %+.4f, %+.4f]  improved %d worse %d'%(x.mean(),y.mean(),d.mean(),ci[0],ci[1],(d>0).sum(),(d<0).sum()))
