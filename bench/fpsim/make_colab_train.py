"""Write casmi26_fpsim_train.ipynb: fine-tune the fold-safe FPNet on real + full-pool simulated spectra (from Drive).

Arms (same budget each, resumable: finished arms are skipped, everything is written to Drive):
  control : real training spectra only
  simA    : + all simulated spectra ([M+H]+ 20/40 eV QTOF, [M+Na]+ 30 eV) of the whole pool
  simB    : + the same minus the simulated spectra of every benchmark answer (clean lower bound)
Scores (bench_np.py: held-out NPs not in MassSpecGym, NP panel, held-out non-NPs) go to Drive and to the private
Kaggle dataset giantwizard/casmi26-jc-fpsim-out. Kill rule: simA or simB >= control + 0.03 MRR on hold_np_clean.
"""
import json, sys

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
md("""# CASMI26: train FPNet on real + simulated spectra

Run this after `casmi26_fpsim_simfull` has finished. Reads the simulated spectra from Google Drive
(`MyDrive/casmi26/fpsim_full/`), trains three arms, scores each, and saves everything to
`MyDrive/casmi26/fpsim_train/`. If Colab disconnects, *Run all* again: finished arms are skipped.

**Before you run:** GPU runtime (L4), the `KAGGLE_API_TOKEN` secret, and allow the Google Drive prompt.
About %.1f hours.
""" % (HOURS * 3 + 0.8))
code(r"""# 1. Drive, credentials, dependencies, inputs
import os, sys, json, glob, subprocess
from google.colab import drive, userdata
drive.mount('/content/drive')
os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
def sh(cmd, check=True):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    if check and r.returncode: raise RuntimeError(cmd)
    return r
sh(f'{sys.executable} -m pip install -q -U kaggle rdkit numba pyarrow')
SIMD = '/content/drive/MyDrive/casmi26/fpsim_full'
OUT = '/content/drive/MyDrive/casmi26/fpsim_train'; os.makedirs(OUT, exist_ok=True)
print('simulated parts on Drive:', len(glob.glob(SIMD + '/*_meta.parquet')))
R = '/content/fp'; os.makedirs(R, exist_ok=True)
def get(ds, dst, f=None):
    if os.path.exists(dst + '/.ok'): return
    sh(f'kaggle datasets download {ds} -p {dst} --unzip' + (f' -f {f}' if f else ''))
    open(dst + '/.ok', 'w').close()
def get_files(ds, dst):
    if os.path.exists(dst + '/.ok'): return
    os.makedirs(dst, exist_ok=True)
    names = [l.split(',')[0] for l in sh(f'kaggle datasets files {ds} --csv --page-size 500').stdout.strip().splitlines()[1:]]
    for n in names:
        if not n.startswith('pilot_part') and not os.path.exists(f'{dst}/{n}'):
            sh(f'kaggle datasets download {ds} -f {n} -p {dst} -q')
            if os.path.exists(f'{dst}/{n}.zip'):
                sh(f'cd {dst} && unzip -oq {n}.zip && rm {n}.zip')
    open(dst + '/.ok', 'w').close()
get('giantwizard/casmi26-jc-fpsim', R + '/in')
get_files('giantwizard/casmi26-jc-fpsim-sim', R + '/in2')
get('ahmedberatozer/casmi26-v2-pool', R + '/pool')
get('ahmedberatozer/casmi26-v4b-models', R + '/v4b')
get('dmitriigluzdov/casmi26-fold-safe-fpnet', R + '/fs', 'fpnet_fold0.pt')
CODE = os.path.dirname(os.path.dirname(glob.glob(R + '/v4b/**/casmi/fpnet.py', recursive=True)[0]))
W = R + '/run/work'; os.makedirs(W, exist_ok=True)
for f in glob.glob(R + '/pool/**/*', recursive=True):
    if os.path.isfile(f) and not os.path.exists(W + '/' + os.path.basename(f)):
        os.symlink(f, W + '/' + os.path.basename(f))
for f in ('spec_off.npy', 'spec_mz.npy', 'spec_it.npy', 'spec_meta.parquet', 'split.parquet'):
    if not os.path.exists(W + '/' + f):
        os.symlink(glob.glob(R + '/in/**/' + f, recursive=True)[0], W + '/' + f)
IN = R + '/in2'
CK0 = glob.glob(R + '/fs/**/fpnet_fold0.pt', recursive=True)[0]
""")
code(r"""# 2. train and score each arm (skips arms already on Drive)
HOURS = %s
SIM = f'{SIMD}/mh,{SIMD}/mna'
COMMON = f'--code {CODE} --run {R}/run --init {CK0} --device cuda --bs 256 --lr 1e-4 --steps 200000 --hours {HOURS} --outdir {OUT}'
BENCH = (f'--code {CODE} --run {R}/run --device cuda --n 1500 '
         f'--msg-keys {IN}/msg_train_keys.json --np-flag {IN}/pool_np.npy')
ARMS = {'control': '--sim_p 0', 'simA': f'--sim {SIM} --sim_p 0.5',
        'simB': f'--sim {SIM} --sim_p 0.5 --exclude_sim_pids {IN}/clean_answer_pids.npy'}
def bench(tag, ck):
    if not os.path.exists(f'{OUT}/res_{tag}.json'):
        sh(f'{sys.executable} {IN}/bench_np.py --ckpt {ck} {BENCH} --out {OUT}/res_{tag}.json > {OUT}/bench_{tag}.log 2>&1', check=False)
        print(open(f'{OUT}/bench_{tag}.log').read()[-1200:])
bench('fold0', CK0)
for tag, extra in ARMS.items():
    if not os.path.exists(f'{OUT}/{tag}.pt'):
        sh(f'{sys.executable} {IN}/train_sim.py {COMMON} {extra} --tag {tag} > {OUT}/train_{tag}.log 2>&1', check=False)
        print(open(f'{OUT}/train_{tag}.log').read()[-1200:])
    if os.path.exists(f'{OUT}/{tag}.pt'):
        bench(tag, f'{OUT}/{tag}.pt')
""" % HOURS)
code(r"""# 3. publish the scores (small files only) to the private Kaggle dataset
import shutil
P = '/content/pub'; os.makedirs(P, exist_ok=True)
for f in glob.glob(OUT + '/res_*.json') + glob.glob(OUT + '/*.log'):
    shutil.copy(f, P)
json.dump({'title': 'casmi26-jc-fpsim-out', 'id': 'giantwizard/casmi26-jc-fpsim-out', 'licenses': [{'name': 'other'}]},
          open(P + '/dataset-metadata.json', 'w'))
if sh(f'kaggle datasets create -p {P} -q', check=False).returncode:
    sh(f'kaggle datasets version -p {P} -m "train results" -q')
for f in sorted(glob.glob(OUT + '/res_*.json')):
    d = json.load(open(f)); print(os.path.basename(f), {k: v for k, v in d.items() if k in ('hold_np_clean', 'np_panel', 'hold_nonnp')})
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open('/Users/b/casmi26/fp/casmi26_fpsim_train.ipynb', 'w'), indent=1)
print('wrote casmi26_fpsim_train.ipynb')
