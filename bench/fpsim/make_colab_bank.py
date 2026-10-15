"""Write casmi26_fpsim_bank.ipynb: fine-tune engine 1's own FP bank (ahmedberatozer/casmi26-fpnet-full1) on real +
full-pool simulated spectra, for deployment as the engine bank in kernels/jc-v7c-simbank.

Differences from casmi26_fpsim_train (the benchmark arms): init from fpnet_full1 instead of the fold-safe fold0, no
holdout (full1 was trained for deployment), smaller learning rate so the v4b ranker's FP features stay close to what it
was trained on, simA recipe (sim_p 0.5, all simulations). No benchmark: full1 has seen the benchmark molecules.
The last cell uploads the checkpoint as the private Kaggle dataset giantwizard/casmi26-jc-fpsim-bank (Jeffery runs it).
"""
import json, sys

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-5
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
md(f"""# CASMI26: simulation-augmented engine FP bank

Fine-tunes `fpnet_full1` (engine 1's FP bank in the 0.443 notebook) on real + simulated spectra from Drive
(`MyDrive/casmi26/fpsim_full/`), about {HOURS:g} h on an L4, saves `full1_sim.pt` to `MyDrive/casmi26/fpsim_bank/`.
**The last cell creates the private Kaggle dataset `giantwizard/casmi26-jc-fpsim-bank`**; skip it if you don't want that.
Before you run: GPU runtime, the `KAGGLE_API_TOKEN` secret, allow the Google Drive prompt.
""")
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
OUT = '/content/drive/MyDrive/casmi26/fpsim_bank'; os.makedirs(OUT, exist_ok=True)
print('simulated parts on Drive:', len(glob.glob(SIMD + '/*_meta.parquet')))
R = '/content/fp'; os.makedirs(R, exist_ok=True)
def get(ds, dst, f=None):
    if os.path.exists(dst + '/.ok'): return
    sh(f'kaggle datasets download {ds} -p {dst} --unzip' + (f' -f {f}' if f else ''))
    for z in glob.glob(dst + '/*.zip'):   # single-file downloads can arrive zipped
        sh(f'cd {dst} && unzip -oq {os.path.basename(z)} && rm {os.path.basename(z)}')
    open(dst + '/.ok', 'w').close()
get('giantwizard/casmi26-jc-fpsim', R + '/in')
get('giantwizard/casmi26-jc-fpsim-sim', R + '/in2', 'train_sim.py')
get('ahmedberatozer/casmi26-v2-pool', R + '/pool')
get('ahmedberatozer/casmi26-v4b-models', R + '/v4b')
get('ahmedberatozer/casmi26-fpnet-full1', R + '/full1')
CODE = os.path.dirname(os.path.dirname(glob.glob(R + '/v4b/**/casmi/fpnet.py', recursive=True)[0]))
W = R + '/run/work'; os.makedirs(W, exist_ok=True)
for f in glob.glob(R + '/pool/**/*', recursive=True):
    if os.path.isfile(f) and not os.path.exists(W + '/' + os.path.basename(f)):
        os.symlink(f, W + '/' + os.path.basename(f))
for f in ('spec_off.npy', 'spec_mz.npy', 'spec_it.npy', 'spec_meta.parquet', 'split.parquet'):
    if not os.path.exists(W + '/' + f):
        os.symlink(glob.glob(R + '/in/**/' + f, recursive=True)[0], W + '/' + f)
TS = glob.glob(R + '/in2/**/train_sim.py', recursive=True)[0]
s = open(TS).read()
if '--holdout' not in s:   # the dataset copy predates the --holdout option; same patch as the repo version
    s = s.replace("a = ap.parse_args()", "ap.add_argument('--holdout', default='hold,np')\na = ap.parse_args()", 1)
    s = s.replace("holdout_folds=('hold', 'np'))", "holdout_folds=tuple(f for f in a.holdout.split(',') if f))")
    assert 'a.holdout' in s
    open(TS, 'w').write(s)
CK = glob.glob(R + '/full1/**/fpnet_full1.pt', recursive=True)[0]
print('init:', CK)
""")
code(r"""# 2. fine-tune (skipped if the checkpoint is already on Drive)
HOURS, LR = %s, %s
SIM = f'{SIMD}/mh,{SIMD}/mna'
if not os.path.exists(f'{OUT}/full1_sim.pt'):
    sh(f'{sys.executable} {TS} --code {CODE} --run {R}/run --init {CK} --device cuda --bs 256 --lr {LR} '
       f'--steps 200000 --hours {HOURS} --holdout "" --sim {SIM} --sim_p 0.5 --tag full1_sim --outdir {OUT} '
       f'> {OUT}/train_full1_sim.log 2>&1', check=False)
print(open(f'{OUT}/train_full1_sim.log').read()[-1500:])
print('checkpoint:', os.path.exists(f'{OUT}/full1_sim.pt'))
""" % (HOURS, LR))
code(r"""# 3. upload as the private Kaggle dataset giantwizard/casmi26-jc-fpsim-bank (used by kernels/jc-v7c-simbank)
import shutil
P = '/content/pub_bank'; os.makedirs(P, exist_ok=True)
shutil.copy(f'{OUT}/full1_sim.pt', P)
json.dump({'title': 'casmi26-jc-fpsim-bank', 'id': 'giantwizard/casmi26-jc-fpsim-bank', 'licenses': [{'name': 'other'}]},
          open(P + '/dataset-metadata.json', 'w'))
if sh(f'kaggle datasets create -p {P} -q', check=False).returncode:
    sh(f'kaggle datasets version -p {P} -m "full1 + sim fine-tune" -q')
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open('/Users/b/casmi26-repo/colab/casmi26_fpsim_bank.ipynb', 'w'), indent=1)
print('wrote colab/casmi26_fpsim_bank.ipynb')
