"""Write the Colab notebook for the simulated-spectra FPNet pilot (casmi26_fpsim_pilot.ipynb).

Arms (all fine-tuned from Gluzdov's fold-safe fpnet_fold0.pt for the same budget):
  control : real training spectra only
  simA    : + GLACIER-simulated spectra of the pilot pool structures (benchmark answers included, as deployment would)
  simB    : + the same, with the benchmark answers' own simulated spectra removed (clean lower bound)
Each arm is scored by bench_np.py on the 600 pilot answers (hold_np_clean) and the NP panel; results and
checkpoints go to the private dataset giantwizard/casmi26-jc-fpsim-out.
"""
import json, sys

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
md("""# CASMI26 simulated-spectra FPNet pilot

Fine-tunes the fold-safe FPNet three ways (real only / + simulated spectra / + simulated spectra without the
benchmark answers) and scores each on held-out natural products. Results upload to the private Kaggle dataset
`giantwizard/casmi26-jc-fpsim-out`.

**Before you run:** *Runtime → Change runtime type → GPU* (L4 recommended), and the `KAGGLE_API_TOKEN` secret
with *Notebook access* on (same as the other CASMI26 Colab notebooks). Then *Runtime → Run all*. About %.0f hours.
""" % (HOURS * 3 + 0.7))
code(r"""# 1. credentials, CLI, dependencies
import os, json, subprocess, sys, time, shutil, glob
from google.colab import userdata
os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
def sh(cmd, check=True):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if check and r.returncode: raise RuntimeError(f'failed: {cmd}')
    return r
sh(f'{sys.executable} -m pip install -q -U kaggle rdkit numba pyarrow')
sh('nvidia-smi --query-gpu=name,memory.total --format=csv')
""")
code(r"""# 2. inputs
R = '/content/fp'; os.makedirs(R, exist_ok=True)
def get(ds, dst, f=None):
    if os.path.exists(dst + '/.ok'): return
    sh(f'kaggle datasets download {ds} -p {dst} --unzip' + (f' -f {f}' if f else ''))
    open(dst + '/.ok', 'w').close()
def get_files(ds, dst):
    # file by file: a freshly created dataset can be 'ready' before its whole-dataset zip exists
    if os.path.exists(dst + '/.ok'): return
    os.makedirs(dst, exist_ok=True)
    names = [l.split(',')[0] for l in sh(f'kaggle datasets files {ds} --csv --page-size 500').stdout.strip().splitlines()[1:]]
    for n in names:
        if not os.path.exists(f'{dst}/{n}'):
            sh(f'kaggle datasets download {ds} -f {n} -p {dst} -q')
            if os.path.exists(f'{dst}/{n}.zip'):
                sh(f'cd {dst} && unzip -oq {n}.zip && rm {n}.zip')
    missing = [n for n in names if not os.path.exists(f'{dst}/{n}')]
    assert not missing, missing
    open(dst + '/.ok', 'w').close()
get('giantwizard/casmi26-jc-fpsim', R + '/in')
get_files('giantwizard/casmi26-jc-fpsim-sim', R + '/in2')
get('ahmedberatozer/casmi26-v2-pool', R + '/pool')
get('ahmedberatozer/casmi26-v4b-models', R + '/v4b')
get('dmitriigluzdov/casmi26-fold-safe-fpnet', R + '/fs', 'fpnet_fold0.pt')
CODE = os.path.dirname(glob.glob(R + '/v4b/**/casmi/fpnet.py', recursive=True)[0])
CODE = os.path.dirname(CODE)
W = R + '/run/work'; os.makedirs(W, exist_ok=True)
for f in glob.glob(R + '/pool/**/*', recursive=True):
    if os.path.isfile(f) and not os.path.exists(W + '/' + os.path.basename(f)):
        os.symlink(f, W + '/' + os.path.basename(f))
for f in ('spec_off.npy', 'spec_mz.npy', 'spec_it.npy', 'spec_meta.parquet', 'split.parquet'):
    if not os.path.exists(W + '/' + f):
        os.symlink(glob.glob(R + '/in/**/' + f, recursive=True)[0], W + '/' + f)
IN = os.path.dirname(glob.glob(R + '/in2/**/train_sim.py', recursive=True)[0])
CK0 = glob.glob(R + '/fs/**/fpnet_fold0.pt', recursive=True)[0]
print('code', CODE, '| inputs', IN, '| work', sorted(os.listdir(W)))
""")
code(r"""# 3. train the three arms and score each
OUT = '/content/out'; os.makedirs(OUT, exist_ok=True)
HOURS = %s
SIM = IN + '/pilot'
COMMON = f'--code {CODE} --run {R}/run --init {CK0} --device cuda --bs 256 --lr 1e-4 --steps 60000 --hours {HOURS} --outdir {OUT}'
BENCH = (f'--code {CODE} --run {R}/run --device cuda --sids {IN}/pilot_sids.npy --only hold_np_clean,np_panel '
         f'--msg-keys {IN}/msg_train_keys.json --np-flag {IN}/pool_np.npy')
ARMS = {'control': '--sim_p 0', 'simA': f'--sim {SIM} --sim_p 0.5',
        'simB': f'--sim {SIM} --sim_p 0.5 --exclude_sim_pids {IN}/pilot_answer_pids.npy'}
def bench(tag, ck):
    if not os.path.exists(f'{OUT}/res_{tag}.json'):
        sh(f'{sys.executable} {IN}/bench_np.py --ckpt {ck} {BENCH} --out {OUT}/res_{tag}.json > {OUT}/bench_{tag}.log 2>&1', check=False)
        print(open(f'{OUT}/bench_{tag}.log').read()[-1500:])
bench('fold0', CK0)
for tag, extra in ARMS.items():
    if not os.path.exists(f'{OUT}/{tag}.pt'):
        sh(f'{sys.executable} {IN}/train_sim.py {COMMON} {extra} --tag {tag} > {OUT}/train_{tag}.log 2>&1', check=False)
        print(open(f'{OUT}/train_{tag}.log').read()[-1500:])
    if os.path.exists(f'{OUT}/{tag}.pt'):
        bench(tag, f'{OUT}/{tag}.pt')
""" % HOURS)
code(r"""# 4. upload results
json.dump({'title': 'casmi26-jc-fpsim-out', 'id': 'giantwizard/casmi26-jc-fpsim-out', 'licenses': [{'name': 'other'}]},
          open(OUT + '/dataset-metadata.json', 'w'))
r = sh(f'kaggle datasets create -p {OUT} -q', check=False)
if r.returncode:
    sh(f'kaggle datasets version -p {OUT} -m "pilot results" -q')
for f in sorted(glob.glob(OUT + '/res_*.json')):
    d = json.load(open(f)); print(os.path.basename(f), {k: v for k, v in d.items() if k in ('hold_np_clean', 'np_panel')})
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open('/Users/b/casmi26/fp/casmi26_fpsim_pilot.ipynb', 'w'), indent=1)
print('wrote casmi26_fpsim_pilot.ipynb')
