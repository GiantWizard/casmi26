"""Write casmi26_fpsim_simfull.ipynb: simulate the whole candidate pool with the MassSpecGym GLACIER on a Colab GPU.

[M+H]+ at 20 and 40 eV and [M+Na]+ at 30 eV for all 710,701 pool structures; chunks go straight to Google Drive
(MyDrive/casmi26/fpsim_full/) so a disconnect loses at most one chunk, and Run all resumes where it stopped.
"""
import json

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
md("""# CASMI26: simulate the whole candidate pool (GLACIER, MassSpecGym checkpoint)

Writes simulated spectra for every pool structure to Google Drive (`MyDrive/casmi26/fpsim_full/`). Safe to stop
and restart: *Run all* resumes from the last finished chunk.

**Before you run:** GPU runtime (L4), the `KAGGLE_API_TOKEN` secret, and allow the Google Drive prompt.
""")
code(r"""# 1. Drive, credentials, dependencies
import os, sys, subprocess, glob, time
from google.colab import drive, userdata
drive.mount('/content/drive')
os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
def sh(cmd, check=True):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    if check and r.returncode: raise RuntimeError(cmd)
    return r
sh(f'{sys.executable} -m pip install -q -U kaggle rdkit pytorch_lightning==2.6.6 torchmetrics einops lightning_utilities pyarrow')
OUT = '/content/drive/MyDrive/casmi26/fpsim_full'; os.makedirs(OUT, exist_ok=True)
R = '/content/fp'; os.makedirs(R, exist_ok=True)
def get(ds, dst, f=None):
    if os.path.exists(dst + '/.ok'): return
    sh(f'kaggle datasets download {ds} -p {dst} --unzip' + (f' -f {f}' if f else ''))
    open(dst + '/.ok', 'w').close()
get('ahmedberatozer/casmi26-glacier', R + '/glacier')
get('ahmedberatozer/casmi26-v2-pool', R + '/pool')
sh(f'kaggle datasets download giantwizard/casmi26-jc-fpsim-sim -f sim_dump.py -p {R}', check=False)
PKG = os.path.dirname(glob.glob(R + '/glacier/**/gl_runner.py', recursive=True)[0])
POOL = glob.glob(R + '/pool/**/pool_meta.parquet', recursive=True)[0]
import numpy as np, pandas as pd
n = len(pd.read_parquet(POOL, columns=['key']))
np.save(R + '/all_pids.npy', np.arange(n))
print('pool', n, '| pkg', PKG)
""")
code(r"""# 2. simulate: [M+H]+ 20/40 eV, then [M+Na]+ 30 eV (each resumes from Drive)
for tag, adduct, ces in (('mh', '[M+H]+', '20,40'), ('mna', '[M+Na]+', '30')):
    sh(f'{sys.executable} {R}/sim_dump.py {R}/all_pids.npy {OUT}/{tag} --pkg {PKG} --pool {POOL} --adduct "{adduct}" '
       f'--ces {ces} --instr QTOF --device cuda --bs 64 --chunk 5000 > {OUT}/{tag}.log 2>&1', check=False)
    print(open(f'{OUT}/{tag}.log').read()[-1500:])
print('parts on Drive:', len(glob.glob(OUT + '/*_meta.parquet')))
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open('/Users/b/casmi26/fp/casmi26_fpsim_simfull.ipynb', 'w'), indent=1)
print('wrote casmi26_fpsim_simfull.ipynb')
