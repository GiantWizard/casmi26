"""Write colab/casmi26_e2fp.ipynb: engine 2's two FPNets retrained with our holdout withheld, plus the MassSpecGym overlap.
All sources are embedded, so nothing has to sync to Drive first. Outputs go to MyDrive/casmi26/e2fp/."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {p: open(os.path.join(HERE, p)).read() for p in
         ['prep_e2.py', 'msg_overlap.py', 'third_party/fpmodel.py', 'third_party/train_fp3.py', 'third_party/train_fp4.py',
          'third_party/LICENSE-SOURCE', 'third_party/NOTICE']}
cells = []
md = lambda s: cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s})
code = lambda s: cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s})

md("""# CASMI26: engine 2 fingerprint models, retrained with fold 0 withheld

Retrains engine 2's two FPNets (`fp_single_s2`, `fp_merged_m1`) with the original trainers (cubres/enveda-casmi-2026,
Apache 2.0), leaving out every structure of our fold-safe split (identity fold 0 + NP panel). Their scores on those
structures are then honest features for the stacked reranker. Also records which withheld structures are in MassSpecGym.

**Before you run:** GPU runtime (L4 or T4) with **High-RAM**, the `KAGGLE_API_TOKEN` secret, and allow the Google Drive prompt.
Then *Run all*. Everything lands in `MyDrive/casmi26/e2fp/`. If Colab disconnects, *Run all* again: finished steps are skipped
and a model in progress resumes from its best checkpoint.

Time: data prep about 30-40 min; single-spectrum model up to 5 h; merged model up to 7 h (the original time caps).""")
code("""# 1. Drive, credentials, dependencies, inputs, embedded sources
import os, sys, glob, subprocess
from google.colab import drive, userdata
drive.mount('/content/drive')
os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
def sh(cmd, check=True):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if check and r.returncode: raise RuntimeError(cmd)
    return r
sh(f'{sys.executable} -m pip install -q -U kaggle rdkit pyarrow huggingface_hub')
OUT = '/content/drive/MyDrive/casmi26/e2fp'; os.makedirs(OUT, exist_ok=True)
R = '/content/e2'; os.makedirs(R, exist_ok=True)
SRC = '/content/e2fp'
FILES = """ + repr(FILES) + """
for p, s in FILES.items():
    os.makedirs(os.path.dirname(f'{SRC}/{p}'), exist_ok=True); open(f'{SRC}/{p}', 'w').write(s)
def get(cmd, dst):
    if not os.path.exists(dst + '/.ok'):
        os.makedirs(dst, exist_ok=True); sh(cmd + f' -p {dst}'); open(dst + '/.ok', 'w').close()
get('kaggle competitions download enveda-CASMI26-molecule-id-mass-spectra -f train.parquet', R + '/comp')
if glob.glob(R + '/comp/*.zip'): sh(f'cd {R}/comp && unzip -oq "*.zip" && rm -f *.zip')
get('kaggle datasets download prvsiyan/coconut-casmi26-candidates --unzip', R + '/coco')
get('kaggle datasets download dmitriigluzdov/casmi26-fold-safe-fpnet -f split.parquet', R + '/split')
if glob.glob(R + '/split/*.zip'): sh(f'cd {R}/split && unzip -oq "*.zip" && rm -f *.zip')
TRAIN = glob.glob(R + '/comp/**/train.parquet', recursive=True)[0]
COCO = os.path.dirname(glob.glob(R + '/coco/**/coco_fp.npy', recursive=True)[0])
SPLIT = glob.glob(R + '/split/**/split.parquet', recursive=True)[0]
print(TRAIN, COCO, SPLIT)
sh('nvidia-smi --query-gpu=name,memory.total --format=csv; free -g; nproc')
""")
code("""# 2. MassSpecGym overlap with the withheld structures (a few minutes)
if not os.path.exists(f'{OUT}/msg_overlap.json'):
    sh(f'{sys.executable} {SRC}/msg_overlap.py {SPLIT} {OUT}/msg_overlap.json', check=False)
""")
code("""# 3. trainer inputs (cached on Drive, so a reconnect skips this)
W = R + '/work'
need = ['ds/spec.npz', 'up_train/fp_meta.pkl', 'pool_fp.npy', 'pool_mass.npy', 'pool_meta.pkl']
if all(os.path.exists(f'{OUT}/prep/{f}') for f in need):
    for f in need:
        os.makedirs(os.path.dirname(f'{W}/{f}'), exist_ok=True)
        if not os.path.exists(f'{W}/{f}'): sh(f'cp {OUT}/prep/{f} {W}/{f}')
else:
    sh(f'{sys.executable} {SRC}/prep_e2.py {TRAIN} {COCO} {SPLIT} {W} --workers {os.cpu_count()} > {OUT}/prep.log 2>&1', check=False)
    print(open(f'{OUT}/prep.log').read()[-3000:])
    for f in need:
        os.makedirs(os.path.dirname(f'{OUT}/prep/{f}'), exist_ok=True); sh(f'cp {W}/{f} {OUT}/prep/{f}')
""")
code("""# 4. train both models (original recipes; each skipped once done, resumed from its best checkpoint otherwise)
RUNS = [('fp_single_s2_f0', 'train_fp3.py', '--steps 24000 --seed 202 --max_minutes 300'),
        ('fp_merged_m1_f0', 'train_fp4.py', '--steps 50000 --seed 404 --merge_p 0.60 --max_minutes 420')]
for tag, trainer, extra in RUNS:
    if os.path.exists(f'{OUT}/{tag}.done'):
        print(tag, 'already done'); continue
    sh(f'cd {W} && PYTHONPATH={SRC}/third_party {sys.executable} -u {SRC}/third_party/{trainer} --spec ds/spec.npz --fpdir up_train '
       f'--pooldir . --bs 256 --lr 2e-4 --K 63 --lam 1.0 --d 512 --layers 6 --val_every 2000 {extra} '
       f'--out {OUT}/{tag}.pt --resume {OUT}/{tag}.pt >> {OUT}/{tag}.log 2>&1', check=False)
    log = open(f'{OUT}/{tag}.log').read()
    print(log[-1500:])
    if 'done. best hardneg-top1' in log: open(f'{OUT}/{tag}.done', 'w').close()
""")
code("""# 5. summary
for tag, _, _ in RUNS:
    log = open(f'{OUT}/{tag}.log').read() if os.path.exists(f'{OUT}/{tag}.log') else ''
    vals = [l for l in log.splitlines() if 'VAL step' in l]
    print(tag, 'done' if os.path.exists(f'{OUT}/{tag}.done') else 'NOT done', '|', vals[-1] if vals else 'no validation yet')
sh(f'ls -la {OUT}')
""")
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'machine_shape': 'hm', 'gpuType': 'L4'},
                                    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}, 'nbformat': 4, 'nbformat_minor': 0}
path = os.path.join(HERE, '..', '..', 'colab', 'casmi26_e2fp.ipynb')
json.dump(nb, open(path, 'w'), indent=1)
print('wrote', os.path.abspath(path))
