"""Write colab/casmi26_scorer.ipynb: train a third spectrum->structure scorer with Saner Turhaner's public kit
(sanpier/casmi-spectrum-model-v1, Apache 2.0), the same recipe bgunshui's public trainer runs, but deliberately different
from the two models already in the 0.481 blend: new seed, wider/deeper network (width 256, 4 layers, 8 heads, embedding
256) and twice the hard negatives. Checkpoints are written straight to Drive; at the end the bundle (final_model.pt,
config.json, kit sources) is uploaded as the private Kaggle dataset giantwizard/casmi26-scorer-ours3.
"""
import json, os

cells = []
md = lambda s: cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s})
code = lambda s: cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s})
md("""# CASMI26: third spectrum-to-structure scorer

Trains a spectrum-candidate scorer with the public kit (Saner Turhaner, Apache 2.0), as bgunshui's public trainer does,
but with a new seed and a bigger network, so it disagrees with the two scorers already blended in the 0.481 notebook.

**Before you run:** GPU runtime (L4 or A100; T4 works but is slower) with **High-RAM**, the `KAGGLE_API_TOKEN` secret, and
allow the Google Drive prompt. Then *Run all*. Up to about 11 hours: data prep, a held-out run (up to 4.5 h) that picks the
epoch count, then a full-data refit (up to 5.5 h). Everything is saved to `MyDrive/casmi26/scorer3/`; the last cell uploads
the finished model to Kaggle. If Colab disconnects, *Run all* again: a finished run is skipped, an unfinished one restarts.""")
code("""# 1. Drive, credentials, inputs
import os, sys, glob, json, shutil, subprocess
from google.colab import drive, userdata
drive.mount('/content/drive')
os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
def sh(cmd, check=True):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    if check and r.returncode: raise RuntimeError(cmd)
    return r
sh(f'{sys.executable} -m pip install -q -U kaggle six')
OUT = '/content/drive/MyDrive/casmi26/scorer3'; os.makedirs(OUT, exist_ok=True)
R = '/content/sc'; os.makedirs(R, exist_ok=True)
def get(cmd, dst):
    if not os.path.exists(dst + '/.ok'):
        os.makedirs(dst, exist_ok=True); sh(cmd + f' -p {dst}')
        for z in glob.glob(dst + '/**/*.zip', recursive=True):
            sh(f'cd {os.path.dirname(z)} && unzip -oq "{z}" && rm -f "{z}"')
        open(dst + '/.ok', 'w').close()
get('kaggle competitions download enveda-CASMI26-molecule-id-mass-spectra -f train.parquet', R + '/comp')
get('kaggle datasets download sanpier/casmi-spectrum-model-v1 --unzip', R + '/kit')
get('kaggle datasets download metric/rdkit-2026-3-3-wheel --unzip', R + '/rdkit')
whl = glob.glob(R + '/rdkit/**/rdkit-2026.3.3-cp312-*.whl', recursive=True)
assert whl and sys.version_info[:2] == (3, 12), (whl, sys.version)
sh(f'{sys.executable} -m pip install -q --no-deps {whl[0]}')
TRAIN = glob.glob(R + '/comp/**/train.parquet', recursive=True)[0]
KIT = os.path.dirname(glob.glob(R + '/kit/**/casmi_spectrum_model_v1/train.py', recursive=True)[0])
print(TRAIN, KIT); sh('nvidia-smi --query-gpu=name,memory.total --format=csv; free -g')
""")
code("""# 2. train (skipped once finished)
CFG_DELTA = dict(seed=20261104, width=256, heads=8, layers=4, embedding=256, hard_negatives=32,
                 epochs=80, holdout_hours=4.5, refit_hours=5.5)
if os.path.exists(f'{OUT}/DONE'):
    print('already trained:', sorted(os.listdir(OUT)))
else:
    stage = '/content/kit_staged'; shutil.copytree(KIT, stage, dirs_exist_ok=True)
    p = f'{stage}/prepare.py'; s = open(p).read()
    s = s.replace("'smiles':['smiles','canonical_smiles']", "'smiles':['smiles','canonical_smiles','normalized_smiles']")
    open(p, 'w').write(s)
    sys.path.insert(0, stage)
    import prepare, train
    cfg = json.load(open(f'{KIT}/config.json'))
    cfg.update(CFG_DELTA, train_path=TRAIN, work='/content/sc/work', output=OUT, external_paths=[])
    print('CFG', json.dumps(cfg))
    audit = prepare.prepare(cfg); print('PREP', json.dumps(audit)[:600], flush=True)
    shutil.copy2('/content/sc/work/data_audit.json', f'{OUT}/data_audit.json')
    result = train.run(cfg); print('RESULT', json.dumps(result), flush=True)
    json.dump(result, open(f'{OUT}/training_result.json', 'w'), indent=2)
    json.dump(cfg, open(f'{OUT}/config.json', 'w'), indent=2)          # this model's own architecture
    for f in glob.glob(f'{KIT}/*.py'):
        shutil.copy2(f, OUT)
    open(f'{OUT}/DONE', 'w').close()
sh(f'ls -la {OUT}')
""")
code("""# 3. upload the bundle to Kaggle (private dataset giantwizard/casmi26-scorer-ours3)
B = '/content/scorer3_ds'; shutil.rmtree(B, ignore_errors=True); os.makedirs(B)
for f in ['final_model.pt', 'config.json', 'training_result.json', 'data_audit.json'] + [os.path.basename(x) for x in glob.glob(f'{OUT}/*.py')]:
    shutil.copy2(f'{OUT}/{f}', B)
json.dump({'title': 'casmi26-scorer-ours3', 'id': 'giantwizard/casmi26-scorer-ours3', 'licenses': [{'name': 'apache-2.0'}]},
          open(f'{B}/dataset-metadata.json', 'w'))
print(json.load(open(f'{OUT}/training_result.json')))
r = sh(f'kaggle datasets create -p {B} 2>&1', check=False)
if 'already' in (r.stdout + r.stderr).lower():
    sh(f'kaggle datasets version -p {B} -m "retrained" 2>&1')
""")
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'machine_shape': 'hm', 'gpuType': 'L4'},
                                    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}}, 'nbformat': 4, 'nbformat_minor': 0}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'colab', 'casmi26_scorer.ipynb')
json.dump(nb, open(path, 'w'), indent=1)
print('wrote', os.path.abspath(path))
