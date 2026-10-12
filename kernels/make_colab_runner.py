"""Generate colab/casmi26_colab_runner.ipynb and the Kaggle jobs dataset it reads.

The runner offloads GPU jobs from Kaggle's weekly quota to Colab:
  1. reads jobs from the private Kaggle dataset giantwizard/casmi26-jc-colab-jobs
     (each job = a Kaggle notebook + its kernel-metadata.json, as generated in kernels/),
  2. rebuilds /kaggle/input from the job's dataset_sources + competition_sources,
  3. runs the notebook with papermill on a Python 3.13 venv (the pinned RDKit wheels are cp313),
  4. uploads the job's small outputs to giantwizard/casmi26-jc-colab-out (one folder per job).

    python kernels/make_colab_runner.py --jobs jc-onp-eval-d2 jc-onp-eval-d3   # writes colab nb + jobs dir
    kaggle datasets version -p colab/jobs -m "jobs" --dir-mode zip                # (or create the first time)
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--jobs", nargs="*", default=[], help="kernel dirs under kernels/ to package as jobs")
args = ap.parse_args()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


cells = [
    md("""
# CASMI26 Colab runner

Runs the CASMI26 pipeline jobs on Colab's GPU instead of Kaggle's weekly quota, then uploads the results to a
private Kaggle dataset (`giantwizard/casmi26-jc-colab-out`) where they are picked up for scoring.

**Before you run (once):**
1. *Runtime → Change runtime type → GPU* (T4 works; L4 or A100 is faster).
2. Click the key icon (Secrets) on the left, add a secret named `KAGGLE_API_TOKEN` with your Kaggle API token
   (the `KGAT_…` token from kaggle.com → Settings → API), and switch on *Notebook access*.

Then *Runtime → Run all*. Each job takes about 2–4 hours on a T4. Jobs already uploaded are skipped, so if
Colab disconnects, just *Run all* again.
"""),
    code(r"""
# 1. Kaggle credentials and CLI
import os, json, subprocess, sys, time, shutil, glob, pathlib
from google.colab import userdata
os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'kaggle'], check=True)
def sh(cmd, check=True):
    print('$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if check and r.returncode: raise RuntimeError(f'failed: {cmd}')
    return r
sh('kaggle competitions list -s casmi | head -4')
sh('nvidia-smi --query-gpu=name,memory.total --format=csv')
"""),
    code(r"""
# 2. Jobs from Kaggle, and results already uploaded
JOBS_DS, OUT_DS = 'giantwizard/casmi26-jc-colab-jobs', 'giantwizard/casmi26-jc-colab-out'
shutil.rmtree('/content/jobs', ignore_errors=True)
sh(f'kaggle datasets download {JOBS_DS} -p /content/jobs --unzip')
for z in glob.glob('/content/jobs/**/*.zip', recursive=True):
    sh(f'cd "{os.path.dirname(z)}" && unzip -oq "{z}" && rm "{z}"')
jobs = json.load(open('/content/jobs/jobs.json'))
OUT = '/content/out'; os.makedirs(OUT, exist_ok=True)
have_out = sh(f'kaggle datasets download {OUT_DS} -p {OUT} --unzip', check=False).returncode == 0
for z in glob.glob(f'{OUT}/*.zip'):
    sh(f'cd {OUT} && unzip -oq "{z}" -d "{z[:-4]}" && rm "{z}"')
done = {d for d in os.listdir(OUT) if os.path.exists(f'{OUT}/{d}/DONE')}
todo = [j for j in jobs if j['name'] not in done]
print('jobs:', [j['name'] for j in jobs], '| already done:', sorted(done), '| to run:', [j['name'] for j in todo])
"""),
    code(r"""
# 3. Python 3.13 environment (the pipeline's pinned RDKit wheels are cp313, as on Kaggle)
if not os.path.exists('/content/py313/bin/python'):
    sh('pip install -q uv')
    sh('uv python install 3.13 && uv venv /content/py313 --python 3.13')
    PY = '/content/py313/bin/python'
    sh(f'uv pip install --python {PY} -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128', check=False)
    sh(f'uv pip install --python {PY} -q "torch>=2.6"', check=False)
    sh(f'uv pip install --python {PY} -q numpy pandas pyarrow scipy scikit-learn lightgbm numba h5py tqdm pyyaml '
       f'networkx matplotlib ipykernel papermill')
    sh(f'{PY} -m ipykernel install --user --name py313 --display-name py313')
PY = '/content/py313/bin/python'
sh(f'{PY} -c "import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())"')
"""),
    code(r"""
# 4. Inputs: /kaggle/input/<slug> for every dataset and competition a job reads (downloaded once, reused)
os.makedirs('/content/kaggle/input', exist_ok=True)
if not os.path.exists('/kaggle'):
    os.symlink('/content/kaggle', '/kaggle')
def fetch_inputs(meta):
    for src in meta.get('dataset_sources', []):
        dst = f"/kaggle/input/{src.split('/')[1]}"
        if os.path.exists(f'{dst}/.ok'): continue
        sh(f'kaggle datasets download {src} -p {dst} --unzip')
        pathlib.Path(f'{dst}/.ok').touch()
    for comp in meta.get('competition_sources', []):
        dst = f'/kaggle/input/{comp}'
        if os.path.exists(f'{dst}/.ok'): continue
        sh(f'kaggle competitions download {comp} -p {dst}')
        for z in glob.glob(f'{dst}/*.zip'):
            sh(f'cd {dst} && unzip -oq "{z}" && rm "{z}"')
        pathlib.Path(f'{dst}/.ok').touch()
"""),
    code(r"""
# 5. Run each job and upload its results after it finishes
KEEP = ('.csv', '.json', '.log', '.txt')
def collect(job_dir, dst):
    os.makedirs(dst, exist_ok=True)
    for p in glob.glob(f'{job_dir}/*') + glob.glob(f'{job_dir}/*_work/*.json'):
        if os.path.isfile(p) and p.endswith(KEEP):
            rel = os.path.relpath(p, job_dir); os.makedirs(os.path.dirname(f'{dst}/{rel}'), exist_ok=True)
            shutil.copy(p, f'{dst}/{rel}')
def upload(msg):
    json.dump({'title': 'casmi26-jc-colab-out', 'id': OUT_DS, 'licenses': [{'name': 'other'}]},
              open(f'{OUT}/dataset-metadata.json', 'w'))
    global have_out
    if have_out:
        sh(f'kaggle datasets version -p {OUT} -m "{msg}" --dir-mode zip -q')
    else:
        sh(f'kaggle datasets create -p {OUT} --dir-mode zip -q'); have_out = True

for job in todo:
    name, t0 = job['name'], time.time()
    meta = json.load(open(f"/content/jobs/{job['dir']}/kernel-metadata.json"))
    fetch_inputs(meta)
    shutil.rmtree('/kaggle/working', ignore_errors=True); os.makedirs('/kaggle/working')
    nb_in = f"/content/jobs/{job['dir']}/{meta['code_file']}"
    log = open('/kaggle/working/run.log', 'w')
    r = subprocess.run([PY, '-m', 'papermill', nb_in, '/kaggle/working/__notebook__.ipynb', '-k', 'py313',
                        '--log-output', '--cwd', '/kaggle/working'], stdout=log, stderr=subprocess.STDOUT, text=True)
    log.close()
    dst = f'{OUT}/{name}'
    shutil.rmtree(dst, ignore_errors=True)
    collect('/kaggle/working', dst)
    status = 'ok' if r.returncode == 0 else f'error {r.returncode}'
    json.dump({'status': status, 'seconds': round(time.time() - t0), 'gpu': sh('nvidia-smi --query-gpu=name --format=csv,noheader').stdout.strip()},
              open(f'{dst}/colab_status.json', 'w'))
    pathlib.Path(f'{dst}/DONE').touch()
    print(name, status, f'{(time.time()-t0)/3600:.1f} h', flush=True)
    print(open('/kaggle/working/run.log').read()[-3000:])
    upload(f'{name}: {status}')
print('all jobs finished')
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
(ROOT / "colab").mkdir(exist_ok=True)
(ROOT / "colab" / "casmi26_colab_runner.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote colab/casmi26_colab_runner.ipynb")

if args.jobs:
    jd = ROOT / "colab" / "jobs"
    shutil.rmtree(jd, ignore_errors=True)
    jd.mkdir(parents=True)
    jobs = []
    for d in args.jobs:
        src = ROOT / "kernels" / d
        shutil.copytree(src, jd / d)
        jobs.append({"name": d, "dir": d})
    (jd / "jobs.json").write_text(json.dumps(jobs, indent=1))
    (jd / "dataset-metadata.json").write_text(json.dumps(
        {"title": "casmi26-jc-colab-jobs", "id": "giantwizard/casmi26-jc-colab-jobs", "licenses": [{"name": "other"}]}))
    print("jobs:", [j["name"] for j in jobs])
