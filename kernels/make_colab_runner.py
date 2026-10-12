"""Generate colab/casmi26_colab_runner.ipynb and the Kaggle jobs dataset it reads.

The runner offloads GPU jobs from Kaggle's weekly quota to Colab:
  1. reads jobs from the private Kaggle dataset giantwizard/casmi26-jc-colab-jobs
     (each job = a Kaggle notebook + its kernel-metadata.json, as generated in kernels/),
  2. rebuilds /kaggle/input from the job's dataset_sources + competition_sources,
  3. runs the notebook as a plain script on a Python 3.13 venv (the pinned RDKit wheels are cp313),
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

**Several sessions at once:** open this notebook in one tab per session, and in the first code cell set
`N_SESSIONS` to the number of tabs and `SESSION` to 0, 1, 2, … in each tab. Each tab then runs only its own share
of the jobs.
"""),
    code(r"""
# 0. Parallel sessions: one tab per session, each with its own SESSION number (0, 1, 2, ...)
N_SESSIONS = 1
SESSION = 0
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
OUT = '/content/out'
shutil.rmtree(OUT, ignore_errors=True)
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
os.makedirs(OUT, exist_ok=True)
have_out = sh(f'kaggle datasets download {OUT_DS} -p {OUT} --unzip', check=False).returncode == 0
for p in [p for p in os.listdir(OUT) if '__' not in p and p != 'dataset-metadata.json']:
    pp = f'{OUT}/{p}'                     # files from the first, folder-less upload format
    shutil.rmtree(pp) if os.path.isdir(pp) else os.remove(pp)
done = {p[:-len('__DONE')] for p in os.listdir(OUT) if p.endswith('__DONE')}
todo = [j for i, j in enumerate(jobs) if j['name'] not in done and i % N_SESSIONS == SESSION]
print('jobs:', [j['name'] for j in jobs], '| already done:', sorted(done), '| to run:', [j['name'] for j in todo])
"""),
    code(r"""
# 3. Python 3.13 environment (the pipeline's pinned RDKit wheels are cp313, as on Kaggle)
if not os.path.exists('/content/py313/bin/python'):
    sh('pip install -q uv')
    sh('uv python install 3.13 && uv venv --seed /content/py313 --python 3.13')
    PY = '/content/py313/bin/python'
    sh(f'uv pip install --python {PY} -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128', check=False)
    sh(f'uv pip install --python {PY} -q "torch>=2.6"', check=False)
    sh(f'uv pip install --python {PY} -q numpy pandas pyarrow scipy scikit-learn lightgbm numba h5py tqdm pyyaml '
       f'networkx matplotlib ipykernel papermill')
    sh(f'{PY} -m ipykernel install --user --name py313 --display-name py313')
PY = '/content/py313/bin/python'
# the pipeline pip-installs its pinned RDKit wheels into this interpreter, so it needs pip
sh(f'uv pip install --python {PY} -q pip')
sh(f'{PY} -m pip --version')
sh(f'{PY} -c "import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())"')
"""),
    code(r"""
# 4. Inputs under /content/kaggle (Colab ships its own read-only /kaggle/input, so the pipeline's /kaggle/ paths are
#    rewritten to /content/kaggle/ in the job notebook and in the downloaded dataset code)
KROOT = '/content/kaggle'
os.makedirs(f'{KROOT}/input', exist_ok=True)
def repath_tree(d):
    sh(f"grep -rlI --include='*.py' --include='*.sh' --include='*.json' '/kaggle/' '{d}' | xargs -r sed -i 's#\\([^t]\\)/kaggle/#\\1/content/kaggle/#g; s#^/kaggle/#/content/kaggle/#'", check=False)
def fetch_inputs(meta):
    for src in meta.get('dataset_sources', []):
        dst = f"{KROOT}/input/{src.split('/')[1]}"
        if os.path.exists(f'{dst}/.ok'): continue
        sh(f'kaggle datasets download {src} -p {dst} --unzip')
        repath_tree(dst)
        pathlib.Path(f'{dst}/.ok').touch()
    for comp in meta.get('competition_sources', []):
        dst = f'{KROOT}/input/{comp}'
        if os.path.exists(f'{dst}/.ok'): continue
        sh(f'kaggle competitions download {comp} -p {dst}')
        for z in glob.glob(f'{dst}/*.zip'):
            sh(f'cd {dst} && unzip -oq "{z}" && rm "{z}"')
        pathlib.Path(f'{dst}/.ok').touch()
def notebook_to_script(src, dst):
    # Run jobs as plain Python on the 3.13 venv: Colab's global IPython config forces its own kernel class,
    # which a separate interpreter can't load. Paths are rewritten from /kaggle/ to /content/kaggle/.
    nb = json.load(open(src))
    parts = ['import builtins\n'
             'def display(*a, **k):\n    print(*a)\n'
             'builtins.display = display\n']
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        t = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        parts.append(f'print("=== cell {i} ===", flush=True)\n' + t.replace('/kaggle/', '/content/kaggle/') + '\n')
    open(dst, 'w').write('\n'.join(parts))
"""),
    code(r"""
# 5. Run each job and upload its results after it finishes (files are flat, prefixed "<job>__")
KEEP = ('.csv', '.json', '.log', '.txt')
def collect(job_dir, name):
    for p in glob.glob(f'{job_dir}/*') + glob.glob(f'{job_dir}/*_work/*.json'):
        if os.path.isfile(p) and p.endswith(KEEP):
            rel = os.path.relpath(p, job_dir).replace('/', '__')
            shutil.copy(p, f'{OUT}/{name}__{rel}')
def upload(msg):
    # merge files another session uploaded meanwhile, then publish a new version
    tmp = f'/content/out_sync_{SESSION}'
    shutil.rmtree(tmp, ignore_errors=True)
    if sh(f'kaggle datasets download {OUT_DS} -p {tmp} --unzip', check=False).returncode == 0:
        for p in glob.glob(f'{tmp}/*'):
            if os.path.isfile(p) and not os.path.exists(f'{OUT}/{os.path.basename(p)}'):
                shutil.copy(p, OUT)
    json.dump({'title': 'casmi26-jc-colab-out', 'id': OUT_DS, 'licenses': [{'name': 'other'}]},
              open(f'{OUT}/dataset-metadata.json', 'w'))
    global have_out
    if have_out:
        sh(f'kaggle datasets version -p {OUT} -m "{msg}" -q')
    else:
        sh(f'kaggle datasets create -p {OUT} -q'); have_out = True

for job in todo:
    name, t0 = job['name'], time.time()
    meta = json.load(open(f"/content/jobs/{job['dir']}/kernel-metadata.json"))
    fetch_inputs(meta)
    WK = f'{KROOT}/working'
    shutil.rmtree(WK, ignore_errors=True); os.makedirs(WK)
    script = f'{WK}/job.py'
    notebook_to_script(f"/content/jobs/{job['dir']}/{meta['code_file']}", script)
    with open(f'{WK}/run.log', 'w') as log:
        r = subprocess.run([PY, '-u', script], cwd=WK, stdout=log, stderr=subprocess.STDOUT, text=True)
    for p in glob.glob(f'{OUT}/{name}__*'):
        os.remove(p)                      # a re-run replaces this job's old files
    collect(WK, name)
    status = 'ok' if r.returncode == 0 else f'error {r.returncode}'
    json.dump({'status': status, 'seconds': round(time.time() - t0),
               'gpu': sh('nvidia-smi --query-gpu=name --format=csv,noheader').stdout.strip()},
              open(f'{OUT}/{name}__colab_status.json', 'w'))
    if status == 'ok':
        pathlib.Path(f'{OUT}/{name}__DONE').touch()
    print(name, status, f'{(time.time()-t0)/3600:.1f} h', flush=True)
    print(open(f'{WK}/run.log').read()[-3000:])
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
