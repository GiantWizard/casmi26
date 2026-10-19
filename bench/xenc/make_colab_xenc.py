"""Write colab/casmi26_xenc.ipynb: train the cross-encoder isomer reranker on an L4. Code and the evaluation-half exclusion
list are read from Drive (MyDrive/casmi26/xenc/, synced from the Mac), checkpoints and the live log are written there,
so the Mac sees progress through Drive-for-desktop. Resumes from the last partial checkpoint after a disconnect.
usage: make_colab_xenc.py [HOURS] [BS]"""
import json, sys
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
import pandas as pd
TS = open('/Users/b/casmi26-repo/bench/xenc/train_xenc.py').read()
EVAL = sorted(int(x) for x in pd.read_parquet('/Users/b/casmi26/rk/np1_evalhalf.parquet').sid)
SETUP = ''.join(json.load(open('/Users/b/casmi26-repo/colab/casmi26_fpsim_train.ipynb'))['cells'][1]['source'])
SETUP = SETUP.replace("OUT = '/content/drive/MyDrive/casmi26/fpsim_train'", "OUT = '/content/drive/MyDrive/casmi26/xenc'")
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
md(f"""# CASMI26: cross-encoder isomer reranker (~{HOURS:g} h)

Trains `XEnc` (spectrum tokens from the frozen fold-safe FPNet + candidate molecular graph) to pick the true structure
among its same-formula isomers, on spectra the FPNet never saw. Code, settings and checkpoints live in
`MyDrive/casmi26/xenc/`. **Run all, keep the tab open.** If it disconnects, Run all again: it resumes.
Needs: L4 GPU runtime, the `KAGGLE_API_TOKEN` secret, the Google Drive prompt.
""")
code(SETUP)
code("# 2. embedded training script + held-back evaluation sids (nothing needs to sync up to Drive)\nimport pandas as pd\nTS = '/content/train_xenc.py'\nopen(TS, 'w').write(" + repr(TS) + ")\nEX = '/content/np1_evalhalf.parquet'\npd.DataFrame({'sid': " + repr(EVAL) + "}).to_parquet(EX)\nprint('wrote', TS, 'and', EX)\n")
code(f"""# 3. train in the background; the log is on Drive (train_xenc.log)
import time
HOURS, BS, TAG = {HOURS}, {BS}, 'xenc'
part = f'{{OUT}}/{{TAG}}_partial.pt'
init = f'--init {{part}}' if os.path.exists(part) else ''
cmd = (f'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {{sys.executable}} {{TS}} --code {{CODE}} --run {{R}}/run --fp {{CK0}} {{init}} --device cuda --bs {{BS}} '
       f'--steps 60000 --hours {{HOURS}} --exclude_sids {{EX}} --profile --save_every 0.5 --tag {{TAG}} '
       f'--outdir {{OUT}} >> {{OUT}}/train_{{TAG}}.log 2>&1')
print(cmd)
proc = subprocess.Popen(cmd, shell=True)
while proc.poll() is None:
    time.sleep(300)
    print(open(f'{{OUT}}/train_{{TAG}}.log').read().strip().splitlines()[-1], flush=True)
print(open(f'{{OUT}}/train_{{TAG}}.log').read()[-2000:])
print('final checkpoint:', os.path.exists(f'{{OUT}}/{{TAG}}.pt'))
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open('/Users/b/casmi26-repo/colab/casmi26_xenc.ipynb', 'w'), indent=1)
print('wrote colab/casmi26_xenc.ipynb')
