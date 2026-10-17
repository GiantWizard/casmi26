"""Write colab/casmi26_fplong.ipynb: an ~11 h fold-safe FPNet run (fold0 init, real + simulated spectra, hardest
same-window negatives), publishing a checkpoint to the private Kaggle dataset giantwizard/casmi26-jc-fplong every hour so
the NP harness (bench/engine/refp.py + compare_banks.py) can score it while it trains. train_sim.py is embedded, so no
upload is needed first. Safe to re-run after a disconnect: training resumes from the last partial checkpoint on Drive.
usage: make_colab_long.py [HOURS] [HARD_K]
"""
import json, sys
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 11.0
HARD_K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
TS = open('/Users/b/casmi26-repo/bench/fpsim/train_sim.py').read()
SETUP = json.load(open('/Users/b/casmi26-repo/colab/casmi26_fpsim_train.ipynb'))['cells'][1]['source']
SETUP = ''.join(SETUP).replace("OUT = '/content/drive/MyDrive/casmi26/fpsim_train'", "OUT = '/content/drive/MyDrive/casmi26/fplong'")
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
md(f"""# CASMI26: long fingerprint-model run (~{HOURS:g} h)

Fine-tunes the fold-safe FPNet on real + simulated spectra with the hardest same-formula-window negatives
({HARD_K} of 63 per query). Every hour the latest checkpoint goes to the private Kaggle dataset
`giantwizard/casmi26-jc-fplong` so it can be scored while training runs.

**Run all, keep this tab open and the computer awake.** If it disconnects, Run all again: it resumes from the last
hourly checkpoint on Drive (`MyDrive/casmi26/fplong/`).
Needs: GPU runtime (L4), the `KAGGLE_API_TOKEN` secret, the Google Drive prompt.
""")
code(SETUP)
code("# 2. training script (embedded copy of bench/fpsim/train_sim.py)\nTS = '/content/train_sim.py'\nopen(TS, 'w').write(" + repr(TS) + ")\nprint('wrote', TS)\n")
code(f"""# 3. start training in the background (resumes from the last partial checkpoint if there is one)
import time
HOURS, HARD_K, TAG = {HOURS}, {HARD_K}, 'fplong'
SIM = f'{{SIMD}}/mh,{{SIMD}}/mna'
part = f'{{OUT}}/{{TAG}}_partial.pt'
init = part if os.path.exists(part) else CK0
done_h = 0.0
if os.path.exists(f'{{OUT}}/hours_done.txt'):
    done_h = float(open(f'{{OUT}}/hours_done.txt').read() or 0)
left = max(0.5, HOURS - done_h)
print('init from', init, '| hours left', round(left, 2))
cmd = (f'{{sys.executable}} {{TS}} --code {{CODE}} --run {{R}}/run --init {{init}} --device cuda --bs 256 --lr 1e-4 '
       f'--steps 65000 --hours {{left}} --sim {{SIM}} --sim_p 0.5 --hard_k {{HARD_K}} --save_every 1 '
       f'--tag {{TAG}} --outdir {{OUT}} >> {{OUT}}/train_{{TAG}}.log 2>&1')
proc = subprocess.Popen(cmd, shell=True)
T_START = time.time()
print('training started, pid', proc.pid)
""")
code("""# 4. every hour: publish the latest checkpoint + log to the private Kaggle dataset; ends when training ends
import shutil
P = '/content/pub_long'; os.makedirs(P, exist_ok=True)
json.dump({'title': 'casmi26-jc-fplong', 'id': 'giantwizard/casmi26-jc-fplong', 'licenses': [{'name': 'other'}]},
          open(P + '/dataset-metadata.json', 'w'))
def publish(note):
    for f in (f'{OUT}/{TAG}_partial.pt', f'{OUT}/{TAG}.pt', f'{OUT}/train_{TAG}.log'):
        if os.path.exists(f):
            shutil.copy(f, P)
    if sh(f'kaggle datasets create -p {P} -q', check=False).returncode:
        sh(f'kaggle datasets version -p {P} -m "{note}" -q', check=False)
last = 0
while proc.poll() is None:
    time.sleep(60)
    open(f'{OUT}/hours_done.txt', 'w').write(str(done_h + (time.time() - T_START) / 3600))
    if time.time() - last > 3600 and os.path.exists(f'{OUT}/{TAG}_partial.pt'):
        publish(f'partial {(time.time() - T_START) / 3600:.1f} h'); last = time.time()
        print(open(f'{OUT}/train_{TAG}.log').read()[-400:], flush=True)
publish('final')
print(open(f'{OUT}/train_{TAG}.log').read()[-1500:])
print('final checkpoint:', os.path.exists(f'{OUT}/{TAG}.pt'))
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open('/Users/b/casmi26-repo/colab/casmi26_fplong.ipynb', 'w'), indent=1)
print('wrote colab/casmi26_fplong.ipynb')
