# Paste into a new cell of the casmi26_e2fp Colab notebook (after cell 1 has run, so Drive and the Kaggle token are set up).
# Saves the two retrained engine-2 FPNets without optimizer state and uploads them as the private Kaggle dataset
# giantwizard/casmi26-e2fp-f0, which the rowgen kernels read.
import os, json, torch, subprocess
D = '/content/e2fp_f0'; os.makedirs(D, exist_ok=True)
for f in ['fp_single_s2_f0.pt', 'fp_merged_m1_f0.pt']:
    ck = torch.load(f'/content/drive/MyDrive/casmi26/e2fp/{f}', map_location='cpu', weights_only=False)
    torch.save({k: ck[k] for k in ('model', 'step', 'nbits', 'd', 'layers')}, f'{D}/{f}')
    print(f, 'step', ck['step'], 'nbits', ck['nbits'], os.path.getsize(f'{D}/{f}') // 2**20, 'MB')
json.dump({'title': 'casmi26-e2fp-f0', 'id': 'giantwizard/casmi26-e2fp-f0', 'licenses': [{'name': 'CC-BY-NC-SA-4.0'}]},
          open(f'{D}/dataset-metadata.json', 'w'))
r = subprocess.run(f'kaggle datasets create -p {D}', shell=True, text=True, capture_output=True)
print(r.stdout, r.stderr)
