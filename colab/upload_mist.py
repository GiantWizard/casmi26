# Run as one Colab cell (any runtime, no GPU; needs the KAGGLE_API_TOKEN secret).
# Downloads the MIST MassSpecGym checkpoint (Zenodo 10.5281/zenodo.23303289, MIT) and builds a wheel of the MIST code
# (github.com/coleygroup/MIST, MIT), then uploads both as the private Kaggle dataset giantwizard/casmi26-mist.
import os, json, hashlib, subprocess, urllib.request
from google.colab import userdata
os.environ['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
sh = lambda c: print(subprocess.run(c, shell=True, text=True, capture_output=True).stdout[-1500:])
sh('pip install -q -U kaggle build')
D = '/content/mist_ds'; os.makedirs(D, exist_ok=True)
ck = f'{D}/mist_msg_noaug.ckpt'
urllib.request.urlretrieve('https://zenodo.org/records/23303289/files/mist_msg_noaug.ckpt', ck)
h = hashlib.sha256(open(ck, 'rb').read()).hexdigest()
assert h == '6f8df582df36aa4a984f120efee14ee0a62f3a7723caa770b856a6e704236727', h
sh('rm -rf /content/MIST && git clone -q https://github.com/coleygroup/MIST.git /content/MIST && git -C /content/MIST rev-parse HEAD')
sh(f'cd /content/MIST && pip wheel -q --no-deps -w {D} . && cp LICENSE.md NOTICE.md {D}/')
print(os.listdir(D))
json.dump({'title': 'casmi26-mist', 'id': 'giantwizard/casmi26-mist', 'licenses': [{'name': 'MIT'}]}, open(f'{D}/dataset-metadata.json', 'w'))
sh(f'kaggle datasets create -p {D} 2>&1')
