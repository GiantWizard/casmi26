"""LB experiment variants of kernels/jc-v7c-e2r2 (copy of the 0.443 public notebook), one parameter family each.

  python kernels/make_v7c_variants.py popx0.5   -> POP_LAM 0.125, POOLPOP_MU 0.075
  python kernels/make_v7c_variants.py popx2     -> POP_LAM 0.5,   POOLPOP_MU 0.3
  python kernels/make_v7c_variants.py simbank   -> engine 1's FP bank = full1_sim.pt (fpnet_full1 fine-tuned on real +
                                                   GLACIER-simulated pool spectra, colab/casmi26_fpsim_bank.ipynb)
Each writes kernels/jc-v7c-<tag>/ ready for `kaggle kernels push`. Purpose: the popularity prior is the knob most
likely to be mis-set for the hidden test (class-2 answers have no public spectra), and no offline set built from
library molecules can measure it.
"""
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
tag = sys.argv[1]
src = ROOT / "kernels/jc-v7c-e2r2/casmi26-jc-v7c-e2r2.ipynb"
nb = json.loads(src.read_text())
extra_ds = []
FULL1 = "_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)   # v4m"
SIMBANK = ("_FULL = glob.glob('/kaggle/input/**/casmi26-jc-fpsim-bank/**/full1_sim.pt', recursive=True)   # jc simbank\n"
           "assert _FULL, 'simulation-trained bank missing: attach giantwizard/casmi26-jc-fpsim-bank'")
n_hit, want = 0, 2
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if tag == 'simbank':
        s2 = s.replace(FULL1, SIMBANK); want = 1
    else:
        mult = {'popx0.5': 0.5, 'popx2': 2.0}[tag]
        s2 = re.sub(r"^POP_LAM, POP_UNION = 0\.25, 200", f"POP_LAM, POP_UNION = {0.25 * mult:g}, 200", s, flags=re.M)
        s2 = re.sub(r"^POOLPOP_MU = 0\.15", f"POOLPOP_MU = {0.15 * mult:g}", s2, flags=re.M)
    if s2 != s:
        n_hit += 1
        c["source"] = s2
assert n_hit == want, n_hit
if tag == 'simbank':
    note = "engine 1's FP bank = fpnet_full1 fine-tuned on real + simulated spectra"
    extra_ds = ["giantwizard/casmi26-jc-fpsim-bank"]
else:
    note = f"both popularity weights x{mult:g}"
nb["cells"][0]["source"] = f"# jc-v7c-{tag}: jc-v7c-e2r2 with {note} (LB experiment)\n\n" + "".join(nb["cells"][0]["source"])
slug = f"casmi26-jc-v7c-{tag.replace('.', 'p')}"
out = ROOT / "kernels" / slug.replace("casmi26-", "")
out.mkdir(parents=True, exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
meta = json.loads((ROOT / "kernels/jc-v7c-e2r2/kernel-metadata.json").read_text())
meta.update(id=f"giantwizard/{slug}", title=slug, code_file=f"{slug}.ipynb",
            dataset_sources=meta["dataset_sources"] + extra_ds)
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", out)
