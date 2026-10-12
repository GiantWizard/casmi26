"""Is the truth in the pipeline's PubChem tier at all? (ahmedberatozer/casmi26-pubchem-tier layout:
pc_mass.npy sorted neutral masses, pc_off.npy byte offsets, pc_smiles.npy one uint8 blob.)

Per truth: ±ppm mass window -> same molecular formula -> InChIKey14 (stereo stripped) match.
    python pubchem_coverage.py --pc DIR --solution solution.csv --out cov.csv
"""
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")
ap = argparse.ArgumentParser()
ap.add_argument("--pc", required=True)
ap.add_argument("--solution", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ppm", type=float, default=10.0)
a = ap.parse_args()

MASS = np.load(f"{a.pc}/pc_mass.npy", mmap_mode="r")
OFF = np.load(f"{a.pc}/pc_off.npy", mmap_mode="r")
BLOB = np.load(f"{a.pc}/pc_smiles.npy", mmap_mode="r")


def ik14(m):
    m = Chem.Mol(m)
    Chem.RemoveStereochemistry(m)
    return Chem.MolToInchiKey(m)[:14]


def check(args):
    mid, smi = args
    t = Chem.MolFromSmiles(smi)
    if t is None:
        return mid, np.nan, 0, 0, False
    mass, form, key = ExactMolWt(t), CalcMolFormula(t), ik14(t)
    tol = mass * a.ppm * 1e-6
    lo, hi = np.searchsorted(MASS, [mass - tol, mass + tol])
    n_form, hit = 0, False
    for i in range(lo, hi):
        s = bytes(BLOB[OFF[i]:OFF[i + 1]]).decode()
        m = Chem.MolFromSmiles(s)
        if m is None or CalcMolFormula(m) != form:
            continue
        n_form += 1
        if ik14(m) == key:
            hit = True
            break
    return mid, mass, int(hi - lo), n_form, hit


sol = pd.read_csv(a.solution)
todo = list(zip(sol.molecule_id, sol.smiles))
with mp.Pool() as pool:
    res = pool.map(check, todo, chunksize=4)
df = pd.DataFrame(res, columns=["molecule_id", "mass", "n_window", "n_formula_scanned", "in_pubchem"]).merge(
    sol[["molecule_id", "cls"]], on="molecule_id")
df.to_csv(a.out, index=False)
print(df.groupby("cls").in_pubchem.agg(["mean", "sum", "size"]))
print("window sizes:", df.n_window.describe().round(0).to_dict())
