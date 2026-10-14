import sys, pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
pat=Chem.MolFromSmarts('[NX3;!$(N-C=O)]-[CX4]-C(=O)[OX2H1,OX1-]')
amide=Chem.MolFromSmarts('C(=O)N[CX4]C(=O)[OX2H1]')
for f in sys.argv[1:]:
    s=pd.read_csv(f)
    def tag(x):
        m=Chem.MolFromSmiles(x)
        if m is None: return 'bad'
        return 'aa-conj' if (m.HasSubstructMatch(pat) or m.HasSubstructMatch(amide)) and m.GetNumHeavyAtoms()>12 else 'other'
    s['tag']=s.smiles.map(tag)
    print(f.split('/')[-2]); print(s.groupby(['cls','tag']).size().unstack(fill_value=0))
