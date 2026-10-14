"""Single-step structural edits (common natural-product biotransformations) of a molecule, for analog generation.

edit_products(smiles) -> {canonical smiles: edit name}. Edits: +/-CH3 (methylation at C/N/O/S-H, demethylation of
terminal CH3), +/-OH (hydroxylation of C-H, removal of a terminal OH), CH2 insertion/removal in acyclic C-C single
bonds (homologues), +OCH3 on aromatic C-H, +/-H2 (C=C <-> C-C, outside aromatic rings), -CO2 (decarboxylation).
"""
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


def _ok(rw):
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(Chem.RemoveHs(m))
    except Exception:
        return None


def edit_products(smiles, max_out=400):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {}
    out = {}

    def add(s, name):
        if s and s not in out and len(out) < max_out:
            out[s] = name
    for a in m.GetAtoms():
        i, sym, nh = a.GetIdx(), a.GetSymbol(), a.GetTotalNumHs()
        if nh > 0 and sym in ('C', 'N', 'O', 'S'):
            rw = Chem.RWMol(m); j = rw.AddAtom(Chem.Atom(6)); rw.AddBond(i, j, Chem.BondType.SINGLE)
            rw.GetAtomWithIdx(i).SetNoImplicit(False); rw.GetAtomWithIdx(i).SetNumExplicitHs(0)
            add(_ok(rw), '+CH2 (methyl)')
        if nh > 0 and sym == 'C':
            rw = Chem.RWMol(m); j = rw.AddAtom(Chem.Atom(8)); rw.AddBond(i, j, Chem.BondType.SINGLE)
            rw.GetAtomWithIdx(i).SetNumExplicitHs(0)
            add(_ok(rw), '+O (hydroxyl)')
            if a.GetIsAromatic():
                rw = Chem.RWMol(m); o = rw.AddAtom(Chem.Atom(8)); c = rw.AddAtom(Chem.Atom(6))
                rw.AddBond(i, o, Chem.BondType.SINGLE); rw.AddBond(o, c, Chem.BondType.SINGLE)
                rw.GetAtomWithIdx(i).SetNumExplicitHs(0)
                add(_ok(rw), '+CH2O (methoxy)')
        if a.GetDegree() == 1 and sym == 'C' and nh == 3:
            rw = Chem.RWMol(m); rw.RemoveAtom(i); add(_ok(rw), '-CH2 (demethyl)')
        if a.GetDegree() == 1 and sym == 'O' and nh == 1:
            nb = a.GetNeighbors()[0]
            if nb.GetSymbol() == 'C' and not any(b.GetBondType() == Chem.BondType.DOUBLE for b in nb.GetBonds()
                                                 if b.GetOtherAtomIdx(nb.GetIdx()) != i and
                                                 b.GetOtherAtom(nb).GetSymbol() == 'O'):
                rw = Chem.RWMol(m); rw.RemoveAtom(i); add(_ok(rw), '-O (dehydroxyl)')
    for b in m.GetBonds():
        x, y = b.GetBeginAtom(), b.GetEndAtom()
        if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing() and x.GetSymbol() == 'C' and y.GetSymbol() == 'C':
            rw = Chem.RWMol(m); rw.RemoveBond(x.GetIdx(), y.GetIdx()); c = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(x.GetIdx(), c, Chem.BondType.SINGLE); rw.AddBond(c, y.GetIdx(), Chem.BondType.SINGLE)
            add(_ok(rw), '+CH2 (homolog)')
        if (b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic() and x.GetSymbol() == 'C' and y.GetSymbol() == 'C'):
            rw = Chem.RWMol(m); rw.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx()).SetBondType(Chem.BondType.SINGLE)
            add(_ok(rw), '+H2 (reduce)')
        if (b.GetBondType() == Chem.BondType.SINGLE and not b.GetIsAromatic() and x.GetSymbol() == 'C' and y.GetSymbol() == 'C'
                and x.GetTotalNumHs() > 0 and y.GetTotalNumHs() > 0):
            rw = Chem.RWMol(m); rw.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
            add(_ok(rw), '-H2 (desaturate)')
    # CH2 removal from acyclic chains: C(H2) with exactly two carbon neighbours, both single bonds
    for a in m.GetAtoms():
        if a.GetSymbol() == 'C' and a.GetTotalNumHs() == 2 and a.GetDegree() == 2 and not a.IsInRing():
            n1, n2 = [n.GetIdx() for n in a.GetNeighbors()]
            rw = Chem.RWMol(m); rw.RemoveAtom(a.GetIdx())
            n1b, n2b = (n1 if n1 < a.GetIdx() else n1 - 1), (n2 if n2 < a.GetIdx() else n2 - 1)
            rw.AddBond(n1b, n2b, Chem.BondType.SINGLE); add(_ok(rw), '-CH2 (homolog)')
    patt = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    for (c, o1, o2) in m.GetSubstructMatches(patt):
        rw = Chem.RWMol(m)
        for idx in sorted((c, o1, o2), reverse=True):
            rw.RemoveAtom(idx)
        add(_ok(rw), '-CO2 (decarboxyl)')
    return out


# reaction-based edits common in plant natural products (stereo-free: the metric strips stereochemistry)
from rdkit.Chem import AllChem as _AC
_RXN = {
    '+hexose (O-glycoside)': '[OX2H1:1][#6:2]>>[#6:2][O:1]C1OC(CO)C(O)C(O)C1O',
    '-hexose (aglycone)': '[#6:2][O:1]C1OC(CO)C(O)C(O)C1O>>[#6:2][OH1:1]',
    '+deoxyhexose': '[OX2H1:1][#6:2]>>[#6:2][O:1]C1OC(C)C(O)C(O)C1O',
    '-deoxyhexose': '[#6:2][O:1]C1OC(C)C(O)C(O)C1O>>[#6:2][OH1:1]',
    '+pentose': '[OX2H1:1][#6:2]>>[#6:2][O:1]C1OCC(O)C(O)C1O',
    '-pentose': '[#6:2][O:1]C1OCC(O)C(O)C1O>>[#6:2][OH1:1]',
    '+acetyl (O)': '[OX2H1:1][#6:2]>>[#6:2][O:1]C(C)=O',
    '-acetyl (O)': '[#6:2][O:1]C(=O)[CH3]>>[#6:2][OH1:1]',
    '+COOH (aromatic)': '[cH:1]>>[c:1]C(=O)O',
    '+glucuronide': '[OX2H1:1][#6:2]>>[#6:2][O:1]C1OC(C(=O)O)C(O)C(O)C1O',
    '+CH2O (hydroxymethyl)': '[CX4H1,CX4H2,CX4H3,cH:1]>>[*:1]CO',
    '+CH2O (aliphatic methoxy)': '[CX4H1,CX4H2:1]>>[*:1]OC',
    '+C2H4 (O-ethyl)': '[OX2H1:1][#6:2]>>[#6:2][O:1]CC',
    '+C2H4 (N-ethyl)': '[NX3;H1,H2;!$(NC=O):1]>>[N:1]CC',
}
_RXNS = {k: _AC.ReactionFromSmarts(v) for k, v in _RXN.items()}


def reaction_products(smiles, max_out=200):
    m = Chem.MolFromSmiles(smiles)
    out = {}
    if m is None:
        return out
    for name, rxn in _RXNS.items():
        try:
            for prods in rxn.RunReactants((m,)):
                p = prods[0]
                try:
                    Chem.SanitizeMol(p)
                    s = Chem.MolToSmiles(p)
                    if s not in out and len(out) < max_out:
                        out[s] = name
                except Exception:
                    pass
        except Exception:
            pass
    return out


_edit_products_basic = edit_products


def edit_products(smiles, max_out=400, reactions=True):
    out = _edit_products_basic(smiles, max_out)
    m0 = Chem.MolFromSmiles(smiles)
    if m0 is not None:
        out.setdefault(Chem.MolToSmiles(m0), '0 (same structure)')
    if reactions:
        for s, n in reaction_products(smiles).items():
            out.setdefault(s, n)
    return out


def positional_isomers(smiles, max_out=300):
    """Same-formula variants: detach one OH / CH3 / OCH3 substituent and re-attach it at every other C-H position."""
    m = Chem.MolFromSmiles(smiles)
    out = {}
    if m is None:
        return out
    groups = []
    for a in m.GetAtoms():
        if a.GetDegree() != 1 or a.GetSymbol() not in ('O', 'C'):
            continue
        nb_ = a.GetNeighbors()[0]
        if a.GetSymbol() == 'O' and a.GetTotalNumHs() == 1 and nb_.GetSymbol() == 'C' and \
                not any(b.GetBondType() == Chem.BondType.DOUBLE for b in nb_.GetBonds()):
            groups.append(('OH', [a.GetIdx()], nb_.GetIdx(), 'O'))
        elif a.GetSymbol() == 'C' and a.GetTotalNumHs() == 3:
            if nb_.GetSymbol() == 'O' and nb_.GetDegree() == 2:
                other = [x for x in nb_.GetNeighbors() if x.GetIdx() != a.GetIdx()][0]
                if other.GetSymbol() == 'C':
                    groups.append(('OCH3', [a.GetIdx(), nb_.GetIdx()], other.GetIdx(), 'OC'))
            elif nb_.GetSymbol() == 'C':
                groups.append(('CH3', [a.GetIdx()], nb_.GetIdx(), 'C'))
    for name, atoms, anchor, frag in groups:
        rw = Chem.RWMol(m)
        for i in sorted(atoms, reverse=True):
            rw.RemoveAtom(i)
        try:
            core = rw.GetMol(); Chem.SanitizeMol(core)
        except Exception:
            continue
        for b in core.GetAtoms():
            if b.GetSymbol() != 'C' or b.GetTotalNumHs() == 0:
                continue
            rw2 = Chem.RWMol(core)
            prev = b.GetIdx()
            for ch in frag:
                j = rw2.AddAtom(Chem.Atom(8 if ch == 'O' else 6)); rw2.AddBond(prev, j, Chem.BondType.SINGLE); prev = j
            rw2.GetAtomWithIdx(b.GetIdx()).SetNumExplicitHs(0)
            s = _ok(rw2)
            if s and s != smiles and s not in out and len(out) < max_out:
                out[s] = f'move {name}'
    return out
