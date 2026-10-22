"""Stacked-reranker rows from the rowgen kernels' saved stages (kernels/make_rowgen.py, 'fusion' dump).

One row per (molecule, candidate structure) over the union of every channel's candidates: engine 1 (before and after the
popularity re-sort), the PubChem-merged main list, engine 2, the PubChem channel, and v465's fused list. Features are each
channel's rank and score (raw, z-scored and gap to the molecule's best), ICEBERG/GLACIER scores (z-scored within the
molecule and within the candidate's formula), popularity and molecule-level context. Label: the candidate has the
answer's metric identity (stereo-stripped, tautomer-canonical InChIKey14, as in the scorer).
usage: build_rows.py ROWGEN_DIR [ROWGEN_DIR ...] --out rows.parquet
"""
import argparse, json, os, pickle
from collections import Counter
from multiprocessing import Pool
import numpy as np, pandas as pd

def ident(smi):
    """(plain stereo-stripped InChIKey14, formula, heavy atoms, heavy-atom composition)"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog('rdApp.*')
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        Chem.RemoveStereochemistry(m)
        comp = tuple(sorted(Counter(x.GetSymbol() for x in m.GetAtoms() if x.GetAtomicNum() > 1).items()))
        return Chem.MolToInchiKey(m)[:14], CalcMolFormula(m), m.GetNumHeavyAtoms(), comp
    except Exception:
        return None


def metric_ik14(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    try:
        m = Chem.MolFromSmiles(smi)
        Chem.RemoveStereochemistry(m)
        return Chem.MolToInchiKey(rdMolStandardize.TautomerEnumerator().Canonicalize(m))[:14]
    except Exception:
        return None


def z(v):
    v = np.asarray(v, float); ok = np.isfinite(v)
    out = np.full(len(v), np.nan)
    if ok.sum() >= 2:
        out[ok] = (v[ok] - v[ok].mean()) / (v[ok].std() + 1e-9)
    elif ok.sum() == 1:
        out[ok] = 0.0
    return out


def load(d):
    R = os.path.join(d, 'rowgen')
    P = lambda n: pickle.load(open(os.path.join(R, n + '_fusion.pkl'), 'rb'))
    S = dict(base_raw=P('BASE_RAW'), base=P('BASE'), eng=P('ENG'), ice=P('ICE_SCORES'), gl=P('GL_BY_MID'),
             pop=P('POOL_POP_BASE'), pre=P('PREFUSION_LISTS'), over=P('FUSION_OVERFLOW'),
             pc=json.load(open(os.path.join(R, 'fusion_pc_lists.json'))),
             fus={str(r.molecule_id): str(r.smiles).split(';') for r in pd.read_csv(os.path.join(R, 'fusion_submission.csv')).itertuples()},
             truth={t['molecule_id']: t for t in json.load(open(os.path.join(R, 'truth.json')))} if os.path.exists(os.path.join(R, 'truth.json')) else {})
    return S


def collect_smiles(S):
    smis = set()
    for v in S['base_raw'].values(): smis.update(v[0])
    for v in S['eng'].values(): smis.update(v['smiles'])
    for v in S['pc'].values(): smis.update(v.get('pc') or [])
    for v in list(S['pre'].values()) + list(S['fus'].values()) + list(S['over'].values()): smis.update(v)
    for v in list(S['ice'].values()) + list(S['gl'].values()): smis.update((v or {}).keys())
    return {x for x in smis if isinstance(x, str) and x and x != 'CCO'}


def mol_frame(S, mid, ID):
    """Candidate feature frame of one molecule (no label), or None."""
    C = {}                                              # plain key -> feature dict

    def add(smi, **f):
        i = ID.get(smi)
        if i is None:
            return
        c = C.setdefault(i[0], dict(smiles=smi, formula=i[1], heavy=i[2], comp=i[3]))
        for k, v in f.items():
            if k not in c or (k.endswith('_rank') and v < c[k]):
                c[k] = v
    br = S['base_raw'].get(mid)
    lib_max = np.nan
    if br:
        sm, _, sc, pids, lib_max = br
        zs = z(sc); top = max(sc) if sc else np.nan
        pops = S['pop'].get(mid) or [None] * len(sm)
        for r, (s, v, zz, pid, pp) in enumerate(zip(sm, sc, zs, pids, pops), 1):
            add(s, e1_rank=r, e1_score=v, e1_z=zz, e1_gap=top - v, e1_gen=float(pid < 0), pop=np.nan if pp is None else pp)
    b = S['base'].get(mid)
    if b:
        for r, s in enumerate(b[0], 1): add(s, e1pp_rank=r)
    for r, s in enumerate(S['pre'].get(mid) or [], 1): add(s, pre_rank=r)
    e = S['eng'].get(mid)
    if e and e['smiles']:
        zb = z(e['blend']); top = max(e['blend'])
        for r, (s, bl, pv, ou, zz) in enumerate(zip(e['smiles'], e['blend'], e['pv'], e['ours'], zb), 1):
            add(s, e2_rank=r, e2_blend=bl, e2_z=zz, e2_gap=top - bl, e2_pv=pv, e2_ours=ou)
    pc = S['pc'].get(mid) or {}
    if pc.get('pc'):
        zp = z(pc['pc_fz']); top = max(pc['pc_fz'])
        for r, (s, v, zz) in enumerate(zip(pc['pc'], pc['pc_fz'], zp), 1):
            add(s, pc_rank=r, pc_S=v, pc_z=zz, pc_gap=top - v)
    for r, s in enumerate(S['fus'].get(mid) or [], 1):
        if s != 'CCO': add(s, fus_rank=r)
    for r, s in enumerate(S['over'].get(mid) or [], 26): add(s, fus_rank=r)
    for nm, src in (('ice', S['ice']), ('gl', S['gl'])):
        for s, v in ((src.get(mid) or {}).items()):
            if v is not None and np.isfinite(v): add(s, **{nm: float(v)})
    if not C:
        return None
    df = pd.DataFrame.from_dict(C, orient='index').reset_index().rename(columns={'index': 'pkey'})
    for col in ['e1_rank', 'e1pp_rank', 'pre_rank', 'e2_rank', 'pc_rank', 'fus_rank', 'e1_score', 'e1_z', 'e1_gap', 'e1_gen', 'pop',
                'e2_blend', 'e2_z', 'e2_gap', 'e2_pv', 'e2_ours', 'pc_S', 'pc_z', 'pc_gap', 'ice', 'gl']:
        if col not in df:
            df[col] = np.nan
    for nm in ('ice', 'gl'):
        df[nm + '_zm'] = z(df[nm])
        df[nm + '_zf'] = df.groupby('formula')[nm].transform(lambda v: pd.Series(z(v), index=v.index))
        df[nm + '_rf'] = df.groupby('formula')[nm].rank(ascending=False)
    df['n_src'] = df[['e1_rank', 'e2_rank', 'pc_rank']].notna().sum(1)
    df['form_n'] = df.groupby('formula').pkey.transform('size')
    df['form_best_fus'] = df.groupby('formula').fus_rank.transform('min')
    df['pop_zm'] = z(df['pop'])
    df['lib_max'] = lib_max
    df['pc_delta'] = (pc.get('fz_top', np.nan) or np.nan) - (pc.get('best_pool_fz', np.nan) or np.nan)
    df['pc_top_pop'] = pc.get('top_pop', np.nan)
    df['n_cand'] = len(df)
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+'); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    shards = [(d, load(d)) for d in a.dirs]
    smis = set()
    for _, S in shards:
        smis |= collect_smiles(S)
        smis.update(t['smiles'] for t in S['truth'].values())
    smis = sorted(smis)
    with Pool(a.workers) as p:
        ID = dict(zip(smis, p.map(ident, smis, chunksize=200)))
    print('identified', len(ID), 'SMILES', flush=True)
    # label: plain key equal to the truth's, or same heavy-atom composition and same metric identity
    tm = {}
    rows = []
    for d, S in shards:
        for mid, t in S['truth'].items():
            tid = ID.get(t['smiles'])
            if tid is None:
                continue
            t_metric = metric_ik14(t['smiles'])
            df = mol_frame(S, mid, ID)
            if df is None:
                continue
            # label
            same = df.pkey == tid[0]
            cand = (~same) & (df.comp.map(lambda c: c == tid[3]))
            lab = same.copy()
            for i in np.flatnonzero(cand.values):
                s = df.smiles.iat[i]
                if s not in tm:
                    tm[s] = metric_ik14(s)
                lab.iat[i] = tm[s] == t_metric
            df['y'] = lab.astype(int)
            df['mid'] = mid; df['shard'] = os.path.basename(os.path.normpath(d)); df['cls'] = t['cls']; df['panel'] = t['panel']
            rows.append(df.drop(columns=['comp']))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(a.out)
    g = out.groupby('mid').y.max()
    print(f'{out.mid.nunique()} molecules, {len(out):,} rows; truth among candidates: {g.mean():.3f}', flush=True)
