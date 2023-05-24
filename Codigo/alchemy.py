import numpy as np

from collections import defaultdict
from dgl import backend as F


# Adapted from https://lifesci.dgl.ai/_modules/dgllife/data/alchemy.html#TencentAlchemyDataset
def alchemy_nodes(mol):
    atom_feats_dict = defaultdict(list)
    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        atom = mol.GetAtomWithIdx(u)
        atom_type = atom.GetAtomicNum()
        atom_feats_dict['h'].append(atom_type)

    atom_feats_dict['h'] = F.tensor(np.array(
        atom_feats_dict['h']).astype(np.int64))

    return atom_feats_dict

def alchemy_edges(mol, self_loop=False):
    bond_feats_dict = defaultdict(list)

    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1
    geom = mol_conformers[0].GetPositions()

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        for v in range(num_atoms):
            if u == v and not self_loop:
                continue
            bond_feats_dict['e'].append(
                np.linalg.norm(geom[u] - geom[v]))
    bond_feats_dict['e'] = F.tensor(
        np.array(bond_feats_dict['e']).astype(np.float32)).reshape(-1 , 1)
    return bond_feats_dict

