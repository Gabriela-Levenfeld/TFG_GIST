from dgllife.utils import CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer, CanonicalBondFeaturizer, AttentiveFPBondFeaturizer, PretrainAtomFeaturizer

import torch
import numpy as np
import dgl.backend as F
from rdkit import Chem
from collections import defaultdict
from sklearn.preprocessing import FunctionTransformer, RobustScaler


def get_atom_featurizer(atom_featurizer):
    if atom_featurizer == 'canonical':
       return CanonicalAtomFeaturizer()
    elif atom_featurizer == 'attentive_featurizer':
        return AttentiveFPAtomFeaturizer()
    elif atom_featurizer == 'pretrain':
        return PretrainAtomFeaturizer()
    elif atom_featurizer == 'alchemy':
        return alchemy_nodes
    else:
        raise ValueError('Invalid atom featurizer')


def get_bond_featurizer(bond_featurizer, self_loop):
    if bond_featurizer == 'canonical':
       return CanonicalBondFeaturizer(self_loop=self_loop)
    elif bond_featurizer == 'attentive_featurizer':
        return AttentiveFPBondFeaturizer(self_loop=self_loop)
    elif bond_featurizer == 'pretrain':
        return GINPretrainBondFeaturizer(self_loop=self_loop)
    elif bond_featurizer == 'alchemy':
        return alchemy_edges
    else:
        raise ValueError('Invalid bond featurizer')


def get_transformer(transformer):
    if transformer == 'none':
        return FunctionTransformer()
    elif transformer == 'robust':
        return RobustScaler()
    else:
        raise ValueError('Invalid transformer')


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


# Class PretrainBondFeaturizer adapted for this specific dataset
class GINPretrainBondFeaturizer(object):
    def __init__(self, bond_types=None, bond_direction_types=None, self_loop=True):
        if bond_types is None:
            bond_types = [
                Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
            ]
        self._bond_types = bond_types

        if bond_direction_types is None:
            bond_direction_types = [
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT,
                # Missing value
                Chem.rdchem.BondDir.EITHERDOUBLE,
                # Added all possible options
                Chem.rdchem.BondDir.BEGINWEDGE,
                Chem.rdchem.BondDir.BEGINDASH,
                Chem.rdchem.BondDir.UNKNOWN
            ]
        self._bond_direction_types = bond_direction_types
        self._self_loop = self_loop


    def __call__(self, mol):
        edge_features = []
        num_bonds = mol.GetNumBonds()
        if num_bonds == 0:
            assert self._self_loop, \
                'The molecule has 0 bonds and we should set self._self_loop to True.'

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            bond_feats = [
                self._bond_types.index(bond.GetBondType()),
                self._bond_direction_types.index(bond.GetBondDir())
            ]
            edge_features.extend([bond_feats, bond_feats.copy()])

        if self._self_loop:
            self_loop_features = torch.zeros((mol.GetNumAtoms(), 2), dtype=torch.int64)
            self_loop_features[:, 0] = len(self._bond_types)

        if num_bonds == 0:
            edge_features = self_loop_features
        else:
            edge_features = np.stack(edge_features)
            edge_features = F.zerocopy_from_numpy(edge_features.astype(np.int64))
            if self._self_loop:
                edge_features = torch.cat([edge_features, self_loop_features], dim=0)

        return {'bond_type': edge_features[:, 0], 'bond_direction_type': edge_features[:, 1]}
