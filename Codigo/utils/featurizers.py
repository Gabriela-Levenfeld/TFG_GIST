from dgllife.utils import CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer, CanonicalBondFeaturizer, \
    AttentiveFPBondFeaturizer

from dgllife.utils import PretrainAtomFeaturizer
import torch
import numpy as np
import dgl.backend as F
from rdkit import Chem
from sklearn.preprocessing import FunctionTransformer, RobustScaler


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
                Chem.rdchem.BondDir.EITHERDOUBLE
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


def get_atom_featurizer(atom_featurizer):
    if atom_featurizer == 'canonical':
       return CanonicalAtomFeaturizer()
    elif atom_featurizer == 'attentive_featurizer':
        return AttentiveFPAtomFeaturizer()
    elif atom_featurizer == 'pretrain':
        return PretrainAtomFeaturizer()
    else:
        raise ValueError('Invalid atom featurizer')


def get_bond_featurizer(bond_featurizer, self_loop):
    if bond_featurizer == 'canonical':
       return CanonicalBondFeaturizer(self_loop=self_loop)
    elif bond_featurizer == 'attentive_featurizer':
        return AttentiveFPBondFeaturizer(self_loop=self_loop)
    elif bond_featurizer == 'pretrain':
        return GINPretrainBondFeaturizer(self_loop=self_loop)
    else:
        raise ValueError('Invalid bond featurizer')


def get_transformer(transformer):
    if transformer == 'none':
        return FunctionTransformer()
    elif transformer == 'robust':
        return RobustScaler()
    else:
        raise ValueError('Invalid transformer')
