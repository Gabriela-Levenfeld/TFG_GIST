from dgllife.utils import CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer, CanonicalBondFeaturizer, \
    AttentiveFPBondFeaturizer
from sklearn.preprocessing import FunctionTransformer, RobustScaler


def get_atom_featurizer(atom_featurizer):
    if atom_featurizer == 'canonical':
       return CanonicalAtomFeaturizer()
    elif atom_featurizer == 'attentive_featurizer':
        return AttentiveFPAtomFeaturizer()
    else:
        raise ValueError('Invalid atom featurizer')


def get_bond_featurizer(bond_featurizer, self_loop):
    if bond_featurizer == 'canonical':
       return CanonicalBondFeaturizer(self_loop=self_loop)
    elif bond_featurizer == 'attentive_featurizer':
        return AttentiveFPBondFeaturizer(self_loop=self_loop)
    else:
        raise ValueError('Invalid bond featurizer')


def get_transformer(transformer):
    if transformer == 'none':
        return FunctionTransformer()
    elif transformer == 'robust':
        return RobustScaler()
    else:
        raise ValueError('Invalid transformer')