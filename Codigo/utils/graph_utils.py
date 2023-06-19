import dgl
import torch
import numpy as np
from dgllife.utils import mol_to_bigraph

from utils.memoization import memorize
from utils.featurizers import get_atom_featurizer, get_bond_featurizer, get_transformer


#@memorize
def build_graph_and_transform_target(train, validation, atom_alg, bond_alg, transformer_alg, self_loop):
    (X_train, y_train) = train
    (X_val, y_val) = validation

    atom_featurizer = get_atom_featurizer(atom_alg)
    bond_featurizer = get_bond_featurizer(bond_alg, self_loop)
    transformer = get_transformer(transformer_alg)

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    if transformer is not None:
        y_train = transformer.fit_transform(y_train)
        y_val = transformer.transform(y_val)

    def featurize(x, y):
        # each item is a duple of type (graph(x), y)
        return (
            mol_to_bigraph(x, node_featurizer=atom_featurizer,
                        edge_featurizer=bond_featurizer,
                        add_self_loop=self_loop),
            y
        )

    train = [featurize(x_i, y_i) for x_i, y_i in zip(X_train, y_train)]
    validation = [featurize(x_i, y_i) for x_i, y_i in zip(X_val, y_val)]
    return train, validation, transformer

def build_test_graph_and_transform_target(test, atom_alg, bond_alg, transformer_alg, self_loop):
    (X_test, y_test) = test

    atom_featurizer = get_atom_featurizer(atom_alg)
    bond_featurizer = get_bond_featurizer(bond_alg, self_loop)
    transformer = get_transformer(transformer_alg)

    y_test = y_test.reshape(-1, 1)
    if transformer is not None:
        y_test = transformer.transform(y_test)

    def featurize(x, y):
        # each item is a duple of type (graph(x), y)
        return (
            mol_to_bigraph(x, node_featurizer=atom_featurizer,
                           edge_featurizer=bond_featurizer,
                           add_self_loop=self_loop),
            y
        )

    test = [featurize(x_i, y_i) for x_i, y_i in zip(X_test, y_test)]
    return test, transformer


def collate_molgraphs(data):
    assert len(data[0]) == 2, 'ooops'
    graphs, labels = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.tensor(np.array(labels), dtype=torch.float32).reshape(-1, 1)
    masks = torch.ones(labels.shape)
    return bg, labels, masks


def to_cuda(bg, labels, masks):
    if torch.cuda.is_available():
        bg = bg.to(torch.device('cuda:0'))
        labels = labels.to('cuda:0')
        masks = masks.to('cuda:0')
    return bg, labels, masks