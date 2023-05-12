import os.path
import time

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgllife.model import GATv2
from dgllife.model.model_zoo.gatv2_predictor import GATv2Predictor
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.data import load_alvadesc_data, load_mols_df
from utils.featurizers import get_atom_featurizer, get_bond_featurizer, get_transformer
from utils.memoization import memorize


#@memorize
def build_graph_and_transform_target(train, test, atom_alg, bond_alg, transformer_alg, self_loop):
    (X_train, y_train) = train
    (X_test, y_test) = test

    atom_featurizer = get_atom_featurizer(atom_alg)
    bond_featurizer = get_bond_featurizer(bond_alg, self_loop)
    transformer = get_transformer(transformer_alg)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    if transformer is not None:
        y_train = transformer.fit_transform(y_train)
        y_test = transformer.transform(y_test)

    def featurize(x, y):
        # each item is a duple of type (graph(x), y)
        return (
            mol_to_bigraph(x, node_featurizer=atom_featurizer,
                        edge_featurizer=bond_featurizer,
                        add_self_loop=self_loop),
            y
        )

    train = [featurize(x_i, y_i) for x_i, y_i in zip(X_train, y_train)]
    test = [featurize(x_i, y_i) for x_i, y_i in zip(X_test, y_test)]
    return train, test, transformer


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


def train_step(reg, bg, labels, masks, loss_criterion, optimizer):
    optimizer.zero_grad()
    # TODO: Review if ndata['h'] is the right thing to pass
    prediction = reg(bg, bg.ndata['h'])
    loss = (loss_criterion(prediction, labels, reduction='none') * (masks != 0).float()).mean()
    loss.backward()
    optimizer.step()
    return loss.data.item()


def eval_step(reg, bg, labels, masks, loss_criterion, transformer):
    """ Compute loss_criterion and the absolute error after undoing the transformation from transformer """
    prediction = reg(bg, bg.ndata['h'])
    loss = (loss_criterion(prediction, labels, reduction='none') * (masks != 0).float()).mean().item()

    prediction = prediction.cpu().numpy().reshape(-1, 1)
    labels = labels.cpu().numpy().reshape(-1, 1)
    if transformer is not None:
        abs_errors = np.abs(
            transformer.inverse_transform(prediction) - transformer.inverse_transform(labels)
        )
    else:
        abs_errors = np.abs(prediction - labels)
    return loss, abs_errors


if __name__ == '__main__':
    batch_size = 256
    fp_size = 1024
    total_epochs = 40
    self_loop = True
    graph_feat_size = 128 #No se usa
    num_layers = 2 #No se usa
    dropout = 0.2
    learning_rate = 1e-3
    SEED = 129767345
    #########################
    rt_scaler = 'robust'
    atom_featurizer = 'attentive_featurizer'
    bond_featurizer = 'attentive_featurizer'
    #######################

    X, y = load_mols_df(n=100) #-> Solo coge 100 muestras para ir más rápido
    #X, y = load_mols_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y) #-> No se puede usar porque una de las 'clases' solo está formada por un componente
    print('Building graphs...', end='')
    start = time.time()
    train, test, transformer = build_graph_and_transform_target(
        (X_train, y_train),
        (X_test, y_test),
        atom_alg=atom_featurizer,
        bond_alg=bond_featurizer,
        transformer_alg=rt_scaler,
        self_loop=self_loop
    )
    print(f'Done! (Ellapsed: {time.time() - start})')

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_molgraphs)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_molgraphs)

    # Get a sample of the graph to know the node_feat_size and the edge_feat_size
    graph, y, masks = next(iter(test_loader))

    # net = GATv2(in_feats=graph.ndata['h'].shape[1],
    #             feat_drops=[0.2 for _ in range(2)],
    #             allow_zero_in_degree=True)

    # Default values for agg_modes and hidden_feats do not work!
    in_feats = graph.ndata['h'].shape[1]
    reg = GATv2Predictor(in_feats=in_feats, agg_modes=["flatten", "mean"], hidden_feats=[32, 32], allow_zero_in_degree=True)


    if torch.cuda.is_available():
        print('using CUDA!')
        reg = reg.cuda()

    optimizer = torch.optim.Adam(reg.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    maes = []
    medaes = []
    loss_criterion = F.smooth_l1_loss


    for epoch in range(1, total_epochs + 1):
        reg.train()
        epoch_losses = []
        for batch_id, (bg, labels, masks) in enumerate(train_loader):
            bg, labels, masks = to_cuda(bg, labels, masks)
            loss = train_step(reg, bg, labels, masks, loss_criterion, optimizer)
            epoch_losses.append(loss)
        train_losses.append(np.mean(epoch_losses))

        reg.eval()
        with torch.no_grad():
            epoch_losses = []
            abs_errors = []
            for batch_id, (bg, labels, masks) in enumerate(test_loader):
                bg, labels, masks = to_cuda(bg, labels, masks)
                loss, absolute_errors = eval_step(reg, bg, labels, masks, loss_criterion, transformer)
                epoch_losses.append(loss)
                abs_errors.append(absolute_errors)

        test_losses.append(np.mean(epoch_losses))
        maes.append(
            np.mean(np.concatenate(abs_errors))
        )
        medaes.append(
            np.median(np.concatenate(abs_errors))
        )

        if epoch % 1 == 0:
            print(f'Epoch:{epoch}, Train loss: {train_losses[-1]}, Test loss: {test_losses[-1]}, Test MEDAE: {medaes[-1]}, Test MAE: {maes[-1]}' )


    losses = pd.DataFrame({
        'epoch': np.arange(len(train_losses)),
        'train_loss': train_losses,
        'test_loss': test_losses,
        'medae': medaes,
        'mae': maes
    })
    losses.index = losses.epoch
    # TODO: change name depending on arguments
    losses.to_csv('losses_GATv2.csv', index=False)
    print('Done')

    import matplotlib.pyplot as plt
    losses[['train_loss', 'test_loss']].plot()
    plt.show()