import time
import dgl
import numpy as np
import optuna
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgllife.utils import mol_to_bigraph
from rdkit import Chem


from utils.data import load_alvadesc_data, load_mols_df
from utils.featurizers import get_atom_featurizer, get_bond_featurizer, get_transformer
from utils.memoization import memorize
from utils.train.model_selection import stratified_train_validation_test_split
from utils.train.param_search import param_search
from models.GNNModel import GATv2Model, AttentiveFPModel, MPNNModel, GINModel


#@memorize
def build_graph_and_transform_target(train, validation, test, atom_alg, bond_alg, transformer_alg, self_loop):
    (X_train, y_train) = train
    (X_val, y_val) = validation
    (X_test, y_test) = test

    atom_featurizer = get_atom_featurizer(atom_alg)
    bond_featurizer = get_bond_featurizer(bond_alg, self_loop)
    transformer = get_transformer(transformer_alg)

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    if transformer is not None:
        y_train = transformer.fit_transform(y_train)
        y_val = transformer.fit_transform(y_val)
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
    validation = [featurize(x_i, y_i) for x_i, y_i in zip(X_val, y_val)]
    test = [featurize(x_i, y_i) for x_i, y_i in zip(X_test, y_test)]
    return train, validation, test, transformer


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


if __name__ == '__main__':
    batch_size = 256
    fp_size = 1024 #No se usa en ningún momento
    total_epochs = 40
    self_loop = True
    learning_rate = 1e-3
    SEED = 129767345
    #########################
    rt_scaler = 'robust'
    atom_featurizer = 'canonical' #attentive_featurizer, pretrain
    bond_featurizer = 'canonical' #attentive_featurizer, pretrain
    #######################

    #X, y = load_mols_df(n=100)
    #X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_validation_test_split(X, y, test_size=0.1, validation_size=0.2, random_state=SEED)

    # Bayesian search
    study = optuna.create_study(study_name=f'{model_name}',
                                direction='minimize',
                                storage='sqlite:///GNNPredict.db',
                                load_if_exists=True)
    attentiveFP_model = AttentiveFPModel()
    n_trials = 10
    attentiveFP_model = param_search(attentiveFP_model, train_loader, study, n_trials)


    print('Building graphs...', end='')
    start = time.time()
    train, validation, test, transformer = build_graph_and_transform_target(
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        atom_alg=atom_featurizer,
        bond_alg=bond_featurizer,
        transformer_alg=rt_scaler,
        self_loop=self_loop
    )
    print(f'Done! (Ellapsed: {time.time() - start})')

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_molgraphs)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, collate_fn=collate_molgraphs)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_molgraphs)

    # Get a sample of the graph to know the node_feat_size and the edge_feat_size
    graph, y, masks = next(iter(test_loader))
    """
    # Default values for GATv2
    reg = GATv2Model(in_feats=graph.ndata['h'].shape[1],
                     agg_modes=["flatten", "mean"],
                     hidden_feats=[32, 32],
                     allow_zero_in_degree=True)
    reg = AttentiveFPModel(node_feat_size=graph.ndata['h'].shape[1],
                           edge_feat_size=graph.edata['e'].shape[1],
                           num_layers=2,
                           graph_feat_size=128,
                           dropout=0.2)
    """
    # Values used for prediction without optimal searching
    reg = MPNNModel(node_in_feats=graph.ndata['h'].shape[1],
                    edge_in_feats=graph.edata['e'].shape[1],
                    node_out_feats=64,
                    edge_hidden_feats=32)
    """
    # Default values for GIN -> Configuración especial: Featurizer='pretrain'
    reg = GINModel(num_node_emb_list=[120, 3],
                   num_edge_emb_list=[6, 3],
                   num_layers=5,
                   emb_dim=300,
                   dropout=0.5,
                   readout='mean')
    """

    if torch.cuda.is_available():
        print('using CUDA!')
        reg = reg.cuda()

    optimizer = torch.optim.Adam(reg._model.parameters(), lr=learning_rate)

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
            loss = reg.train_step(reg, bg, labels, masks, loss_criterion, optimizer)
            epoch_losses.append(loss)
        train_losses.append(np.mean(epoch_losses))

        reg.eval()
        with torch.no_grad():
            epoch_losses = []
            abs_errors = []
            for batch_id, (bg, labels, masks) in enumerate(test_loader):
                bg, labels, masks = to_cuda(bg, labels, masks)
                loss, absolute_errors = reg.eval_step(reg, bg, labels, masks, loss_criterion, transformer)
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
