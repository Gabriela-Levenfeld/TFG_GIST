import os.path
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.GNNModel import GATv2Model, AttentiveFPModel, MPNNModel, GINModel
from utils.graph_utils import build_graph_and_transform_target, build_test_graph_and_transform_target, collate_molgraphs, to_cuda



def csv_best_params():
    cwd = os.getcwd()
    df = pd.read_csv(f'{cwd}/../../param_search_results/trials_dataframe.csv')
    lowest_value_row = df[df['value'] == df['value'].min()]
    return lowest_value_row

def load_dataset(filename):
    cwd = os.getcwd()
    with open(f'{cwd}/../../data/datasets/{filename}', "rb") as f:
        X, y = pickle.load(f)
    return X, y

def generate_model(best_params, test_loader):
    # Get a sample of the graph to know the node_feat_size and the edge_feat_size
    graph, y, masks = next(iter(test_loader))

    if best_params['model_name'].values[0] == 'GATv2':
        num_layers = best_params['num_layers'].values[0]
        hidden_f = best_params['hidden_feats'].values[0]
        n_attention_heads = best_params['num_heads'].values[0]
        dropout_input_feats = best_params['feat_drops'].values[0]
        dropout_edge = best_params['attn_drops'].values[0]
        alpha = best_params['alphas'].values[0]
        r = best_params['residuals'].values[0]
        s_w = best_params['share_weights'].values[0]
        aggregate_modes = best_params['agg_modes'].values[0]
        reg = GATv2Model(in_feats=graph.ndata['h'].shape[1],
                         hidden_feats=[hidden_f] * num_layers,
                         num_heads=[n_attention_heads] * num_layers,
                         feat_drops=[dropout_input_feats] * num_layers,
                         attn_drops=[dropout_edge] * num_layers,
                         alphas=[alpha] * num_layers,
                         residuals=[r] * num_layers,
                         allow_zero_in_degree=best_params['allow_zero_in_degree'].values[0],
                         share_weights=[s_w] * num_layers,
                         agg_modes=[aggregate_modes] * num_layers,
                         predictor_out_feats=best_params['predictor_out_feats'].values[0],
                         predictor_dropout=best_params['predictor_dropout'].values[0])
    elif best_params['model_name'].values[0] == 'AttentiveFP':
        reg = AttentiveFPModel(node_feat_size=graph.ndata['h'].shape[1],
                               edge_feat_size=graph.edata['e'].shape[1],
                               num_layers=best_params['num_layers'].values[0],
                               graph_feat_size=best_params['graph_feat_size'].values[0],
                               dropout=best_params['dropout'].values[0])
    elif best_params['model_name'].values[0] == 'MPNN':
        reg = MPNNModel(node_in_feats=graph.ndata['h'].shape[1],
                        edge_in_feats=graph.edata['e'].shape[1],
                        node_out_feats=best_params['node_out_feats'].values[0],
                        edge_hidden_feats=best_params['edge_hidden_feats'].values[0])
    elif best_params['model_name'].values[0] == 'GIN':
        reg = GINModel(num_node_emb_list=best_params['num_node_emb_list'].values[0],
                       num_edge_emb_list=best_params['num_edge_emb_list'].values[0],
                       num_layers=best_params['num_layers'].values[0],
                       emb_dim=best_params['emb_dim'].values[0],
                       JK=best_params['JK'].values[0],
                       dropout=best_params['dropout'].values[0],
                       readout=best_params['readout'].values[0])
    else:
        raise ValueError(f"Invalid model name {best_params['model_name'].values[0]}")

    if torch.cuda.is_available():
        print('using CUDA!')
        reg = reg.cuda()
    return reg

def test_results(best_params, filename_train, filename_val, filename_test):

    X_train, y_train = load_dataset(filename_train)
    X_val, y_val = load_dataset(filename_val)
    X_test, y_test = load_dataset(filename_test)

    print('Building graphs...', end='')
    start = time.time()
    train, validation, transformer = build_graph_and_transform_target(
        (X_train, y_train),
        (X_val, y_val),
        atom_alg=best_params['atom_featurizer'].values[0],
        bond_alg=best_params['bond_featurizer'].values[0],
        transformer_alg=best_params['rt_scaler'].values[0],
        self_loop=best_params['self_loop'].values[0]
    )
    test, transformer = build_test_graph_and_transform_target(
        (X_test, y_test),
        atom_alg=best_params['atom_featurizer'].values[0],
        bond_alg=best_params['bond_featurizer'].values[0],
        transformer_alg=best_params['rt_scaler'].values[0],
        self_loop=best_params['self_loop'].values[0]
    )
    print(f'Done! (Ellapsed: {time.time() - start})')

    # Evaluate the model on TEST set
    train_loader = DataLoader(train, batch_size=best_params['params_batch_size'].values[0], shuffle=True, collate_fn=collate_molgraphs)
    val_loader = DataLoader(validation, batch_size=best_params['params_batch_size'].values[0], shuffle=False, collate_fn=collate_molgraphs)
    test_loader = DataLoader(test, batch_size=best_params['params_batch_size'].values[0], shuffle=False, collate_fn=collate_molgraphs)

    reg = generate_model(best_params, test_loader)

    optimizer = best_params['optimizer'].values[0]

    train_losses = []
    val_losses = []
    maes = []
    medaes = []
    loss_criterion = F.smooth_l1_loss
    total_epochs = best_params['total_epochs'].values[0]

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
            for batch_id, (bg, labels, masks) in enumerate(val_loader):
                bg, labels, masks = to_cuda(bg, labels, masks)
                loss, absolute_errors = reg.eval_step(reg, bg, labels, masks, loss_criterion, transformer)
                epoch_losses.append(loss)
                abs_errors.append(absolute_errors)

        val_losses.append(np.mean(epoch_losses))
        maes.append(np.mean(np.concatenate(abs_errors)))
        medaes.append(np.median(np.concatenate(abs_errors)))

        if epoch % 1 == 0:
            print(f'Epoch:{epoch}, Train loss: {train_losses[-1]}, Test loss: {val_losses[-1]}, Test MEDAE: {medaes[-1]}, Test MAE: {maes[-1]}' )

    # Plot loss curves
    losses = pd.DataFrame({
        'epoch': np.arange(len(train_losses)),
        'train_loss': train_losses,
        'test_loss': val_losses,
        'medae': medaes,
        'mae': maes
    })
    losses.index = losses.epoch
    model_name = best_params['model_name'].values[0]
    losses.to_csv(f'losses{model_name}.csv', index=False)
    print('Done')

    import matplotlib.pyplot as plt
    losses[['train_loss', 'test_loss']].plot()
    plt.show()

    # TEST: Evaluation on the test set
    test_losses = []
    test_maes = []
    test_medaes = []
    reg.eval()
    with torch.no_grad():
        for batch_id, (bg, labels, masks) in enumerate(test_loader):
            bg, labels, masks = to_cuda(bg, labels, masks)
            loss, absolute_errors = reg.eval_step(reg, bg, labels, masks, loss_criterion, transformer)
            test_losses.append(loss)
            test_maes.append(np.mean(absolute_errors))
            test_medaes.append(np.median(absolute_errors))
    average_test_loss = np.mean(test_losses)
    average_test_mae = np.mean(test_maes)
    average_test_medae = np.mean(test_medaes)

    print('========== TEST Results ==========')
    print(f'Average Test Loss: {average_test_loss: .4f}')
    print(f'Average Test MAE: {average_test_mae: .4f}')
    print(f'Average Test MEDAE: {average_test_medae}: .4f')
    print('==================================')


if __name__ == '__main__':
    filename_train = 'train.pkl'
    filename_val = 'val.pkl'
    filename_test = 'test_set.pkl'

    best_params = csv_best_params()

    test_results(best_params, filename_train, filename_val, filename_test)
