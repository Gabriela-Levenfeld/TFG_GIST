import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.GNNModel import GATv2Model, AttentiveFPModel, MPNNModel, GINModel
from utils.graph_utils import build_test_graph_and_transform_target, collate_molgraphs, to_cuda


def load_test_set(filename):
    with open(filename, "rb") as f:
        X_test, y_test = pickle.load(f)
    return X_test, y_test

def generate_model(best_params, test_loader):
    # Get a sample of the graph to know the node_feat_size and the edge_feat_size
    graph, y, masks = next(iter(test_loader))

    if best_params['model_name'] == 'GATv2':
        num_layers = best_params['num_layers']
        hidden_f = best_params['hidden_feats']
        n_attention_heads = best_params['num_heads']
        dropout_input_feats = best_params['feat_drops']
        dropout_edge = best_params['attn_drops']
        alpha = best_params['alphas']
        r = best_params['residuals']
        s_w = best_params['share_weights']
        aggregate_modes = best_params['agg_modes']
        reg = GATv2Model(in_feats=graph.ndata['h'].shape[1],
                         hidden_feats=[hidden_f] * num_layers,
                         num_heads=[n_attention_heads] * num_layers,
                         feat_drops=[dropout_input_feats] * num_layers,
                         attn_drops=[dropout_edge] * num_layers,
                         alphas=[alpha] * num_layers,
                         residuals=[r] * num_layers,
                         allow_zero_in_degree=best_params['allow_zero_in_degree'],
                         share_weights=[s_w] * num_layers,
                         agg_modes=[aggregate_modes] * num_layers,
                         predictor_out_feats=best_params['predictor_out_feats'],
                         predictor_dropout=best_params['predictor_dropout'])
    elif best_params['model_name'] == 'AttentiveFP':
        reg = AttentiveFPModel(node_feat_size=graph.ndata['h'].shape[1],
                               edge_feat_size=graph.edata['e'].shape[1],
                               num_layers=best_params['num_layers'],
                               graph_feat_size=best_params['graph_feat_size'],
                               dropout=best_params['dropout'])
    elif best_params['model_name'] == 'MPNN':
        reg = MPNNModel(node_in_feats=graph.ndata['h'].shape[1],
                        edge_in_feats=graph.edata['e'].shape[1],
                        node_out_feats=best_params['node_out_feats'],
                        edge_hidden_feats=best_params['edge_hidden_feats'])
    elif best_params['model_name'] == 'GIN':
        num_layers = best_params['num_layers']
        num_nodes = best_params['num_node_emb_list']
        num_edges = best_params['num_edge_emb_list']
        reg = GINModel(num_node_emb_list=[num_nodes] * num_layers,
                       num_edge_emb_list=[num_edges] * num_layers,
                       num_layers=num_layers,
                       emb_dim=best_params['emb_dim'],
                       JK=best_params['JK'],
                       dropout=best_params['dropout'],
                       readout=best_params['readout'])
    else:
        raise ValueError(f"Invalid model name {best_params['model_name']}")

    if torch.cuda.is_available():
        print('using CUDA!')
        reg = reg.cuda()
    return reg

def test_results(best_params, filename):
    X_test, y_test = load_test_set(filename)

    # TODO: Establecer rt_scaler con los best_params no por fuerza bruta
    rt_scaler = 'robust'
    print('Building graphs...', end='')
    start = time.time()
    test, transformer = build_test_graph_and_transform_target(
        (X_test, y_test),
        atom_alg=best_params['atom_featurizer'],
        bond_alg=best_params['bond_featurizer'],
        transformer_alg=rt_scaler,
        self_loop=best_params['self_loop']
    )
    print(f'Done! (Ellapsed: {time.time() - start})')

    # Evaluate the model on TEST set
    test_loader = DataLoader(test, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_molgraphs)

    reg = generate_model(best_params, test_loader)

    # Performace predictions on the Test set
    total_epochs = best_params['total_epochs']
    test_losses = []
    maes = []
    medaes = []
    loss_criterion = F.smooth_l1_loss
    for epoch in range(1, total_epochs + 1):
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
        maes.append(np.mean(np.concatenate(abs_errors)))
        medaes.append(np.median(np.concatenate(abs_errors)))


        if epoch % 1 == 0:
            print(f'Epoch:{epoch}, Test loss: {test_losses[-1]}, Test MEDAE: {medaes[-1]}, Test MAE: {maes[-1]}')

    losses = pd.DataFrame({
        'epoch': np.arange(len(test_losses)),
        'test_loss': test_losses,
        'medae': medaes,
        'mae': maes
    })
    losses.index = losses.epoch
    model_name = best_params['model_name']
    losses.to_csv(f'{model_name}_losses.csv', index=False)
    print('Done - Results saved')

    import matplotlib.pyplot as plt
    losses[['test_loss']].plot()
    plt.show()
