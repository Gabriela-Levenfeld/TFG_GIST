import torch

import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from optuna.trial import TrialState

from models.GNNModel import GATv2Model, AttentiveFPModel, MPNNModel, GINModel
from utils.graph_utils import build_graph_and_transform_target, collate_molgraphs,to_cuda



def suggest_params_gatv2(trial):
    # Default values for agg_modes and hidden_feats do not work!
    # Setting up based on GAT -> https://github.com/awslabs/dgl-lifesci/blob/master/examples/property_prediction/csv_data_configuration/hyper.py
    num_layers = trial.suggest_int('num_layers', 3, 6)
    hidden_f = trial.suggest_int('hidden_feats', 32, 256)
    n_attention_heads = trial.suggest_int('num_heads', 4, 8)
    dropout_input_feats = trial.suggest_float('feat_drops', 0, 0.5)
    dropout_edge = trial.suggest_float('attn_drops', 0, 0.5)
    alpha =trial.suggest_float('alphas', 0, 1)
    r = trial.suggest_categorical('residuals', [True, False])
    s_w = trial.suggest_categorical('share_weights', [True, False])
    aggregate_modes = trial.suggest_categorical('agg_modes', ['flatten', 'mean']) # This are the only 2 options that GATv2 knows how to manage
    params = {
        'hidden_feats': [hidden_f] * num_layers,
        'num_heads': [n_attention_heads] * num_layers,
        'feat_drops': [dropout_input_feats] * num_layers,
        'attn_drops': [dropout_edge] * num_layers,
        'alphas': [alpha] * num_layers,
        'residuals': [r] * num_layers,
        'allow_zero_in_degree': trial.suggest_categorical('allow_zero_in_degree', [True, False]),
        'share_weights': [s_w] * num_layers,
        'agg_modes': [aggregate_modes] * num_layers,
        'predictor_out_feats': trial.suggest_int('predictor_out_feats', 16, 512), #Deafult value to 128
        'predictor_dropout': trial.suggest_float('predictor_dropout', 0, 0.5)
    }
    return params

def suggest_params_attentiveFP(trial):
    # Values selected according to Supplementary Table 4 and 6
    # of "Pushing the boundaries of molecular representation for drug discovery with graph attention mechanism"
    params = {
        'num_layers': trial.suggest_int('num_layers', 2, 5),
        'graph_feat_size': trial.suggest_int('graph_feat_size', 50, 500),
        'dropout': trial.suggest_float('dropout', 0.1, 0.6),
    }
    return params

def suggest_params_mpnn(trial):
    params = {
        'node_out_feats': trial.suggest_int('node_out_feats', 32, 256),
        'edge_hidden_feats': trial.suggest_int('edge_hidden_feats', 16, 256),
    }
    return params

def suggest_params_gin(trial):
    num_layers = trial.suggest_int('num_layers', 3, 6)
    num_nodes = trial.suggest_int('num_node_emb_list', 3, 120)
    num_edges = trial.suggest_int('num_edge_emb_list', 3, 6)
    params = {
        'num_node_emb_list': [num_nodes] * num_layers,
        'num_edge_emb_list': [num_edges] * num_layers,
        'num_layers': num_layers,
        'emb_dim': trial.suggest_int('emb_dim', 64, 512),
        'JK': trial.suggest_categorical('JK', ['concat', 'last', 'max', 'sum']),
        'dropout':trial.suggest_float('dropout', 0.1, 0.6),
        'readout': trial.suggest_categorical('readout', ['sum', 'mean', 'max', 'attention', 'set2set'])
    }
    return params


def generate_model(m, params, train_loader):
    # Get a sample of the graph to know the node_feat_size and the edge_feat_size
    graph, y, masks = next(iter(train_loader))

    if m == 'GATv2':
        reg = GATv2Model(in_feats=graph.ndata['h'].shape[1],
                         hidden_feats=params['hidden_feats'],
                         num_heads=params['num_heads'],
                         feat_drops=params['feat_drops'],
                         attn_drops= params['attn_drops'],
                         alphas=params['alphas'],
                         residuals=params['residuals'],
                         allow_zero_in_degree=params['allow_zero_in_degree'],
                         share_weights=params['share_weights'],
                         agg_modes=params['agg_modes'],
                         predictor_out_feats=params['predictor_out_feats'],
                         predictor_dropout=params['predictor_dropout'])
    elif m == 'AttentiveFP':
        reg = AttentiveFPModel(node_feat_size=graph.ndata['h'].shape[1],
                               edge_feat_size=graph.edata['e'].shape[1],
                               num_layers=params['num_layers'],
                               graph_feat_size=params['graph_feat_size'],
                               dropout=params['dropout'])
    elif m == 'MPNN':
        reg = MPNNModel(node_in_feats=graph.ndata['h'].shape[1],
                        edge_in_feats=graph.edata['e'].shape[1],
                        node_out_feats=params['node_out_feats'],
                        edge_hidden_feats=params['edge_hidden_feats'])
    elif m == 'GIN':
        reg = GINModel(num_node_emb_list=params['num_node_emb_list'],
                       num_edge_emb_list=params['num_edge_emb_list'],
                       num_layers=params['num_layers'],
                       emb_dim=params['emb_dim'],
                       JK=params['JK'],
                       dropout=params['dropout'],
                       readout=params['readout'])
    else:
        raise ValueError(f"Invalid model name {m}")

    if torch.cuda.is_available():
        print('using CUDA!')
        reg = reg.cuda()
    return reg

def bo_train(reg, train_loader, optimizer, loss_criterion):
    train_losses = []
    reg.train()
    epoch_losses = []
    for batch_id, (bg, labels, masks) in enumerate(train_loader):
        bg, labels, masks = to_cuda(bg, labels, masks)
        loss = reg.train_step(reg, bg, labels, masks, loss_criterion, optimizer)
        epoch_losses.append(loss)
    train_losses.append(np.mean(epoch_losses))
    return train_losses

def bo_validation(reg, val_loader, loss_criterion, transformer):
    reg.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_abs_error = 0.0
        for batch_id, (bg, labels, masks) in enumerate(val_loader):
            bg, labels, masks = to_cuda(bg, labels, masks)
            loss, absolute_errors = reg.eval_step(reg, bg, labels, masks, loss_criterion, transformer)
            total_loss += loss
            total_abs_error += absolute_errors
    return total_loss, total_abs_error


def create_objective(train_dataset, validation_dataset):
    def objective(trial):
        params = dict()
        # TODO: Arreglar GIN model
        #model_name = trial.suggest_categorical('model_name', ['GATv2', 'AttentiveFP', 'MPNN', 'GIN'])
        model_name = trial.suggest_categorical('model_name', ['GATv2', 'AttentiveFP', 'MPNN'])

        # Generate the graphs
        rt_scaler = 'robust'
        if model_name == 'GIN':
            atom_featurizer = 'pretrain'
            bond_featurizer = 'pretrain'
        else:
            atom_featurizer = trial.suggest_categorical('atom_featurizer', ['canonical', 'attentive_featurizer'])
            bond_featurizer = trial.suggest_categorical('bond_featurizer', ['canonical', 'attentive_featurizer'])
        self_loop = trial.suggest_categorical('self_loop', [True, False])

        train, validation, transformer= build_graph_and_transform_target(
            train=train_dataset,
            validation=validation_dataset,
            atom_alg=atom_featurizer,
            bond_alg=bond_featurizer,
            transformer_alg=rt_scaler,
            self_loop=self_loop
        )
        params.update({
            'model_name': model_name,
            'rt_scaler': rt_scaler,
            'atom_featurizer': atom_featurizer,
            'bond_featurizer': bond_featurizer,
            'self_loop': self_loop
        })

        # Generate the specific hyperparameters
        if model_name == 'GATv2':
            params.update(suggest_params_gatv2(trial))
        elif model_name == 'AttentiveFP':
            params.update(suggest_params_attentiveFP(trial))
        elif model_name == 'MPNN':
            params.update(suggest_params_mpnn(trial))
        elif model_name == 'GIN':
            params.update(suggest_params_gin(trial))

        batch_size = trial.suggest_int('batch_size', 32, 512)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_molgraphs)
        val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, collate_fn=collate_molgraphs)

        # Generate the model
        reg = generate_model(model_name, params, train_loader)

        # Generate the common hyperparameters
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        common_hyperparameters = {
            'batch_size': batch_size,
            'total_epochs': trial.suggest_int('total_epochs', 40, 250),
            # By now, total_epoch=40 and sometimes it is not enough
            'learning_rate': learning_rate,
            'optimizer': getattr(torch.optim, optimizer_name)(reg._model.parameters(), lr=learning_rate)
        }
        params.update(common_hyperparameters)

        # Training of the model
        train_losses = []
        loss_criterion = F.smooth_l1_loss
        for epoch in range(1, params['total_epochs'] + 1):
            train_loss = bo_train(reg, train_loader, params['optimizer'], loss_criterion)
            train_losses.append(train_loss)
        val_loss, abs_errors = bo_validation(reg, val_loader, loss_criterion, transformer)
        return val_loss
    return objective


def param_search(train, validation, study, n_trials):
    objective = create_objective(train, validation)
    trials = [trial for trial in study.get_trials() if trial.state == TrialState.COMPLETE]
    n_trials = max(0, n_trials-len(trials))
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials, catch=(Exception, ))
    best_params = load_best_params(study)
    return best_params


def load_best_params(study):
    try:
        return study.best_params
    except Exception as e:
        print(f'Study does not exist')
        raise e
