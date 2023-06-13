import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from optuna.trial import TrialState
from sklearn.base import clone
from functools import singledispatch

from models.GNNModel import GATv2Model, AttentiveFPModel, MPNNModel, GINModel
from utils.train.loss import truncated_rmse_scorer
from utils.graph_utils import build_graph_and_transform_target, collate_molgraphs,to_cuda
"""""
Parameters haven't used yet
'weight_decay': trial.suggest_uniform('weight_decay', 0, 3e-3),
'patience': trial.suggest_int('patience', 5, 20),
"""""
@singledispatch
def suggest_params(estimator, trial):
    raise NotImplementedError

@suggest_params.register
def _(estimator: GATv2Model, trial):
    # Default values for agg_modes and hidden_feats do not work!
    # Setting up based on GAT -> https://github.com/awslabs/dgl-lifesci/blob/master/examples/property_prediction/csv_data_configuration/hyper.py
    num_layers = trial.suggest_int('num_layer', 3, 6)
    hidden_f = trial.suggest_int('hidden_feats', 32, 256)
    n_attention_heads = trial.suggest_int('num_heads', 4, 8)
    dropout_input_feats = trial.suggest_uniform('feat_drops', 0, 0.5)
    dropout_edge = trial.suggest_uniform('attn_drops', 0, 0.5)
    alpha =trial.suggest_uniform('alphas', 0, 1)
    r = trial.suggest_categorical('residuals', [True, False])
    s_w = trial.suggest_categorical('share_weights', [True, False])
    aggregate_modes = trial.suggest_categorical('agg_modes', ['flatten', 'mean', 'max', 'min', 'sum'])
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
        'predictor_dropout': trial.suggest_uniform('predictor_dropout', 0, 0.5)
    }
    return params

@suggest_params.register
def _(estimator: AttentiveFPModel, trial):
    # Values selected according to Supplementary Table 4 and 6
    # of "Pushing the boundaries of molecular representation for drug discovery with graph attention mechanism"
    params = {
        'num_layers': trial.suggest_int('num_layer', 2, 5),
        'graph_feat_size': trial.suggest_int('graph_feat_size', 50, 500),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.6),
    }
    return params

@suggest_params.register
def _(estimator: MPNNModel, trial):
    params = {
        'node_out_feats': trial.suggest_int('node_out_feats', 32, 256),
        'edge_hidden_feats': trial.suggest_int('edge_hidden_feats', 16, 256),
    }
    return params

@suggest_params.register
def _(estimator: GINModel, trial):
    num_layers = trial.suggest_int('num_layer', 3, 6)
    num_nodes = trial.suggest_int('num_node_emb_list', 3, 120)
    num_edges = trial.suggest_int('num_edge_emb_list', 3, 6)
    params = {
        'num_node_emb_list': [num_nodes] * num_layers,
        'num_edge_emb_list': [num_edges] * num_layers,
        'num_layers': num_layers,
        'emb_dim': trial.suggest_int('emb_dim', 64, 512),
        'JK': trial.suggest_categorical('JK', ['concat', 'last', 'max', 'sum']),
        'dropout':trial.suggest_uniform('dropout', 0.1, 0.6),
        'readout': trial.suggest_categorical('readout', ['sum', 'mean', 'max', 'attention', 'set2set'])
    }
    return params


def generate_model(estimator, train_loader):
    # Get a sample of the graph to know the node_feat_size and the edge_feat_size
    graph, y, masks = next(iter(train_loader))

    if isinstance(estimator, GATv2Model):
        reg = GATv2Model(in_feats=graph.ndata['h'].shape[1],
                         hidden_feats=estimator.hidden_feats,
                         num_heads=estimator.num_heads,
                         feat_drops=estimator.feat_drops,
                         attn_drops= estimator.attn_drops,
                         alphas=estimator.alphas,
                         residuals=estimator.residuals,
                         allow_zero_in_degree=estimator.allow_zero_in_degree,
                         share_weights=estimator.share_weights,
                         agg_modes=estimator.agg_modes,
                         predictor_out_feats=estimator.predictor_out_feats,
                         predictor_dropout=estimator.predictor_dropout)
    elif isinstance(estimator, AttentiveFPModel):
        reg = AttentiveFPModel(node_feat_size=graph.ndata['h'].shape[1],
                               edge_feat_size=graph.edata['e'].shape[1],
                               num_layers=estimator.num_layers,
                               graph_feat_size=estimator.graph_feat_size,
                               dropout=estimator.dropout)
    elif isinstance(estimator, MPNNModel):
        reg = MPNNModel(node_in_feats=graph.ndata['h'].shape[1],
                        edge_in_feats=graph.edata['e'].shape[1],
                        node_out_feats=estimator.node_out_feats,
                        edge_hidden_feats=estimator.edge_hidden_feats)
    elif isinstance(estimator,GINModel):
        reg = GINModel(num_node_emb_list=estimator.num_node_emb_list,
                       num_edge_emb_list=estimator.num_edge_emb_list,
                       num_layers=estimator.num_layers,
                       emb_dim=estimator.emb_dim,
                       dropout=estimator.dropout,
                       readout=estimator.readout)

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
        epoch_losses = []
        abs_errors = []
        for batch_id, (bg, labels, masks) in enumerate(val_loader):
            bg, labels, masks = to_cuda(bg, labels, masks)
            loss, absolute_errors = reg.eval_step(reg, bg, labels, masks, loss_criterion, transformer)
            epoch_losses.append(loss)
            abs_errors.append(absolute_errors)
    return epoch_losses, abs_errors


def create_objective(estimator, train, validation, scoring):
    estimator_factory = lambda: clone(estimator)
    def objective(trial):
        estimator = estimator_factory()
        # Generate the specific hyperparameters
        params = estimator.suggest_params(estimator, trial)

        # Generate the graphs
        rt_scaler = 'robust'
        if isinstance(estimator, GINModel):
            atom_featurizer = 'pretrain'
            bond_featurizer = 'pretrain'
        else:
            atom_featurizer = trial.suggest_categorical('atom_featurizer', ['canonical', 'attentive_featurizer'])
            bond_featurizer = trial.suggest_categorical('bond_featurizer', ['canonical', 'attentive_featurizer'])
        self_loop = trial.suggest_categorical('self_loop', [True, False])

        (X_train, y_train), (X_val, y_val), transformer, _ = build_graph_and_transform_target(
            train=train,
            validation=validation,
            atom_alg=atom_featurizer,
            bond_alg=bond_featurizer,
            transformer_alg=rt_scaler,
            self_loop= self_loop
        )
        params.update({
            'rt_scaler': rt_scaler,
            'atom_featurizer': atom_featurizer,
            'bond_featurizer': bond_featurizer,
            'self_loop': self_loop
        })

        # Generate the common hyperparameters
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop', 'SGD'])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        common_hyperparameters = {
            'batch_size': trial.suggest_int('batch_size', 32, 512),
            'total_epoch': trial.suggest_int('total_epoch', 40, 100), # By now, total_epoch=40 and sometimes it is not enough
            'learning_rate': learning_rate,
            'optimizer': getattr(torch.optim, optimizer_name)(estimator.parameters(), lr=learning_rate)
        }
        params.update(common_hyperparameters)
        estimator.set_params(**params)

        # Load the dataset
        train_loader = DataLoader((X_train, y_train), batch_size=estimator.batch_size, shuffle=True, collate_fn=collate_molgraphs)
        val_loader = DataLoader((X_val, y_val), batch_size=estimator.batch_size, shuffle=False, collate_fn=collate_molgraphs)

        # Generate the model
        reg = generate_model(estimator, train_loader)

        # Training of the model
        train_losses = []
        val_losses = []
        loss_criterion = F.smooth_l1_loss
        for epoch in range(1, estimator.total_epochs + 1):
            train_loss = bo_train(reg, train_loader, estimator.optimizer, loss_criterion)
            val_loss, abs_errors = bo_validation(reg, val_loader, loss_criterion, transformer)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        return val_losses
    return objective


def param_search(estimator, train, validation, study, n_trials, scoring=truncated_rmse_scorer):
    objective = create_objective(estimator, train, validation, scoring)
    trials = [trial for trial in study.get_trials() if trial.state == TrialState.COMPLETE]
    n_trials = max(0, n_trials-len(trials))
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)
    best_params = load_best_params(study)
    #estimator.set_params(**best_params)
    return best_params


def load_best_params(study):
    try:
        return study.best_params
    except Exception as e:
        print(f'Study does not exist')
        raise e
