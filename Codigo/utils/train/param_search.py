import numpy as np
import torch.nn.functional as F

from optuna.trial import TrialState
from sklearn.base import clone
from functools import singledispatch

from models.GNNModel import GATv2Model, AttentiveFPModel, MPNNModel, GINModel
from utils.train.loss import truncated_rmse_scorer


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
    # Setting up based on GAT -> https://github.com/awslabs/dgl-lifesci/blob/master/examples/property_prediction/csv_data_configuration/hyper.py
    params = {
        'hidden_feats': trial.suggest_int('hidden_feats', 32, 256), #list of int
        'num_heads': trial.suggest_int('num_heads', 4, 8), #list of int
        'feat_drops': trial.suggest_uniform('feat_drops', 0, 0.5), #list of float of len = num_layers
        'attn_drops': trial.suggest_uniform('attn_drops', 0, 0.5), #list of float of len = num_layers
        'alphas': trial.suggest_uniform('alphas', 0, 1), #len(alphas) = num_layers
        'residuals': trial.suggest_categorical('residuals', [True, False]),
        'allow_zero_in_degree': trial.suggest_categorical('allow_zero_in_degree', [True, False]),
        'share_weights': trial.suggest_categorical('share_weights', [True, False]), #len(share_weights)=num_layers
        'agg_modes': trial.suggest_categorical('agg_modes', ['flatten', 'mean', 'max', 'min', 'sum']), #len(agg_modes)=num_layers -> Eliminar este parámetro y usar por defecto
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
    params = {
        'num_node_emb_list': trial.suggest_int('num_node_emb_list', 3, 120),
        'num_edge_emb_list': trial.suggest_int('num_edge_emb_list', 3, 6),
        'num_layers': trial.suggest_int('num_layer', 3, 6),
        'emb_dim': trial.suggest_int('emb_dim', 64, 512),
        'JK': trial.suggest_categorical('JK', ['concat', 'last', 'max', 'sum']),
        'dropout':trial.suggest_uniform('dropout', 0.1, 0.6),
        'readout': trial.suggest_categorical('readout', ['sum', 'mean', 'max', 'attention', 'set2set'])
    }
    return params


def create_objective(estimator, X, y, scoring):
    estimator_factory = lambda: clone(estimator)
    def objective(trial):
        estimator = estimator_factory()
        params = estimator.suggest_params(trial)

        if estimator.isinstance(GINModel):
            atom_featurizer = 'pretrain'
            bond_featurizer = 'pretrain'
        else: # Canonical o Attentive
            atom_featurizer = trial.suggest_categorical('atom_featurizer', ['canonical', 'attentive_featurizer'])
            bond_featurizer = trial.suggest_categorical('bond_featurizer', ['canonical', 'attentive_featurizer'])
        common_hyperparameters = {
            'batch_size': trial.suggest_int('batch_size', 32, 512),
            'total_epoch': trial.suggest_int('total_epoch', 40, 100), # By now, total_epoch=40 and sometimes it is not enough
            'self_loop': trial.suggest_categorical('self_loop', [True, False]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),

            'rt_scaler': 'robust',
            'atom_featurizer': atom_featurizer,
            'bond_featurizer': bond_featurizer
        }
        params.update(common_hyperparameters)
        estimator.set_params(**params)
        return scoring(estimator, X, y)
    return objective


def param_search(estimator, X, y, study, n_trials, scoring=truncated_rmse_scorer):
    objective = create_objective(estimator, X, y, scoring)
    trials = [trial for trial in study.get_trials() if trial.state == TrialState.COMPLETE]
    n_trials = max(0, n_trials-len(trials))
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)
    best_params = load_best_params(study)
    estimator.set_params(**best_params)
    return estimator


def load_best_params(study):
    try:
        return study.best_params
    except Exception as e:
        print(f'Study does not exist')
        raise e
