import numpy as np
import optuna
import torch.nn.functional as F
from optuna.trial import TrialState
from sklearn.base import clone

from models.SkGNN import SkGNN, handle_patience
from utils.train.loss import truncated_rmse_scorer
from utils.train.model_selection import stratify_y



def create_objective(estimator: SkGNN, X, y, cv, scoring):
    estimator_factory = lambda: clone(estimator)
    def objective(trial):
        estimator = estimator_factory()
        params = estimator.suggest_params(trial)
        estimator.set_params(**params)
        return cross_val_score_with_pruning(estimator, X, y, cv=cv, scoring=scoring, trial=trial)
    return objective


def cross_val_score_with_pruning(estimator, X, y, cv, scoring, trial):
    cross_val_scores = []
    strats = stratify_y(y, 6)
    for step, (train_index, test_index) in enumerate(cv.split(X, strats)):
        est = clone(estimator)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        est.fit(X_train, y_train)
        cross_val_scores.append(scoring(est, X_test, y_test))
        intermediate_value = np.mean(cross_val_scores)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(cross_val_scores)


def param_search(estimator, X, y, cv, study, n_trials, scoring=truncated_rmse_scorer, keep_going=False):
    objective = create_objective(estimator, X, y, cv, scoring)
    trials = [trial for trial in study.get_trials() if trial.state in [TrialState.COMPLETE, TrialState.PRUNED]]
    if not keep_going:
        n_trials = n_trials - len(trials)
    if n_trials > 0:
        print(f"Starting {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)
    best_params = handle_patience(load_best_params(study))
    estimator.set_params(**best_params)
    return estimator


def load_best_params(study):
    try:
        return study.best_params
    except Exception as e:
        print(f'Study does not exist')
        raise e
