import time
import numpy as np
import optuna
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rdkit import Chem


from utils.data import load_alvadesc_data, load_mols_df
from utils.train.model_selection import stratified_train_validation_test_split
from utils.train.param_search import param_search
from models.GNNModel import GATv2Model, AttentiveFPModel, MPNNModel, GINModel
from utils.graph_utils import build_test_graph_and_transform_target, collate_molgraphs, to_cuda


def create_estimators():
    estimator = ['GATv2', 'AttentiveFP', 'MPNN', 'GIN']
    return estimator

def dummy_create_estimators():
    estimator = ['AttentiveFP']
    return estimator

if __name__ == '__main__':
    SEED = 129767345
    #########################

    X, y = load_mols_df(n=100)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_validation_test_split(X, y, test_size=0.1, validation_size=0.2, random_state=SEED)

    isDUMMY = True
    if isDUMMY:
        print("Searching dummy configuration")
        estimators = dummy_create_estimators()
        n_trials = 2
    else:
        print("Searching configuration")
        estimators = create_estimators
        n_trials = 10

    study = optuna.create_study(study_name='GLS_TFG',
                                direction='minimize',
                                storage='sqlite:///GNNPredict.db', # Esto se sobreescribe (?)
                                load_if_exists=True)

    # Bayesian search
    for estimator in estimators:
        name_model = estimator[0]
        best_params = param_search(estimator, (X_train, y_train), (X_val, y_val), study, n_trials)
        # best_params contiene parámetros para la construccion del grafo y parámetros para testear el modelo
        estimator.set_params(**best_params)

        print('Building graphs...', end='')
        start = time.time()
        test, transformer = build_test_graph_and_transform_target(
            (X_test, y_test),
            atom_alg=best_params['atom_featurizer'],
            bond_alg=best_params['bond_featurizer'],
            transformer_alg=best_params['rt_scaler'],
            self_loop=best_params['self_loop']
        )
        print(f'Done! (Ellapsed: {time.time() - start})')

        # Evaluate the estimator on TEST set
        test_loader = DataLoader(test, batch_size=best_params['batch_size'], shuffle=False, collate_fn=collate_molgraphs)
        # Get a sample of the graph to know the node_feat_size and the edge_feat_size
        graph, y, masks = next(iter(test_loader))

        if torch.cuda.is_available():
            print('using CUDA!')
            estimator = estimator.cuda()

        optimizer = best_params['optimizer']
        total_epochs = best_params['total_epochs']
        test_losses = []
        maes = []
        medaes = []
        loss_criterion = F.smooth_l1_loss
        for epoch in range(1, total_epochs + 1):
            estimator.eval()
            with torch.no_grad():
                epoch_losses = []
                abs_errors = []
                for batch_id, (bg, labels, masks) in enumerate(test_loader):
                    bg, labels, masks = to_cuda(bg, labels, masks)
                    loss, absolute_errors = estimator.eval_step(estimator, bg, labels, masks, loss_criterion, transformer)
                    epoch_losses.append(loss)
                    abs_errors.append(absolute_errors)

            test_losses.append(np.mean(epoch_losses))
            maes.append(np.mean(np.concatenate(abs_errors)))
            medaes.append(np.median(np.concatenate(abs_errors)))
            if epoch % 1 == 0:
                print(f'Epoch:{epoch}, Test loss: {test_losses[-1]}, Test MEDAE: {medaes[-1]}, Test MAE: {maes[-1]}')

        losses = pd.DataFrame({
            'epoch': np.arange(len(test_losses)),
            'train_loss': test_losses,
            'test_loss': test_losses,
            'medae': medaes,
            'mae': maes
        })
        losses.index = losses.epoch
        # TODO: change name depending on arguments
        losses.to_csv(f'{name_model}.csv', index=False)
        print('Done')

        import matplotlib.pyplot as plt
        losses[['train_loss', 'test_loss']].plot()
        plt.show()
