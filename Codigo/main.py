import pickle
import optuna

from utils.data import load_alvadesc_data, load_mols_df
from utils.train.model_selection import stratified_train_validation_test_split
from utils.train.param_search import param_search
from utils.test.test_metrics import test_results


if __name__ == '__main__':
    SEED = 129767345
    #########################

    isDUMMY = True
    if isDUMMY:
        print("Searching dummy configuration")
        X, y = load_mols_df(n=50)
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_validation_test_split(X, y, test_size=0.1,
                                                                                                validation_size=0.2,
                                                                                                random_state=SEED)
        n_trials = 2
        storage = 'sqlite:///basura.db'
    else:
        print("Searching configuration")
        X, y = load_mols_df()
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_validation_test_split(X, y, test_size=0.1,
                                                                                                validation_size=0.2,
                                                                                                random_state=SEED)
        n_trials = 10
        storage = 'sqlite:///GNNPredict.db'

    study = optuna.create_study(study_name='GLS_TFG',
                                direction='minimize',
                                storage=storage,
                                load_if_exists=True)

    # Bayesian search
    best_params = param_search((X_train, y_train), (X_val, y_val), study, n_trials)

    filename = 'data/test_params/test_set.pkl'
    with open(filename, "wb") as f:
        pickle.dump((X_test, y_test), f)

    # Final Test
    test_results(best_params, filename)
