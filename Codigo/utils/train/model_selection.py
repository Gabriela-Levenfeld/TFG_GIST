import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def stratify_y(y, n_strats=6):
    ps = np.linspace(0, 1, n_strats)
    # make sure last is 1, to avoid rounding issues
    ps[-1] = 1
    quantiles = np.quantile(y, ps)
    cuts = pd.cut(y, quantiles, include_lowest=True)
    codes, _ = cuts.factorize()
    return codes


def stratified_train_test_split(X, y, *, test_size, n_strats=6):
    return train_test_split(X, y, test_size=test_size, stratify=stratify_y(y, n_strats))


def stratified_train_validation_test_split(X, y, test_size, validation_size, random_state, n_strats=6):
    # Split data into: train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_y(y, n_strats))
    # Compute the new % for splitting right
    validation_ratio = validation_size/(1-test_size)
    # Split train into: train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_ratio, random_state=random_state, stratify=stratify_y(y_train, n_strats))

    return X_train, X_val, X_test, y_train, y_val, y_test
