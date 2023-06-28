import errno
import hashlib
import os
import pickle
from functools import wraps

import numpy as np
import pandas as pd


def memorize(fun):
    """ Memoization decorator, intended to cache the results for the graph building function to disk. np.arrays
    or pandas dataframes are hex-hashed. The other arguments are expected to be convertable to strings
    """
    CACHE_PATH = os.path.expanduser("~/.tfg_gnns_cache")
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    def hash_argument(arg):
        if hasattr(arg, '__name__'):
            return arg.__name__
        if isinstance(arg, pd.DataFrame):
            # Python's hash function is randomized for security reasons. Hence, use hashlib.
            # Use only first 10 characters to avoid "name too long errors"
            return hashlib.sha1(arg.values.tobytes()).hexdigest()[:10]
        if isinstance(arg, np.ndarray):
            try:
                return hashlib.sha1(arg.tobytes()).hexdigest()[:10]
            except ValueError:
                # In case the numpy array is not C-ordered, fix this
                return hashlib.sha1(arg.copy(order='C').tobytes()).hexdigest()[:10]
        arg_str = str(arg)[:10].replace('/', '_')
        arg_str = arg_str.replace('[', '')
        arg_str = arg_str.replace('<', '-')
        return arg_str

    @wraps(fun)
    def new_fun(*args, **kwargs):
        string_args = ''
        if len(args) > 0:
            string_args += '_' + '_'.join([hash_argument(arg) for arg in args])
        if len(kwargs) > 0:
            string_args += '_' + '_'.join([(str(k)[:10] + hash_argument(v)) for k, v in kwargs.items()])

        filename = '.cache_{}{}.pickle'.format(fun.__name__, string_args)
        filepath = os.path.join(CACHE_PATH, filename)
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as file:
                    result = pickle.load(file)
            else:
                result = fun(*args, **kwargs)
                os.makedirs(CACHE_PATH, exist_ok=True)
                with open(filepath, 'wb') as file:
                    pickle.dump(result, file)
        except OSError as e:
            if e.errno == errno.ENAMETOOLONG:
                print('Error: Filename too long')
            else:
                print(f'Error: OSError happend - {str(e)}')
            result = fun(*args, **kwargs)
        return result
    return new_fun
