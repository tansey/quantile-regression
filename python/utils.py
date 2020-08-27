import numpy as np


def create_folds(X, k):
    if isinstance(X, int) or isinstance(X, np.integer):
        indices = np.arange(X)
    elif hasattr(X, '__len__'):
        indices = np.arange(len(X))
    else:
        indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in range(k):
        start = end
        end = start + len(indices) // k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[start:end])
    return folds

def fold_complement(N, fold):
    '''Returns all the indices from 0 to N that are not in fold.'''
    mask = np.ones(N, dtype=bool)
    mask[fold] = False
    return np.arange(N)[mask]

def train_test_split(test_indices, *splittables):
    if len(splittables) == 0:
        return []
    N = len(splittables[0])
    train_indices = fold_complement(N, test_indices)
    return ([v[train_indices] for v in splittables],
            [v[test_indices] for v in splittables])

def batches(indices, batch_size, shuffle=True):
    order = np.copy(indices)
    if shuffle:
        np.random.shuffle(order)
    nbatches = int(np.ceil(len(order) / float(batch_size)))
    for b in range(nbatches):
        idx = order[b*batch_size:min((b+1)*batch_size, len(order))]
        yield idx


