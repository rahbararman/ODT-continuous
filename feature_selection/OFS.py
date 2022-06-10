import numpy as np


def truncate_weights(weights, num_features):
    weights = np.copy(weights)
    if np.count_nonzero(weights) > num_features:
        weights[np.flip(np.argsort(weights))[num_features:]] = 0
    
    return weights

def ofs_partial(R, num_features, eps, stepsize):
    '''
        R: maximum L2 norm
        num_features: number of selected features (B in the paper)
        eps: exploration exploitation tradeoff (
        .. math:: 
        \sum
        in the paper)

    '''
    
x = ofs_partial()