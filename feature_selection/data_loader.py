#test implementation

#1. load the dataset: X, y
#2. calculate the predications
#3. find the accuracy (total number of mistakes) and compare with the paper

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import dsdl


def load_magic04():
    data_magic = pd.read_csv('magic04.data.csv', header=None)
    X = data_magic.to_numpy()[:, :-1]
    Y = data_magic.to_numpy()[:, -1]
    Y = np.array(list(map(lambda x: 1 if x=='g' else -1, Y)))
    X = normalize(X)
    return X, Y

def load_svmguide3():
    ds = dsdl.load("svmguide3")
    X, Y = ds.get_train()
    X = X.toarray()
    X = normalize(X)
    return X,Y




