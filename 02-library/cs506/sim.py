from operator import le
import numpy as np
from scipy.spatial.distance import pdist


def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)

def manhattan_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += abs(x[i] - y[i])
    return res
    # raise NotImplementedError()

def jaccard_dist(x, y):
    X=np.vstack([x,y])
    return pdist(X,'jaccard')
    # raise NotImplementedError()

def cosine_sim(x, y):
    res = 0
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    # raise NotImplementedError()

# Feel free to add more
