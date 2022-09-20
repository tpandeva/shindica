# BSD 2-Clause License

import numpy as np
from sklearn.utils.extmath import randomized_svd
import torch.nn as nn
import geotorch
import scipy

class MSIICA(nn.Module):
    def __init__(self, n_in, n_out, U=None, ortho=True):
        super().__init__()
        self.W = nn.Linear(n_in, n_out, bias=False)
        if ortho:
            geotorch.orthogonal(self.W, "weight")
        else:
            geotorch.sln(self.W, "weight")
        if U is not None:
            self.W.weight = U.contiguous()


    def forward(self, Xw):
        S = self.W(Xw)
        return S

def whiten(X,n_components):
    D, n_features, n_samples = X.shape
    XT = np.zeros((D,n_components,n_samples))

    K = np.zeros((D,n_components,n_features))
    for i in range(len(X)):
        X_mean = X[i].mean(axis=-1)
        X[i] -= X_mean[:, np.newaxis]

        u, d, _ = randomized_svd(X[i], n_components=n_components)

        del _
        Ki = (u / d).T
        Xw = np.dot(Ki, X[i])
        Xw *= np.sqrt(n_samples/2)
        XT[i] = Xw
        K[i]=Ki


    return K, XT

def find_ordering(S_list):
    # taken from https://github.com/hugorichard/multiviewica
    n_pb = len(S_list)
    p = None
    for i in range(n_pb):
        p = S_list[i].shape[0] if p is None else np.min((p, S_list[i].shape[0]))

    for i in range(len(S_list)):
        S_list[i] /= np.linalg.norm(S_list[i], axis=1, keepdims=1)
    S = S_list[0].copy()
    order = np.arange(p)[None, :] * np.ones(n_pb, dtype=int)[:, None]
    for i, s in enumerate(S_list[1:]):
            M = np.dot(S, s.T)
            u, orders = scipy.optimize.linear_sum_assignment(-abs(M.T))
            order[i + 1] = orders
            vals= abs(M[orders, u])

    print("Max coverage", np.max(vals))
    print("Mean coverage", np.mean(vals))
    print("Min coverage", np.min(vals))
    print("Coverage>0.7", sum(vals>0.7))
    return u,orders, vals

def amari(W, A):
    # taken from https://github.com/hugorichard/multiviewica
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)

    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])
