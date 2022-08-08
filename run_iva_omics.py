import pickle
from independent_vector_analysis import iva_l_sos, consistent_iva
# input iva_l_loss N x T x K (N features, T samples, K views)
import numpy as np
import pandas as pd
import argparse

from model.utils import whiten



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', default=10, type=int, help='number of sources')
    args = parser.parse_args()
    return args

def iva(args):
    X1 = pd.read_csv("data/dataset2.csv", sep=",")
    X2 = pd.read_csv("data/dataset1.csv", sep=",")
    X1 = X1.drop(X1.columns[0], axis=1)

    X2 = X2.drop(X2.columns[0], axis=1)
    X1 = X1.to_numpy()
    X2 = X2.to_numpy()
    k = args.k

    X1 = X1.T
    X2 = X2.T

    K1, Xw1, = whiten(X1.reshape(1, X1.shape[0], X1.shape[1]), k)
    K2, Xw2 = whiten(X2.reshape(1, X2.shape[0], X2.shape[1]), k)
    X = np.zeros((2, 3994, k))
    X[0,:,:] = Xw1[0,:,:].T
    X[1,:,:] = Xw2[0,:,:].T
    iva_results = consistent_iva(X, which_iva='iva_l_sos', n_runs=5)
    open_file = open(f'log/omics/resIVA_{k}.pickle', "wb")
    pickle.dump(iva_results,open_file)
    open_file.close()
if __name__ == '__main__':
    args = parse_args()
    iva(args)