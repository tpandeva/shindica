import os

os.environ['R_HOME'] = "path/to/your/R/installation"
import pandas as pd
import numpy as np
from rpy2.robjects import r as r
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=80, type=int, help='number of sources')
    parser.add_argument('--isIVA',action='store_true', help='true: load IVA output, false: load our method output')
    args = parser.parse_args()
    return args

def eval(wi):
    wi = 0.5*(wi+wi.T)
    r('''source('run_glasso.R')''')
    get_stat = robjects.globalenv['get_stat']
    rpy2.robjects.numpy2ri.activate()
    dt = get_stat(np.array(wi))
    return np.array(dt)

def ebic(s, lg, n):
    print("Start R: glasso")
    r.library("huge")

    # Prepare the call for GLasso
    glasso = r("huge.glasso")

    rpy2.robjects.numpy2ri.activate()
    results = glasso(np.array(s), lg)
    wi = results[4][0]
    ks = wi @ s
    ch = np.linalg.cholesky(wi)
    loglik = 2 * np.log(np.diag(ch)).sum() - np.trace(ks)
    return -n * 0.5 * loglik + np.log(n) * results[2] + 4 * 0.5 * np.log(s.shape[0]) * results[2], wi


def main(cfg):

    if cfg.isIVA:

       open_file = open(f"log/omics/resIVA_{cfg.index}.pickle", "rb")
       res = pickle.load(open_file)
       open_file.close()
       s = np.concatenate((res['S']), axis=1)
    else:
        Sp1 = pd.read_csv(f"log/omics/S1_{cfg.index}.csv").to_numpy()
        Sp1 = Sp1[:, 1:].T
        Sp2 = pd.read_csv(f"log/omics/S2_{cfg.index}.csv").to_numpy()
        Sp2 = Sp2[:, 1:].T

        s = np.concatenate((Sp1, Sp2), axis=1)
    n = s.shape[1]
    s = np.corrcoef(s)
    out = []
    lmax = max((s - np.eye(s.shape[0])).max(), -(s - np.eye(s.shape[0])).min())
    sa = np.abs(s - np.eye(s.shape[0]))
    sa = 0.5 * (sa + sa.T)
    sa = sa.reshape(-1, )
    sa = np.unique(sa)
    sa = np.sort(sa)
    lmin = sa[round(0.95 * len(sa))]
    lglasso = np.exp(np.linspace(np.log(lmin), np.log(lmax), 30))
    results_glasso = []

    items = [(s, lg, n) for lg in lglasso]

    for j in items:
        si, lgi, ni = j
        results_glasso.append(ebic(si, lgi, ni))



    ebic_score = []
    stats = []
    for k in range(len(results_glasso)):
        ebic_score.append(results_glasso[k][0])

    ebic_score = np.array(ebic_score).reshape(-1)
    ind = np.argsort((np.array(ebic_score)))[:10]
    for l in ind:
        wi = results_glasso[l][1]
        stats.append(eval(wi))
    lambdas = list(lglasso[l] for l in ind.tolist())
    out.append( tuple([stats, lambdas]))

    open_file = open(f"log/omics/res{cfg.index}.pickle", "wb")

    pickle.dump(out, open_file)

    open_file.close()
    return out


if __name__ == '__main__':
    args = parse_args()
    print(args)
    res = main(args)
