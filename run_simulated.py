import pickle

import numpy as np
import argparse
import torch
from model import siica
from model import shica, shica_ml, groupica, infomax, multiviewica
from model.utils import find_ordering, amari
import time
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=5000, type=int, help='number of total epochs to run')
    parser.add_argument('--a', default=100, type=int,
                        help='number of sources')
    parser.add_argument('--m', default=2, type=int,
                        help='number of views')
    parser.add_argument('--c', default=50, type=int,
                        help='number of shared sources')
    parser.add_argument('--noise', default=0, type=float,
                        help='noise std')
    parser.add_argument('--isRandom', action='store_true',
                        help='if noise is sampled randomly from interval [1,2]')
    parser.add_argument('--n', default=1000, type=int,
                        help='sample size')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed')
    parser.add_argument('--lam', default=1, type=float,
                        help='lambda hyperparameter')
    parser.add_argument('--fig', default=1, type=int,
                        help='type of experiment (2) is figure 1 and (3) is figure 3')
    parser.add_argument('--shml', default=1, type=int,
                        help='type of experiment (2) is figure 1 and (3) is figure 3')
    args = parser.parse_args()
    return args



def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sigma = args.noise
    m = args.m
    n = args.n
    a = args.a
    c = args.c
    l = args.lam

    distsh = []
    distshml=[]
    distsi=[]
    distinfo = []
    distgroup = []
    distmica = []

    timesh = []
    timeshml=[]
    timesi=[]
    timeinfo = []
    timegroup = []
    timemica = []


    for s in range(50):
            print(f'sigma={sigma}, seed={s}')
            torch.manual_seed(s)
            np.random.seed(s)
            Sm = np.random.laplace(0,1,size=c * n).reshape(n, c)
            Ss = np.random.laplace(size=(a - c) * n * m).reshape(m, n, (a - c))
            A = np.random.normal(1, 0.1, size=a * c * m).reshape(m, a, c)
            N = np.random.normal(0, 1, size=n * a * m).reshape(m, n, a)
            B = np.random.normal(1, 0.1, size=a * (a - c) * m).reshape(m, a, (a - c))
            D = np.concatenate((A, B), axis=2)
            X = np.array([(Sm.dot(Ai.T)).T for Ai in A]) + np.array([(Si.dot(Bi.T)).T for (Bi, Si) in zip(B, Ss)])
            if args.isRandom:
                X += np.array([(np.random.uniform(1,2) * ni.dot(Ci.T)).T for (Ci, ni) in zip(D, N)])
            else:
                X += np.array([(sigma*ni.dot(Ci.T)).T for (Ci, ni) in zip(D, N)])
            Sst = np.concatenate((Ss), axis=1)
            St = np.concatenate((Sst,Sm),axis=1)

            start = time.time()
            if c==a:
                Km, WK, S = siica(X, max_iter=args.epochs,  hyper = l, init=True)
            else:
                Km, WK, S = siica(X, max_iter=args.epochs, c=c,hyper=l, init=True)
            end = time.time()
            timesi.append(end - start)
            W = [WK[i] @ Km[i] for i in range(m)]
            print("ShIndICA", np.mean([amari(W[i], D[i]) for i in range(m)]))
            distsi.append(np.mean([amari(W[i], D[i]) for i in range(m)]))


            start = time.time()
            Wsh,_,_ = shica(X, use_scaling=False)
            end = time.time()
            timesh.append(end - start)
            distsh.append(np.mean([amari(Wsh[i], D[i]) for i in range(m)]))
            print("shica", np.mean([amari(Wsh[i], D[i]) for i in range(m)]))

            if args.shml == 1:
                start = time.time()
                Wshml, _, _ = shica_ml(X, init="shica_j", )
                end = time.time()
                distshml.append(np.mean([amari(Wshml[i], D[i]) for i in range(m)]))
                print("shica-ml",np.mean([amari(Wshml[i], D[i]) for i in range(m)]))
                timeshml.append(end - start)


            start = time.time()
            _, Wpermica, _ = infomax(X)
            end = time.time()
            timeinfo.append(end - start)
            distinfo.append(np.mean([amari(Wpermica[i], D[i]) for i in range(m)]))
            print("infomax", np.mean([amari(Wpermica[i], D[i]) for i in range(m)]))

            start = time.time()
            _, Wmica, _ = multiviewica(X)
            end = time.time()
            timemica.append(end - start)
            distmica.append(np.mean([amari(Wmica[i], D[i]) for i in range(m)]))
            print("multiview", np.mean([amari(Wmica[i], D[i]) for i in range(m)]))

            start = time.time()
            _,Wgroupica, _ = groupica(X )
            end = time.time()
            timegroup.append(end - start)
            distgroup.append(np.mean([amari(Wgroupica[i], D[i]) for i in range(m)]))
            print("groupica",np.mean([amari(Wgroupica[i], D[i]) for i in range(m)]))



    return distsh, distshml, distsi, distinfo, distgroup, distmica, timesh, timeshml, timesi,timeinfo ,timegroup, timemica
if __name__ == '__main__':
    args = parse_args()
    print(args)
    res= main(args)
    file = open(f"log/out_{args.m}_{args.a}_{args.c}_lam_{args.lam}_fig_{args.fig}.pickle","wb")
    pickle.dump(res, file)
    file.close()

