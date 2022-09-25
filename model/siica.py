# BSD 2-Clause License

import numpy as np
from .utils import whiten, MSIICA
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.cross_decomposition import CCA
from torch.autograd import Variable
import time

def siica(
    X,
    n_components=None,
    max_iter=1000,
    c=None,
    approx=True,
    hyper=None,
    init=True
):
    if n_components is None:
        n_components = X.shape[1]
    P, X = whiten(
        X, n_components=n_components
    )


    if c is None:
        W, S, t = _siica_main(
        X, n_components,max_iter=max_iter, hyper=hyper, init=init
    )
    else:
        W, S = _siica_si_main(
            X, n_components,c, max_iter=max_iter,init=init, approx=approx, hyper=hyper
        )
    return P, W, S

def _siica_main(X,n_components, max_iter=10, hyper=None, init=True):
    if hyper is None:
        hyper=1
    device = torch.device('cpu')

    D, p, n = X.shape
    X_copy=list()

    for i in range(D):
        Xw = torch.from_numpy(X[i].T).float().to(device)
        X_copy.append(Xw)
    X =X_copy
    del X_copy
    model = list()
    param = list()
    if init:
        ca = CCA(n_components=p)
        ca.fit(X[0].detach().cpu().numpy(), X[1].detach().cpu().numpy())
        U0 = torch.from_numpy(ca.x_weights_).float()  # .to(device)
        model.append(MSIICA(p, n_components, U0.T).to(device))
        param += list(model[0].parameters())

        for i in range(1, D):
            ca = CCA(n_components=p)
            ca.fit(X[0].detach().cpu().numpy(), X[i].detach().cpu().numpy())
            Ui = torch.from_numpy(ca.y_weights_).float()  # .to(device)
            model.append(MSIICA(p, n_components, Ui.T).to(device))
            param += list(model[i].parameters())

    else:
        for i in range(D):
            model.append(MSIICA(p, n_components).to(device))
            param += list(model[i].parameters())
    start = time.time()
    train_data = TensorDataset(*X)
    train_loader = DataLoader(dataset=train_data, batch_size=n, shuffle=True)
    optim = torch.optim.LBFGS(param, line_search_fn="strong_wolfe", max_iter=max_iter)

    Wo = np.zeros((D, p, p))


    for i in range(D):
        Wo[i] = model[i].W.weight.detach().cpu().numpy()
    for t in range(1):
        for x_batch in train_loader:
            for i in range(D):
                x_batch[i] = Variable(x_batch[i].contiguous())

            def loss_closure():

                optim.zero_grad()


                s0 = model[0](x_batch[0])
                loss = torch.sum(torch.log(torch.cosh(s0)))
                sc = s0.view(1, *s0.shape)
                for i in range(1, D):
                  si = model[i](x_batch[i])
                  loss+=torch.sum(torch.log(torch.cosh(si)))

                  loss-=hyper* torch.trace(si.T @ torch.mean(sc,dim=0))
                  sc = torch.cat((sc,si.view(1, *si.shape)))


                loss.backward()
                return loss

            optim.step(loss_closure)

    W = np.zeros((D, p, p))
    S = np.zeros((D, p, n))
    s=0
    for i in range(D):
        sp = model[i](X[i])
        sp = sp.detach().cpu().numpy()
        W[i] =  model[i].W.weight.detach().cpu().numpy()
        S[i] = sp.T
        s+=sp.T/D
    end = time.time()
    return W, s,end - start


def _siica_si_main(X, n_components, c, max_iter = 10,init=True, approx=True, hyper=None):
    if hyper is None:
        hyper=1
    device = torch.device('cpu')
    D, p, n = X.shape
    X_copy=list()

    for i in range(D):
        Xw = torch.from_numpy(X[i].T).float().to(device)
        X_copy.append(Xw)

    X =X_copy
    del X_copy
    model = list()
    param = list()
    if init:
        ca = CCA(n_components=p)
        ca.fit(X[0].detach().cpu().numpy(), X[1].detach().cpu().numpy())
        U0 = torch.from_numpy(ca.x_weights_).float()
        model.append(MSIICA(p, n_components, U0.T).to(device))
        param += list(model[0].parameters())

        for i in range(1, D):
            ca = CCA(n_components=p)
            ca.fit(X[0].detach().cpu().numpy(), X[i].detach().cpu().numpy())
            Ui = torch.from_numpy(ca.y_weights_).float()
            model.append(MSIICA(p, n_components, Ui.T).to(device))
            param += list(model[i].parameters())
    else:
        for i in range(D):
            model.append(MSIICA(p, n_components).to(device))
            param += list(model[i].parameters())

    train_data = TensorDataset(*X)

    train_loader = DataLoader(dataset=train_data, batch_size=n, shuffle=True)
    optim = torch.optim.LBFGS(param, line_search_fn="strong_wolfe",
                              max_iter=max_iter)


    Wo = np.zeros((D, p, p))


    for i in range(D):
        Wo[i] = model[i].W.weight.detach().cpu().numpy()


    for t in range(1):

        for x_batch in train_loader:
            for i in range(D):
                x_batch[i] = Variable(x_batch[i].contiguous())

            def loss_closure():
                optim.zero_grad()

                s0 = model[0](x_batch[0])
                sc = s0[:,:c].view(1, -1,c)
                loss = torch.sum(torch.log(torch.cosh(s0[:,c:])))
                m = torch.mean(sc, dim=0)
                for i in range(1, D):
                    l = 1 if i+1==D else 0
                    si = model[i](x_batch[i])
                    loss += torch.sum(torch.log(torch.cosh(si[:,c:])))

                   # loss += l*torch.sum(torch.log(torch.cosh(m)))
                    loss -= hyper * torch.trace(si[:,:c].T @ m)
                    sc = torch.cat((sc, (si[:,:c]).view(1, -1,c)))
                    m = torch.sum(sc, dim=0)

                loss += torch.sum(torch.log(torch.cosh(m)))
                loss.backward()
                return loss

            optim.step(loss_closure)



    W = np.zeros((D, p, p))
    S = np.zeros((D, p, n))
    s=0
    for i in range(D):
        sp = model[i](X[i])
        sp = sp.detach().cpu().numpy()

        W[i] = model[i].W.weight.detach().cpu().numpy()
        S[i] = sp.T
        s+=sp[:,:c].T/D
    if approx:
        return W, s
    else:
        return W, np.vstack(S)