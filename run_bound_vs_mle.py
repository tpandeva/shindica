import numpy as np
from sklearn.cross_decomposition import CCA
import torch
import torch.nn as nn
from model.utils import find_ordering, amari
from model.utils import MSIICA, whiten
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from torch.autograd import Variable
torch.manual_seed(6)
np.random.seed(6)
from model import siica

Am1 = []
Am2 = []
device = torch.device('cpu')
a = 100
m = 2
c = k = 50
n = 1000
for _ in range(50):
    Sm = np.random.laplace(0, 1, size=c * n).reshape(n, c)
    Ss = np.random.laplace(size=(a - c) * n * m).reshape(m, n, (a - c))
    A = np.random.normal(1, 0.1, size=a * c * m).reshape(m, a, c)
    N = np.random.normal(0, 1, size=n * a * m).reshape(m, n, a)
    B = np.random.normal(1, 0.1, size=a * (a - c) * m).reshape(m, a, (a - c))
    D = np.concatenate((A, B), axis=2)
    X = np.array([(Sm.dot(Ai.T)).T for Ai in A]) + np.array([(Si.dot(Bi.T)).T for (Bi, Si) in zip(B, Ss)])
    X_save = X.copy()

    am1, am2= [],[]
    for it in range(100,1100,100):
        X =X_save.copy()
        Km, WK, S = siica(X, max_iter=it, c=k, hyper=1, init=True)
        W = [WK[i] @ Km[i] for i in range(m)]
        am1.append(np.mean([amari(W[i], D[i]) for i in range(m)]))


        X = X_save.copy()
        X_copy = list()
        Ks = list()
        #D = 2
        for i in range(m):
            Ki, Xwi = whiten(X[i].reshape(1,*X[i].shape ), a)
            Ks.append(Ki)
            Xw = torch.from_numpy(Xwi[0,:,:].T).float().to(device)
            X_copy.append(Xw)
        X = X_copy
        del X_copy
        model = list()
        param = list()
        ca = CCA(n_components=a)
        ca.fit(X[0].detach().cpu().numpy(), X[1].detach().cpu().numpy())
        U0 = torch.from_numpy(ca.x_weights_).float()  # .to(device)
        model.append(MSIICA(a, a, U0.T).to(device))
        param += list(model[0].parameters())

        for i in range(1, m):
            ca = CCA(n_components=a)
            ca.fit(X[0].detach().cpu().numpy(), X[i].detach().cpu().numpy())
            Ui = torch.from_numpy(ca.y_weights_).float()  # .to(device)
            model.append(MSIICA(a, a, (Ui.T).contiguous()).to(device))
            param += list(model[i].parameters())
        train_data = TensorDataset(*X)
        optim = torch.optim.LBFGS(param, line_search_fn="strong_wolfe", max_iter=it)

        Wo = model[0].W.weight.detach()


        train_loader = DataLoader(dataset=train_data, batch_size=1000, shuffle=True)
        mloss = nn.MSELoss()

        for t in range(1):
            for x_batch in train_loader:
                for i in range(m):
                    x_batch[i] = Variable(x_batch[i].contiguous())


                def loss_closure():

                    optim.zero_grad()

                    s1 = model[0](x_batch[0])
                    s2 = model[1](x_batch[1])
                    sc = 0.5 * (s1 + s2)
                    loss = torch.sum(torch.log(torch.cosh(s1))) + torch.sum(torch.log(torch.cosh(s2))) + torch.mean(
                        mloss(s1[:, :k], sc[:, :k])) + torch.mean(mloss(s2[:, :k], sc[:, :k]))

                    loss.backward()
                    return loss


                optim.step(loss_closure)


            sc = 0
            W = []
            for i in range(m):
                sp = model[i](X[i])
                W.append(model[i].W.weight.detach().cpu().numpy())

            W = [W[i] @ Ks[i][0,:,:] for i in range(m)]
            am2.append(np.mean([amari(W[i], D[i]) for i in range(m)]))
    Am1.append(am1)
    Am2.append(am2)
Am1 = [np.array(A) for A in Am1]
Am1 = np.vstack(Am1)
Am2 = [np.array(A) for A in Am2]
Am2 = np.vstack(Am2)
with open("log/Am1.npy", "wb") as f:
    np.save(f, Am1)
with open("log/Am2.npy", "wb") as f:
    np.save(f, Am2)