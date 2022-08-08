# BSD 2-Clause License
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from model.utils import *
from sklearn.cross_decomposition import CCA
import pandas as pd


def main():
    print(os.getcwd())
    seed = 0
    a1 = 80 #total number of sources in dataset1
    a2 = 80 #total number of sources in dataset2
    c = 40 #shared sources

    torch.manual_seed(seed)
    np.random.seed(seed)
    folder = "log/omics"
    if not os.path.exists(folder):
        os.makedirs(folder)
    stats_file1 = folder+"/S1_"+str(a1)+".csv"
    stats_file2 = folder + "/S2_"+str(a2)+".csv"
    X1 = pd.read_csv("data/dataset2.csv", sep=",")
    X2 = pd.read_csv("data/dataset1.csv", sep=",")
    X1 = X1.drop(X1.columns[0], axis=1)


    X2 = X2.drop(X2.columns[0], axis=1)
    X1 = X1.to_numpy()
    X2 = X2.to_numpy()
    k1 = np.linalg.matrix_rank(X1)
    k2 = np.linalg.matrix_rank(X2)
    device = torch.device('cpu')
    X1 = X1.T
    X2 = X2.T

    K1, Xw1 = whiten(X1.reshape(1,X1.shape[0],X1.shape[1]), k1)
    K2, Xw2 = whiten(X2.reshape(1,X2.shape[0],X2.shape[1]), k2)
    ca = CCA(n_components=a2)
    ca.fit(Xw1[0].T, Xw2[0].T)
    U2 = ca.y_weights_
    U2 = torch.from_numpy(U2).float()
    U1 = ca.x_weights_
    U1 = torch.from_numpy(U1).float()
    model1 = MSIICA(k1, a1, (U1[:,:a1]).T).to(device)
    assert torch.allclose(model1.W.weight.cpu(), (U1[:,:a1]).T)

    model2 = MSIICA(k2, a2, U2.T).to(device)
    assert torch.allclose(model2.W.weight.cpu(), U2.T)

    Xw1 = torch.from_numpy(Xw1[0].T).float().to(device)
    Xw2 = torch.from_numpy(Xw2[0].T).float().to(device)



    train_data = TensorDataset(Xw1, Xw2)
    train_loader = DataLoader(dataset=train_data, batch_size=3994, shuffle=True)
    optim = torch.optim.LBFGS(list(model1.parameters()) + list(model2.parameters()), line_search_fn="strong_wolfe",
                              max_iter=5000)
    for t in range(10):
        for x1_batch, x2_batch in train_loader:
            def loss_closure():
                Sp1 = model1(x1_batch)
                Sp2 = model2(x2_batch)
                optim.zero_grad()
                loss1 = torch.sum(torch.log(torch.cosh(Sp1))) +  torch.sum(torch.log(torch.cosh(Sp2))) - torch.trace(
                      Sp1[:, :c].T @ Sp2[:, :c])

                loss1.backward()
                return loss1

            optim.step(loss_closure)


    Sp1 = model1(Xw1)
    Sp2 = model2(Xw2)
    S_list = [Sp1.detach().cpu().numpy().T, Sp2.detach().cpu().numpy().T]
    u, order, vals = find_ordering(S_list)
    pd.DataFrame(Sp1.detach().cpu().numpy().T).to_csv(stats_file1)
    pd.DataFrame(Sp2.detach().cpu().numpy().T).to_csv(stats_file2)
    return Sp1,Sp2
if __name__ == '__main__':
    main()
