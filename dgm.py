import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.optim import Adagrad
from torch.utils.tensorboard import SummaryWriter

from spiral_dataset import spiral_data_2d

rng = np.random.default_rng(seed=100)

n_samples  = 20000

X, _ = make_circles(n_samples, random_state=123)
X = np.reshape(spiral_data_2d(5, n_samples), (-1, 2))
rng.shuffle(X)
feature_size = X.shape[-1]
n_exports = 8
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(X[:, 0], X[:, 1])
plt.show()

X = torch.from_numpy(X).float()


class DemNet(nn.Module):
    def __init__(self,n_exports, feature_size):
        super(DemNet, self).__init__()
        self.n_exports = n_exports
        self.feature_size = feature_size
        self.lin1 = nn.Linear(feature_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, n_exports)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.U = Parameter(torch.ones(self.feature_size))
        self.B = Parameter(torch.ones(self.feature_size))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        init.constant_(self.U, 0)
        init.constant_(self.B, 1)
        if isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight, 1 )
            if module.bias is not None:
                init.constant_(module.bias, 0)

    def energy(self, x):
        h = self.forward(x)
        return torch.mean(((x-self.U)@(x-self.U).t()).sum(-1) - x@self.B.t() - self.softplus(h).sum(-1))

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        return self.lin3(x)


class DgmNet(nn.Module):
    def __init__(self,n_exports, feature_size) -> None:
        super(DgmNet, self).__init__()
        self.n_exports = n_exports
        self.feature_size = feature_size
        self.lin1 = nn.Linear(n_exports, 128)
        self.bn1 = nn.BatchNorm1d(128, affine=False)
        self.lin2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128, affine=False)
        self.lin3 = nn.Linear(128, feature_size)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight, 1)
            if module.bias is not None:
                init.constant_(module.bias, 0)

    def entropy(self) -> float:
        dummy  = 2 *np.pi * np.e
        return 0.5 * (torch.log(dummy * self.bn1.running_var).sum() +
                      torch.log(dummy * self.bn2.running_var).sum())

    def decode(self, x) -> torch.Tensor:
        x = self.lin3.weight

    def forward(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x

dem_net = DemNet(n_exports, feature_size)
dgm_net = DgmNet(n_exports, feature_size)


opt_dgm = Adagrad(dgm_net.parameters(), lr=0.002, weight_decay=1e-5)
opt_dem = Adagrad(dem_net.parameters(), lr=0.002, weight_decay=1e-5)

batch_size = 20
n_epochs = 1000
grad_clip_value = 5.0
grad_clip_normn = 10.0
writer = SummaryWriter("")
for i in range(n_epochs):
    dgm_net.train()
    n_batchs = n_samples // batch_size
    for j in range(n_batchs):
        global_iix = j + n_batchs * i

        opt_dgm.zero_grad()
        opt_dem.zero_grad()
        # train dem
        # x = X[j:j+batch_size]
        x = X[rng.choice(n_samples, batch_size)]
        z = torch.normal(torch.zeros(batch_size, n_exports))
        energy_g = dem_net.energy(dgm_net(z).detach())
        energy_x = dem_net.energy(x)
        loss = energy_x - energy_g
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dem_net.parameters(), grad_clip_normn)
        opt_dem.step()

        writer.add_scalar("dem_loss", loss.item(), global_iix)
        writer.add_scalar("dem_energy_g", energy_g.item(), global_iix)
        writer.add_scalar("dem_energy_x", energy_x.item(), global_iix)


        # train dgm
        z = torch.normal(torch.zeros(batch_size, n_exports))
        loss = dem_net.energy(dgm_net(z)) - dgm_net.entropy()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dem_net.parameters(), grad_clip_normn)
        opt_dgm.step()

        writer.add_scalar("dgm_loss", loss.item(), global_iix)


    z = torch.normal(torch.zeros(500 * batch_size, n_exports))
    dgm_net.eval()
    x = dgm_net(z).detach().data

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x[:,0],x[:,1])
    plt.show()
    writer.add_figure("test_{}".format(i), fig, i)
    # plt.savefig

