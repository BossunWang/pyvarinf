import numpy as np
import torch
from torch import nn
from sklearn import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm

w0 = 0.125
b0 = 5.
x_range = [-20, 60]

def load_dataset(n=1500, n_tst=150):
    np.random.seed(43)

    def s(x):
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g**2.)

    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1. + np.sin(x)) + b0) + eps
    y = (y - y.mean()) / y.std()
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    return y[:, None], x[:, None]

y, x = load_dataset()

plt.figure("training data")
plt.scatter(x, y)
plt.show()


# Go to pytorch world
X = torch.tensor(x, dtype=torch.float)
Y = torch.tensor(y, dtype=torch.float)

# Maximum likelihood estimate
class MaximumLikelihood(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.out(x)


epochs = 1500
m = MaximumLikelihood()
optim = torch.optim.Adam(m.parameters(), lr=0.01)

for epoch in range(epochs):
    optim.zero_grad()
    y_pred = m(X)
    loss = (0.5 * (y_pred - Y) ** 2).mean()
    print(loss)
    loss.backward()
    optim.step()


plt.figure(figsize=(16, 6))
plt.scatter(x, y)
plt.scatter(x, y_pred.cpu().detach().numpy())
plt.show()

# Variational regression
class VI(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var


def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2))* (y-mu)**2


def elbo(y_pred, y, mu, log_var):
    # likelihood of observing y given Variational mu and sigma
    likelihood = ll_gaussian(y, mu, log_var)

    # prior probability of y_pred
    log_prior = ll_gaussian(y_pred, 0, torch.log(torch.tensor(1.)))

    # variational probability of y_pred
    log_p_q = ll_gaussian(y_pred, mu, log_var)

    # by taking the mean we approximate the expectation
    return (likelihood + log_prior - log_p_q).mean()


def det_loss(y_pred, y, mu, log_var):
    return -elbo(y_pred, y, mu, log_var)


epochs = 1500
batch_size = 256

m = VI().cuda()
optim = torch.optim.Adam(m.parameters(), lr=0.005)
loss_list = []

for epoch in tqdm(range(epochs)):
    optim.zero_grad()
    rand_list = np.arange(len(x))
    np.random.shuffle(rand_list)
    slice_size = len(x) // batch_size
    random_idx = rand_list[:slice_size * batch_size].reshape(slice_size, batch_size)

    for batch_idx in random_idx:
        X = torch.from_numpy(x[batch_idx]).float().cuda()
        Y = torch.from_numpy(y[batch_idx]).float().cuda()
        y_pred, mu, log_var = m(X)
        loss = det_loss(y_pred, Y, mu, log_var)
        print(loss.item())
        loss.backward()
        optim.step()
        loss_list.append(loss.item())

X = torch.from_numpy(x).float().cuda()
Y = torch.from_numpy(y).float().cuda()
# draw samples from Q(theta)
with torch.no_grad():
    y_pred = torch.cat([m(X)[0] for _ in range(1000)], dim=1)

X = X.cpu().detach().numpy()
Y = Y.cpu().detach().numpy()
y_pred = y_pred.cpu().detach().numpy()

# Get some quantiles
q1, mu, q2 = np.quantile(y_pred, [0.05, 0.5, 0.95], axis=1)

plt.figure(figsize=(16, 6))
plt.scatter(X, Y)
plt.plot(X, mu)
plt.fill_between(X.flatten(), q1, q2, alpha=0.2)

loss_list = np.array(loss_list)
iter_list = np.array([a for a in range(len(loss_list))])
plt.figure("loss")
plt.plot(loss_list)
plt.show()
