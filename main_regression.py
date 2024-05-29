import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import pyvarinf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from IPython import display
import pylab as pl

use_cuda = torch.cuda.is_available()

# Building the model

n_units = 32


class Model(nn.Module):
    """ The model we are going to use """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units),
            nn.ReLU())
        self.mean = nn.Linear(n_units, 1)
        self.rho = nn.Linear(n_units, 1)

    def forward(self, *inputs):
        h = self.model(*inputs)
        return self.mean(h), self.rho(h)


model = Model()
var_model = pyvarinf.Variationalize(model)

if use_cuda:
    var_model = var_model.cuda()

# Generating dataset

def generate_data(n_samples):
    """ Generate n_samples regression datapoints """
    x = np.random.normal(size=(n_samples, 1))
    y = np.cos(x * 3) + np.random.normal(size=(n_samples, 1)) * np.abs(x) / 2
    return x, y


def batch_iterator(x, y):
    """ Provides an iterator given data and labels """
    n_samples = x.shape[0]

    def _iterator(batch_size):
        sample_indices = np.random.randint(0, high=n_samples, size=batch_size)
        return x[sample_indices], y[sample_indices]

    return _iterator


n_train_data = 5000
n_test_data = 100
train_x, train_y = generate_data(n_train_data)
test_x, test_y = generate_data(n_test_data)

# Normalize outputs
train_y = train_y / np.std(train_y)
test_y = test_y / np.std(test_y)

train_iterator = batch_iterator(train_x, train_y)
plt.scatter(train_x, train_y)
plt.show()

# Define plotting function
n_epochs = 300
n_samples = 4


def init_plot():
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-10, 10)
    ax1.set_xlabel("x", fontsize=20)
    ax1.set_ylabel("y", fontsize=20)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim(0, n_epochs)
    ax2.set_ylim(-1.25, 1.25)
    ax2.set_xlabel("Epochs", fontsize=20)
    ax2.set_ylabel("Variational bound estimate", fontsize=20)
    lines = [ax1.plot(test_x, test_y, ls='', marker='o', alpha=0.7)]
    for _ in range(n_samples):
        lines += ax1.plot([0], [0], ls='', marker='o', alpha=0.3)
    loss_line, = ax2.plot([0.], [0.], alpha=0.7)
    prior_loss_line, = ax2.plot([0.], [0.], alpha=0.7)
    mle_loss_line, = ax2.plot([0.], [0.], alpha=0.7)
    ax2.legend([loss_line, prior_loss_line, mle_loss_line],
               ["Loss", "Prior loss", "MLE loss"])
    return lines, loss_line, prior_loss_line, mle_loss_line


def plot(lines, loss_line, prior_loss_line, mle_loss_line,
         epoch, loss, prior_loss, mle_loss):
    """ Plotting utility """
    # sample n models from the posterior
    if lines is None:
        lines, loss_line, prior_loss_line, mle_loss_line = init_plot()
    l0, lines = lines[0], lines[1:]
    n_samples = 10
    nets = [pyvarinf.Sample(var_model) for _ in range(n_samples)]
    for net in nets:
        net.draw()
    x_space = np.random.normal(size=(1000,))
    y_spaces = []
    for net in nets:
        inputs = Variable(torch.Tensor(np.expand_dims(x_space, 1)))
        noise = Variable(torch.randn(1000, 1))
        if use_cuda:
            noise = noise.cuda()
            inputs = inputs.cuda()
        mean, rho = net(inputs)
        outputs = mean + torch.log(1 + torch.exp(rho)) * noise
        y_spaces += [outputs.squeeze().data.cpu().numpy()]
    for l, y_space in zip(lines, y_spaces):
        l.set_data(x_space, y_space)
    lines = [l0] + lines

    epochs = loss_line.get_xdata()
    losses = loss_line.get_ydata()
    prior_losses = prior_loss_line.get_ydata()
    mle_losses = mle_loss_line.get_ydata()
    epochs = list(epochs) + [epoch]
    losses = list(losses) + [loss]
    prior_losses = list(prior_losses) + [prior_loss]
    mle_losses = list(mle_losses) + [mle_loss]
    loss_line.set_data(epochs, losses)
    prior_loss_line.set_data(epochs, prior_losses)
    mle_loss_line.set_data(epochs, mle_losses)
    return lines, loss_line, prior_loss_line, mle_loss_line

# Define log likelihood loss
def gaussian_fit_loss(data, mean, rho):
    """ Compute log likelihood of data, assuming
    a N(mean, log(1 + e^rho)) model.
    """
    sigma = torch.log(1 + torch.exp(rho))
    return torch.mean((mean - data) ** 2 / (2 * sigma ** 2)) + torch.mean(torch.log(sigma))


# Training
n_iterations_per_epochs = 100
batch_size = 256
mle_samples = 1

optimizer = torch.optim.Adam(var_model.parameters(), lr=5e-3)

lines = None
loss_line, prior_loss_line, mle_loss_line = None, None, None
for e in range(n_epochs):
    for i in range(n_iterations_per_epochs):
        batch_x, batch_y = train_iterator(batch_size)
        batch_x, batch_y = [Variable(torch.Tensor(arr)) for arr in [batch_x, batch_y]]
        if use_cuda:
            batch_x, batch_y = [arr.cuda() for arr in [batch_x, batch_y]]
        mle_loss = 0
        for _ in range(mle_samples):
            mean, rho = var_model(batch_x)
            mle_loss += gaussian_fit_loss(batch_y, mean, rho)
        mle_loss = mle_loss / mle_samples
        prior_loss = var_model.prior_loss() / n_train_data
        loss = mle_loss + prior_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch:{e}, loss:{loss.item()}')
    display.clear_output(wait=True)

    plt.figure()
    lines, loss_line, prior_loss_line, mle_loss_line = plot(
        lines, loss_line, prior_loss_line, mle_loss_line,
        e, loss.item(), prior_loss.item(), mle_loss.item())
    plt.show()