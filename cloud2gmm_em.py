import torch
from torch import tensor
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from math import acos, degrees
from matplotlib.pyplot import cm

"""
EM algo 2D demo, in pytorch
"""

def plot(x, h, n_stdev_bands=3,final=False):

    variance, transform = torch.linalg.eigh(h.covariance_matrix,UPLO='U')
    stdev = variance.sqrt()
    ax = fig.add_subplot(111, aspect='equal')
    max_x, min_x, max_y, min_y = 0.0, 0.0, 0.0, 0.0
    legend = []
    cmap = cm.rainbow(torch.linspace(0, 1, h.mean.size(0)))
    for mean, stdev, transform, color in zip(h.mean, stdev, transform, cmap):
        legend += [mpatches.Patch(color=color, label=f'mu {mean[0].item():.2f}, {mean[1].item():.2f} '
                                                     f'sigma {stdev[0].item():.2f} {stdev[1].item():.2f}')]
        for j in range(1, n_stdev_bands+1):
            ell = Ellipse(xy=(mean[0], mean[1]),
                          width=stdev[0] * j * 2, height=stdev[1] * j * 2,
                          angle=degrees(acos(transform[0, 0].item())),
                          alpha=1.0,
                          edgecolor=color,
                          fc='none')
            ax.add_artist(ell)
            max_x = max(max_x, mean[0] + stdev[0] * j * 2)
            max_y = max(max_y, mean[1] + stdev[1] * j * 2)
            min_x = min(min_x, mean[0] - stdev[0] * j * 2)
            min_y = min(min_y, mean[1] - stdev[1] * j * 2)

    ax.scatter(x[:, 0], x[:, 1])
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    plt.gca().set_aspect('equal', adjustable='box')


    plt.legend(handles=legend)
    if final:
        plt.show()
        plt.pause(5.0)
    else:
        plt.show(block=False)
        plt.pause(0.05)
        ax.cla()

def sample(n, mu, c):
    z = torch.randn(dims, n)
    return (mu.view(-1, 1) - c.matmul(z)).T

def load_vol_file(file):
    with open(file, "rb") as f:
        raw = f.read()
    f.close()
    decoded_vol = raw[:3].decode("utf-8")
    
    return file


if __name__ == '__main__':
    f = load_vol_file("volume.vol")
    plt.ion()
    fig = plt.figure()
    n = 50  # must be even number
    k = 4
    dims = 2
    eps = torch.finfo(torch.float32).eps

    x1 = sample(n//5, tensor([3.0, 0.0]), torch.eye(2) * 1/4)
    x2 = sample(n//5, tensor([0.0, 1.0]), torch.eye(2) * 1/4)
    x3 = sample(n//5, tensor([2.0, 4.0]), torch.eye(2) * 1/4)
    x4 = sample(n//5, tensor([1.0, 3.0]), torch.eye(2) * 1/4)
    x5 = sample(n//5, tensor([0.0, 2.0]), torch.eye(2) * 1/4)
    x = torch.cat((x1, x2, x3, x4, x5)).unsqueeze(1)

    mu = torch.randn(k, dims)
    covar = torch.stack(k * [torch.eye(dims)])
    prior = torch.tensor([1/k, 1/k, 1/k, 1/k]).log()
    converged = False
    i = 0
    h = None

    while not converged:

        prev_mu = mu.clone()
        prev_covar = covar.clone()

        if i%5 == 0:
            # resample
            x1 = sample(n//5, tensor([3.0, 0.0]), torch.eye(2) * 1/4)
            x2 = sample(n//5, tensor([0.0, 1.0]), torch.eye(2) * 1/4)
            x3 = sample(n//5, tensor([2.0, 4.0]), torch.eye(2) * 1/4)
            x4 = sample(n//5, tensor([1.0, 3.0]), torch.eye(2) * 1/4)
            x5 = sample(n//5, tensor([0.0, 2.0]), torch.eye(2) * 1/4)
            x = torch.cat((x1, x2, x3, x4, x5)).unsqueeze(1)

        h = MultivariateNormal(mu, covar)
        llhood = h.log_prob(x)
        weighted_llhood = llhood + prior
        log_sum_lhood = torch.logsumexp(weighted_llhood, dim=1, keepdim=True)
        log_posterior = weighted_llhood - log_sum_lhood

        # if i % 5 == 0:
        plot(x.squeeze(), h)

        pi = torch.exp(log_posterior.reshape(n, k, 1))
        pi = pi * (1- k * eps) + eps
        mu = torch.sum(x * pi, dim=0) / torch.sum(pi, dim=0)

        delta = pi * (x - mu)
        covar = torch.matmul(delta.permute(1, 2, 0), delta.permute(1, 0, 2)) / torch.sum(pi, dim=0).reshape(k, 1, 1)

        converged = torch.allclose(mu, prev_mu) and torch.allclose(covar, prev_covar)
        i += 1

    plot(x.squeeze(), h,final=True)