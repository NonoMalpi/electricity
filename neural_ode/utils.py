import torch

import matplotlib.pyplot as plt
import numpy as np

class RunningAverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_batch(y_true, t,  batch_size, trajectory_time):
    s = torch.from_numpy(np.random.choice(
        np.arange(t.shape[0] - trajectory_time, dtype=np.int64), batch_size, replace=False
    ))

    samp_trajs = []
    for i in range(batch_size):
        t0 = s[i]
        sample = y_true[t0: t0 + trajectory_time]
        samp_trajs.append(sample)

    samp_trajs = torch.from_numpy(np.stack(samp_trajs, axis=0)).float()

    samp_ts = t[:trajectory_time].float()

    return samp_trajs, samp_ts, s


def log_normal_pdf(x, mean, logvar):
    """ Return log (PDF_normal)
    log ((1/(sigma*sqrt(2*pi))*exp(-1/2*(x-mean)**2/sigma**2)) =
    = -0.5 * (log(2*pi) + log(sigma**2) + (x-mean)**2 / sigma**2)

    :param x: true trajectory, shape: (nspiral, nsamples, nobs)
    :param mean: mean of the predicted trajectory distribution, shape: (nspiral, nsamples, nobs)
    :param logvar: log of variance of the predicted trajectory distribution, shape: (nspiral, nsamples, nobs)
    :return: log of normal pdf, shape: (nspiral, nsamples, nobs)
    """
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, logvar1, mu2, logvar2):
    v1 = torch.exp(logvar1)
    v2 = torch.exp(logvar2)
    lstd1 = logvar1 / 2.
    lstd2 = logvar2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def plot_y_y_pred(y_true, y_pred, t):

    fig, ax = plt.subplots()
    ax.plot(t, y_true.reshape(-1), color="blue")
    ax.plot(t, y_pred.reshape(-1).detach().numpy(), color="orange")
    plt.show()