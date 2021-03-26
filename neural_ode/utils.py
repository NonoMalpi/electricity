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


def get_random_batch(y_true, t,  batch_size, trajectory_time):
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


def get_norm_batch(y_true, start, end):

    trajectory = y_true.iloc[start: end]
    mean = trajectory.mean()
    std = trajectory.std()

    scaled_trajectory = np.array((trajectory - mean) / std).reshape(-1, 1)
    samp_traj = torch.from_numpy(np.stack([scaled_trajectory], axis=0)).float()

    samp_ts = torch.arange(start=start, end=end, step=1, dtype=torch.float)

    return samp_traj, samp_ts, mean, std


def get_norm_batch_with_noise(y_true, start, end, noise_std):

    samp_traj, samp_ts, mean, std = get_norm_batch(y_true=y_true, start=start, end=end)

    noise = torch.from_numpy(np.random.randn(*samp_traj.shape) * noise_std)

    samp_traj += noise

    return samp_traj, samp_ts, mean, std


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
    ax.scatter(t, y_true.reshape(-1), color="blue")
    ax.plot(t, y_pred.reshape(-1).detach().numpy(), color="orange")
    plt.show()


def plot_prediction_true_scale(y_true, y_pred, samp_ts, t_limit, mean, std):

    y_pred_scale = y_pred * std + mean

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(samp_ts, y_true.reshape(-1), color="blue")
    ax.plot(samp_ts, y_pred_scale.reshape(-1), color="orange")
    ax.axvline(x=t_limit)
    plt.show()
