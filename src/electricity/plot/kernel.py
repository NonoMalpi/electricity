from typing import NoReturn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from electricity.ml import GaussianKernel
from .colors import ImperialColors


def plot_samples_kernel_mean_expected(ax: plt.Axes,
                                      gk: GaussianKernel,
                                      historical_mean: pd.Series,
                                      ymax_lim: float = None,
                                      lw: float = 1) -> NoReturn:
    """ Plot simultaneously the observations, the kernel density, the expected value function from the kernel
    and the frequentist mean.

    Parameters
    ----------
    ax: plt.Axes
        The axis to plot.

    gk: GaussianKernel
        The fitted kernel contained the samples, the grid and the sampled probability density function.

    historical_mean: pd.Series
        A series containing the frequentist mean of the samples used for the fitted kernel.

    ymax_lim: float
        Maximum y tick to represent, default = None.

    lw: float
        Line width for the representation of the historical mean and the expected value from the kernel, default = 1.
    """

    ax.plot(gk.samples[0], gk.samples[1], 'k.', markersize=1, alpha=0.2)
    historical_mean.plot(ax=ax, lw=lw, color=ImperialColors.red.value)
    ax.plot(gk.expected_value_function[0, :], gk.expected_value_function[1, :], lw=lw,
            color=ImperialColors.dark_grey.value)
    cs = ax.contourf(gk.grid[0], gk.grid[1], gk.p,
                     levels=100, cmap=plt.cm.gist_earth_r, alpha=0.5, antialiased=True)
    if ymax_lim:
        ax.set_ylim(0, ymax_lim)


def plot_potential(ax: plt.axis, gk: GaussianKernel, historical_mean: pd.Series) -> NoReturn:
    """ Plot potential coming from the fitted kernel and the frequentist mean.

    Parameters
    ----------
    ax: plt.Axes
        The axis to plot.

    gk: GaussianKernel
        The fitted kernel contained the samples, the grid and the sampled probability density function.

    historical_mean: pd.Series
        A series containing the frequentist mean of the samples used for the fitted kernel.
    """
    dx = gk.expected_value_function[0, 1] - gk.expected_value_function[0, 0]
    ax.plot(gk.expected_value_function[0, :], (-1) * np.cumsum(gk.expected_value_function[1, :] * dx),
            color=ImperialColors.blue.value)
    ax.plot(historical_mean.index, (-1) * np.cumsum(historical_mean.values),
            color=ImperialColors.red.value)