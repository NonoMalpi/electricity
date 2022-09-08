from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd
import torch


class SignalDimension(Enum):
    """ Enum to indicate whether training the neural ODE with a univariate or multivariate signal. """
    Univariate = 1
    Multivariate = 2


class ScenarioParams:
    """ Stores simulation parameters and auxiliary information for the external drift learning scenario.

    Attributes
    ----------
    sim_periods: int
        Number of time steps [days] to simulate trajectories.

    n_sim: int
        Number of independent simulated trajectories.

    delta_t: int
        Length of the time step.

    seed: int
        Random seed to use during simulations.

    batch_size: int
        Number of samples per batch during training phase

    obs_dim: int
        Number of dimensions of the observable time series

    epochs: int
        Number of epochs to train neural ODE

    lr: float
        Learning rate to train neural ODE
    """
    def __init__(self,
                 sim_periods: int,
                 n_sim: int,
                 delta_t: int,
                 seed: int,
                 batch_size: int,
                 obs_dim: int,
                 epochs: int,
                 lr: float):
        self.sim_periods = sim_periods
        self.n_sim = n_sim
        self.delta_t = delta_t
        self.random_seed = seed
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.epochs = epochs
        self.lr = lr


def get_multivariate_batch(train_df: pd.DataFrame, time_period: int, params: ScenarioParams) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Obtain a random batch tensor from training set with length time_period and size equal to params.batch_size.

    Parameters
    ----------
    train_df: pd.DataFrame
        The training set with MultiIndex(hour, time step) and number of simulation as columns.

    time_period: int
        Number of time steps considered.

    params: ScenarioParams
        Class containing simulation parameters and auxiliary information.

    Returns
    -------
    batch_y0: torch.Tensor
        Tensor containing the initial values of the random batch (batch_size, 1, obs_dim).

    batch_t: torch.Tensor
        Tensor containing the time steps (time_period + 1).

    batch_y: torch.Tensor
        Tensor containing the whole trajectory of the random batch (time_period + 1, batch_size, 1, obs_dim).
    """
    batch_y = np.zeros((time_period + 1, params.batch_size, 1, params.obs_dim))  # shape: (time, batch_size, 1, obs_dim)
    batch_t = torch.from_numpy(np.arange(time_period + 1, dtype=float))
    col_indexes = np.random.choice(train_df.columns, size=params.batch_size, replace=False)  # pick batch_size columns
    # iterate through each timestamp to get array of shape (batch_zise, 1, obs_dim)
    for h in range(time_period + 1):
        batch_y[h] = train_df[col_indexes].xs(h, level=1, drop_level=False).T.values.reshape(params.batch_size, 1, params.obs_dim)
    batch_y0 = torch.from_numpy(batch_y[0]).float()
    batch_y = torch.from_numpy(batch_y)

    return batch_y0, batch_t, batch_y


def get_mean_tensor_from_training_set(train_df: pd.DataFrame, time_step: int) -> torch.Tensor:
    """ Compute multivariate mean at a given time step from training set.

    Parameters
    ----------
    train_df: pd.DataFrame
        The training set with MultiIndex(hour, time step) and number of simulation as columns.

    time_step: int
        The time step at which compute the multivariate mean.

    Returns
    -------
    mean_tensor: torch.Tensor
        Multivariate mean tensor (1, 1, obs_dim)
    """
    mean_tensor = train_df.xs(time_step, level=1, drop_level=False).mean(axis=1).values
    mean_tensor = torch.from_numpy(mean_tensor.reshape(1, 1, -1)).float()

    return mean_tensor

