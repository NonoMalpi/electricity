from typing import Tuple

import numpy as np
import pandas as pd
import torch


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
    """
    def __init__(self,
                 sim_periods: int,
                 n_sim: int,
                 delta_t: int,
                 seed: int,
                 batch_size: int,
                 obs_dim: int,
                 epochs: int):
        self.sim_periods = sim_periods
        self.n_sim = n_sim
        self.delta_t = delta_t
        self.random_seed = seed
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.epochs = epochs


def generate_training_set(sim_df: pd.DataFrame, actual_series: pd.Series, params: ScenarioParams) -> pd.DataFrame:
    """ Generate training set by computing the difference between the simulated and trajectories and the actual trajectory.

    The difference can be considered as the external drift that is not captured by the stationary formulation.
    In addition, this function splits the whole external drift into as many dimensions considered in the stationary
    formulation, e.g. 24.

    Parameters
    ----------
    sim_df: pd.DataFrame
        Dataframe containing the simulated trajectories (sim_periods, n_sim).

    actual_series: pd.Series
        Series containing the actual trajectory (sim_periods,).

    params: ScenarioParams
        Class containing simulation parameters and auxiliary information.

    Returns
    -------
    hour_ts_diff_df: pd.DataFrame
        Dataframe containing the training set with MultiIndex(hour, time step) and number of simulation as columns.
    """
    diff_df = sim_df.subtract(actual_series, axis="index")
    diff_df.index = pd.RangeIndex(start=1, stop=diff_df.index.max() + 2, step=1)

    hour_indexes = np.vstack([np.arange(1, 25, 1) + 24 * i for i in range(params.sim_periods)]).T

    hours_list = np.arange(1, 25)
    ts = np.arange(0, params.sim_periods)
    multi_idx = pd.MultiIndex.from_product([hours_list, ts])
    hour_ts_diff_df = pd.DataFrame(index=multi_idx, columns=diff_df.columns, dtype=float)

    for i, h in enumerate(hours_list):
        hour_ts_diff_df.loc[h] = diff_df.loc[hour_indexes[i]].values

    return hour_ts_diff_df


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

