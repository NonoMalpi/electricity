import numpy as np
import pandas as pd
import torch

from .utils import ScenarioParams


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

    hour_indexes = np.vstack([
        np.arange(1, params.obs_dim + 1, 1) + params.obs_dim * i for i in range(params.sim_periods)
    ]).T

    hours_list = np.arange(1, 25)
    ts = np.arange(0, params.sim_periods)
    multi_idx = pd.MultiIndex.from_product([hours_list, ts])
    hour_ts_diff_df = pd.DataFrame(index=multi_idx, columns=diff_df.columns, dtype=float)

    for i, h in enumerate(hours_list):
        hour_ts_diff_df.loc[h] = diff_df.loc[hour_indexes[i]].values

    return hour_ts_diff_df


def apply_learnt_external_drift_to_sim(external_drift_tensor: torch.Tensor, sim_df: pd.DataFrame) -> pd.DataFrame:
    """ Include learnt external drift from neural ODE into simulated trajectories.

    The learnt external drift is a combination of training in-samples and out-of-samples prediction.
    The last time step of the external drift is an out-of-sample prediction.

    Parameters
    ----------
    external_drift_tensor: torch.Tensor
        The external drift time series predicted (time steps, 1, 1, obs_dim).

    sim_df: pd.DataFrame
        Dataframe containing the simulated trajectories (sim_periods, n_sim).

    Returns
    -------
    sim_ext_drift_df: pd.Dataframe
        Dataframe containing the simulated trajectories including the external drift prediction (time steps * obs_dim, n_sim).
    """

    ext_drift_series = pd.Series(external_drift_tensor.flatten())
    sim_ext_drift_df = sim_df.subtract(ext_drift_series, axis="index").dropna()

    return sim_ext_drift_df
