from typing import NoReturn

import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from electricity.external_drift.utils import ScenarioParams
from .colors import ImperialColors


def plot_training_evaluation(pred_tensor: torch.Tensor,
                             train_df: pd.DataFrame,
                             training_ts: int,
                             params: ScenarioParams,
                             rows: int = 3,
                             columns: int = 8
                             ) -> NoReturn:
    """ Plot multivariate predicted and actual trajectories.

    The trajectory corresponds to the mean between the differences of the actual price and the simulated prices.
    This mean represents the external drift that is not captured by the stationary formulation.

    Parameters
    ----------
    pred_tensor: torch.Tensor
        Tensor containing the multivariate predicted trajectories (training_ts+2, 1, 1, 24).

    train_df: pd.Dataframe
        Dataframe containing training set (differences between the simulations and the actual price)
        with MultiIndex(hour, time step) and number of simulation as columns.

    training_ts: int
        Last time step used in the training process.

    params: ScenarioParams
        Class containing simulation parameters and auxiliary information.

    rows: int
        Number of rows to display in figure, default = 3.

    columns: int
        Number of columns to display in figure, default = 8.
    """

    assert rows*columns == params.obs_dim, \
        f"Please change the number of rows and columns, the total number of figures to display must be {params.obs_dim}."

    fig, axis = plt.subplots(rows, columns, figsize=(60, 45))

    pred_df = pd.DataFrame(pred_tensor.reshape(-1, params.obs_dim).T, index=np.arange(1, params.obs_dim+1))

    true_mean_series = train_df.mean(axis=1)
    y_max = true_mean_series.max().round()
    y_min = true_mean_series.min().round()

    for i, ax_list in enumerate(axis):
        for j, ax in enumerate(ax_list):
            h = 1 + i*columns + j
            true_mean_series.loc[h].plot(ax=ax, lw=5, color=ImperialColors.blue.value)
            pred_df.loc[h].plot(ax=ax, lw=5, color=ImperialColors.dark_green.value)
            ax.axvline(x=training_ts, color=ImperialColors.cool_grey.value)

            ax.set_xlabel("")
            ax.set_title(f"Hour = {h}", fontsize=40)
            ax.set_xlim(0, pred_df.shape[-1])
            ax.set_ylim(y_min, y_max)
            ax.tick_params(axis="both", labelsize=30)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    fig.text(0.5, 0.095, "Day", ha="center", va="center", fontdict={"size": 50})
    fig.text(0.09, 0.5, "Difference (€/MWh)", ha="center", va="center", fontdict={"size": 50}, rotation=90)
