from typing import Dict, NoReturn


import matplotlib.pyplot as plt

from .colors import ImperialColors


def fill_ax_simulation(ax: plt.Axes, sim_dict: Dict) -> NoReturn:
    """ Plot simulation in axes according to parameters from sim_dict parameter.

    Parameters
    ----------
    ax: plt.Axes
        The matplotlib axis to draw.

    sim_dict: Dict[str, ]
        Dictionary with parameters for simulation.
    """

    sim_df = sim_dict["sim_df"]
    actual_df = sim_dict["actual_df"]
    num_sim_show = sim_dict.get("num_sim_show", 0)
    quantile_regions = sim_dict.get("quantile_regions", True)
    ymax = sim_dict.get("ymax", None)

    if num_sim_show > 0:
        sim_df.iloc[24:, : num_sim_show].plot(alpha=0.05, ax=ax, legend=False)

    sim_df.mean(axis=1).plot(lw=2, color=ImperialColors.blue.value, ax=ax)
    actual_df.reset_index()["spain"].plot(lw=2, color=ImperialColors.dark_grey.value, ax=ax)
    intervals_df = sim_df.iloc[24:, :].quantile(q=[0.1, 0.25, 0.75, 0.9], axis=1)

    if quantile_regions:
        ax.fill_between(intervals_df.columns, intervals_df.iloc[0, :], intervals_df.iloc[-1, :],
                        facecolor=ImperialColors.light_blue.value, alpha=0.5)
        ax.fill_between(intervals_df.columns, intervals_df.iloc[1, :], intervals_df.iloc[-2, :],
                        facecolor=ImperialColors.light_blue.value, alpha=1)

    ax.set_xlabel("time [hours]", fontsize=14)
    ax.set_ylabel("spot price [€/MWh]", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if ymax:
        ax.set_ylim(0, ymax)


def plot_simulation(sim_dict: Dict, filename: str = None) -> NoReturn:
    """ Plot and save a simulation for the electricity price time series.

    Parameters
    ----------
    sim_dict: Dict[str, ]
        Dictionary containing parameters for simulations, it must contain:
            - sim_df [pd.DataFrame]: Simulations results.
            - actual_df [pd.DataFrame]:  Real price time series.
        Optional arguments:
            - num_sim_show [int]: number of random trajectories to plot with very light alpha in background, default = 0.
            - quantile_regions [bool]: Whether show percentile shaded areas, default = True.
            - ymax [floar]: maximum y tick to show, default = None.

    filename: str
        Save figure in this filename, default = None meaning the figure is not saved.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fill_ax_simulation(ax=ax, sim_dict=sim_dict)

    if filename:
        fig.savefig(filename, bbox_inches='tight')


def plot_simulation_comparison(sim_left_dict: Dict, sim_right_dict: Dict, filename: str = None):
    """ Plot and save a comparison between simulations for the electricity price time series.

    Parameters
    ----------
    sim_left_dict: Dict[str, ]
        Dictionary containing parameters for simulations, it must contain:
            - sim_df [pd.DataFrame]: Simulations results.
            - actual_df [pd.DataFrame]:  Real price time series.
        Optional arguments:
            - num_sim_show [int]: number of random trajectories to plot with very light alpha in background, default = 0.
            - quantile_regions [bool]: Whether show percentile shaded areas, default = True.
            - ymax [floar]: maximum y tick to show, default = None.
            - title [str]: Title of the figure.

    sim_right_dict: Dict[str, ]
        Idem.

    filename: str
        Save figure in this filename, default = None meaning the figure is not saved.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 7))

    fill_ax_simulation(ax=ax0, sim_dict=sim_left_dict)
    ax0.set_title(sim_left_dict.get("title", ""), fontsize=16)
    fill_ax_simulation(ax=ax1, sim_dict=sim_right_dict)
    ax1.set_title(sim_right_dict.get("title", ""), fontsize=16)

    if filename:
        fig.savefig(filename, bbox_inches='tight')
