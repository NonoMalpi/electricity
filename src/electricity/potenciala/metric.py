from typing import List

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.figsize'] = [8.0, 6.0]


class Metric:

    def __init__(self,
                 df: pd.DataFrame,
                 base_cols: List[str],
                 x: str,
                 metric_cols: List[str],
                 q: List[float]):

        self.x_col = x
        self.metric_cols = metric_cols

        self.df = df[base_cols + [x] + metric_cols]
        self.mean = self._compute_mean()
        self.percentiles = self._compute_percentiles(q=q)

    def _compute_mean(self) -> pd.DataFrame:
        return self.df.groupby(self.x_col)[self.metric_cols].mean()

    def _compute_percentiles(self, q: List[float]) -> pd.DataFrame:
        df = self.df.groupby(self.x_col)[self.metric_cols].quantile(q=q).unstack(-1)
        col_names = ["p_" + "{0:.2f}".format(i).split(".")[-1] for i in q]
        df.columns.set_levels(levels=col_names, level=1, inplace=True)
        return df

    def _plot(self, series: pd.DataFrame) -> (plt.Figure, plt.Axes):
        fig, ax = plt.subplots()
        series.plot(ax=ax)
        ax.set_xlabel(self.x_col)
        return fig, ax

    def plot_mean(self, series_names: str = None) -> (plt.Figure, plt.Axes):
        fig, ax = self._plot(self.mean if not series_names else self.mean[series_names])
        return fig, ax

    def plot_percentiles(self, series_names: str = None) -> (plt.Figure, plt.Axes):
        fig, ax = self._plot(self.percentiles if not series_names else self.percentiles[series_names])
        return fig, ax

    def plot_mean_percentiles(self, series_names: str) -> (plt.Figure, plt.Axes):
        mean_aux = self.mean[series_names]
        mean_aux.name = "mean"
        df_aux = pd.concat([mean_aux, self.percentiles[series_names]], axis=1)
        fig, ax = self._plot(series=df_aux)
        return fig, ax
