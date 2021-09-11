from typing import Dict, List, NoReturn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from potenciala.metric import Metric


class BucketMethod:

    Cut = "cut"
    Round = "round"


class PotencialaBase:

    def __init__(self,
                 df: pd.DataFrame,
                 signal_name: str,
                 metric_lag_time: List[int],
                 bucket_method=BucketMethod.Cut,
                 bin_size=2,
                 time_change: bool = True,
                 x_col_name: str = "x_label"):

        self.signal_name = signal_name
        self.metric_lag_time = metric_lag_time
        self.bin_size = bin_size
        self.x_col_name = x_col_name

        self.df = self._preprocess_input_df(df=df, time_change=time_change)
        self._bucketise_signal(method=bucket_method, x_col_name=self.x_col_name)

    def _preprocess_input_df(self, df: pd.DataFrame, time_change: bool) -> pd.DataFrame:

        if not time_change:
            return df.copy(deep=True)
        else:
            min_date, max_date = df["date"].min(), df["date"].max()
            max_date = (pd.to_datetime(max_date) + pd.Timedelta("1 day")).strftime(format="%Y-%m-%d")
            date_index = pd.date_range(start=min_date, end=max_date, freq="H")

            aux_df = pd.DataFrame()
            aux_df["date_hour"] = date_index
            aux_df = aux_df[aux_df["date_hour"] < max_date]
            aux_df["date"] = aux_df["date_hour"].dt.date.astype(str)
            aux_df["hour"] = (aux_df["date_hour"].dt.hour + 1).astype(int)

            result_df = aux_df.merge(df, how="left", on=["date", "hour"])
            result_df.drop("date_hour", axis=1, inplace=True)

            return result_df

    def _bucketise_signal(self, method: str, x_col_name: str) -> NoReturn:

        if method == BucketMethod.Cut:

            x_axis = np.arange(
                self.df[self.signal_name].min(),
                self.df[self.signal_name].max() + 2*self.bin_size,
                self.bin_size
            )

            self.df[x_col_name] = pd.cut(
                x=self.df[self.signal_name], bins=x_axis, labels=x_axis[:-1], right=False
            ).astype(float)

        elif method == BucketMethod.Round:

            self.df[x_col_name] = self.df[self.signal_name].round(decimals=0)

    def _compute_drift(self) -> NoReturn:
        self.drift_cols = []
        for i in self.metric_lag_time:
            drift_col_name = f"drift_{i}"
            self.df[drift_col_name] = self.df[self.signal_name].shift(-i) - self.df[self.signal_name]
            self.drift_cols.append(drift_col_name)


class SingleTimeSeries(PotencialaBase):

    def __init__(self,
                 df: pd.DataFrame,
                 signal_name: str,
                 metric_lag_time: List[int],
                 quantile: List[float] = [0.10, 0.25, 0.50, 0.75, 0.90],
                 bucket_method=BucketMethod.Cut,
                 bin_size=2,
                 time_change: bool = True,
                 x_col_name: str = "x_label"):

        super().__init__(df=df,
                         signal_name=signal_name,
                         metric_lag_time=metric_lag_time,
                         bucket_method=bucket_method,
                         bin_size=bin_size,
                         time_change=time_change,
                         x_col_name=x_col_name)

        base_cols = ["year", "month", "day", "hour"]

        self._compute_drift()
        self._compute_diffusion()

        self.drift = Metric(
            df=self.df, base_cols=base_cols, x=self.x_col_name, metric_cols=self.drift_cols, q=quantile
        )

        self.diffusion = Metric(
            df=self.df, base_cols=base_cols, x=self.x_col_name, metric_cols=self.diffusion_cols, q=quantile
        )

        self.potential = self._compute_potential()
        self.volatility = self._compute_volatility()

    def _compute_diffusion(self) -> NoReturn:
        self.diffusion_cols = []
        for i in self.metric_lag_time:
            diffusion_col_name = f"diffusion_{i}"
            self.df[diffusion_col_name] = self.df[f"drift_{i}"] ** 2
            self.diffusion_cols.append(diffusion_col_name)

    def _compute_potential(self) -> pd.DataFrame:
        potential_df = (-1) * self.drift.mean.cumsum()
        potential_df.columns = [f"potential_{i}" for i in self.metric_lag_time]
        self.potential_cols = potential_df.columns
        return potential_df

    def _compute_volatility(self) -> Dict:
        volatility = {}
        for i in self.metric_lag_time:
            volatility[f"vol_{i}"] = self.df[f"drift_{i}"].std()

        return volatility

    def get_potential_percentiles(self, drift_col: str) -> pd.DataFrame:
        return (-1) * self.drift.percentiles[drift_col].cumsum()


class VectorTimeSeries(PotencialaBase):

    def __init__(self,
                 df: pd.DataFrame,
                 signal_name: str,
                 bucket_method=BucketMethod.Cut,
                 bin_size=2,
                 time_change: bool = True,
                 x_col_name: str = "x_label"):

        super().__init__(df=df,
                         signal_name=signal_name,
                         metric_lag_time=[24],
                         bucket_method=bucket_method,
                         bin_size=bin_size,
                         time_change=time_change,
                         x_col_name=x_col_name)

        self.df_vector = self._compute_vector_df()

        self._compute_drift()

        self.samples_hour_x = self._compute_samples_by_hour_x()
        self.drift_hour_x = self._compute_mean_drift_by_hour_x()
        self.potential_hour_x = self._compute_potential()

    def _compute_vector_df(self) -> pd.DataFrame:
        df_vector = self.df.pivot(values=self.signal_name, index=["hour"], columns=["date"])
        df_vector.sort_index(axis=1, inplace=True)
        return df_vector

    def _get_group_by_hour_x(self) -> pd.core.groupby.generic.SeriesGroupBy:
        return self.df.groupby(["hour", self.x_col_name])[self.drift_cols[0]]

    def _compute_samples_by_hour_x(self) -> pd.DataFrame:
        return self._get_group_by_hour_x().count().unstack(-1)

    def _compute_mean_drift_by_hour_x(self) -> pd.DataFrame:
        return self._get_group_by_hour_x().mean().unstack(-1)

    def _compute_potential(self) -> pd.DataFrame:
        return (-1) * self.drift_hour_x.cumsum(axis=1)

    def plot_hourly_boxplot(self):
        fig, ax = plt.subplots(figsize=(20, 7))
        self.df_vector.T.boxplot(ax=ax)
        ax.set_title("Hourly price boxplot")
        ax.set_xlabel("hour")
        ax.set_ylabel("€/MWh")

