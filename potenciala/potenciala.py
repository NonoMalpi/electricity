from typing import Dict, List, NoReturn

import numpy as np
import pandas as pd

from potenciala.metric import Metric


class BucketMethod:

    Cut = "cut"
    Round = "round"


class Potenciala:

    def __init__(self,
                 df: pd.DataFrame,
                 signal_name: str,
                 metric_lag_time: List[int],
                 bucket_method=BucketMethod.Cut,
                 bin_size=2,
                 time_change: bool = True):

        self.df = self._preprocess_input_df(df=df, time_change=time_change)
        self.signal_name = signal_name
        self.metric_lag_time = metric_lag_time
        self.bin_size = bin_size

        base_cols = ["year", "month", "day", "hour"]
        x_col_name = "x_label"

        self._compute_drift()
        self._compute_diffusion()
        self._bucketise_signal(method=bucket_method, x_col_name=x_col_name)

        q = [0.10, 0.25, 0.50, 0.75, 0.90]
        self.drift = Metric(
            df=self.df, base_cols=base_cols, x=x_col_name, metric_cols=self.drift_cols, q=q
        )

        self.diffusion = Metric(
            df=self.df, base_cols=base_cols, x=x_col_name, metric_cols=self.diffusion_cols, q=q
        )

        self.potential = self._compute_potential()
        self.volatility = self._compute_volatility()

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

    def _compute_drift(self) -> NoReturn:
        self.drift_cols = []
        for i in self.metric_lag_time:
            drift_col_name = f"drift_{i}"
            self.df[drift_col_name] = self.df[self.signal_name].shift(-i) - self.df[self.signal_name]
            self.drift_cols.append(drift_col_name)

    def _compute_diffusion(self) -> NoReturn:
        self.diffusion_cols = []
        for i in self.metric_lag_time:
            diffusion_col_name = f"diffusion_{i}"
            self.df[diffusion_col_name] = self.df[f"drift_{i}"] ** 2
            self.diffusion_cols.append(diffusion_col_name)

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

    def _compute_potential(self) -> pd.DataFrame:
        return (-1) * self.drift.mean.cumsum()

    def _compute_volatility(self) -> Dict:
        volatility = {}
        for i in self.metric_lag_time:
            volatility[f"vol_{i}"] = self.df[f"drift_{i}"].std()

        return volatility

    def get_potential_percentiles(self, drift_col: str) -> pd.DataFrame:
        return (-1) * self.drift.percentiles[drift_col].cumsum()


