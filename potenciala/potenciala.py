from typing import List, NoReturn

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
                 bin_size=2):

        self.df = df
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
            ).astype(int)

        elif method == BucketMethod.Round:

            self.df[x_col_name] = self.df[self.signal_name].round(decimals=0)

    def _compute_potential(self) -> pd.DataFrame:
        return (-1) * self.drift.mean.cumsum()

    def get_potential_percentiles(self, drift_col: str) -> pd.DataFrame:
        return (-1) * self.drift.percentiles[drift_col].cumsum()
