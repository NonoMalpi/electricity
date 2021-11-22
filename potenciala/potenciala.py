from enum import Enum
from typing import Dict, List, NoReturn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.linalg import sqrtm

from potenciala.metric import Metric
from potenciala.transformers import TransformerFactory


class BucketMethod(Enum):
    Cut = "cut"
    Round = "round"
    NoBucket = "no_bucket"


class FigureShape(Enum):
    TwoDim = "2D"
    ThreeDim = "3D"


class PotencialaBase:

    def __init__(self,
                 df: pd.DataFrame,
                 signal_name: str,
                 metric_lag_time: List[int],
                 bucket_method=BucketMethod.Cut,
                 bin_size=2,
                 time_change: bool = True,
                 x_col_name: str = "x_label",
                 signal_transformation: str = None,
                 x_transformation: str = None):

        self.signal_transformer = TransformerFactory.build(transformer_type=signal_transformation)
        self.x_transformer = TransformerFactory.build(transformer_type=x_transformation)

        self.signal_name, df = self._transform_signal(df=df, signal_name=signal_name)
        self.metric_lag_time = metric_lag_time
        self.bin_size = bin_size
        self.x_col_name = x_col_name

        self.df = self._preprocess_input_df(df=df, time_change=time_change)
        self._bucketise_x(method=bucket_method)

    def _transform_signal(self, df: pd.DataFrame, signal_name: str):
        df = df.copy(deep=True)
        new_name = self.signal_transformer.rename_signal(signal_name=signal_name)
        df[new_name] = self.signal_transformer.transform(series=df[signal_name])
        return new_name, df

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

    def _bucketise_x(self, method: BucketMethod) -> NoReturn:

        x_transformed = self.x_transformer.transform(self.df[self.signal_name])

        if method == BucketMethod.Cut:
            x_axis = np.arange(
                x_transformed.min(),
                x_transformed.max() + 2*self.bin_size,
                self.bin_size
            )

            self.df[self.x_col_name] = pd.cut(
                x=x_transformed, bins=x_axis, labels=x_axis[:-1], right=False
            ).astype(float)

        elif method == BucketMethod.Round:
            self.df[self.x_col_name] = x_transformed.round(decimals=0)

        elif method == BucketMethod.NoBucket:
            self.df[self.x_col_name] = x_transformed

    def _compute_drift(self) -> NoReturn:
        self.drift_cols = []
        for i in self.metric_lag_time:
            drift_col_name = f"drift_{i}"
            self.df[drift_col_name] = self.df[self.signal_name].shift(-i) - self.df[self.signal_name]
            self.drift_cols.append(drift_col_name)

    def _plot_ts(self, ts: pd.DataFrame, signal_name: str) -> NoReturn:
        ts_index = (pd.to_datetime(ts["date"]) + pd.to_timedelta(ts["hour"] - 1, unit="H"))
        ts_df = pd.DataFrame(index=ts_index, data=ts[signal_name].values)

        fig, ax = plt.subplots(figsize=(20, 7))
        ts_df.plot(ax=ax)
        ax.get_legend().remove()

    def plot_ts(self, signal_name: str = None, period_filter: str = None):
        if not signal_name:
            signal_name = self.signal_name
        self._plot_ts(self.df if not period_filter else self.df.query(period_filter), signal_name=signal_name)


class SingleTimeSeries(PotencialaBase):

    def __init__(self,
                 df: pd.DataFrame,
                 signal_name: str,
                 metric_lag_time: List[int],
                 quantile: List[float] = [0.10, 0.25, 0.50, 0.75, 0.90],
                 bucket_method=BucketMethod.Cut,
                 bin_size=2,
                 time_change: bool = True,
                 x_col_name: str = "x_label",
                 signal_transformation: str = None,
                 x_transformation: str = None):

        super().__init__(df=df,
                         signal_name=signal_name,
                         metric_lag_time=metric_lag_time,
                         bucket_method=bucket_method,
                         bin_size=bin_size,
                         time_change=time_change,
                         x_col_name=x_col_name,
                         signal_transformation=signal_transformation,
                         x_transformation=x_transformation)

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
                 x_col_name: str = "x_label",
                 diff_matrix_xi_xj_computation: bool = False,
                 signal_transformation: str = None,
                 x_transformation: str = None):

        super().__init__(df=df,
                         signal_name=signal_name,
                         metric_lag_time=[24],
                         bucket_method=bucket_method,
                         bin_size=bin_size,
                         time_change=time_change,
                         x_col_name=x_col_name,
                         signal_transformation=signal_transformation,
                         x_transformation=x_transformation)

        self.df_vector = self._compute_vector_df()

        self._compute_drift()
        self._compute_diffusion()
        self._add_x_two_label()

        self.samples_hour_x = self._compute_samples_by_hour_x()
        self.drift_hour_x = self._compute_mean_drift_by_hour_x()
        self.potential_hour_x = self._compute_potential()

        self.diffusion_matrix = self._compute_diffusion_matrix()
        self.sqrt_diff_matrix = self._compute_sqrt_diffusion_matrix()

        if diff_matrix_xi_xj_computation:
            self.diff_xi_xj = self._compute_diff_matrix_components()

    def _compute_vector_df(self) -> pd.DataFrame:
        df_vector = self.df.pivot(values=self.signal_name, index=["hour"], columns=["date"])
        df_vector.sort_index(axis=1, inplace=True)
        return df_vector

    def _compute_diffusion(self) -> NoReturn:
        df_date_hour_drift = self.df.pivot(values=self.drift_cols[0], index=["date"], columns=["hour"])
        self.diffusion_cols = [f"diffusion_h_{i}" for i in df_date_hour_drift.columns]
        df_date_hour_drift.columns = self.diffusion_cols
        self.df = self.df.merge(df_date_hour_drift, how="left", on="date")
        self.df[self.diffusion_cols] = self.df[self.diffusion_cols].multiply(self.df[self.drift_cols[0]], axis="index")

    def _add_x_two_label(self) -> NoReturn:
        df_date_hour_x = self.df.pivot(values=self.x_col_name, index=["date"], columns=["hour"])
        self.x_two_col_names = [f"{self.x_col_name}_2_h_{i}" for i in df_date_hour_x.columns]
        df_date_hour_x.columns = self.x_two_col_names
        self.df = self.df.merge(df_date_hour_x, how="left", on="date")

    def _get_group_by_hour_x(self) -> pd.core.groupby.generic.SeriesGroupBy:
        return self.df.groupby(["hour", self.x_col_name])[self.drift_cols[0]]

    def _compute_samples_by_hour_x(self) -> pd.DataFrame:
        return self._get_group_by_hour_x().count().unstack(-1)

    def _compute_mean_drift_by_hour_x(self) -> pd.DataFrame:
        drift_hour_x = self._get_group_by_hour_x().mean().unstack(-1).fillna(0)
        # fill missing x values with zero drift
        min_x = drift_hour_x.columns.min()
        max_x = drift_hour_x.columns.max()
        for col in list(set(np.arange(min_x, max_x, self.bin_size, dtype="float64")) - set(drift_hour_x.columns.tolist())):
            drift_hour_x[col] = 0
        drift_hour_x.sort_index(axis=1, inplace=True)
        return drift_hour_x

    def _compute_potential(self) -> pd.DataFrame:
        return (-1) * self.drift_hour_x.cumsum(axis=1)

    def _compute_diffusion_matrix(self) -> pd.DataFrame:
        return self.df.groupby(["hour"])[self.diffusion_cols].mean()

    def _compute_sqrt_diffusion_matrix(self) -> np.ndarray:
        return sqrtm(self.diffusion_matrix)

    def _compute_diff_xi_xj(self, i: int, j: int) -> pd.DataFrame:
        diff_df = self.df.set_index("hour").loc[i].groupby([
            self.x_col_name, self.x_two_col_names[j - 1]
        ])[self.diffusion_cols[j - 1]].mean().unstack(-1)

        min_col_x = diff_df.columns.min()
        min_row_x = diff_df.index.min()
        min_x = min(min_col_x, min_row_x)

        max_col_x = diff_df.columns.max()
        max_row_x = diff_df.index.max()
        max_x = max(max_col_x, max_row_x)

        # fill missing xi, xj values with zero diff
        for col in list(set(np.arange(min_x, max_x, self.bin_size, dtype="float64")) - set(diff_df.columns.tolist())):
            diff_df[col] = 0
        for row in list(set(np.arange(min_x, max_x, self.bin_size, dtype="float64")) - set(diff_df.index.tolist())):
            diff_df.loc[row] = 0
        diff_df.index.name = f"X_{i}"
        diff_df.columns.name = f"X_{j}"
        return diff_df.sort_index(axis=0).sort_index(axis=1)

    def _compute_diff_matrix_components(self) -> np.ndarray:
        diff_components_array = np.zeros(shape=(24, 24), dtype=pd.DataFrame)
        for i in range(1, 25):
            for j in range(1, i+1):
                diff_components_array[i-1, j-1] = self._compute_diff_xi_xj(i=i, j=j)
        return diff_components_array

    def plot_hourly_boxplot(self) -> NoReturn:
        fig, ax = plt.subplots(figsize=(20, 7))
        self.df_vector.T.boxplot(ax=ax)
        ax.set_title("Hourly price boxplot")
        ax.set_xlabel("hour")
        ax.set_ylabel("€/MWh")
        fig.show()

    def _plot_diffusion_2D(self, diff_df: pd.DataFrame, i: int, j: int) -> NoReturn:
        diff_df = diff_df.unstack(-1)
        fig, ax = plt.subplots()
        sns.heatmap(diff_df, cmap='crest', ax=ax)
        ax.set_xlabel(rf"$X_{{{j}}}$")
        ax.set_ylabel(rf"$X_{{{i}}}$")
        ax.set_title("Mean diffusion")
        fig.show()

    def _plot_diffusion_3D(self, diff_df: pd.DataFrame, i: int, j: int) -> NoReturn:
        diff_df = diff_df.reset_index()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 10))
        ax.scatter(diff_df.iloc[:, 0], diff_df.iloc[:, 1], diff_df.iloc[:, 2])
        ax.set_ylabel(rf"$X_{{{j}}}$")
        ax.set_xlabel(rf"$X_{{{i}}}$")
        ax.set_zlabel("Mean diffusion")
        fig.show()

    def plot_diffusion_element(self, i: int, j: int, method: FigureShape = FigureShape.TwoDim) -> NoReturn:
        diff_df = self.df.set_index("hour").loc[i].groupby([
            self.x_col_name, self.x_two_col_names[j-1]
        ])[self.diffusion_cols[j-1]].mean()

        if method == FigureShape.TwoDim:
            self._plot_diffusion_2D(diff_df=diff_df, i=i, j=j)
        elif method == FigureShape.ThreeDim:
            self._plot_diffusion_3D(diff_df=diff_df, i=i, j=j)
