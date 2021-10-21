from typing import NoReturn

import numpy as np
import pandas as pd


class Coefficient:

    def __init__(self, df: pd.DataFrame):

        self.df = df

    def _check_dimensions(self) -> NoReturn:
        if self.df.shape[1] == 1:
            self.df = self.df.T

    def _rename_index_names(self) -> NoReturn:
        num_dim = len(self.df.index)
        self.df.index = range(1, num_dim+1)

    def get_value(self, x, t):
        pass


class SpatialDriftMultivariate(Coefficient):

    def __init__(self, df: pd.DataFrame):
        super(SpatialDriftMultivariate, self).__init__(df=df)
        self._check_dimensions()
        self._rename_index_names()

    def get_value(self, x: np.ndarray, t: int) -> np.ndarray:
        x = x[:, t-1].round()
        return self.df.lookup(self.df.index, x)


class SpatialDifferenceDriftMultivariate(Coefficient):

    def __init__(self, df: pd.DataFrame):
        super(SpatialDifferenceDriftMultivariate, self).__init__(df=df)
        self._check_dimensions()
        self._rename_index_names()

    def get_value(self, x: np.ndarray, t: int) -> np.ndarray:
        x = (x[:, t-1] - x[:, t-2]).round()
        return self.df.lookup(self.df.index, x)


class SpatialDiffusionUnivariate(Coefficient):

    def __init__(self, df: pd.DataFrame):
        super(SpatialDiffusionUnivariate, self).__init__(df=df)
        self._check_dimensions()
        self._rename_index_names()
        self.values = self.df.values

    def get_value(self, x: np.ndarray, t: int) -> np.ndarray:
        x = x[:, t-1].round()
        value = self.df.lookup(self.df.index, x)
        return np.sqrt(value)


class ConstantDiffusionMultivariate(Coefficient):

    def __init__(self, df: pd.DataFrame):
        super(ConstantDiffusionMultivariate, self).__init__(df=df)
        self._check_square_matrix()
        self.values = self.df.values

    def _check_square_matrix(self):
        n_row = self.df.shape[0]
        n_col = self.df.shape[1]
        assert n_row == n_col, f"Diffusion matrix must be squared, ({n_row}, {n_col}) matrix provided."

    def get_value(self, x: np.ndarray, t: int) -> np.ndarray:
        return self.values
