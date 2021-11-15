from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd

from numba import jit
from scipy import stats


class ComputationMode(Enum):
    Numba = "numba"
    Simple = "simple"


class GaussianKernel:

    def __init__(self,
                 samples: pd.DataFrame,
                 grid_shape: Tuple[int, ...],
                 xmin: float = None, xmax: float = None,
                 ymin: float = None, ymax: float = None,
                 zmin: float = None, zmax: float = None,
                 computation_mode: ComputationMode = ComputationMode.Simple
                 ):

        self.computation_mode = computation_mode

        self.kernel, self.samples = self._fit_gaussian_kernel(samples_df=samples)
        mesh, grid = self._generate_mesh(grid_shape=grid_shape,
                                         xmin=xmin, xmax=xmax,
                                         ymin=ymin, ymax=ymax,
                                         zmin=zmin, zmax=zmax)
        p = self._compute_mesh_prob(mesh=mesh, new_shape=grid[0].shape)
        self.expected_value_function = self._compute_expected_value_function(grid=grid, p=p)
        self.expected_value = self._compute_expected_value()
        self.most_likely, most_likely_indexes = self._get_most_likely(grid=grid, p=p)
        self.expected_value_from_most_likely = self._get_expected_value_from_most_likely(
            indexes=most_likely_indexes, grid=grid
        )

    def _fit_gaussian_kernel(self, samples_df: pd.DataFrame) -> Tuple[stats.kde.gaussian_kde, np.ndarray]:
        values = samples_df.values.T
        kernel = stats.gaussian_kde(values)
        return kernel, values

    def _generate_mesh(self,
                       grid_shape: Tuple[int, ...],
                       xmin: float, xmax: float,
                       ymin: float, ymax: float,
                       zmin: float, zmax: float) -> [np.ndarray, Tuple[np.ndarray, ...]]:
        #TODO: Refactor method

        xmin = self.samples[0].min().round(2) if not xmin else xmin
        xmax = self.samples[0].max().round(2) if not xmax else xmax

        ymin = self.samples[1].min().round(2) if not ymin else ymin
        ymax = self.samples[1].max().round(2) if not ymax else ymax

        x_mesh = np.linspace(xmin, xmax, grid_shape[0])
        y_mesh = np.linspace(ymin, ymax, grid_shape[1])

        if self.samples.shape[0] == 3:
            zmin = self.samples[2].min().round(2) if not zmin else zmin
            zmax = self.samples[2].max().round(2) if not zmax else zmax

            z_mesh = np.linspace(zmin, zmax, grid_shape[2])

            X, Y, Z = np.meshgrid(*(x_mesh, y_mesh, z_mesh), indexing="ij")
            mesh = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

            return mesh, (X, Y, Z)

        else:
            X, Y = np.meshgrid(*(x_mesh, y_mesh), indexing="ij")
            mesh = np.vstack([X.ravel(), Y.ravel()])

            return mesh, (X, Y)

    def _compute_mesh_prob(self, mesh: np.ndarray, new_shape: Tuple[int, ...]) -> np.ndarray:
        return np.reshape(self.kernel(mesh).T, new_shape)

    @staticmethod
    def _compute_expectation(values: np.ndarray, weights: np.ndarray) -> float:
        return np.average(values, weights=weights)

    @staticmethod
    @jit(nopython=True)
    def _compute_expectation_numba(grid: Tuple[np.ndarray, ...], p: np.ndarray) -> List[np.ndarray]:
        # TODO: Refactor method

        if len(grid) == 2:
            x_range = grid[0][:, 0]
            expected_value_x = np.zeros_like(x_range)
            for i in range(x_range.shape[0]):
                expected_value_x[i] = sum(grid[1][i] * p[i]) / sum(p[i])

            return [x_range, expected_value_x]

        elif len(grid) == 3:
            x, y, z = grid[0], grid[1], grid[2]
            x_range = x[:, :, 0]
            y_range = y[:, :, 0]

            expected_value_x_y = np.zeros_like(x_range.ravel())
            for i in range(x.shape[0]):
                for j in range(y[i].shape[0]):
                    # if sum of probabilities across z is 0 do not compute average
                    if sum(p[i][j]) == 0:
                        expected_value_x_y[i * x.shape[1] + j] = 0
                    else:
                        expected_value_x_y[i * x.shape[1] + j] = sum(z[i][j] * p[i][j]) / sum(p[i][j])

            return [x_range.ravel(), y_range.ravel(), expected_value_x_y]

    def _compute_expected_value_function(self, grid: Tuple[np.ndarray, ...], p: np.ndarray) -> np.ndarray:

        if self.computation_mode == ComputationMode.Numba:
            return np.vstack(self._compute_expectation_numba(grid=grid, p=p))

        elif self.computation_mode == ComputationMode.Simple:
            return np.vstack(self._compute_expectation_numba.py_func(grid=grid, p=p))

    def _compute_expected_value(self) -> float:
        p_expected_elements = self.kernel(self.expected_value_function)
        return np.average(self.expected_value_function[-1, :], weights=p_expected_elements)

    def _get_most_likely(self, grid: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        #TODO: Refactor method
        indexes = np.unravel_index(p.argmax(), p.shape)

        if len(indexes) == 2:
            x_index, y_index = indexes[0], indexes[1]
            x, y = grid[0], grid[1]
            most_likely = np.array([
                x[x_index][y_index],
                y[x_index][y_index]
            ])
            return most_likely, indexes

        if len(indexes) == 3:
            x_index, y_index, z_index = indexes[0], indexes[1], indexes[2]
            x, y, z = grid[0], grid[1], grid[2]
            most_likely = np.array([
                x[x_index][y_index][z_index],
                y[x_index][y_index][z_index],
                z[x_index][y_index][z_index]
            ])

            return most_likely, indexes

    def _get_expected_value_from_most_likely(self,
                                             indexes: Tuple[int, ...],
                                             grid: Tuple[np.ndarray, ...]) -> np.ndarray:

        if len(indexes) == 2:
            x_index = indexes[0]
            return self.expected_value_function[:, x_index]

        if len(indexes) == 3:
            x_index, y_index = indexes[0], indexes[1]
            x = grid[0]
            return self.expected_value_function[:, x_index*x.shape[1]+y_index]

