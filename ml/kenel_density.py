from typing import Tuple

import numpy as np
import pandas as pd

from scipy import stats


class GaussianKernel:

    def __init__(self,
                 samples: pd.DataFrame,
                 grid_shape: Tuple[int, ...],
                 xmin: float = None, xmax: float = None,
                 ymin: float = None, ymax: float = None,
                 zmin: float = None, zmax: float = None
                 ):

        self.kernel, self.samples = self._fit_gaussian_kernel(samples_df=samples)
        mesh, grid = self._generate_mesh(grid_shape=grid_shape,
                                         xmin=xmin, xmax=xmax,
                                         ymin=ymin, ymax=ymax,
                                         zmin=zmin, zmax=zmax)
        p = self._compute_mesh_prob(mesh=mesh, new_shape=grid[0].shape)
        self.expected_value_function = self._compute_expected_value_function(grid=grid, p=p)
        self.expected_value = self._compute_expected_value()

    def _fit_gaussian_kernel(self, samples_df: pd.DataFrame) -> Tuple[stats.kde.gaussian_kde, np.ndarray]:
        values = samples_df.values.T
        kernel = stats.gaussian_kde(values)
        return kernel, values

    def _generate_mesh(self,
                       grid_shape: Tuple[int, ...],
                       xmin: float, xmax: float,
                       ymin: float, ymax: float,
                       zmin: float, zmax: float) -> [np.ndarray, Tuple[np.ndarray, ...]]:

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

    def _compute_mesh_prob(self, mesh: np.ndarray, new_shape: Tuple[int, ...]):
        return np.reshape(self.kernel(mesh).T, new_shape)

    def _compute_expected_value_function(self, grid: Tuple[np.ndarray, ...], p: np.ndarray):

        if len(grid) == 2:
            x_range = grid[0][:, 0]
            expected_value_x = np.zeros_like(x_range)
            for i in range(x_range.shape[0]):
                expected_value_x[i] = np.average(grid[1][i], weights=p[i])

            return np.vstack([x_range, expected_value_x])
            #return pd.Series(expected_value_x, index=x_range)

        elif len(grid) == 3:
            x, y, z = grid[0], grid[1], grid[2]
            x_range = x[:, :, 0]
            y_range = y[:, :, 0]

            expected_value_x_y = np.zeros(x_range.ravel())
            for i in range(x.shape[0]):
                for j in range(y[i].shape[0]):
                    if np.sum(p[i][j]) == 0:
                        expected_value_x_y[i*x.shape[0] + j] = 0
                    else:
                        expected_value_x_y[i*x.shape[0] + j] = np.average(z[i][j], weights=p[i][j])

            return np.vstack([x_range.ravel(), y_range.ravel(), expected_value_x_y])

    def _compute_expected_value(self):
        p_expected_elements = self.kernel(self.expected_value_function)
        return np.average(self.expected_value_function[-1, :], weights=p_expected_elements)
