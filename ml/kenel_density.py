from enum import Enum
from typing import List, Tuple

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from numba import jit
from scipy import stats


class ComputationMode(Enum):
    Numba = "numba"
    Simple = "simple"


class GaussianKernel:
    """ Fit a Gaussian Kernel density estimation over a dataset and extract features from the probability density function.

    The input dataset has shape (observations, dimensions) (n, d).
    This class supports up to 3 dimensions. The last column is treated as a dependent variable while
    the rest of (d-1) dimensions are independent variables. The resulting kernel acts as an estimation
    of the joint probability density function (PDF). Then, the expected and most likely values of the
    dependent variable are computed sampling from the resulting PDF. For such purpose, a d-dimensional mesh
    is generated.

    Parameters
    ----------
    samples: pd.DataFrame
        The observations to fit the Gaussian Kernel, shape = (observations, dimensions).
        The last dimension (column) corresponds to the dependent variable we are interested to
        extract expected and most likely values. The first (n-1)th dimensions are the independent variables.

    grid_shape: Tuple[int, ...]
        Tuple containing the number of points per dimension to sample from the fitted kernel.
        Each number must be an integer value times the mesh_chunks parameter.

    x_min: float
        The minimum value of the first dimension to generate the sampling grid, default = None.
        By default, the minimum value of the first dimension from the samples provided is used.

    x_max: float
        The maximum value of the first dimension to generate the sampling grid, default = None.
        By default, the maximum value of the first dimension from the samples provided is used.

    y_min: float
        The minimum value of the second dimension to generate the sampling grid, default = None.
        By default, the minimum value of the second dimension from the samples provided is used.

    y_max: float
        The maximum value of the second dimension to generate the sampling grid, default = None.
        By default, the maximum value of the second dimension from the samples provided is used.

    z_min: float
        The minimum value of the third dimension to generate the sampling grid, default = None.
        By default, the minimum value of the third dimension from the samples provided is used.

    z_max: float
        The maximum value of the third dimension to generate the sampling grid, default = None.
        By default, the maximum value of the third dimension from the samples provided is used.

    computation_mode: ComputationMode
        Compute the expected value function using Numpy or Numba.

    mesh_chunks: int
        Number of partitions of the mesh to sample and compute the probability density function.
        It must be a divisor of each element of the grid_shape parameter.

    Attributes
    ----------
    computation_mode: ComputationMode
        Compute the expected value function using Numpy or Numba.

    mesh_chunks: int
        Number of partitions of the mesh to sample and compute the probability density function.
        It must be a divisor of each element of the grid_shape parameter.

    kernel: stats.kde.gaussian_kde
        The fitted Gaussian Kernel.

    samples: np.ndarray
        Transposed ndarray of the input dataset, shape = (dimensions, observations).
        The last dimension corresponds to the dependent variable.

    grid: Tuple[ndarray, ...]
        Tuple of n elements containing ndarrays of shape grid_shape with values for the dth dimension.

    p: np.ndarray
        Ndarray containing the probability over the sampled mesh, shape = grid_shape.

    expected_value_function: np.ndarray
        Expected value of the dependent variable as function of the (d-1)th independent variables,
        shape = (number of independent variables + 1, product of grid_shape elements for independent variables.

    expected_value: float
        Expected value of the dependent variable.

    most_likely: np.ndarray
        Most likely value extracted from the maximum of p, shape = (number of dimensions, ).

    expected_value_from_most_likely: np.ndarray
        Expected value of the dependent variable for the most likely values of the independent variables,
        shape = (number of dimensions, )
    """

    def __init__(self,
                 samples: pd.DataFrame,
                 grid_shape: Tuple[int, ...],
                 xmin: float = None, xmax: float = None,
                 ymin: float = None, ymax: float = None,
                 zmin: float = None, zmax: float = None,
                 computation_mode: ComputationMode = ComputationMode.Simple,
                 mesh_chunks: int = 8
                 ):

        self.computation_mode = computation_mode
        self.mesh_chunks = mesh_chunks

        self.kernel, self.samples = self._fit_gaussian_kernel(samples_df=samples)
        mesh, self.grid = self._generate_mesh(grid_shape=grid_shape,
                                         xmin=xmin, xmax=xmax,
                                         ymin=ymin, ymax=ymax,
                                         zmin=zmin, zmax=zmax)
        self.p = self._compute_mesh_prob(mesh=mesh)
        self.expected_value_function = self._compute_expected_value_function()
        self.expected_value = self._compute_expected_value()
        self.most_likely, most_likely_indexes = self._get_most_likely()
        self.expected_value_from_most_likely = self._get_expected_value_from_most_likely(
            indexes=most_likely_indexes
        )

    def _fit_gaussian_kernel(self, samples_df: pd.DataFrame) -> Tuple[stats.kde.gaussian_kde, np.ndarray]:
        """ Fit the Gaussian kernel to the input samples.

        Parameters
        ----------
        samples_df: pd.DataFrame
            The observations to fit the Gaussian Kernel, shape = (observations, dimensions).

        Returns
        -------
        kernel: stats.kde.gaussian_kde
            The fitted Gaussian kernel.

        values: np.array
            The samples used to fit the kernel, shape = (dimensions, observations).
        """
        values = samples_df.values.T
        kernel = stats.gaussian_kde(values)
        return kernel, values

    def _generate_mesh(self,
                       grid_shape: Tuple[int, ...],
                       xmin: float, xmax: float,
                       ymin: float, ymax: float,
                       zmin: float, zmax: float) -> [np.ndarray, Tuple[np.ndarray, ...]]:
        """ Create a mesh of shape grid_shape.

        Parameters
        ----------
        grid_shape: Tuple[int, ...]
            Tuple containing the number of points per dimension to sample from the fitted kernel.
            Each number must be an integer value times the mesh_chunks parameter.

        x_min: float
            The minimum value of the first dimension to generate the sampling grid, default = None.
            By default, the minimum value of the first dimension from the samples provided is used.

        x_max: float
            The maximum value of the first dimension to generate the sampling grid, default = None.
            By default, the maximum value of the first dimension from the samples provided is used.

        y_min: float
            The minimum value of the second dimension to generate the sampling grid, default = None.
            By default, the minimum value of the second dimension from the samples provided is used.

        y_max: float
            The maximum value of the second dimension to generate the sampling grid, default = None.
            By default, the maximum value of the second dimension from the samples provided is used.

        z_min: float
            The minimum value of the third dimension to generate the sampling grid, default = None.
            By default, the minimum value of the third dimension from the samples provided is used.

        z_max: float
            The maximum value of the third dimension to generate the sampling grid, default = None.
            By default, the maximum value of the third dimension from the samples provided is used.

        Returns
        -------
        mesh: np.ndarray
            Ndarray containing the values of the mesh, shape = (d, product of elements in grid_shape parameter).

        Tuple[np.ndarray, ...]
            Tuple of n elements containing ndarrays of shape grid_shape with values for the dth dimension.
        """
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

    def _compute_mesh_chunk_prob(self, mesh_chunk: np.ndarray) -> np.ndarray:
        """ Return the estimated probability for the mesh chunk.

        Parameters
        ----------
        mesh_chunk: np.ndarray
            Ndarray containing the values of the mesh, shape = (d, product of elements in grid_shape parameter / mesh_chunks).

        Returns
        -------
        np.ndarray
            Probability array, shape = (product of elements in grid_shape parameter / mesh_chunks, )
        """
        return self.kernel(mesh_chunk)

    def _compute_mesh_prob(self, mesh: np.ndarray) -> np.ndarray:
        """ Sample the fitted kernel with the constructed mesh to approximate the PDF.

        Parameters
        ----------
        mesh: np.ndarray
            Ndarray containing the values of the mesh, shape = (d, product of elements in grid_shape parameter).

        Returns
        -------
        np.ndarray
            Ndarray containing the probability over the sampled mesh, shape = grid_shape.
        """
        mesh_chunks_array = np.split(mesh, indices_or_sections=self.mesh_chunks, axis=1)
        prob_chunks = Parallel(n_jobs=-1, verbose=1)(
            delayed(self._compute_mesh_chunk_prob)(mesh_chunk=chunk)
            for chunk in mesh_chunks_array
        )

        prob = np.concatenate(prob_chunks)
        return np.reshape(prob.T, self.grid[0].shape)

    @staticmethod
    @jit(nopython=True)
    def _compute_expectation_numba(grid: Tuple[np.ndarray, ...], p: np.ndarray) -> List[np.ndarray]:
        """ Compute the expected value of the dependent variable as function of the independent variables.

        Parameters
        ----------
        grid: Tuple[ndarray, ...]
            Tuple of n elements containing ndarrays of shape grid_shape with values for the dth dimension.

        p: np.ndarray
            Ndarray containing the probability over the sampled mesh, shape = grid_shape.

        Returns
        -------
        List[np.ndarray]
            List of arrays with values of the independent variables, the last array correspond to the expected
            value of the dependent variable sampled from the PDF.
        """
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

    def _compute_expected_value_function(self) -> np.ndarray:
        """ Obtain the expected value of the dependent variable as a function of the independent variables.

        Returns
        -------
        np.ndarray
            Expected value of the dependent variable as function of the (d-1)th independent variables,
            shape = (number of independent variables + 1, product of grid_shape elements for independent variables.
        """

        if self.computation_mode == ComputationMode.Numba:
            return np.vstack(self._compute_expectation_numba(grid=self.grid, p=self.p))

        elif self.computation_mode == ComputationMode.Simple:
            return np.vstack(self._compute_expectation_numba.py_func(grid=self.grid, p=self.p))

    def _compute_expected_value(self) -> float:
        """ Obtain expected value of the dependent variable using the expected value function.

        Returns
        -------
        float
            Expected value of the dependent variable.
        """
        p_expected_elements = self.kernel(self.expected_value_function)
        return np.average(self.expected_value_function[-1, :], weights=p_expected_elements)

    def _get_most_likely(self) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """ Returns the datapoint with the maximum likelihood value and the indexes indicating that point.

        Returns
        -------
        most_likely: np.ndarray
            Most likely value extracted from the maximum of p, shape = (number of dimensions, ).

        indexes: Tuple[int, ...]
            Tuple indicating the indexes of p at which the maximum value is reached.
        """
        #TODO: Refactor method
        indexes = np.unravel_index(self.p.argmax(), self.p.shape)

        if len(indexes) == 2:
            x_index, y_index = indexes[0], indexes[1]
            x, y = self.grid[0], self.grid[1]
            most_likely = np.array([
                x[x_index][y_index],
                y[x_index][y_index]
            ])
            return most_likely, indexes

        if len(indexes) == 3:
            x_index, y_index, z_index = indexes[0], indexes[1], indexes[2]
            x, y, z = self.grid[0], self.grid[1], self.grid[2]
            most_likely = np.array([
                x[x_index][y_index][z_index],
                y[x_index][y_index][z_index],
                z[x_index][y_index][z_index]
            ])

            return most_likely, indexes

    def _get_expected_value_from_most_likely(self, indexes: Tuple[int, ...]) -> np.ndarray:
        """ Return the expected value of dependent variable for the most likely value of the independent variables.

        Parameters
        ----------
        indexes: Tuple[int, ...]
            Tuple indicating the indexes of p at which the maximum value is reached.

        Returns
        -------
        np.ndarray
            Expected value from the expected value function for the most likely independent variables.
        """

        if len(indexes) == 2:
            x_index = indexes[0]
            return self.expected_value_function[:, x_index]

        if len(indexes) == 3:
            x_index, y_index = indexes[0], indexes[1]
            x = self.grid[0]
            return self.expected_value_function[:, x_index*x.shape[1]+y_index]
