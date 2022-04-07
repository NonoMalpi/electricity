from typing import Dict

import numpy as np
import pandas as pd

from ml import GaussianKernel


class KramersMoyal:

    @staticmethod
    def obtain_first_coefficient_vector(kernels_dict: Dict[int, GaussianKernel]) -> pd.DataFrame:
        """ Generate the 1st Kramers-Moyal coefficient (drift vector) from the first moments of the fitted kernels.

        This method assumes the drift is a function of the independent variable.

        Params
        ------
        kernels_dict: Dict[int, GaussianKernel]
            Dictionary containing the fitted kernels with the expected values.

        Returns
        -------
        first_km_kde_df: pd.DataFrame
            The expected drift vector as function of the independent variable.
            The number of rows is equal to the components (hours)
            The number of columns depends on the grid of the fitted kernel.
        """
        first_km_df_list = []
        hours = list(kernels_dict.keys())
        hours.sort()
        for h in hours:
            first_km_df_list.append(
                pd.DataFrame(data={h: kernels_dict[h].expected_value_function[1, :]}).T
            )

        first_km_kde_df = pd.concat(first_km_df_list)
        first_km_kde_df.columns = kernels_dict[hours[0]].expected_value_function[0, :]

        return first_km_kde_df

    @staticmethod
    def obtain_second_coefficient_matrix(diag_kernels_dict: Dict[int, GaussianKernel],
                                         non_diag_kernels_dict: Dict[int, Dict[int, GaussianKernel]]) -> np.ndarray:
        """ Generate the 2nd Kramers-Moyal coefficient matrix (proportional to squared diffusion matrix)
        from the second moments of the fitted kernels.

        This method assumes the diffusion is constant.
        The diffusion matrix will be the squared root of 2 times this result.

        Parameters
        ----------
        diag_kernels_dict: Dict[int, GaussianKernel]
            Dictionary containing the fitted kernels for elements in the diagonal, h_i = h_j.

        non_diag_kernels_dict: Dict[int, Dict[int, GaussianKernel]]
            Dictionary containing the hour-fitted kernels dictionaries for elements not in the diagonal, h_i <> h_j.

        Returns
        -------
        second_km_kde_matrix: np.ndarray
            The constant 2nd Kramers-Moyal coefficients.
        """

        second_km_kde_matrix = np.zeros((24, 24))

        for hour_i, v in non_diag_kernels_dict.items():
            for hour_j, v_j in v.items():
                second_km_kde_matrix[hour_i - 1, hour_j - 1] = v_j.expected_value

        # fill symmetric triangular matrix
        second_km_kde_matrix = second_km_kde_matrix + np.tril(second_km_kde_matrix, k=-1).T

        for hour_i, v in diag_kernels_dict.items():
            second_km_kde_matrix[hour_i - 1][hour_i - 1] = v.expected_value

        # add 1/2 to consider the second Kramers-Moyal coefficient definition
        second_km_kde_matrix = 1 / 2 * second_km_kde_matrix

        return second_km_kde_matrix
