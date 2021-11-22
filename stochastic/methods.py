from typing import Tuple

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from stochastic.coefficients import Coefficient


class EulerMaruyama:

    def __init__(self,
                 num_sim: int,
                 periods: int,
                 delta_t: float,
                 nu: float,
                 drift: Coefficient,
                 diffusion: Coefficient
                 ):

        self.num_sim = num_sim
        self.periods = periods
        self.delta_t = delta_t
        self.sqrt_delta_t = np.sqrt(delta_t)
        self.nu = nu
        self.drift = drift
        self.diffusion = diffusion

    def _check_x0_input(self, x0: np.ndarray) -> Tuple[np.ndarray, int, int]:

        if x0.ndim == 1:
            x0 = x0.reshape(-1, 1)
        elif x0.ndim == 2:
            pass
        else:
            raise ValueError(f"Wrong x0 dimension: {x0.shape}")

        x0_dim = x0.shape[0]
        x0_time_step = x0.shape[1]

        return x0, x0_dim, x0_time_step

    def _set_y_array(self, x0: np.ndarray, x0_dim: int, x0_time_step: int) -> np.ndarray:
        y = np.zeros((x0_dim, self.periods))
        y[:, :x0_time_step] = x0

        return y

    def simulate_path(self, x0: np.ndarray, x0_dim: int, x0_time_step: int) -> np.ndarray:
        y = self._set_y_array(x0=x0, x0_dim=x0_dim, x0_time_step=x0_time_step)
        for t in range(x0_time_step, self.periods):
            y_t_1 = y[:, t - 1]
            y_t = y_t_1 + self.drift.get_value(x=y, t=t) * self.delta_t + \
                  (1 / 2) * np.dot(self.diffusion.get_value(x=y, t=t),
                                   np.random.normal(loc=self.nu, scale=self.sqrt_delta_t, size=(x0_dim, 1))
                                   ).reshape(-1)
            y_t = np.where(y_t < 0, 0, y_t)
            y[:, t] = y_t
        return y.flatten("F")

    def simulate(self, x0: np.ndarray, random_seed: float = 0) -> pd.DataFrame:
        """

        x0: np.ndarray (n_dim, ) or (n_dim, t)
            Initial value with (n_dim,) for one single timestep or (n_dim, t) for multiple timesteps
        """

        np.random.seed(seed=random_seed)
        x0, x0_dim, x0_time_step = self._check_x0_input(x0=x0)

        paths = Parallel(n_jobs=-1, verbose=0)(
            delayed(self.simulate_path)(x0=x0, x0_dim=x0_dim, x0_time_step=x0_time_step)
            for _ in range(self.num_sim)
        )

        sim_df = pd.DataFrame(paths).T

        return sim_df

