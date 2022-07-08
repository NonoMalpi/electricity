import warnings

from typing import Tuple

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from stochastic.coefficients import Coefficient

warnings.simplefilter(action='ignore', category=FutureWarning)


class EulerMaruyama:
    """ Numerical solution of Stochastic Differential Equation (SDE) through Euler-Maruyama method.

    Considering a SDE of the form: dX_t = a(X_t, t)dt + b(X_t, t)dW_t, the solution of this SDE over
    a time interval [0, T] can be approximated as follows:
                    X_{t+1} = X_t + a(X_t, t)Delta_t + b(X_t, t)DeltaW_t
    with initial condition X_0 = x_0 and where the time interval is discretised:
                    0 = tau_0 < tau_1 <  ... < tau_t < ... < tau_N = T
    with Delta_t = tau_{t+1} - tau_t = T / N and DeltaW_t = W_{t+1} - W_t ~ N(0, Delta_t^{1/2})
    because W_t is a Wiener process.

    Parameters
    ----------
    num_sim: int
        Number of random trajectories to simulate.

    periods: int
        Number of steps for each trajectory.

    delta_t: float
        Interval of time / periods.

    nu: float
        Mean value of the normal random variable.

    drift: Coefficient
        The expected drift.

    diffusion: Coefficient
        The diffusion that scales the random variable.

    Attributes
    ----------
    num_sim: int
        Number of random trajectories to simulate.

    periods: int
        Number of steps for each trajectory.

    delta_t: float
        Interval of time / periods.

    sqrt_delta_t: float
        Squared root of delta_t, this is the standard deviation of the normal random variable.

    nu: float
        Mean value of the normal random variable.

    drift: Coefficient
        The expected drift.

    diffusion: Coefficient
        The diffusion that scales the random variable.

    Methods
    -------
    simulate
    """

    def __init__(self,
                 num_sim: int,
                 periods: int,
                 delta_t: float,
                 nu: float,
                 drift: Coefficient,
                 diffusion: Coefficient):

        self.num_sim = num_sim
        self.periods = periods
        self.delta_t = delta_t
        self.sqrt_delta_t = np.sqrt(delta_t)
        self.nu = nu
        self.drift = drift
        self.diffusion = diffusion

    def _check_x0_input(self, x0: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """ Rearrange initial condition shape if one single time step is provided.

        Parameters
        ----------
        x0: np.ndarray (n_dim, ) or (n_dim, t)
            Initial value with (n_dim,) for one single time step or (n_dim, t) for multiple time steps.

        Returns
        -------
        x0: np.ndarray
            Initial condition with shape (n_dim, t), if only one time step is provided, t = 1.

        x0_dim: float
            The number of dimensions = n_dim.

        x0_time_step: float
            The number of time step after the initial condition.
        """

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
        """ Initialise simulation array.

        Parameters
        ----------
        x0: np.ndarray
            Initial condition with shape (n_dim, t), if only one time step is provided, t = 1.

        x0_dim: float
            The number of dimensions = n_dim.

        x0_time_step: float
            The number of time step after the initial condition.

        Returns
        -------
        y: np.ndarray
            The simulation array with the initial condition, shape = (n_dim, periods)
        """
        y = np.zeros((x0_dim, self.periods))
        y[:, :x0_time_step] = x0

        return y

    def _simulate_path(self, x0: np.ndarray, x0_dim: int, x0_time_step: int, seed: int) -> np.ndarray:
        """ Simulate a single trajectory through Euler-Maruyama method.

        Parameters
        ----------
        x0: np.ndarray
            Initial condition with shape (n_dim, t), if only one time step is provided, t = 1.

        x0_dim: float
            The number of dimensions = n_dim.

        x0_time_step: float
            The number of time step after the initial condition.

        seed: int
            The random seed to generate the path.

        Returns
        -------
        np.ndarray
            The resulting trajectory, shape = (n_dim * self.periods, )
        """
        np.random.seed(seed=seed)
        y = self._set_y_array(x0=x0, x0_dim=x0_dim, x0_time_step=x0_time_step)

        for t in range(x0_time_step, self.periods):
            y_t_1 = y[:, t - 1]

            y_t = y_t_1 + self.drift.get_value(x=y, t=t) * self.delta_t + \
                np.dot(self.diffusion.get_value(x=y, t=t),
                       np.random.normal(loc=self.nu, scale=self.sqrt_delta_t, size=(x0_dim, 1))
                ).reshape(-1)

            # boundary condition: if price plunges beyond 0, set 0
            y_t = np.where(y_t < 0, 0, y_t)

            y[:, t] = y_t
        return y.flatten("F")

    def simulate(self, x0: np.ndarray, random_seed: float = 0) -> pd.DataFrame:
        """ Simulate several trajectories of the SDE for the initial condition provided.

        Parameters
        ----------
        x0: np.ndarray (n_dim, ) or (n_dim, t)
            Initial value with (n_dim,) for one single time step or (n_dim, t) for multiple time steps.

        random_seed: float
            The random seed to use, default = 0.

        Returns
        -------
        sim_df: pd.DataFrame
            A dataframe with the resulting trajectories, shape = (n_dim * self.periods, self.num_sim)
        """

        # generate seeds to keep reproducible simulations working with Joblib
        # source: https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
        np.random.seed(seed=random_seed)
        random_seeds = np.random.randint(np.iinfo(np.int32).max, size=self.num_sim)

        x0, x0_dim, x0_time_step = self._check_x0_input(x0=x0)

        paths = Parallel(n_jobs=-1, verbose=0)(
            delayed(self._simulate_path)(x0=x0, x0_dim=x0_dim, x0_time_step=x0_time_step, seed=seed)
            for seed in random_seeds
        )

        sim_df = pd.DataFrame(paths).T

        return sim_df
