import numpy as np
import pandas as pd

from stochastic.coefficients import Coefficient, SpatialDriftMultivariate, ConstantDiffusionMultivariate


class EulerMaruyama:

    def __init__(self,
                 num_sim: int,
                 period: int,
                 delta_t: float,
                 nu: float,
                 drift_df: pd.DataFrame,
                 diffusion_df: pd.DataFrame
                 ):

        self.num_sim = num_sim
        self.period = period
        self.delta_t = delta_t
        self.nu = nu
        self.drift = self._load_drift(df=drift_df)
        self.diffusion = self._load_diffusion(df=diffusion_df)

    def _load_drift(self, df: pd.DataFrame) -> Coefficient:
        return SpatialDriftMultivariate(df=df)

    def _load_diffusion(self, df: pd.DataFrame) -> Coefficient:
        return ConstantDiffusionMultivariate(df=df)

    def simulate(self, x0: np.ndarray, random_seed: float = 0):

        sim_df = pd.DataFrame()

        np.random.seed(seed=random_seed)
        x0_dim = x0.shape[0]

        for i in range(self.num_sim):
            y = np.zeros((x0_dim, self.period))
            y[:, 0] = x0
            for t in range(1, self.period):
                y_t_1 = y[:, t-1]
                y_t = y_t_1 + self.drift.get_value(x=y, t=t)*self.delta_t + \
                      np.dot(self.diffusion.get_value(x=y, t=t),
                             np.random.normal(loc=self.nu, scale=self.delta_t, size=(x0_dim, 1))
                             ).reshape(-1)
                y_t = np.where(y_t < 0, 0, y_t)
                y[:, t] = y_t
            sim_df[i] = y.flatten("F")

        return sim_df
