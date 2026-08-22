import logging
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


class CovarianceEstimator:
    """Computes sample and regularized covariance matrices for portfolio optimization."""

    def __init__(self, returns: pd.DataFrame, annualize_factor: int = 252):
        self.returns = returns
        self.annualize_factor = annualize_factor
        self.tickers = list(returns.columns)

    def sample_covariance(self) -> pd.DataFrame:
        """Computes standard sample annualized covariance matrix."""
        cov = self.returns.cov() * self.annualize_factor
        return cov

    def ledoit_wolf_covariance(self) -> Tuple[pd.DataFrame, float]:
        """
        Computes Ledoit-Wolf Shrinkage annualized covariance matrix.
        Returns: (shrinkage_covariance_df, shrinkage_intensity_delta)
        """
        lw = LedoitWolf()
        lw.fit(self.returns.values)
        
        cov_matrix = lw.covariance_ * self.annualize_factor
        shrinkage_intensity = float(lw.shrinkage_)

        cov_df = pd.DataFrame(cov_matrix, index=self.tickers, columns=self.tickers)
        return cov_df, shrinkage_intensity

    def exponential_covariance(self, half_life: int = 63) -> pd.DataFrame:
        """
        Computes Exponentially Weighted Moving Average (EWMA) covariance matrix.
        half_life: Number of days for weight to decay to 50% (default: 63 days / 1 quarter).
        """
        decay = np.log(2) / half_life
        weights = np.exp(-decay * np.arange(len(self.returns))[::-1])
        weights /= weights.sum()

        mean_adj = self.returns - self.returns.mean()
        weighted_cov = np.zeros((len(self.tickers), len(self.tickers)))

        for t in range(len(self.returns)):
            r_t = mean_adj.iloc[t].values.reshape(-1, 1)
            weighted_cov += weights[t] * (r_t @ r_t.T)

        weighted_cov *= self.annualize_factor
        return pd.DataFrame(weighted_cov, index=self.tickers, columns=self.tickers)

    @staticmethod
    def calculate_condition_number(matrix: pd.DataFrame) -> float:
        """
        Calculates condition number kappa(Sigma) = lambda_max / lambda_min.
        High condition numbers indicate ill-conditioned matrices that amplify inversion errors.
        """
        eigenvalues = np.linalg.eigvalsh(matrix.values)
        min_eig = max(np.min(eigenvalues), 1e-12)
        max_eig = np.max(eigenvalues)
        return float(max_eig / min_eig)