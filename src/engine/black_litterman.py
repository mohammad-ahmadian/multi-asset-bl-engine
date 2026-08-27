import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BlackLittermanModel:
    """
    Executes Black-Litterman Bayesian asset allocation calculations.
    Combines Market Equilibrium returns with tactical investor views.
    """

    def __init__(
        self,
        implied_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        tau: float = 0.05
    ):
        self.tickers = list(covariance_matrix.columns)
        self.Pi = implied_returns.reindex(self.tickers).values.reshape(-1, 1)
        self.Sigma = covariance_matrix.reindex(index=self.tickers, columns=self.tickers).values
        self.tau = tau
        self.n = len(self.tickers)

    def calibrate_omega_idzorek(self, P: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """
        Calibrates view uncertainty matrix Omega using Idzorek's method:
        Omega_ii = P_i * (tau * Sigma) * P_i^T * ((1 - c_i) / c_i)
        """
        k = P.shape[0]
        Omega = np.zeros((k, k))
        tau_sigma = self.tau * self.Sigma

        for i in range(k):
            p_row = P[i : i + 1, :]  # 1 x N
            # Extract scalar cleanly using .item() to support all NumPy versions
            view_variance = float((p_row @ tau_sigma @ p_row.T).item())
            c = float(confidences[i].item())
            
            # Idzorek confidence scaling
            c_clamped = float(np.clip(c, 0.01, 0.9999))
            uncertainty_scale = (1.0 - c_clamped) / c_clamped
            Omega[i, i] = max(view_variance * uncertainty_scale, 1e-6)

        return Omega

    def calculate_posterior(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Computes Black-Litterman posterior expected returns E[R] and posterior covariance Sigma_post.
        """
        k = P.shape[0]

        # Case 1: No views specified -> Posterior = Prior
        if k == 0:
            return (
                pd.Series(self.Pi.flatten(), index=self.tickers, name="posterior_expected_return"),
                pd.DataFrame(self.Sigma, index=self.tickers, columns=self.tickers),
                pd.DataFrame()
            )

        # 1. Calibrate Omega
        Omega = self.calibrate_omega_idzorek(P, confidences)

        tau_sigma = self.tau * self.Sigma  # (N x N)
        tau_sigma_pt = tau_sigma @ P.T     # (N x K)

        # 2. Inversion core: [P * (tau * Sigma) * P^T + Omega]
        M_inv = P @ tau_sigma_pt + Omega    # (K x K)
        M = np.linalg.inv(M_inv)

        # 3. Posterior Expected Returns: E[R] = Pi + tau*Sigma*P^T * M * (Q - P*Pi)
        excess_view_returns = Q - (P @ self.Pi)
        posterior_mean = self.Pi + (tau_sigma_pt @ M @ excess_view_returns)

        # 4. Posterior Covariance: Sigma_post = Sigma + M_post
        M_post = tau_sigma - (tau_sigma_pt @ M @ tau_sigma_pt.T)
        posterior_cov = self.Sigma + M_post

        # Wrap in pandas structures
        posterior_returns_series = pd.Series(
            posterior_mean.flatten(),
            index=self.tickers,
            name="posterior_expected_return"
        )
        posterior_cov_df = pd.DataFrame(
            posterior_cov,
            index=self.tickers,
            columns=self.tickers
        )
        omega_df = pd.DataFrame(Omega)

        return posterior_returns_series, posterior_cov_df, omega_df

    def compute_unconstrained_weights(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        delta: float
    ) -> pd.Series:
        """
        Computes analytical unconstrained Markowitz weights: w = (1 / delta) * Sigma^-1 * E[R]
        """
        cov_inv = np.linalg.inv(covariance.values)
        mu = expected_returns.values.reshape(-1, 1)
        
        unconstrained_w = (1.0 / delta) * (cov_inv @ mu)
        unconstrained_w = unconstrained_w.flatten()
        normalized_w = unconstrained_w / np.sum(unconstrained_w)

        return pd.Series(normalized_w, index=self.tickers, name="unconstrained_weight")