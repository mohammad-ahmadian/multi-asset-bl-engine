import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EquilibriumEstimator:
    """Calculates market risk aversion and CAPM-implied equilibrium returns (Black-Litterman Prior)."""

    def __init__(
        self,
        returns: pd.DataFrame,
        covariance_matrix: pd.DataFrame,
        benchmark_weights: pd.Series,
        risk_free_rate: float,
        annualize_factor: int = 252
    ):
        self.returns = returns
        self.covariance = covariance_matrix
        self.annualize_factor = annualize_factor
        self.risk_free_rate = risk_free_rate

        # Align benchmark weights with return columns
        self.tickers = list(covariance_matrix.columns)
        self.benchmark_weights = benchmark_weights.reindex(self.tickers).fillna(0.0)
        self.benchmark_weights = self.benchmark_weights / self.benchmark_weights.sum()

    def calibrate_risk_aversion(self) -> float:
        """
        Calibrates market risk aversion coefficient delta = (E[R_mkt] - R_f) / Var(R_mkt).
        Standard institutional values range from 2.0 to 4.0.
        """
        w = self.benchmark_weights.values
        sigma = self.covariance.values

        # Benchmark portfolio annualized variance: w^T * Sigma * w
        bench_variance = float(w.T @ sigma @ w)
        bench_volatility = np.sqrt(bench_variance)

        # Historical benchmark portfolio daily return series
        bench_daily_returns = self.returns @ self.benchmark_weights
        bench_annual_return = float(bench_daily_returns.mean() * self.annualize_factor)

        # Excess return over risk-free rate
        excess_return = bench_annual_return - self.risk_free_rate

        # Safeguard: if historical excess return is excessively low, floor delta at 2.5
        if excess_return <= 0.01:
            delta = 2.5
            logger.info(f"Historical excess return low ({excess_return:.2%}). Calibrated delta defaulted to: {delta:.2f}")
        else:
            delta = float(excess_return / bench_variance)

        return float(np.clip(delta, 1.5, 5.0))

    def compute_implied_equilibrium_returns(self, delta: float = None) -> pd.Series:
        """
        Calculates reverse-optimized market implied equilibrium returns:
        Pi = delta * Sigma * w_mkt
        """
        if delta is None:
            delta = self.calibrate_risk_aversion()

        w = self.benchmark_weights.values
        sigma = self.covariance.values

        # Reverse optimization formula
        implied_excess_returns = delta * (sigma @ w)
        implied_total_returns = implied_excess_returns + self.risk_free_rate

        return pd.Series(implied_total_returns, index=self.tickers, name="implied_equilibrium_return")

    def compute_historical_statistics(self) -> pd.DataFrame:
        """Computes annualized historical mean return, annualized volatility, and Sharpe ratios."""
        hist_annual_returns = self.returns.mean() * self.annualize_factor
        hist_annual_vol = self.returns.std() * np.sqrt(self.annualize_factor)
        sharpe_ratios = (hist_annual_returns - self.risk_free_rate) / hist_annual_vol

        df = pd.DataFrame({
            "historical_mean_return": hist_annual_returns,
            "annualized_volatility": hist_annual_vol,
            "historical_sharpe": sharpe_ratios,
            "benchmark_weight": self.benchmark_weights
        })
        return df