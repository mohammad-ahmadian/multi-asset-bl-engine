import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

logger = logging.getLogger(__name__)


@dataclass
class TailRiskMetrics:
    """Statistical tail-risk metrics container."""
    skewness: float
    excess_kurtosis: float
    hist_var_95: float
    hist_var_99: float
    param_var_95: float
    param_var_99: float
    cornish_fisher_var_95: float
    cornish_fisher_var_99: float
    cvar_95: float
    cvar_99: float


class RiskEngine:
    """Computes advanced non-normal tail-risk metrics for portfolio return distributions."""

    def __init__(self, returns: pd.Series, annualize_factor: int = 252):
        self.returns = returns.dropna()
        self.ann_factor = annualize_factor
        self.mu = float(self.returns.mean())
        self.sigma = float(self.returns.std())

    def compute_tail_risk(self) -> TailRiskMetrics:
        """Calculates historical, parametric, and Cornish-Fisher VaR and CVaR."""
        r = self.returns.values
        s = float(skew(r))
        k = float(kurtosis(r, fisher=True)) # Excess kurtosis (Fisher=True means Normal = 0)

        # 1. Historical VaR & CVaR (1-Day horizon)
        h_var_95 = -float(np.percentile(r, 5.0))
        h_var_99 = -float(np.percentile(r, 1.0))

        tail_95 = r[r <= -h_var_95]
        tail_99 = r[r <= -h_var_99]

        cvar_95 = -float(np.mean(tail_95)) if len(tail_95) > 0 else h_var_95
        cvar_99 = -float(np.mean(tail_99)) if len(tail_99) > 0 else h_var_99

        # 2. Parametric Gaussian VaR
        z_95 = norm.ppf(0.05) # -1.6449
        z_99 = norm.ppf(0.01) # -2.3263

        p_var_95 = -(self.mu + z_95 * self.sigma)
        p_var_99 = -(self.mu + z_99 * self.sigma)

        # 3. Cornish-Fisher Modified VaR
        cf_z_95 = z_95 + (1/6)*(z_95**2 - 1)*s + (1/24)*(z_95**3 - 3*z_95)*k - (1/36)*(2*z_95**3 - 5*z_95)*(s**2)
        cf_z_99 = z_99 + (1/6)*(z_99**2 - 1)*s + (1/24)*(z_99**3 - 3*z_99)*k - (1/36)*(2*z_99**3 - 5*z_99)*(s**2)

        cf_var_95 = -(self.mu + cf_z_95 * self.sigma)
        cf_var_99 = -(self.mu + cf_z_99 * self.sigma)

        return TailRiskMetrics(
            skewness=s,
            excess_kurtosis=k,
            hist_var_95=h_var_95,
            hist_var_99=h_var_99,
            param_var_95=p_var_95,
            param_var_99=p_var_99,
            cornish_fisher_var_95=cf_var_95,
            cornish_fisher_var_99=cf_var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99
        )


class StressTestEngine:
    """Executes historical crisis replays and hypothetical macro factor stress scenarios."""

    HISTORICAL_SCENARIOS = {
        "2020 COVID Liquidity Shock": ("2020-02-19", "2020-03-23"),
        "2022 Global Rate Spike & Inflation": ("2022-01-03", "2022-10-14"),
        "2018 Q4 Tech & Growth Sell-Off": ("2018-10-01", "2018-12-24"),
        "2015-16 EM & Commodity Crisis": ("2015-08-10", "2016-02-11")
    }

    HYPOTHETICAL_SHOCKS = {
        "Global Equity Crash (-15%)": {
            "SPY": -0.15, "VGK": -0.18, "EEM": -0.22, "TLT": 0.05, "IEF": 0.03,
            "LQD": -0.04, "HYG": -0.10, "GLD": 0.08, "VNQ": -0.16, "BNDX": 0.02
        },
        "Stagflation Shock (Inflation + Yield Spike)": {
            "SPY": -0.10, "VGK": -0.12, "EEM": -0.14, "TLT": -0.12, "IEF": -0.06,
            "LQD": -0.08, "HYG": -0.09, "GLD": 0.15, "VNQ": -0.12, "BNDX": -0.05
        },
        "Monetary Tightening / Rate Shock": {
            "SPY": -0.06, "VGK": -0.07, "EEM": -0.08, "TLT": -0.16, "IEF": -0.08,
            "LQD": -0.09, "HYG": -0.05, "GLD": -0.04, "VNQ": -0.10, "BNDX": -0.06
        }
    }

    def __init__(self, raw_returns: pd.DataFrame):
        self.raw_returns = raw_returns

    def run_historical_replays(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series
    ) -> pd.DataFrame:
        """
        Replays exact historical crisis periods by applying weights to asset returns during the shock.
        """
        records = []
        tickers = list(portfolio_weights.index)
        w_p = portfolio_weights.values
        w_b = benchmark_weights.reindex(tickers).fillna(0.0).values

        for scenario_name, (start_dt, end_dt) in self.HISTORICAL_SCENARIOS.items():
            if start_dt not in self.raw_returns.index or end_dt not in self.raw_returns.index:
                # Find closest matching dates
                slice_df = self.raw_returns.loc[start_dt:end_dt, tickers]
            else:
                slice_df = self.raw_returns.loc[start_dt:end_dt, tickers]

            if slice_df.empty:
                continue

            # Calculate cumulative asset returns over the crisis window
            cum_asset_rets = (1.0 + slice_df).prod() - 1.0

            port_cum_ret = float(w_p @ cum_asset_rets.values)
            bench_cum_ret = float(w_b @ cum_asset_rets.values)

            # Max drawdown during the crisis slice
            port_daily = slice_df @ w_p
            bench_daily = slice_df @ w_b

            port_mdd = float(((1.0 + port_daily).cumprod() / (1.0 + port_daily).cumprod().cummax() - 1.0).min())
            bench_mdd = float(((1.0 + bench_daily).cumprod() / (1.0 + bench_daily).cumprod().cummax() - 1.0).min())

            records.append({
                "Crisis Scenario": scenario_name,
                "Window": f"{start_dt} to {end_dt}",
                "Portfolio Return": port_cum_ret,
                "Benchmark Return": bench_cum_ret,
                "Active Delta": port_cum_ret - bench_cum_ret,
                "Portfolio MaxDD": port_mdd,
                "Benchmark MaxDD": bench_mdd
            })

        return pd.DataFrame(records)

    def run_hypothetical_shocks(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series
    ) -> pd.DataFrame:
        """
        Calculates instantaneous portfolio impact under macro factor stress scenarios.
        """
        records = []
        tickers = list(portfolio_weights.index)
        w_p = portfolio_weights.values
        w_b = benchmark_weights.reindex(tickers).fillna(0.0).values

        for shock_name, shock_dict in self.HYPOTHETICAL_SHOCKS.items():
            shock_vector = np.array([shock_dict.get(t, 0.0) for t in tickers])

            port_impact = float(w_p @ shock_vector)
            bench_impact = float(w_b @ shock_vector)

            records.append({
                "Macro Shock Scenario": shock_name,
                "Portfolio Impact": port_impact,
                "Benchmark Impact": bench_impact,
                "Resilience Delta": port_impact - bench_impact
            })

        return pd.DataFrame(records)