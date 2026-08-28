import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioAnalytics:
    """
    Computes institutional-grade risk, return, drawdown, and benchmark-relative metrics.
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        annualize_factor: int = 252
    ):
        self.port_ret = portfolio_returns.dropna()
        self.bench_ret = benchmark_returns.reindex(self.port_ret.index).dropna() if benchmark_returns is not None else None
        self.rf = risk_free_rate
        self.rf_daily = (1.0 + self.rf) ** (1.0 / annualize_factor) - 1.0
        self.ann_factor = annualize_factor

    def compute_cagr(self) -> float:
        """Calculates Compound Annual Growth Rate (CAGR)."""
        cumulative = (1.0 + self.port_ret).prod()
        n_years = len(self.port_ret) / self.ann_factor
        if n_years <= 0:
            return 0.0
        return float(cumulative ** (1.0 / n_years) - 1.0)

    def compute_volatility(self) -> float:
        """Calculates annualized standard deviation of returns."""
        return float(self.port_ret.std() * np.sqrt(self.ann_factor))

    def compute_sharpe_ratio(self) -> float:
        """Calculates annualized Sharpe Ratio using excess returns over risk-free rate."""
        excess_daily = self.port_ret - self.rf_daily
        mean_excess_ann = excess_daily.mean() * self.ann_factor
        vol = self.compute_volatility()
        return float(mean_excess_ann / vol) if vol > 0 else 0.0

    def compute_sortino_ratio(self) -> float:
        """Calculates Sortino Ratio based on downside semi-variance below risk-free rate."""
        excess_daily = self.port_ret - self.rf_daily
        downside_returns = excess_daily[excess_daily < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_std = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(self.ann_factor)
        return float((self.compute_cagr() - self.rf) / downside_std) if downside_std > 0 else 0.0

    def compute_omega_ratio(self, threshold: float = 0.0) -> float:
        """Calculates Omega Ratio (gain-to-loss probability mass)."""
        threshold_daily = (1.0 + threshold) ** (1.0 / self.ann_factor) - 1.0
        excess = self.port_ret - threshold_daily
        gains = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()
        return float(gains / losses) if losses > 0 else 0.0

    def compute_drawdown_profile(self) -> Tuple[float, float, int, pd.Series]:
        """
        Computes Max Drawdown, Average Drawdown, Max Drawdown Duration (in days),
        and the full underwater drawdown series.
        """
        cumulative = (1.0 + self.port_ret).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak

        max_dd = float(drawdown.min())
        avg_dd = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

        # Calculate max drawdown duration in trading days
        is_in_dd = drawdown < 0
        dd_durations = []
        current_len = 0

        for in_dd in is_in_dd:
            if in_dd:
                current_len += 1
            else:
                if current_len > 0:
                    dd_durations.append(current_len)
                current_len = 0
        if current_len > 0:
            dd_durations.append(current_len)

        max_duration = max(dd_durations) if dd_durations else 0

        return max_dd, avg_dd, max_duration, drawdown

    def compute_calmar_ratio(self) -> float:
        """Calculates Calmar Ratio = CAGR / |Max Drawdown|."""
        max_dd, _, _, _ = self.compute_drawdown_profile()
        return float(self.compute_cagr() / abs(max_dd)) if max_dd != 0 else 0.0

    def compute_benchmark_relative_metrics(self) -> Dict[str, float]:
        """
        Computes Beta, Alpha, Tracking Error, Information Ratio,
        Treynor Ratio, and Up/Down Capture against benchmark.
        """
        if self.bench_ret is None or len(self.bench_ret) == 0:
            return {}

        aligned_p, aligned_b = self.port_ret.align(self.bench_ret, join="inner")
        
        # 1. Beta & Alpha
        cov_pb = np.cov(aligned_p, aligned_b)[0, 1]
        var_b = np.var(aligned_b)
        beta = float(cov_pb / var_b) if var_b > 0 else 1.0

        cagr_p = self.compute_cagr()
        bench_cagr = float((1.0 + aligned_b).prod() ** (self.ann_factor / len(aligned_b)) - 1.0)
        
        # Jensen's Alpha: CAGR_p - [Rf + Beta * (CAGR_b - Rf)]
        alpha = cagr_p - (self.rf + beta * (bench_cagr - self.rf))

        # 2. Tracking Error & Information Ratio
        excess_vs_bench = aligned_p - aligned_b
        te = float(excess_vs_bench.std() * np.sqrt(self.ann_factor))
        ir = float((cagr_p - bench_cagr) / te) if te > 0 else 0.0

        # 3. Treynor Ratio
        treynor = float((cagr_p - self.rf) / beta) if beta != 0 else 0.0

        # 4. Up / Down Capture Ratios
        up_idx = aligned_b > 0
        down_idx = aligned_b < 0

        up_capture = float(aligned_p[up_idx].mean() / aligned_b[up_idx].mean()) if up_idx.any() else 1.0
        down_capture = float(aligned_p[down_idx].mean() / aligned_b[down_idx].mean()) if down_idx.any() else 1.0
        capture_ratio = float(up_capture / down_capture) if down_capture != 0 else 1.0

        return {
            "Beta": beta,
            "Jensen Alpha": alpha,
            "Tracking Error": te,
            "Information Ratio": ir,
            "Treynor Ratio": treynor,
            "Up Capture": up_capture,
            "Down Capture": down_capture,
            "Capture Ratio": capture_ratio
        }

    def compute_all_metrics(self) -> Dict[str, float]:
        """Consolidates complete risk, return, and benchmark statistics into a single dictionary."""
        max_dd, avg_dd, max_dur, _ = self.compute_drawdown_profile()
        
        metrics = {
            "Total Cumulative Return": float((1.0 + self.port_ret).prod() - 1.0),
            "CAGR": self.compute_cagr(),
            "Annualized Volatility": self.compute_volatility(),
            "Sharpe Ratio": self.compute_sharpe_ratio(),
            "Sortino Ratio": self.compute_sortino_ratio(),
            "Omega Ratio": self.compute_omega_ratio(threshold=self.rf),
            "Max Drawdown": max_dd,
            "Avg Drawdown": avg_dd,
            "Max Drawdown Days": max_dur,
            "Calmar Ratio": self.compute_calmar_ratio()
        }

        if self.bench_ret is not None:
            metrics.update(self.compute_benchmark_relative_metrics())

        return metrics

    def compute_rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """
        Computes 12-Month rolling Sharpe Ratio, rolling Volatility, and rolling Beta.
        """
        roll_mean = self.port_ret.rolling(window).mean() * self.ann_factor
        roll_vol = self.port_ret.rolling(window).std() * np.sqrt(self.ann_factor)
        roll_sharpe = (roll_mean - self.rf) / roll_vol

        df_rolling = pd.DataFrame({
            "rolling_volatility": roll_vol,
            "rolling_sharpe": roll_sharpe
        }, index=self.port_ret.index)

        if self.bench_ret is not None:
            aligned_p, aligned_b = self.port_ret.align(self.bench_ret, join="inner")
            roll_cov = aligned_p.rolling(window).cov(aligned_b)
            roll_var_b = aligned_b.rolling(window).var()
            df_rolling["rolling_beta"] = roll_cov / roll_var_b
            
            diff = aligned_p - aligned_b
            df_rolling["rolling_tracking_error"] = diff.rolling(window).std() * np.sqrt(self.ann_factor)

        return df_rolling.dropna()