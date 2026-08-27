import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np

from src.engine.data_loader import DataLoader
from src.engine.covariance import CovarianceEstimator
from src.engine.equilibrium import EquilibriumEstimator
from src.engine.views import ViewsManager
from src.engine.black_litterman import BlackLittermanModel
from src.engine.constraints import PortfolioConstraints
from src.engine.optimizer import PortfolioOptimizer


def calculate_portfolio_metrics(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float, bench_w: pd.Series) -> dict:
    """Computes annualized return, volatility, Sharpe, Tracking Error, and Active Share."""
    w = weights.values
    wb = bench_w.values
    sigma = cov.values

    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ sigma @ w))
    sharpe = (ret - rf) / vol

    active_w = w - wb
    tracking_error = float(np.sqrt(active_w @ sigma @ active_w))
    active_share = float(0.5 * np.sum(np.abs(active_w)))

    return {
        "Expected Return": ret,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Tracking Error": tracking_error,
        "Active Share": active_share
    }


def main():
    print("\n" + "=" * 110)
    print("           INSTITUTIONAL PORTFOLIO OPTIMIZATION & UCITS MANDATE AUDIT")
    print("=" * 110)

    # 1. Load Data & Assets Metadata
    loader = DataLoader()
    returns = loader.get_returns_matrix()
    benchmark_weights = loader.get_benchmark_weights()
    rf_rate = loader.get_risk_free_rate()

    tickers = list(benchmark_weights.index)
    asset_classes = {
        "SPY": "Equity", "VGK": "Equity", "EEM": "Equity",
        "TLT": "Fixed Income", "IEF": "Fixed Income", "LQD": "Fixed Income", "HYG": "Fixed Income", "BNDX": "Fixed Income",
        "GLD": "Commodity", "VNQ": "Real Estate"
    }

    # 2. Covariance & Prior Setup
    cov_engine = CovarianceEstimator(returns)
    lw_cov, _ = cov_engine.ledoit_wolf_covariance()

    eq_engine = EquilibriumEstimator(returns, lw_cov, benchmark_weights, rf_rate)
    delta = eq_engine.calibrate_risk_aversion()
    implied_prior = eq_engine.compute_implied_equilibrium_returns(delta)
    hist_returns = returns.mean() * 252

    # 3. Ingest Tactical Views (VGK over SPY +1.5%, GLD 9.0%)
    views_mgr = ViewsManager(tickers)
    views_mgr.add_relative_view("VGK", "SPY", expected_outperformance=0.015, confidence=0.65)
    views_mgr.add_absolute_view("GLD", expected_return=0.090, confidence=0.75)
    P, Q, conf = views_mgr.build_matrices()

    bl_model = BlackLittermanModel(implied_prior, lw_cov, tau=0.05)
    posterior_mu, posterior_cov, _ = bl_model.calculate_posterior(P, Q, conf)

    # 4. Define UCITS & BaFin Fund Mandate Constraints
    mandate = PortfolioConstraints(
        tickers=tickers,
        asset_classes=asset_classes,
        min_weight=0.0,      # No short selling
        max_weight=0.35,     # Single asset max
        custom_asset_bounds={
            "SPY": (0.10, 0.35),
            "VGK": (0.05, 0.25),
            "EEM": (0.00, 0.10),
            "GLD": (0.00, 0.10),
            "VNQ": (0.00, 0.08)
        },
        asset_class_bounds={
            "Equity": (0.35, 0.60),        # Total Equity limit
            "Fixed Income": (0.30, 0.55),  # Total Fixed Income limit
        },
        max_tracking_error=0.035,          # 3.50% Active Risk Budget
        benchmark_weights=benchmark_weights
    )

    # 5. Run Optimizations
    opt_hist = PortfolioOptimizer(hist_returns, lw_cov, rf_rate)
    w_mvo_hist = opt_hist.optimize_utility(risk_aversion=delta, constraints=mandate)

    opt_bl = PortfolioOptimizer(posterior_mu, posterior_cov, rf_rate)
    w_bl_constrained = opt_bl.optimize_utility(risk_aversion=delta, constraints=mandate)

    w_equal = pd.Series(1.0 / len(tickers), index=tickers)

    # 6. Comparative Allocation Table
    alloc_table = pd.DataFrame({
        "Asset": tickers,
        "Class": [asset_classes[t] for t in tickers],
        "Benchmark (Cap)": benchmark_weights.map(lambda x: f"{x:.1%}"),
        "Equal Weight (1/N)": w_equal.map(lambda x: f"{x:.1%}"),
        "Constrained MVO (Hist)": w_mvo_hist.map(lambda x: f"{x:.1%}"),
        "Constrained BL (Views)": w_bl_constrained.map(lambda x: f"{x:.1%}")
    })

    print("\n" + "=" * 110)
    print("                     ASSET CLASS ALLOCATION COMPARISON")
    print("=" * 110)
    print(alloc_table.to_string(index=False))

    # 7. Portfolio Performance Metrics Comparison
    metrics_bench = calculate_portfolio_metrics(benchmark_weights, posterior_mu, posterior_cov, rf_rate, benchmark_weights)
    metrics_equal = calculate_portfolio_metrics(w_equal, posterior_mu, posterior_cov, rf_rate, benchmark_weights)
    metrics_mvo = calculate_portfolio_metrics(w_mvo_hist, posterior_mu, posterior_cov, rf_rate, benchmark_weights)
    metrics_bl = calculate_portfolio_metrics(w_bl_constrained, posterior_mu, posterior_cov, rf_rate, benchmark_weights)

    summary_metrics = pd.DataFrame([metrics_bench, metrics_equal, metrics_mvo, metrics_bl], index=[
        "Global Benchmark (Cap)",
        "Equal Weight (1/N)",
        "Constrained MVO (Hist Returns)",
        "Constrained Black-Litterman (Optimal)"
    ])

    formatted_metrics = pd.DataFrame({
        "Exp Return": summary_metrics["Expected Return"].map(lambda x: f"{x:.2%}"),
        "Volatility": summary_metrics["Volatility"].map(lambda x: f"{x:.2%}"),
        "Sharpe Ratio": summary_metrics["Sharpe Ratio"].map(lambda x: f"{x:.2f}"),
        "Tracking Error": summary_metrics["Tracking Error"].map(lambda x: f"{x:.2%}"),
        "Active Share": summary_metrics["Active Share"].map(lambda x: f"{x:.1%}")
    }, index=summary_metrics.index)

    print("\n" + "=" * 110)
    print("                     STRATEGY RISK & PERFORMANCE AUDIT")
    print("=" * 110)
    print(formatted_metrics.to_string())
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()