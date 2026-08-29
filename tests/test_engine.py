import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.engine.covariance import CovarianceEstimator
from src.engine.equilibrium import EquilibriumEstimator
from src.engine.views import ViewsManager
from src.engine.black_litterman import BlackLittermanModel
from src.engine.constraints import PortfolioConstraints
from src.engine.optimizer import PortfolioOptimizer
from src.engine.stress_testing import RiskEngine


@pytest.fixture
def synthetic_market_data():
    """Generates synthetic multi-asset return data for deterministic testing."""
    np.random.seed(42)
    n_days = 500
    tickers = ["SPY", "VGK", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "VNQ", "BNDX"]
    
    # Generate correlated random returns
    mean_daily = np.array([0.0004, 0.0003, 0.0003, 0.0001, 0.0001, 0.0002, 0.0002, 0.0003, 0.0003, 0.0001])
    vol_daily = np.array([0.010, 0.011, 0.013, 0.009, 0.005, 0.006, 0.007, 0.010, 0.012, 0.004])
    
    corr = np.eye(len(tickers))
    corr[0, 1] = corr[1, 0] = 0.75 # SPY - VGK
    corr[3, 4] = corr[4, 3] = 0.85 # TLT - IEF
    
    cov = np.diag(vol_daily) @ corr @ np.diag(vol_daily)
    returns_matrix = np.random.multivariate_normal(mean_daily, cov, size=n_days)
    
    df_returns = pd.DataFrame(returns_matrix, columns=tickers)
    
    # Market Cap benchmark weights
    benchmark_weights = pd.Series({
        "SPY": 0.40, "VGK": 0.15, "EEM": 0.08, "TLT": 0.07, "IEF": 0.08,
        "LQD": 0.08, "HYG": 0.04, "GLD": 0.03, "VNQ": 0.03, "BNDX": 0.04
    })
    
    return df_returns, benchmark_weights


def test_covariance_ledoit_wolf_properties(synthetic_market_data):
    """Verifies that Ledoit-Wolf produces symmetric, positive-definite matrices with lower condition numbers."""
    returns, _ = synthetic_market_data
    cov_engine = CovarianceEstimator(returns)
    
    sample_cov = cov_engine.sample_covariance()
    lw_cov, shrinkage_delta = cov_engine.ledoit_wolf_covariance()
    
    # 1. Test Symmetry
    np.testing.assert_allclose(lw_cov.values, lw_cov.values.T, atol=1e-8)
    
    # 2. Test Positive Definiteness (all eigenvalues > 0)
    eigenvalues = np.linalg.eigvalsh(lw_cov.values)
    assert np.all(eigenvalues > 0), "Ledoit-Wolf covariance must be strictly positive-definite."
    
    # 3. Test Condition Number Reduction
    cond_sample = cov_engine.calculate_condition_number(sample_cov)
    cond_lw = cov_engine.calculate_condition_number(lw_cov)
    assert cond_lw <= cond_sample, "Ledoit-Wolf must improve or maintain matrix condition number."
    assert 0.0 <= shrinkage_delta <= 1.0, "Shrinkage intensity delta must be bounded in [0, 1]."


def test_black_litterman_zero_views_equilibrium(synthetic_market_data):
    """Verifies that when no views are provided, Black-Litterman posterior equals the market equilibrium prior."""
    returns, benchmark_weights = synthetic_market_data
    cov_engine = CovarianceEstimator(returns)
    lw_cov, _ = cov_engine.ledoit_wolf_covariance()
    
    eq_engine = EquilibriumEstimator(returns, lw_cov, benchmark_weights, risk_free_rate=0.02)
    delta = eq_engine.calibrate_risk_aversion()
    implied_prior = eq_engine.compute_implied_equilibrium_returns(delta)
    
    bl_model = BlackLittermanModel(implied_prior, lw_cov, tau=0.05)
    
    # Empty views
    P_empty = np.empty((0, len(returns.columns)))
    Q_empty = np.empty((0, 1))
    conf_empty = np.empty((0, 1))
    
    post_mu, post_cov, _ = bl_model.calculate_posterior(P_empty, Q_empty, conf_empty)
    
    # Posterior must match Prior exactly
    np.testing.assert_allclose(post_mu.values, implied_prior.values, atol=1e-7)
    np.testing.assert_allclose(post_cov.values, lw_cov.values, atol=1e-7)


def test_black_litterman_relative_view_tilt(synthetic_market_data):
    """Verifies that an outperformance view on VGK vs SPY strictly increases VGK return relative to SPY."""
    returns, benchmark_weights = synthetic_market_data
    cov_engine = CovarianceEstimator(returns)
    lw_cov, _ = cov_engine.ledoit_wolf_covariance()
    
    eq_engine = EquilibriumEstimator(returns, lw_cov, benchmark_weights, risk_free_rate=0.02)
    delta = eq_engine.calibrate_risk_aversion()
    implied_prior = eq_engine.compute_implied_equilibrium_returns(delta)
    
    views_mgr = ViewsManager(list(returns.columns))
    views_mgr.add_relative_view("VGK", "SPY", expected_outperformance=0.02, confidence=0.70)
    P, Q, conf = views_mgr.build_matrices()
    
    bl_model = BlackLittermanModel(implied_prior, lw_cov, tau=0.05)
    post_mu, _, _ = bl_model.calculate_posterior(P, Q, conf)
    
    prior_spread = implied_prior["VGK"] - implied_prior["SPY"]
    post_spread = post_mu["VGK"] - post_mu["SPY"]
    
    assert post_spread > prior_spread, "Bullish relative view on VGK vs SPY must widen the expected return spread."


def test_optimizer_ucits_mandate_constraints(synthetic_market_data):
    """Verifies that quadratic optimizer strictly adheres to full investment, long-only, and single-asset caps."""
    returns, benchmark_weights = synthetic_market_data
    cov_engine = CovarianceEstimator(returns)
    lw_cov, _ = cov_engine.ledoit_wolf_covariance()
    mu = returns.mean() * 252
    
    tickers = list(returns.columns)
    asset_classes = {
        "SPY": "Equity", "VGK": "Equity", "EEM": "Equity",
        "TLT": "Fixed Income", "IEF": "Fixed Income", "LQD": "Fixed Income", "HYG": "Fixed Income", "BNDX": "Fixed Income",
        "GLD": "Commodity", "VNQ": "Real Estate"
    }
    
    max_cap = 0.35
    mandate = PortfolioConstraints(
        tickers=tickers,
        asset_classes=asset_classes,
        min_weight=0.0,
        max_weight=max_cap,
        asset_class_bounds={"Equity": (0.35, 0.60), "Fixed Income": (0.30, 0.55)},
        max_tracking_error=0.04,
        benchmark_weights=benchmark_weights
    )
    
    opt = PortfolioOptimizer(mu, lw_cov, risk_free_rate=0.02)
    w_opt = opt.optimize_utility(risk_aversion=2.85, constraints=mandate)
    
    # 1. Full Investment Budget: sum(w) == 1.0
    assert np.isclose(w_opt.sum(), 1.0, atol=1e-5), f"Portfolio weights must sum to 1.0. Got: {w_opt.sum()}"
    
    # 2. Long-Only: w_i >= 0
    assert (w_opt >= -1e-6).all(), "All weights must be non-negative (no short-selling)."
    
    # 3. Single-Asset Cap: w_i <= max_cap
    assert (w_opt <= max_cap + 1e-5).all(), f"No single asset weight may exceed {max_cap:.1%}."
    
    # 4. Group Bounds: Equity in [0.35, 0.60]
    equity_tickers = [t for t, c in asset_classes.items() if c == "Equity"]
    eq_wgt = w_opt.reindex(equity_tickers).sum()
    assert 0.35 - 1e-4 <= eq_wgt <= 0.60 + 1e-4, f"Equity exposure ({eq_wgt:.2%}) must be within [35%, 60%]."


def test_tail_risk_metrics_ordering(synthetic_market_data):
    """Verifies that CVaR is strictly greater than or equal to VaR at 99% confidence."""
    returns, benchmark_weights = synthetic_market_data
    port_daily_ret = returns @ benchmark_weights
    
    risk_engine = RiskEngine(port_daily_ret)
    metrics = risk_engine.compute_tail_risk()
    
    assert metrics.cvar_95 >= metrics.hist_var_95, "CVaR (95%) must be >= Historical VaR (95%)."
    assert metrics.cvar_99 >= metrics.hist_var_99, "CVaR (99%) must be >= Historical VaR (99%)."
    assert metrics.cornish_fisher_var_99 > 0, "Cornish-Fisher VaR must be positive."