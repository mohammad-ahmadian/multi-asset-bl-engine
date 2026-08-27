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


def main():
    print("\n" + "=" * 105)
    print("                     BLACK-LITTERMAN BAYESIAN ALLOCATION CALCULATION")
    print("=" * 105)

    # 1. Load Data & Prior
    loader = DataLoader()
    returns = loader.get_returns_matrix()
    benchmark_weights = loader.get_benchmark_weights()
    rf_rate = loader.get_risk_free_rate()

    cov_engine = CovarianceEstimator(returns)
    lw_cov, _ = cov_engine.ledoit_wolf_covariance()

    eq_engine = EquilibriumEstimator(
        returns=returns,
        covariance_matrix=lw_cov,
        benchmark_weights=benchmark_weights,
        risk_free_rate=rf_rate
    )
    delta = eq_engine.calibrate_risk_aversion()
    implied_prior = eq_engine.compute_implied_equilibrium_returns(delta=delta)

    # 2. Formulate Tactical Views
    views_mgr = ViewsManager(tickers=list(lw_cov.columns))

    # View 1: Relative View (European Equities outperform US Equities by +1.50% with 65% confidence)
    views_mgr.add_relative_view(
        asset_long="VGK",
        asset_short="SPY",
        expected_outperformance=0.015,
        confidence=0.65
    )

    # View 2: Absolute View (Gold achieves 9.00% annualized return with 75% confidence)
    views_mgr.add_absolute_view(
        asset="GLD",
        expected_return=0.090,
        confidence=0.75
    )

    # Sync views to PostgreSQL
    views_mgr.sync_to_database()
    P, Q, confidences = views_mgr.build_matrices()

    print(f"\n[INFO] Tactical Views Ingested: {len(views_mgr.views)}")
    for i, v in enumerate(views_mgr.views, 1):
        target = f"{v.asset_long} over {v.asset_short}" if v.view_type == "RELATIVE" else v.asset_long
        print(f"  • View #{i} ({v.view_type}): {target} => {v.expected_outperformance:+.2%} (Confidence: {v.confidence:.0%})")

    # 3. Compute Black-Litterman Posterior
    bl_engine = BlackLittermanModel(
        implied_returns=implied_prior,
        covariance_matrix=lw_cov,
        tau=0.05
    )

    posterior_returns, posterior_cov, omega = bl_engine.calculate_posterior(P, Q, confidences)

    # Unconstrained weights to inspect directional shifts
    prior_unconstrained_w = bl_engine.compute_unconstrained_weights(implied_prior, lw_cov, delta)
    post_unconstrained_w = bl_engine.compute_unconstrained_weights(posterior_returns, posterior_cov, delta)

    # 4. Consolidate and Print Shift Analysis
    comparison_df = pd.DataFrame({
        "Asset": lw_cov.columns,
        "Benchmark Weight": benchmark_weights.reindex(lw_cov.columns).map(lambda x: f"{x:.1%}"),
        "Prior Return (Pi)": implied_prior.map(lambda x: f"{x:.2%}"),
        "Posterior Return E[R]": posterior_returns.map(lambda x: f"{x:.2%}"),
        "Return Delta": (posterior_returns - implied_prior).map(lambda x: f"{x:+.2%}"),
        "Prior Weights": prior_unconstrained_w.map(lambda x: f"{x:.1%}"),
        "Posterior Weights": post_unconstrained_w.map(lambda x: f"{x:.1%}"),
        "Weight Shift": (post_unconstrained_w - prior_unconstrained_w).map(lambda x: f"{x:+.1%}")
    })

    print("\n" + "=" * 105)
    print("                     BLACK-LITTERMAN POSTERIOR SHIFT ANALYSIS")
    print("=" * 105)
    print(comparison_df.to_string(index=False))
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()