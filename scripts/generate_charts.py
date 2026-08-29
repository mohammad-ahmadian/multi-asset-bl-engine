import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np

from src.engine.pipeline import PortfolioPipeline
from src.engine.optimizer import PortfolioOptimizer
from src.engine.constraints import PortfolioConstraints
from src.analytics.visualize import PortfolioVisualizer


def main():
    print("\n" + "=" * 95)
    print("        GENERATING INSTITUTIONAL PUBLICATION VISUALIZATION FIGURES")
    print("=" * 95)

    figures_dir = ROOT_DIR / "docs" / "figures"
    visualizer = PortfolioVisualizer(output_dir=figures_dir)
    pipeline = PortfolioPipeline()

    # 1. Run Pipeline Stages
    calib = pipeline.run_calibration_stage()
    bl_res = pipeline.run_black_litterman_stage(calib)
    target_weights = pipeline.run_optimization_stage(calib, bl_res)
    backtest_results = pipeline.run_backtest_stage(calib)
    risk_res = pipeline.run_risk_and_stress_stage(calib, target_weights)

    # -------------------------------------------------------------
    # Figure 1: Cumulative Performance & Underwater Drawdown
    # -------------------------------------------------------------
    print("\n[1/4] Generating Cumulative Performance & Drawdown Profile...")
    visualizer.plot_cumulative_performance_and_drawdown(
        strategy_results=backtest_results,
        filename="cumulative_returns.png"
    )

    # -------------------------------------------------------------
    # Figure 2: Efficient Frontier Shift
    # -------------------------------------------------------------
    print("[2/4] Generating Efficient Frontier Shift...")
    mandate = PortfolioConstraints(
        tickers=pipeline.universe,
        asset_classes=pipeline.asset_classes,
        min_weight=0.0,
        max_weight=0.35,
        benchmark_weights=calib["benchmark_weights"]
    )

    opt_prior = PortfolioOptimizer(calib["implied_prior"], calib["lw_cov"], calib["rf_rate"])
    frontier_prior = opt_prior.generate_efficient_frontier(mandate, num_points=20)

    opt_post = PortfolioOptimizer(bl_res["posterior_mu"], bl_res["posterior_cov"], calib["rf_rate"])
    frontier_post = opt_post.generate_efficient_frontier(mandate, num_points=20)

    w_prior = opt_prior.optimize_utility(calib["delta"], mandate)
    opt_prior_pt = (
        float(np.sqrt(w_prior.values @ calib["lw_cov"].values @ w_prior.values)),
        float(w_prior.values @ calib["implied_prior"].values)
    )

    opt_post_pt = (
        float(np.sqrt(target_weights.values @ bl_res["posterior_cov"].values @ target_weights.values)),
        float(target_weights.values @ bl_res["posterior_mu"].values)
    )

    hist_stats = pd.DataFrame({
        "annualized_volatility": calib["returns"].std() * np.sqrt(252),
        "historical_mean_return": calib["returns"].mean() * 252
    })

    visualizer.plot_efficient_frontier_shift(
        frontier_prior=frontier_prior,
        frontier_post=frontier_post,
        opt_prior_pt=opt_prior_pt,
        opt_post_pt=opt_post_pt,
        asset_stats=hist_stats,
        filename="efficient_frontier.png"
    )

    # -------------------------------------------------------------
    # Figure 3: Historical Asset Allocation Drift
    # -------------------------------------------------------------
    print("[3/4] Generating Historical Allocation Drift Stacked Area...")
    bl_weights_history = backtest_results["BLACK_LITTERMAN"].weights_history
    visualizer.plot_historical_asset_allocation(
        weights_history=bl_weights_history,
        filename="asset_allocation_drift.png"
    )

    # -------------------------------------------------------------
    # Figure 4: Crisis Stress Test Comparison
    # -------------------------------------------------------------
    print("[4/4] Generating Historical Crisis Replay Bar Chart...")
    visualizer.plot_stress_test_comparison(
        stress_df=risk_res["hist_scenarios"],
        filename="stress_test_comparison.png"
    )

    print("\n" + "=" * 95)
    print(f"[SUCCESS] All 4 institutional figures successfully saved to:\n  -> {figures_dir.resolve()}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()