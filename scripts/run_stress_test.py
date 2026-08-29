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
from src.engine.stress_testing import RiskEngine, StressTestEngine


def main():
    print("\n" + "=" * 115)
    print("               PORTFOLIO TAIL-RISK, MODIFIED VAR & CRISIS STRESS-TESTING")
    print("=" * 115)

    # 1. Load Data
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

    # 2. Derive Current Optimal Black-Litterman Portfolio
    cov_engine = CovarianceEstimator(returns)
    lw_cov, _ = cov_engine.ledoit_wolf_covariance()

    eq_engine = EquilibriumEstimator(returns, lw_cov, benchmark_weights, rf_rate)
    delta = eq_engine.calibrate_risk_aversion()
    implied_prior = eq_engine.compute_implied_equilibrium_returns(delta)

    views_mgr = ViewsManager(tickers)
    views_mgr.add_relative_view("VGK", "SPY", 0.015, 0.65)
    views_mgr.add_absolute_view("GLD", 0.090, 0.75)
    P, Q, conf = views_mgr.build_matrices()

    bl_model = BlackLittermanModel(implied_prior, lw_cov, tau=0.05)
    post_mu, post_cov, _ = bl_model.calculate_posterior(P, Q, conf)

    mandate = PortfolioConstraints(
        tickers=tickers,
        asset_classes=asset_classes,
        min_weight=0.0,
        max_weight=0.35,
        custom_asset_bounds={"SPY": (0.10, 0.35), "VGK": (0.05, 0.25), "GLD": (0.0, 0.10)},
        asset_class_bounds={"Equity": (0.35, 0.60), "Fixed Income": (0.30, 0.55)},
        max_tracking_error=0.035,
        benchmark_weights=benchmark_weights
    )

    opt = PortfolioOptimizer(post_mu, post_cov, rf_rate)
    w_bl = opt.optimize_utility(risk_aversion=delta, constraints=mandate)

    # 3. Generate Daily Portfolio Return Series for Statistical Audit
    port_daily_rets = returns @ w_bl
    bench_daily_rets = returns @ benchmark_weights

    risk_p = RiskEngine(port_daily_rets).compute_tail_risk()
    risk_b = RiskEngine(bench_daily_rets).compute_tail_risk()

    # 4. Format Tail-Risk Metrics Table
    tail_df = pd.DataFrame({
        "Metric": [
            "Skewness", "Excess Kurtosis",
            "Parametric Gaussian VaR (95%)", "Cornish-Fisher Modified VaR (95%)", "Historical VaR (95%)", "Expected Shortfall CVaR (95%)",
            "Parametric Gaussian VaR (99%)", "Cornish-Fisher Modified VaR (99%)", "Historical VaR (99%)", "Expected Shortfall CVaR (99%)"
        ],
        "Black-Litterman Portfolio": [
            f"{risk_p.skewness:.3f}", f"{risk_p.excess_kurtosis:.3f}",
            f"{risk_p.param_var_95:.2%}", f"{risk_p.cornish_fisher_var_95:.2%}", f"{risk_p.hist_var_95:.2%}", f"{risk_p.cvar_95:.2%}",
            f"{risk_p.param_var_99:.2%}", f"{risk_p.cornish_fisher_var_99:.2%}", f"{risk_p.hist_var_99:.2%}", f"{risk_p.cvar_99:.2%}"
        ],
        "Global Benchmark": [
            f"{risk_b.skewness:.3f}", f"{risk_b.excess_kurtosis:.3f}",
            f"{risk_b.param_var_95:.2%}", f"{risk_b.cornish_fisher_var_95:.2%}", f"{risk_b.hist_var_95:.2%}", f"{risk_b.cvar_95:.2%}",
            f"{risk_b.param_var_99:.2%}", f"{risk_b.cornish_fisher_var_99:.2%}", f"{risk_b.hist_var_99:.2%}", f"{risk_b.cvar_99:.2%}"
        ]
    })

    print("\n" + "=" * 115)
    print("                     STATISTICAL TAIL-RISK & VALUE-AT-RISK AUDIT (1-DAY)")
    print("=" * 115)
    print(tail_df.to_string(index=False))

    # 5. Execute Historical Crisis Replays
    stress_engine = StressTestEngine(returns)
    df_hist_scenarios = stress_engine.run_historical_replays(w_bl, benchmark_weights)

    formatted_hist = pd.DataFrame({
        "Crisis Scenario": df_hist_scenarios["Crisis Scenario"],
        "Portfolio Return": df_hist_scenarios["Portfolio Return"].map(lambda x: f"{x:.2%}"),
        "Benchmark Return": df_hist_scenarios["Benchmark Return"].map(lambda x: f"{x:.2%}"),
        "Active Delta": df_hist_scenarios["Active Delta"].map(lambda x: f"{x:+.2%}"),
        "Portfolio MaxDD": df_hist_scenarios["Portfolio MaxDD"].map(lambda x: f"{x:.2%}"),
        "Benchmark MaxDD": df_hist_scenarios["Benchmark MaxDD"].map(lambda x: f"{x:.2%}")
    })

    print("\n" + "=" * 115)
    print("                     HISTORICAL CRISIS SCENARIO REPLAY AUDIT")
    print("=" * 115)
    print(formatted_hist.to_string(index=False))

    # 6. Execute Hypothetical Factor Shocks
    df_hypo_shocks = stress_engine.run_hypothetical_shocks(w_bl, benchmark_weights)

    formatted_hypo = pd.DataFrame({
        "Macro Factor Shock": df_hypo_shocks["Macro Shock Scenario"],
        "Portfolio Impact": df_hypo_shocks["Portfolio Impact"].map(lambda x: f"{x:.2%}"),
        "Benchmark Impact": df_hypo_shocks["Benchmark Impact"].map(lambda x: f"{x:.2%}"),
        "Resilience Delta": df_hypo_shocks["Resilience Delta"].map(lambda x: f"{x:+.2%}")
    })

    print("\n" + "=" * 115)
    print("                     HYPOTHETICAL MACRO FACTOR STRESS TEST")
    print("=" * 115)
    print(formatted_hypo.to_string(index=False))
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()