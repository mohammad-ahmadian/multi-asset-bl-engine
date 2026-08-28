import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np

from src.engine.data_loader import DataLoader
from src.engine.backtest import BacktestEngine
from src.engine.analytics import PortfolioAnalytics


def main():
    print("\n" + "=" * 115)
    print("                 INSTITUTIONAL PERFORMANCE & ACTIVE RISK TEAR-SHEET")
    print("=" * 115)

    # 1. Load Data
    loader = DataLoader()
    returns = loader.get_returns_matrix()
    benchmark_weights = loader.get_benchmark_weights()
    rf_rate = loader.get_risk_free_rate()

    asset_classes = {
        "SPY": "Equity", "VGK": "Equity", "EEM": "Equity",
        "TLT": "Fixed Income", "IEF": "Fixed Income", "LQD": "Fixed Income", "HYG": "Fixed Income", "BNDX": "Fixed Income",
        "GLD": "Commodity", "VNQ": "Real Estate"
    }

    # 2. Run Backtest Simulation
    engine = BacktestEngine(
        returns=returns,
        benchmark_weights=benchmark_weights,
        asset_classes=asset_classes,
        lookback_days=756,
        transaction_cost_bps=10.0
    )

    strategies = ["BENCHMARK_CAP", "EQUAL_WEIGHT", "HISTORICAL_MVO", "BLACK_LITTERMAN"]
    strategy_labels = {
        "BENCHMARK_CAP": "Global Benchmark (Cap-Weighted)",
        "EQUAL_WEIGHT": "Equal Weight (1/N)",
        "HISTORICAL_MVO": "Constrained Historical MVO",
        "BLACK_LITTERMAN": "Dynamic Constrained Black-Litterman"
    }

    strategy_results = {}

    print(f"\n[INFO] Simulating strategies from {engine.rebalance_dates[0].strftime('%Y-%m-%d')} to {engine.rebalance_dates[-1].strftime('%Y-%m-%d')}...")
    for strat in strategies:
        strategy_results[strat] = engine.run_strategy(strat)

    bench_daily_returns = strategy_results["BENCHMARK_CAP"].daily_returns

    # 3. Compute Analytics
    analytics_records = {}
    for strat, res in strategy_results.items():
        bench_series = bench_daily_returns if strat != "BENCHMARK_CAP" else None
        analytics = PortfolioAnalytics(
            portfolio_returns=res.daily_returns,
            benchmark_returns=bench_series,
            risk_free_rate=rf_rate
        )
        analytics_records[strategy_labels[strat]] = analytics.compute_all_metrics()

    df_raw = pd.DataFrame(analytics_records).T

    # 4. Format Display Table (Align directly with df_raw.index)
    formatted_table = pd.DataFrame({
        "Total Return": df_raw["Total Cumulative Return"].map(lambda x: f"{x:.2%}"),
        "CAGR": df_raw["CAGR"].map(lambda x: f"{x:.2%}"),
        "Volatility": df_raw["Annualized Volatility"].map(lambda x: f"{x:.2%}"),
        "Sharpe (Rf)": df_raw["Sharpe Ratio"].map(lambda x: f"{x:.2f}"),
        "Sortino": df_raw["Sortino Ratio"].map(lambda x: f"{x:.2f}"),
        "Omega": df_raw["Omega Ratio"].map(lambda x: f"{x:.2f}"),
        "Max DD": df_raw["Max Drawdown"].map(lambda x: f"{x:.2%}"),
        "Max DD Days": df_raw["Max Drawdown Days"].map(lambda x: f"{int(x)}d"),
        "Calmar": df_raw["Calmar Ratio"].map(lambda x: f"{x:.2f}"),
        "Beta": df_raw["Beta"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "1.00"),
        "Alpha": df_raw["Jensen Alpha"].map(lambda x: f"{x:+.2%}" if pd.notnull(x) else "0.00%"),
        "Tracking Error": df_raw["Tracking Error"].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "0.00%"),
        "Info Ratio": df_raw["Information Ratio"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "0.00"),
        "Capture Ratio": df_raw["Capture Ratio"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "1.00")
    }, index=df_raw.index)

    print("\n" + "=" * 115)
    print("                     EXECUTIVE STRATEGY PERFORMANCE & RISK MATRIX")
    print("=" * 115)
    print(formatted_table.to_string())
    print("=" * 115)

    # 5. Executive Insights for Senior PM
    bl_metrics = analytics_records["Dynamic Constrained Black-Litterman"]
    bench_mdd_days = int(df_raw.loc["Global Benchmark (Cap-Weighted)", "Max Drawdown Days"])
    
    print("\n[PORTFOLIO ANALYST EXECUTIVE INSIGHTS]")
    print(f"  1. Risk-Adjusted Alpha: Black-Litterman generated Jensen's Alpha of {bl_metrics['Jensen Alpha']:+.2%} p.a.")
    print(f"  2. Active Efficiency: Information Ratio of {bl_metrics['Information Ratio']:.2f} with Tracking Error of {bl_metrics['Tracking Error']:.2%}")
    print(f"  3. Asymmetric Capture: Up-Market Capture = {bl_metrics['Up Capture']:.1%}, Down-Market Capture = {bl_metrics['Down Capture']:.1%} (Net Ratio: {bl_metrics['Capture Ratio']:.2f})")
    print(f"  4. Capital Protection: Max Drawdown Duration: {int(bl_metrics['Max Drawdown Days'])} days (Benchmark: {bench_mdd_days} days).\n")


if __name__ == "__main__":
    main()