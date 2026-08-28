import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np

from src.engine.data_loader import DataLoader
from src.engine.backtest import BacktestEngine, BacktestResult


def calculate_performance_summary(res: BacktestResult, rf_annual: float = 0.02) -> dict:
    """Calculates comprehensive institutional portfolio performance and drawdown statistics."""
    rets = res.daily_returns.dropna()
    nav = res.equity_curve.dropna()
    
    # 1. Return Metrics
    total_return = float((nav.iloc[-1] / nav.iloc[0]) - 1.0)
    n_years = len(rets) / 252.0
    cagr = float((1.0 + total_return) ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
    
    # 2. Risk Metrics
    ann_vol = float(rets.std() * np.sqrt(252))
    sharpe = float((cagr - rf_annual) / ann_vol) if ann_vol > 0 else 0.0
    
    # Downside deviation for Sortino
    neg_rets = rets[rets < 0]
    downside_vol = float(neg_rets.std() * np.sqrt(252)) if len(neg_rets) > 0 else ann_vol
    sortino = float((cagr - rf_annual) / downside_vol) if downside_vol > 0 else 0.0
    
    # 3. Drawdown Analysis
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    max_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # 4. Turnover & Friction
    mean_monthly_turnover = float(res.turnover_history.mean()) if not res.turnover_history.empty else 0.0
    annualized_turnover = mean_monthly_turnover * 12.0

    return {
        "Total Return": total_return,
        "CAGR (Ann. Return)": cagr,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
        "Ann. Turnover": annualized_turnover,
        "Total TX Costs (bps)": res.total_tx_costs
    }


def main():
    print("\n" + "=" * 115)
    print("                HISTORICAL WALK-FORWARD ROLLING BACKTEST (LOOKAHEAD-FREE)")
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

    # 2. Instantiate Backtesting Engine
    engine = BacktestEngine(
        returns=returns,
        benchmark_weights=benchmark_weights,
        asset_classes=asset_classes,
        lookback_days=756,          # 3-year calibration lookback
        rebalance_freq="M",         # Monthly rebalance
        transaction_cost_bps=10.0   # 10 bps friction
    )

    print(f"\n[INFO] Universe: {len(returns.columns)} assets | Calibration Lookback: 3 Years (756 trading days)")
    print(f"[INFO] Backtest Execution Window: {engine.rebalance_dates[0].strftime('%Y-%m-%d')} to {engine.rebalance_dates[-1].strftime('%Y-%m-%d')}")
    print(f"[INFO] Total Monthly Rebalance Checkpoints: {len(engine.rebalance_dates)}")

    # 3. Execute 4 Strategies
    strategies = ["BENCHMARK_CAP", "EQUAL_WEIGHT", "HISTORICAL_MVO", "BLACK_LITTERMAN"]
    strategy_labels = {
        "BENCHMARK_CAP": "Global Benchmark (Cap-Weighted)",
        "EQUAL_WEIGHT": "Equal Weight (1/N)",
        "HISTORICAL_MVO": "Constrained Historical MVO",
        "BLACK_LITTERMAN": "Dynamic Constrained Black-Litterman"
    }

    summaries = {}
    for strat in strategies:
        print(f"  • Simulating {strat}...")
        res = engine.run_strategy(strat)
        summaries[strategy_labels[strat]] = calculate_performance_summary(res, rf_annual=rf_rate)
        # Log rebalances to database
        engine.log_rebalances_to_db(res)

    # 4. Format & Display Results Table
    df_perf = pd.DataFrame(summaries).T
    
    formatted_df = pd.DataFrame({
        "Total Return": df_perf["Total Return"].map(lambda x: f"{x:.2%}"),
        "CAGR": df_perf["CAGR (Ann. Return)"].map(lambda x: f"{x:.2%}"),
        "Volatility": df_perf["Ann. Volatility"].map(lambda x: f"{x:.2%}"),
        "Sharpe (Rf)": df_perf["Sharpe Ratio"].map(lambda x: f"{x:.2f}"),
        "Sortino": df_perf["Sortino Ratio"].map(lambda x: f"{x:.2f}"),
        "Max Drawdown": df_perf["Max Drawdown"].map(lambda x: f"{x:.2%}"),
        "Calmar": df_perf["Calmar Ratio"].map(lambda x: f"{x:.2f}"),
        "Ann. Turnover": df_perf["Ann. Turnover"].map(lambda x: f"{x:.1%}"),
        "TX Costs (bps)": df_perf["Total TX Costs (bps)"].map(lambda x: f"{x:.1f}")
    }, index=df_perf.index)

    print("\n" + "=" * 115)
    print("                       OUT-OF-SAMPLE STRATEGY PERFORMANCE AUDIT")
    print("=" * 115)
    print(formatted_df.to_string())
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()