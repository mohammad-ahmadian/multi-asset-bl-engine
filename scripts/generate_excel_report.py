import sys
from pathlib import Path
from datetime import datetime

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
from src.engine.stress_testing import RiskEngine
from src.analytics.rebalance_report import RebalanceReportGenerator


def main():
    print("\n" + "=" * 110)
    print("           GENERATING INSTITUTIONAL EXCEL REBALANCING ORDER SHEET")
    print("=" * 110)

    # 1. Load Data
    loader = DataLoader()
    returns = loader.get_returns_matrix()
    prices = loader.get_price_matrix()
    benchmark_weights = loader.get_benchmark_weights()
    rf_rate = loader.get_risk_free_rate()
    tickers = list(benchmark_weights.index)

    asset_names = {
        "SPY": "SPDR S&P 500 ETF Trust",
        "VGK": "Vanguard FTSE Europe ETF",
        "EEM": "iShares MSCI Emerging Markets ETF",
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "IEF": "iShares 7-10 Year Treasury Bond ETF",
        "LQD": "iShares iBoxx $ Inv Grade Corporate Bond ETF",
        "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
        "GLD": "SPDR Gold Shares",
        "VNQ": "Vanguard Real Estate ETF",
        "BNDX": "Vanguard Total International Bond ETF"
    }

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
    target_weights = opt.optimize_utility(risk_aversion=delta, constraints=mandate)

    # 3. Simulate Current Drifting Weights (Month-End state before rebalance)
    # Drift benchmark weights slightly to simulate 1 month of price movement
    drift_noise = np.array([0.02, -0.01, -0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.00])
    current_weights = benchmark_weights + drift_noise
    current_weights = current_weights / current_weights.sum()

    latest_prices = prices.iloc[-1]

    # 4. Generate Trade Execution Table
    AUM = 25_000_000.0 # $25 Million Fund
    generator = RebalanceReportGenerator(portfolio_aum=AUM, base_currency="USD")

    trades_df = generator.generate_trade_orders_table(
        current_prices=latest_prices,
        current_weights=current_weights,
        target_weights=target_weights,
        asset_names=asset_names,
        asset_classes=asset_classes,
        tx_cost_bps=10.0
    )

    # 5. Compute Risk & Mandate Metrics
    port_ret_series = returns @ target_weights
    risk_stats = RiskEngine(port_ret_series).compute_tail_risk()
    
    active_w = target_weights - benchmark_weights
    te = float(np.sqrt(active_w.values @ lw_cov.values @ active_w.values))
    active_share = float(0.5 * np.sum(np.abs(active_w.values)))
    exp_ret = float(target_weights.values @ post_mu.values)
    port_vol = float(np.sqrt(target_weights.values @ post_cov.values @ target_weights.values))
    total_tx_cost = float(trades_df["tx_cost"].sum())

    risk_metrics = {
        "expected_return": exp_ret,
        "volatility": port_vol,
        "tracking_error": te,
        "active_share": active_share,
        "cf_var_99": risk_stats.cornish_fisher_var_99,
        "total_tx_cost": total_tx_cost
    }

    # 6. Asset Class Mandate Summary Table
    ac_summary = []
    mandate_limits = {
        "Equity": (0.35, 0.60),
        "Fixed Income": (0.30, 0.55),
        "Commodity": (0.00, 0.10),
        "Real Estate": (0.00, 0.08)
    }

    for ac, (min_l, max_l) in mandate_limits.items():
        ac_tickers = [t for t, c in asset_classes.items() if c == ac]
        curr_ac_wgt = float(current_weights.reindex(ac_tickers).sum())
        targ_ac_wgt = float(target_weights.reindex(ac_tickers).sum())
        
        status = "COMPLIANT (PASS)" if min_l <= targ_ac_wgt <= max_l else "BREACH"
        ac_summary.append({
            "asset_class": ac,
            "current_wgt": curr_ac_wgt,
            "target_wgt": targ_ac_wgt,
            "min_limit": min_l,
            "max_limit": max_l,
            "status": status
        })

    df_ac_summary = pd.DataFrame(ac_summary)

    # 7. Export to Excel File
    timestamp_str = datetime.now().strftime("%Y%m%d")
    output_file = ROOT_DIR / "reports" / f"Rebalance_Order_Sheet_{timestamp_str}.xlsx"
    generator.export_to_excel(trades_df, risk_metrics, df_ac_summary, output_file)

    print(f"\n[SUCCESS] Institutional Excel Order Sheet generated successfully!")
    print(f"  • File Location: {output_file.resolve()}")
    print(f"  • Total Portfolio AUM: ${AUM:,.2f}")
    print(f"  • Total Rebalancing Volume: ${trades_df['trade_value'].sum():,.2f}")
    print(f"  • Total Estimated Friction / Costs: ${total_tx_cost:,.2f} ({total_tx_cost / AUM * 10000:.1f} bps)\n")


if __name__ == "__main__":
    main()