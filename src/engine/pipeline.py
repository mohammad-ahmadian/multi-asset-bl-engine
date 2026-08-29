import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime
import yaml
import pandas as pd
import numpy as np

from src.db.connection import get_db_engine
from src.engine.data_loader import DataLoader
from src.engine.covariance import CovarianceEstimator
from src.engine.equilibrium import EquilibriumEstimator
from src.engine.views import ViewsManager
from src.engine.black_litterman import BlackLittermanModel
from src.engine.constraints import PortfolioConstraints
from src.engine.optimizer import PortfolioOptimizer
from src.engine.backtest import BacktestEngine, BacktestResult
from src.engine.analytics import PortfolioAnalytics
from src.engine.stress_testing import RiskEngine, StressTestEngine
from src.analytics.rebalance_report import RebalanceReportGenerator

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]


class PortfolioPipeline:
    """
    End-to-End Multi-Asset Black-Litterman Portfolio Execution Pipeline.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or ROOT_DIR / "config" / "config.yaml"
        self.config = self._load_config()
        self.engine = get_db_engine()

        # Extract core config elements
        self.universe = self.config["universe"]["tickers"]
        self.asset_names = self.config["universe"]["asset_names"]
        self.asset_classes = self.config["universe"]["asset_classes"]
        self.fund_params = self.config["fund_parameters"]
        self.model_params = self.config["model_parameters"]
        self.views_config = self.config["tactical_views"]
        self.mandate_config = self.config["mandate_constraints"]

    def _load_config(self) -> Dict[str, Any]:
        """Loads and parses the YAML configuration file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def run_calibration_stage(self) -> Dict[str, Any]:
        """Loads data from PostgreSQL and computes Ledoit-Wolf Covariance & Market Equilibrium."""
        loader = DataLoader(self.engine)
        returns = loader.get_returns_matrix()
        bench_weights = loader.get_benchmark_weights()
        rf_rate = loader.get_risk_free_rate()

        cov_engine = CovarianceEstimator(returns)
        lw_cov, shrinkage_delta = cov_engine.ledoit_wolf_covariance()

        eq_engine = EquilibriumEstimator(returns, lw_cov, bench_weights, rf_rate)
        delta = eq_engine.calibrate_risk_aversion()
        implied_prior = eq_engine.compute_implied_equilibrium_returns(delta)

        return {
            "loader": loader,
            "returns": returns,
            "benchmark_weights": bench_weights,
            "rf_rate": rf_rate,
            "lw_cov": lw_cov,
            "shrinkage_delta": shrinkage_delta,
            "delta": delta,
            "implied_prior": implied_prior
        }

    def run_black_litterman_stage(self, calib: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs views and computes Black-Litterman posterior distribution."""
        views_mgr = ViewsManager(self.universe, self.engine)

        for v in self.views_config:
            if v["view_type"].upper() == "RELATIVE":
                views_mgr.add_relative_view(
                    v["asset_long"],
                    v["asset_short"],
                    float(v["expected_outperformance"]),
                    float(v["confidence"])
                )
            elif v["view_type"].upper() == "ABSOLUTE":
                views_mgr.add_absolute_view(
                    v["asset_long"],
                    float(v["expected_outperformance"]),
                    float(v["confidence"])
                )

        views_mgr.sync_to_database()
        P, Q, conf = views_mgr.build_matrices()

        bl_model = BlackLittermanModel(
            implied_returns=calib["implied_prior"],
            covariance_matrix=calib["lw_cov"],
            tau=float(self.model_params.get("black_litterman_tau", 0.05))
        )

        posterior_mu, posterior_cov, omega = bl_model.calculate_posterior(P, Q, conf)

        return {
            "views_mgr": views_mgr,
            "bl_model": bl_model,
            "posterior_mu": posterior_mu,
            "posterior_cov": posterior_cov,
            "omega": omega
        }

    def run_optimization_stage(self, calib: Dict[str, Any], bl_res: Dict[str, Any]) -> pd.Series:
        """Solves the constrained quadratic optimization problem under UCITS limits."""
        # Convert custom bounds from list to tuple
        custom_bounds = {k: tuple(v) for k, v in self.mandate_config["custom_asset_bounds"].items()}
        class_bounds = {k: tuple(v) for k, v in self.mandate_config["asset_class_bounds"].items()}

        mandate = PortfolioConstraints(
            tickers=self.universe,
            asset_classes=self.asset_classes,
            min_weight=float(self.mandate_config.get("min_weight", 0.0)),
            max_weight=float(self.mandate_config.get("max_weight", 0.35)),
            custom_asset_bounds=custom_bounds,
            asset_class_bounds=class_bounds,
            max_tracking_error=float(self.mandate_config.get("max_tracking_error", 0.035)),
            benchmark_weights=calib["benchmark_weights"]
        )

        opt = PortfolioOptimizer(
            expected_returns=bl_res["posterior_mu"],
            covariance_matrix=bl_res["posterior_cov"],
            risk_free_rate=calib["rf_rate"]
        )

        target_weights = opt.optimize_utility(
            risk_aversion=calib["delta"],
            constraints=mandate
        )

        return target_weights

    def run_backtest_stage(self, calib: Dict[str, Any]) -> Dict[str, BacktestResult]:
        """Runs rolling out-of-sample walk-forward simulations for all strategies."""
        engine = BacktestEngine(
            returns=calib["returns"],
            benchmark_weights=calib["benchmark_weights"],
            asset_classes=self.asset_classes,
            lookback_days=int(self.model_params.get("lookback_days", 756)),
            rebalance_freq=self.fund_params.get("rebalance_frequency", "M"),
            transaction_cost_bps=float(self.fund_params.get("transaction_cost_bps", 10.0)),
            engine=self.engine
        )

        strategies = ["BENCHMARK_CAP", "EQUAL_WEIGHT", "HISTORICAL_MVO", "BLACK_LITTERMAN"]
        results = {}

        for s in strategies:
            res = engine.run_strategy(s)
            engine.log_rebalances_to_db(res)
            results[s] = res

        return results

    def run_risk_and_stress_stage(self, calib: Dict[str, Any], target_weights: pd.Series) -> Dict[str, Any]:
        """Computes statistical tail-risk, historical crisis replays, and factor shocks."""
        port_daily_ret = calib["returns"] @ target_weights
        risk_engine = RiskEngine(port_daily_ret)
        tail_metrics = risk_engine.compute_tail_risk()

        stress_engine = StressTestEngine(calib["returns"])
        hist_scenarios = stress_engine.run_historical_replays(target_weights, calib["benchmark_weights"])
        hypo_shocks = stress_engine.run_hypothetical_shocks(target_weights, calib["benchmark_weights"])

        return {
            "tail_metrics": tail_metrics,
            "hist_scenarios": hist_scenarios,
            "hypo_shocks": hypo_shocks
        }

    def export_excel_report(
        self,
        calib: Dict[str, Any],
        bl_res: Dict[str, Any],
        target_weights: pd.Series,
        risk_res: Dict[str, Any]
    ) -> Path:
        """Generates the institutional Excel Rebalancing Order Sheet."""
        prices = calib["loader"].get_price_matrix()
        latest_prices = prices.iloc[-1]

        # Simulate month-end drifting current weights
        bench_w = calib["benchmark_weights"]
        drift_noise = np.array([0.02, -0.01, -0.01, -0.01, 0.00, 0.01, 0.00, 0.01, -0.01, 0.00])
        current_weights = bench_w + drift_noise
        current_weights = current_weights / current_weights.sum()

        generator = RebalanceReportGenerator(
            portfolio_aum=float(self.fund_params["initial_aum"]),
            base_currency=self.fund_params["base_currency"],
            fund_name=self.config["project"]["fund_name"]
        )

        trades_df = generator.generate_trade_orders_table(
            current_prices=latest_prices,
            current_weights=current_weights,
            target_weights=target_weights,
            asset_names=self.asset_names,
            asset_classes=self.asset_classes,
            tx_cost_bps=float(self.fund_params["transaction_cost_bps"])
        )

        active_w = target_weights - bench_w
        te = float(np.sqrt(active_w.values @ calib["lw_cov"].values @ active_w.values))
        active_share = float(0.5 * np.sum(np.abs(active_w.values)))
        exp_ret = float(target_weights.values @ bl_res["posterior_mu"].values)
        port_vol = float(np.sqrt(target_weights.values @ bl_res["posterior_cov"].values @ target_weights.values))
        total_tx_cost = float(trades_df["tx_cost"].sum())

        risk_metrics = {
            "expected_return": exp_ret,
            "volatility": port_vol,
            "tracking_error": te,
            "active_share": active_share,
            "cf_var_99": risk_res["tail_metrics"].cornish_fisher_var_99,
            "total_tx_cost": total_tx_cost
        }

        # Asset class compliance summary
        ac_summary = []
        for ac, bounds in self.mandate_config["asset_class_bounds"].items():
            min_l, max_l = bounds
            ac_tickers = [t for t, c in self.asset_classes.items() if c == ac]
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

        timestamp_str = datetime.now().strftime("%Y%m%d")
        output_file = ROOT_DIR / "reports" / f"Rebalance_Order_Sheet_{timestamp_str}.xlsx"
        generator.export_to_excel(trades_df, risk_metrics, df_ac_summary, output_file)
        return output_file

    def run_full_pipeline(self, export_excel: bool = True) -> Dict[str, Any]:
        """Executes the complete pipeline end-to-end."""
        start_time = time.time()
        logger.info(">>> Initializing Master Portfolio Engine Pipeline...")

        # Stage 1: Calibration
        calib = self.run_calibration_stage()
        logger.info("Stage 1/5 Complete: Market Equilibrium & Covariance calibrated.")

        # Stage 2: Black-Litterman
        bl_res = self.run_black_litterman_stage(calib)
        logger.info("Stage 2/5 Complete: Black-Litterman posterior derived.")

        # Stage 3: Optimization
        target_weights = self.run_optimization_stage(calib, bl_res)
        logger.info("Stage 3/5 Complete: UCITS-compliant target allocations solved.")

        # Stage 4: Rolling Backtest
        backtest_results = self.run_backtest_stage(calib)
        logger.info("Stage 4/5 Complete: Walk-forward backtests executed & logged.")

        # Stage 5: Risk & Stress-Testing
        risk_res = self.run_risk_and_stress_stage(calib, target_weights)
        logger.info("Stage 5/5 Complete: Tail-risk and crisis stress tests evaluated.")

        # Optional Stage: Export Excel
        excel_path = None
        if export_excel:
            excel_path = self.export_excel_report(calib, bl_res, target_weights, risk_res)
            logger.info(f"Report Generated: {excel_path.name}")

        elapsed = time.time() - start_time
        logger.info(f">>> Full Pipeline Execution Completed in {elapsed:.2f} seconds.")

        return {
            "calibration": calib,
            "black_litterman": bl_res,
            "target_weights": target_weights,
            "backtest_results": backtest_results,
            "risk_results": risk_res,
            "excel_path": excel_path,
            "elapsed_seconds": elapsed
        }