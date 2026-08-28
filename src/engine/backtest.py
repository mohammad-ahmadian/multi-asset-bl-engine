import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db.connection import get_db_engine
from src.engine.covariance import CovarianceEstimator
from src.engine.equilibrium import EquilibriumEstimator
from src.engine.views import ViewsManager
from src.engine.black_litterman import BlackLittermanModel
from src.engine.constraints import PortfolioConstraints
from src.engine.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Holds backtest simulation outputs."""
    strategy_name: str
    equity_curve: pd.Series        # Daily NAV series starting at 100.0
    daily_returns: pd.Series       # Net daily returns after transaction costs
    weights_history: pd.DataFrame  # Rebalanced target weights over time
    turnover_history: pd.Series    # One-way turnover at each rebalance point
    total_tx_costs: float          # Cumulative transaction costs in basis points
    total_trades_count: int


class BacktestEngine:
    """
    Executes walk-forward rolling backtests for multi-asset strategies.
    Handles lookahead-free parameter estimation, weight drift, and transaction costs.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        benchmark_weights: pd.Series,
        asset_classes: Dict[str, str],
        lookback_days: int = 756,          # 3-year rolling calibration window
        rebalance_freq: str = "M",         # Monthly rebalancing
        transaction_cost_bps: float = 10.0,# 10 bps per trade (0.10%)
        initial_capital: float = 100.0,
        engine: Engine = None
    ):
        self.returns = returns
        self.tickers = list(returns.columns)
        self.benchmark_weights = benchmark_weights.reindex(self.tickers).fillna(0.0)
        self.benchmark_weights = self.benchmark_weights / self.benchmark_weights.sum()
        self.asset_classes = asset_classes
        self.lookback_days = lookback_days
        self.rebalance_freq = rebalance_freq
        self.tx_cost_rate = float(transaction_cost_bps / 10000.0)
        self.initial_capital = float(initial_capital)
        self.engine = engine or get_db_engine()

        # Identify monthly rebalance date indices
        self.rebalance_dates = self._identify_rebalance_dates()

    def _identify_rebalance_dates(self) -> List[pd.Timestamp]:
        """Identifies month-end business dates within the backtest timeframe."""
        dates = self.returns.index[self.lookback_days:]
        rebalance_dates = []
        
        # Group dates by year and month, picking the last available trading day of each month
        df_dates = pd.DataFrame({"date": dates})
        df_dates["year_month"] = df_dates["date"].dt.to_period("M")
        
        for _, group in df_dates.groupby("year_month"):
            rebalance_dates.append(group["date"].iloc[-1])

        return sorted(rebalance_dates)

    def _generate_dynamic_views(self, hist_returns: pd.DataFrame) -> ViewsManager:
        """
        Systematic Tactical View Overlay based on rolling 12-month Momentum & Valuation:
        1. Overweight top-performing equity region over bottom equity region (+1.50%).
        2. Absolute Gold tactical view (+8.00% if 12M trend is positive).
        """
        views_mgr = ViewsManager(self.tickers)
        
        # Compute 12-month (252-day) cumulative performance
        perf_12m = (1.0 + hist_returns.iloc[-252:]).prod() - 1.0

        # Relative Equity View: VGK vs SPY
        vgk_perf = float(perf_12m.get("VGK", 0.0))
        spy_perf = float(perf_12m.get("SPY", 0.0))
        if vgk_perf > spy_perf:
            views_mgr.add_relative_view("VGK", "SPY", expected_outperformance=0.015, confidence=0.60)
        else:
            views_mgr.add_relative_view("SPY", "VGK", expected_outperformance=0.015, confidence=0.60)

        # Absolute Gold View
        gld_perf = float(perf_12m.get("GLD", 0.0))
        if gld_perf > 0.05:
            views_mgr.add_absolute_view("GLD", expected_return=0.085, confidence=0.70)
        else:
            views_mgr.add_absolute_view("GLD", expected_return=0.055, confidence=0.50)

        return views_mgr

    def run_strategy(self, strategy_type: str) -> BacktestResult:
        """
        Executes rolling walk-forward simulation for a specified strategy:
        'BENCHMARK_CAP', 'EQUAL_WEIGHT', 'HISTORICAL_MVO', 'BLACK_LITTERMAN'
        """
        dates = self.returns.index[self.lookback_days:]
        
        daily_portfolio_returns = pd.Series(index=dates, dtype=float)
        nav_series = pd.Series(index=dates, dtype=float)
        weights_records = []
        turnover_records = {}

        current_weights = pd.Series(0.0, index=self.tickers)
        current_nav = self.initial_capital
        total_tx_costs_bps = 0.0
        trades_count = 0

        for t_idx, current_date in enumerate(dates):
            # Check if current_date is a scheduled rebalance date
            if current_date in self.rebalance_dates or t_idx == 0:
                # Slice historical data up to t (Strictly Lookahead-Free)
                hist_slice = self.returns.loc[:current_date].iloc[-(self.lookback_days + 1):-1]
                
                # Estimate Ledoit-Wolf Covariance
                cov_engine = CovarianceEstimator(hist_slice)
                lw_cov, _ = cov_engine.ledoit_wolf_covariance()

                # Determine target weights based on strategy
                if strategy_type == "BENCHMARK_CAP":
                    target_w = self.benchmark_weights.copy()

                elif strategy_type == "EQUAL_WEIGHT":
                    target_w = pd.Series(1.0 / len(self.tickers), index=self.tickers)

                elif strategy_type == "HISTORICAL_MVO":
                    hist_mu = hist_slice.mean() * 252
                    mandate = PortfolioConstraints(
                        tickers=self.tickers,
                        asset_classes=self.asset_classes,
                        min_weight=0.0,
                        max_weight=0.35,
                        custom_asset_bounds={"SPY": (0.10, 0.35), "VGK": (0.05, 0.25), "GLD": (0.0, 0.10)},
                        asset_class_bounds={"Equity": (0.35, 0.60), "Fixed Income": (0.30, 0.55)}
                    )
                    opt = PortfolioOptimizer(hist_mu, lw_cov, risk_free_rate=0.02)
                    target_w = opt.optimize_utility(risk_aversion=2.85, constraints=mandate)

                elif strategy_type == "BLACK_LITTERMAN":
                    eq_engine = EquilibriumEstimator(hist_slice, lw_cov, self.benchmark_weights, risk_free_rate=0.02)
                    delta = eq_engine.calibrate_risk_aversion()
                    implied_prior = eq_engine.compute_implied_equilibrium_returns(delta)

                    views_mgr = self._generate_dynamic_views(hist_slice)
                    P, Q, conf = views_mgr.build_matrices()

                    bl_model = BlackLittermanModel(implied_prior, lw_cov, tau=0.05)
                    post_mu, post_cov, _ = bl_model.calculate_posterior(P, Q, conf)

                    mandate = PortfolioConstraints(
                        tickers=self.tickers,
                        asset_classes=self.asset_classes,
                        min_weight=0.0,
                        max_weight=0.35,
                        custom_asset_bounds={"SPY": (0.10, 0.35), "VGK": (0.05, 0.25), "GLD": (0.0, 0.10)},
                        asset_class_bounds={"Equity": (0.35, 0.60), "Fixed Income": (0.30, 0.55)},
                        max_tracking_error=0.040, # 4.0% TE active budget
                        benchmark_weights=self.benchmark_weights
                    )
                    opt = PortfolioOptimizer(post_mu, post_cov, risk_free_rate=0.02)
                    target_w = opt.optimize_utility(risk_aversion=delta, constraints=mandate)

                else:
                    raise ValueError(f"Unknown strategy: {strategy_type}")

                # Calculate Turnover: sum(|w_target - w_current|) as native float
                turnover = float(np.sum(np.abs(target_w.values - current_weights.values)).item())
                tx_cost = float(turnover * self.tx_cost_rate)
                total_tx_costs_bps += float(tx_cost * 10000.0)
                trades_count += int(np.sum(np.abs(target_w.values - current_weights.values) > 0.001))

                # Apply transaction cost deduction to NAV
                current_nav *= (1.0 - tx_cost)
                current_weights = target_w.copy()
                
                turnover_records[current_date] = turnover
                weights_records.append(pd.Series(target_w, name=current_date))

            # Daily Return & Weight Drift Simulation
            r_daily = self.returns.loc[current_date, self.tickers]
            port_daily_ret = float(np.sum(current_weights.values * r_daily.values).item())

            # Update NAV
            current_nav *= (1.0 + port_daily_ret)
            nav_series[current_date] = current_nav
            daily_portfolio_returns[current_date] = port_daily_ret

            # Drift weights for the next trading day: w_i * (1 + r_i) / (1 + r_port)
            drifted_unnorm = current_weights * (1.0 + r_daily)
            current_weights = drifted_unnorm / (1.0 + port_daily_ret)

        df_weights = pd.DataFrame(weights_records)
        s_turnover = pd.Series(turnover_records)

        return BacktestResult(
            strategy_name=strategy_type,
            equity_curve=nav_series,
            daily_returns=daily_portfolio_returns,
            weights_history=df_weights,
            turnover_history=s_turnover,
            total_tx_costs=total_tx_costs_bps,
            total_trades_count=trades_count
        )

    def log_rebalances_to_db(self, result: BacktestResult):
        """Saves historical rebalance allocations and turnover to PostgreSQL rebalance_history using batch insert."""
        if result.weights_history.empty:
            return

        batch_records = []
        for date_idx, row in result.weights_history.iterrows():
            rdate = date_idx.strftime("%Y-%m-%d")
            turnover_val = float(result.turnover_history.get(date_idx, 0.0))
            tx_cost_val = float(turnover_val * self.tx_cost_rate * 100.0) # in $ per $100 AUM

            for ticker in self.tickers:
                w = float(row.get(ticker, 0.0))
                batch_records.append({
                    "rdate": rdate,
                    "sname": str(result.strategy_name),
                    "ticker": str(ticker),
                    "weight": float(w),
                    "turnover": float(turnover_val),
                    "tx_cost": float(tx_cost_val)
                })

        with self.engine.begin() as conn:
            # Delete previous records for this strategy to keep clean state
            conn.execute(
                text("DELETE FROM rebalance_history WHERE strategy_name = :sname;"),
                {"sname": str(result.strategy_name)}
            )
            
            insert_query = text("""
                INSERT INTO rebalance_history (rebalance_date, strategy_name, ticker, weight, turnover, transaction_cost)
                VALUES (:rdate, :sname, :ticker, :weight, :turnover, :tx_cost);
            """)

            conn.execute(insert_query, batch_records)

        logger.info(f"Persisted {len(result.weights_history)} rebalancing events for {result.strategy_name} into PostgreSQL.")