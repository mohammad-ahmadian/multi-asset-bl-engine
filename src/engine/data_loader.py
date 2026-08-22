import logging
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db.connection import get_db_engine

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and preprocesses asset prices, benchmark weights, and risk-free rates from PostgreSQL."""

    def __init__(self, engine: Engine = None):
        self.engine = engine or get_db_engine()

    def get_price_matrix(self, tickers: List[str] = None) -> pd.DataFrame:
        """
        Extracts adjusted closing prices for investable assets and pivots into a wide DataFrame.
        Index: price_date, Columns: ticker.
        """
        query = text("""
            SELECT price_date, ticker, adj_close
            FROM daily_prices
            WHERE ticker != '^IRX'
            ORDER BY price_date ASC, ticker ASC;
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            raise ValueError("No price data found in daily_prices table. Run scripts/run_ingest.py first.")

        # Pivot to wide format: rows = dates, columns = tickers
        price_pivot = df.pivot(index="price_date", columns="ticker", values="adj_close")
        price_pivot.index = pd.to_datetime(price_pivot.index)

        if tickers:
            price_pivot = price_pivot[[t for t in tickers if t in price_pivot.columns]]

        # Forward-fill any minor holiday gaps, then drop remaining NaNs
        price_pivot = price_pivot.ffill().dropna()
        return price_pivot

    def get_returns_matrix(self, method: str = "simple") -> pd.DataFrame:
        """
        Calculates daily returns across all assets.
        method: 'simple' (arithmetic) or 'log' (continuous).
        """
        prices = self.get_price_matrix()
        if method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        return returns.dropna()

    def get_benchmark_weights(self) -> pd.Series:
        """
        Retrieves the latest global market capitalization proxy weights from benchmark_weights table.
        """
        query = text("""
            SELECT ticker, market_cap_weight
            FROM benchmark_weights
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM benchmark_weights);
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            raise ValueError("No benchmark weights found in benchmark_weights table.")

        weights_series = df.set_index("ticker")["market_cap_weight"]
        # Normalize to ensure sum is exactly 1.0
        weights_series = weights_series / weights_series.sum()
        return weights_series

    def get_risk_free_rate(self) -> float:
        """
        Calculates the average annualized risk-free rate over the last 12 months using ^IRX.
        Returns rate in decimal form (e.g., 0.045 for 4.5%).
        """
        query = text("""
            SELECT adj_close
            FROM daily_prices
            WHERE ticker = '^IRX'
            ORDER BY price_date DESC
            LIMIT 252;
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty or df["adj_close"].dropna().empty:
            logger.warning("No ^IRX data found. Defaulting risk-free rate to 2.0% (0.02).")
            return 0.02

        # ^IRX is in annual percentage points (e.g. 5.25 -> 0.0525)
        mean_yield = df["adj_close"].mean() / 100.0
        return float(max(mean_yield, 0.0001)) # Floor at 0.01%