import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db.connection import get_db_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config() -> Dict[str, Any]:
    """Loads configuration parameters from config.yaml."""
    config_path = ROOT_DIR / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class MarketDataIngestor:
    """Handles downloading and ingesting multi-asset market data into PostgreSQL."""

    def __init__(self, engine: Engine = None):
        self.engine = engine or get_db_engine()
        self.config = load_config()
        self.tickers: List[str] = self.config["universe"]["tickers"]
        self.start_date: str = self.config["parameters"]["start_date"]
        self.end_date: str = self.config["parameters"]["end_date"]

    def fetch_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Downloads historical daily OHLCV data for a given ticker from Yahoo Finance.
        Handles timezone stripping and data cleaning.
        """
        logger.info(f"Downloading historical data for {ticker} from {self.start_date} to {self.end_date}...")
        
        df = yf.download(
            tickers=ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            logger.warning(f"No data returned for ticker: {ticker}")
            return pd.DataFrame()

        # Flatten multi-level columns if returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        
        # Standardize column naming
        df.rename(columns={
            "Date": "price_date",
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Adj Close": "adj_close",
            "Volume": "volume"
        }, inplace=True)

        # Clean date and types
        df["price_date"] = pd.to_datetime(df["price_date"]).dt.date
        df["ticker"] = ticker
        
        # Drop rows with NaN in essential columns
        df = df.dropna(subset=["adj_close", "price_date"])
        
        # Ensure volume is integer
        df["volume"] = df["volume"].fillna(0).astype(int)

        # Select target schema columns
        target_cols = ["ticker", "price_date", "open_price", "high_price", "low_price", "close_price", "adj_close", "volume"]
        return df[target_cols]

    def upsert_prices(self, df: pd.DataFrame) -> int:
        """
        Inserts or updates price records in the PostgreSQL daily_prices table.
        """
        if df.empty:
            return 0

        records = df.to_dict(orient="records")

        upsert_query = text("""
            INSERT INTO daily_prices (ticker, price_date, open_price, high_price, low_price, close_price, adj_close, volume)
            VALUES (:ticker, :price_date, :open_price, :high_price, :low_price, :close_price, :adj_close, :volume)
            ON CONFLICT (ticker, price_date) DO UPDATE 
            SET open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                adj_close = EXCLUDED.adj_close,
                volume = EXCLUDED.volume;
        """)

        with self.engine.begin() as conn:
            conn.execute(upsert_query, records)

        return len(records)

    def run_universe_ingestion(self) -> Dict[str, int]:
        """
        Iterates over all tickers defined in the universe and ingests their price history.
        """
        summary = {}
        logger.info(f"Starting ingestion for universe: {self.tickers}")

        for ticker in self.tickers:
            try:
                df = self.fetch_ticker_data(ticker)
                rows_inserted = self.upsert_prices(df)
                summary[ticker] = rows_inserted
                logger.info(f"Successfully processed {ticker}: {rows_inserted} daily records upserted.")
            except Exception as e:
                logger.error(f"Error ingesting {ticker}: {str(e)}", exc_info=True)
                summary[ticker] = 0

        return summary