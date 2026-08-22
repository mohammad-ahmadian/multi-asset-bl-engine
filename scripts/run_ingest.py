import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
from sqlalchemy import text
from src.db.connection import get_db_engine
from src.db.ingest_prices import MarketDataIngestor


def print_database_summary():
    """Queries PostgreSQL and displays an overview of ingested data."""
    engine = get_db_engine()
    query = text("""
        SELECT 
            a.ticker,
            a.asset_name,
            a.asset_class,
            COUNT(p.id) AS total_trading_days,
            MIN(p.price_date) AS start_date,
            MAX(p.price_date) AS end_date,
            ROUND(AVG(p.adj_close), 2) AS avg_adj_close,
            ROUND(MAX(p.adj_close), 2) AS max_adj_close,
            ROUND(MIN(p.adj_close), 2) AS min_adj_close
        FROM assets a
        LEFT JOIN daily_prices p ON a.ticker = p.ticker
        GROUP BY a.ticker, a.asset_name, a.asset_class
        ORDER BY a.asset_class, a.ticker;
    """)

    with engine.connect() as conn:
        df_summary = pd.read_sql(query, conn)

    print("\n" + "=" * 105)
    print("                      DATABASE INGESTION & DATA INTEGRITY AUDIT")
    print("=" * 105)
    print(df_summary.to_string(index=False))
    print("=" * 105 + "\n")


def main():
    print("\n[INFO] Initializing Market Data Ingestion Pipeline...")
    ingestor = MarketDataIngestor()
    
    # 1. Ingest Universe Tickers
    results = ingestor.run_universe_ingestion()
    
    # 2. Ingest Risk-Free Rate Proxy (^IRX)
    rf_ticker = ingestor.config["parameters"].get("risk_free_rate_ticker", "^IRX")
    print(f"\n[INFO] Ingesting Risk-Free Rate Proxy: {rf_ticker}...")
    
    # Insert ^IRX into assets table if not already present
    engine = get_db_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO assets (ticker, asset_name, asset_class, sub_class, currency)
            VALUES (:ticker, :name, :asset_class, :sub_class, 'USD')
            ON CONFLICT (ticker) DO NOTHING;
        """), {
            "ticker": rf_ticker,
            "name": "13-Week Treasury Bill Yield Index",
            "asset_class": "Cash Equivalent",
            "sub_class": "Risk-Free Proxy"
        })
    
    rf_df = ingestor.fetch_ticker_data(rf_ticker)
    rf_rows = ingestor.upsert_prices(rf_df)
    print(f"[INFO] Ingested {rf_rows} daily data points for {rf_ticker}.")

    # 3. Print Data Verification Summary Table
    print_database_summary()


if __name__ == "__main__":
    main()