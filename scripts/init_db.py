import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from sqlalchemy import text
from src.db.connection import get_db_engine

# Multi-Asset Universe Definition
ASSET_METADATA = [
    {"ticker": "SPY", "asset_name": "SPDR S&P 500 ETF Trust", "asset_class": "Equity", "sub_class": "US Large Cap", "currency": "USD"},
    {"ticker": "VGK", "asset_name": "Vanguard FTSE Europe ETF", "asset_class": "Equity", "sub_class": "Europe Developed", "currency": "USD"},
    {"ticker": "EEM", "asset_name": "iShares MSCI Emerging Markets ETF", "asset_class": "Equity", "sub_class": "Emerging Markets", "currency": "USD"},
    {"ticker": "TLT", "asset_name": "iShares 20+ Year Treasury Bond ETF", "asset_class": "Fixed Income", "sub_class": "US Sovereign Long", "currency": "USD"},
    {"ticker": "IEF", "asset_name": "iShares 7-10 Year Treasury Bond ETF", "asset_class": "Fixed Income", "sub_class": "US Sovereign Interm", "currency": "USD"},
    {"ticker": "LQD", "asset_name": "iShares iBoxx $ Inv Grade Corporate Bond ETF", "asset_class": "Fixed Income", "sub_class": "US IG Corporate", "currency": "USD"},
    {"ticker": "HYG", "asset_name": "iShares iBoxx $ High Yield Corporate Bond ETF", "asset_class": "Fixed Income", "sub_class": "US High Yield", "currency": "USD"},
    {"ticker": "GLD", "asset_name": "SPDR Gold Shares", "asset_class": "Commodity", "sub_class": "Precious Metals", "currency": "USD"},
    {"ticker": "VNQ", "asset_name": "Vanguard Real Estate ETF", "asset_class": "Real Estate", "sub_class": "US REITs", "currency": "USD"},
    {"ticker": "BNDX", "asset_name": "Vanguard Total International Bond ETF", "asset_class": "Fixed Income", "sub_class": "Global Ex-US Debt", "currency": "USD"}
]

# Baseline Global Capitalization Weights (Used for BL Market Equilibrium Implied Returns)
INITIAL_BENCHMARK_WEIGHTS = [
    {"ticker": "SPY", "as_of_date": "2024-01-01", "market_cap_weight": 0.40},
    {"ticker": "VGK", "as_of_date": "2024-01-01", "market_cap_weight": 0.15},
    {"ticker": "EEM", "as_of_date": "2024-01-01", "market_cap_weight": 0.08},
    {"ticker": "TLT", "as_of_date": "2024-01-01", "market_cap_weight": 0.07},
    {"ticker": "IEF", "as_of_date": "2024-01-01", "market_cap_weight": 0.08},
    {"ticker": "LQD", "as_of_date": "2024-01-01", "market_cap_weight": 0.08},
    {"ticker": "HYG", "as_of_date": "2024-01-01", "market_cap_weight": 0.04},
    {"ticker": "GLD", "as_of_date": "2024-01-01", "market_cap_weight": 0.03},
    {"ticker": "VNQ", "as_of_date": "2024-01-01", "market_cap_weight": 0.03},
    {"ticker": "BNDX", "as_of_date": "2024-01-01", "market_cap_weight": 0.04}
]

def initialize_database():
    engine = get_db_engine()
    schema_file = ROOT_DIR / "sql" / "schema.sql"
    
    print(f"Connecting to database and executing schema: {schema_file}")
    with open(schema_file, "r") as f:
        schema_sql = f.read()

    with engine.begin() as conn:
        conn.execute(text(schema_sql))
        print("Schema tables created successfully.")

        # Seed Assets
        print("Seeding asset metadata...")
        for asset in ASSET_METADATA:
            upsert_asset = text("""
                INSERT INTO assets (ticker, asset_name, asset_class, sub_class, currency)
                VALUES (:ticker, :asset_name, :asset_class, :sub_class, :currency)
                ON CONFLICT (ticker) DO UPDATE 
                SET asset_name = EXCLUDED.asset_name,
                    asset_class = EXCLUDED.asset_class,
                    sub_class = EXCLUDED.sub_class;
            """)
            conn.execute(upsert_asset, asset)

        # Seed Benchmark Weights
        print("Seeding global benchmark market cap weights...")
        for weight in INITIAL_BENCHMARK_WEIGHTS:
            upsert_weight = text("""
                INSERT INTO benchmark_weights (ticker, as_of_date, market_cap_weight)
                VALUES (:ticker, :as_of_date, :market_cap_weight)
                ON CONFLICT (ticker, as_of_date) DO UPDATE 
                SET market_cap_weight = EXCLUDED.market_cap_weight;
            """)
            conn.execute(upsert_weight, weight)

    print("Database initialization and universe seeding successfully completed.")

if __name__ == "__main__":
    initialize_database()