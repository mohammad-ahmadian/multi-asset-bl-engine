-- ====================================================================
-- Institutional Multi-Asset Portfolio Schema
-- Author: Mohammad Ahmadian (https://github.com/mohammad-ahmadian)
-- ====================================================================

-- 1. Asset Universe Metadata
CREATE TABLE IF NOT EXISTS assets (
    ticker VARCHAR(10) PRIMARY KEY,
    asset_name VARCHAR(100) NOT NULL,
    asset_class VARCHAR(50) NOT NULL,
    sub_class VARCHAR(50),
    currency VARCHAR(3) DEFAULT 'USD',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Daily Price Series (OHLCV + Adjusted Close)
CREATE TABLE IF NOT EXISTS daily_prices (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL REFERENCES assets(ticker) ON DELETE CASCADE,
    price_date DATE NOT NULL,
    open_price NUMERIC(14, 4),
    high_price NUMERIC(14, 4),
    low_price NUMERIC(14, 4),
    close_price NUMERIC(14, 4),
    adj_close NUMERIC(14, 4) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_ticker_date UNIQUE (ticker, price_date)
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON daily_prices (ticker, price_date DESC);

-- 3. Benchmark Market Weights (for Black-Litterman Reverse Optimization)
CREATE TABLE IF NOT EXISTS benchmark_weights (
    ticker VARCHAR(10) NOT NULL REFERENCES assets(ticker) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,
    market_cap_weight NUMERIC(8, 6) NOT NULL,
    PRIMARY KEY (ticker, as_of_date)
);

-- 4. Black-Litterman Subjective Investor Views
CREATE TABLE IF NOT EXISTS bl_views (
    view_id SERIAL PRIMARY KEY,
    view_date DATE NOT NULL,
    view_type VARCHAR(20) NOT NULL, -- 'ABSOLUTE' or 'RELATIVE'
    asset_long VARCHAR(10) NOT NULL REFERENCES assets(ticker),
    asset_short VARCHAR(10) REFERENCES assets(ticker),
    expected_outperformance NUMERIC(8, 6) NOT NULL, -- Magnitude (e.g. 0.025 = 2.5%)
    confidence NUMERIC(5, 4) NOT NULL,              -- Range: (0.0, 1.0]
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Rebalancing History and Target Weights
CREATE TABLE IF NOT EXISTS rebalance_history (
    id SERIAL PRIMARY KEY,
    rebalance_date DATE NOT NULL,
    strategy_name VARCHAR(50) NOT NULL, -- 'EQUILIBRIUM', 'MVO', 'BL_CONSTRAINED'
    ticker VARCHAR(10) NOT NULL REFERENCES assets(ticker),
    weight NUMERIC(8, 6) NOT NULL,
    shares_held NUMERIC(14, 4),
    turnover NUMERIC(8, 6),
    transaction_cost NUMERIC(10, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rebalance_strat_date ON rebalance_history (strategy_name, rebalance_date DESC);