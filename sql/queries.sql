-- ====================================================================
-- Analytical & Validation Queries for multi-asset-bl-engine
-- ====================================================================

-- 1. Data Integrity & Date Alignment Check
SELECT 
    a.ticker,
    a.asset_name,
    a.asset_class,
    COUNT(p.id) AS trading_days,
    MIN(p.price_date) AS earliest_date,
    MAX(p.price_date) AS latest_date
FROM assets a
LEFT JOIN daily_prices p ON a.ticker = p.ticker
GROUP BY a.ticker, a.asset_name, a.asset_class
ORDER BY a.asset_class, a.ticker;


-- 2. Daily Log Returns for Investable Assets (Excluding Risk-Free Yield Index)
WITH price_lags AS (
    SELECT 
        ticker,
        price_date,
        adj_close,
        LAG(adj_close, 1) OVER (PARTITION BY ticker ORDER BY price_date) AS prev_adj_close
    FROM daily_prices
    WHERE ticker != '^IRX' -- Only calculate price returns for investable assets
)
SELECT 
    ticker,
    price_date,
    adj_close,
    prev_adj_close,
    CASE 
        WHEN prev_adj_close > 0 AND adj_close > 0 
        THEN ROUND(LN(adj_close / prev_adj_close), 6)
        ELSE NULL 
    END AS log_return
FROM price_lags
WHERE prev_adj_close IS NOT NULL
ORDER BY price_date DESC, ticker
LIMIT 20;


-- 3. Risk-Free Rate Daily Equivalent Check (^IRX)
-- (^IRX is in annual percentage points, e.g., 5.25 -> daily rf = (5.25 / 100) / 252)
SELECT 
    price_date,
    adj_close AS annual_yield_pct,
    ROUND((adj_close / 100.0) / 252.0, 8) AS daily_risk_free_rate
FROM daily_prices
WHERE ticker = '^IRX'
ORDER BY price_date DESC
LIMIT 10;