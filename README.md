
# Multi-Asset Allocation Engine & Black-Litterman Framework

**Author:** Mohammad Ahmadian
**GitHub Profile:** [mohammad-ahmadian](https://github.com/mohammad-ahmadian)
**Target Role:** Portfolio Analyst / Quantitative Risk Analyst

## Project Overview

An institutional-grade portfolio construction and optimization system integrating:

* **Relational Database:** PostgreSQL schema for multi-asset OHLCV data, views, and rebalancing logs.
* **Asset Allocation:** Black-Litterman framework combining global market equilibrium returns with investor views.
* **Risk & Analytics:** Regularized covariance estimation (Ledoit-Wolf), portfolio constraints, and automated Excel reporting.

## Project Structure

* `config/`: Universe definitions and hyperparameters.
* `sql/`: DDL schemas and analytical views.
* `src/db/`: Database connection management.
* `src/engine/`: Optimization algorithms and rebalancing engine.
* `scripts/`: Data ingestion and execution pipelines.
