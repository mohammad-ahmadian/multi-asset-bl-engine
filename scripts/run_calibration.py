import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
from src.engine.data_loader import DataLoader
from src.engine.covariance import CovarianceEstimator
from src.engine.equilibrium import EquilibriumEstimator


def main():
    print("\n" + "=" * 90)
    print("      MULTI-ASSET RISK CALIBRATION & MARKET EQUILIBRIUM (BLACK-LITTERMAN PRIOR)")
    print("=" * 90)

    # 1. Load Data
    loader = DataLoader()
    returns = loader.get_returns_matrix(method="simple")
    benchmark_weights = loader.get_benchmark_weights()
    rf_rate = loader.get_risk_free_rate()

    print(f"\n[INFO] Total Observations: {len(returns)} trading days.")
    print(f"[INFO] Annualized Risk-Free Rate Proxy (^IRX): {rf_rate:.2%}")

    # 2. Covariance Estimation & Shrinkage
    cov_engine = CovarianceEstimator(returns)
    sample_cov = cov_engine.sample_covariance()
    lw_cov, shrinkage_delta = cov_engine.ledoit_wolf_covariance()

    sample_cond = cov_engine.calculate_condition_number(sample_cov)
    lw_cond = cov_engine.calculate_condition_number(lw_cov)

    print(f"\n[COVARIANCE AUDIT]")
    print(f"  • Ledoit-Wolf Optimal Shrinkage Intensity (delta): {shrinkage_delta:.4f}")
    print(f"  • Sample Covariance Condition Number:              {sample_cond:.2f}")
    print(f"  • Ledoit-Wolf Covariance Condition Number:         {lw_cond:.2f} ({(1 - lw_cond/sample_cond)*100:.1f}% improvement)")

    # 3. Market Equilibrium Calibration (Pi)
    eq_engine = EquilibriumEstimator(
        returns=returns,
        covariance_matrix=lw_cov,
        benchmark_weights=benchmark_weights,
        risk_free_rate=rf_rate
    )

    delta = eq_engine.calibrate_risk_aversion()
    implied_returns = eq_engine.compute_implied_equilibrium_returns(delta=delta)
    hist_stats = eq_engine.compute_historical_statistics()

    # 4. Consolidate Results Table
    results_df = hist_stats.copy()
    results_df["implied_equilibrium_return (Pi)"] = implied_returns

    # Formatting
    formatted_df = pd.DataFrame({
        "Asset Class": results_df.index,
        "Bench Weight": results_df["benchmark_weight"].map(lambda x: f"{x:.1%}"),
        "Hist Volatility": results_df["annualized_volatility"].map(lambda x: f"{x:.2%}"),
        "Hist Mean Return": results_df["historical_mean_return"].map(lambda x: f"{x:.2%}"),
        "Implied Return (Pi)": results_df["implied_equilibrium_return (Pi)"].map(lambda x: f"{x:.2%}"),
        "Hist Sharpe": results_df["historical_sharpe"].map(lambda x: f"{x:.2f}")
    })

    print(f"\n[MARKET RISK AVERSION] Calibrated Delta (delta): {delta:.3f}")
    print("\n" + "=" * 90)
    print("                    ASSET UNIVERSE EQUILIBRIUM & RISK PARAMETERS")
    print("=" * 90)
    print(formatted_df.to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()