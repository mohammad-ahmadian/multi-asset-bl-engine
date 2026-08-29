import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
from src.engine.pipeline import PortfolioPipeline


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-Asset Black-Litterman Portfolio Engine CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "calibrate", "optimize", "backtest", "report"],
        default="all",
        help="Execution mode."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--export-excel",
        action="store_true",
        default=True,
        help="Generate institutional Excel Rebalance Order Sheet."
    )
    return parser.parse_args()


def print_banner():
    print("\n" + "=" * 115)
    print("      MULTI-ASSET BLACK-LITTERMAN PORTFOLIO ENGINE — PRODUCTION CLI PIPELINE")
    print("      Author: Mohammad Ahmadian  |  Target: Portfolio Analyst / Quantitative Allocation")
    print("=" * 115 + "\n")


def main():
    args = parse_arguments()
    print_banner()

    config_path = ROOT_DIR / args.config
    pipeline = PortfolioPipeline(config_path)

    if args.mode == "all":
        results = pipeline.run_full_pipeline(export_excel=args.export_excel)
        
        # Display Final Optimal Allocation
        w_opt = results["target_weights"]
        bench_w = results["calibration"]["benchmark_weights"]
        mu_post = results["black_litterman"]["posterior_mu"]
        
        summary_df = pd.DataFrame({
            "Asset": w_opt.index,
            "Description": [pipeline.asset_names.get(t, t) for t in w_opt.index],
            "Asset Class": [pipeline.asset_classes.get(t, "Other") for t in w_opt.index],
            "Benchmark Wgt": bench_w.map(lambda x: f"{x:.1%}"),
            "Target Wgt": w_opt.map(lambda x: f"{x:.1%}"),
            "Active Tilt": (w_opt - bench_w).map(lambda x: f"{x:+.1%}"),
            "Posterior E[R]": mu_post.map(lambda x: f"{x:.2%}")
        })

        print("\n" + "=" * 115)
        print("                      OPTIMAL UCITS CONSTRAINED TARGET PORTFOLIO")
        print("=" * 115)
        print(summary_df.to_string(index=False))
        print("=" * 115)
        
        if results["excel_path"]:
            print(f"\n[INFO] Institutional Excel Order Sheet created at:\n  -> {results['excel_path'].resolve()}")
        print(f"[INFO] Pipeline finished successfully in {results['elapsed_seconds']:.2f} seconds.\n")

    elif args.mode == "calibrate":
        calib = pipeline.run_calibration_stage()
        print(f"Calibration successful. Shrinkage Delta: {calib['shrinkage_delta']:.4f}, Risk Aversion (Delta): {calib['delta']:.2f}")

    elif args.mode == "optimize":
        calib = pipeline.run_calibration_stage()
        bl_res = pipeline.run_black_litterman_stage(calib)
        target_w = pipeline.run_optimization_stage(calib, bl_res)
        print("Optimal Target Allocations:")
        print(target_w)

    elif args.mode == "backtest":
        calib = pipeline.run_calibration_stage()
        res = pipeline.run_backtest_stage(calib)
        print(f"Backtests completed for: {list(res.keys())}")

    elif args.mode == "report":
        calib = pipeline.run_calibration_stage()
        bl_res = pipeline.run_black_litterman_stage(calib)
        target_w = pipeline.run_optimization_stage(calib, bl_res)
        risk_res = pipeline.run_risk_and_stress_stage(calib, target_w)
        excel_path = pipeline.export_excel_report(calib, bl_res, target_w, risk_res)
        print(f"Report exported to: {excel_path}")


if __name__ == "__main__":
    main()