import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Configure Institutional Typography and Styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#EBEBEB"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7


class PortfolioVisualizer:
    """Generates publication-quality charts for institutional portfolio reporting."""

    COLORS = {
        "BLACK_LITTERMAN": "#1B365D",    # Deep Navy
        "BENCHMARK_CAP": "#5F6368",      # Slate Gray
        "EQUAL_WEIGHT": "#E37400",       # Amber Orange
        "HISTORICAL_MVO": "#C5221F",     # Crimson Red
        "PRIOR_FRONTIER": "#80868B",     # Muted Gray
        "POST_FRONTIER": "#137333"       # Forest Green
    }

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_cumulative_performance_and_drawdown(
        self,
        strategy_results: Dict[str, Any],
        filename: str = "cumulative_returns.png"
    ) -> Path:
        """
        Creates a 2-panel chart: Upper panel shows cumulative NAV;
        lower panel shows the underwater drawdown profile.
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.08}
        )

        labels = {
            "BLACK_LITTERMAN": "Dynamic Constrained Black-Litterman",
            "BENCHMARK_CAP": "Global Cap Benchmark",
            "EQUAL_WEIGHT": "Equal Weight (1/N)",
            "HISTORICAL_MVO": "Constrained Historical MVO"
        }

        # 1. Equity Curves
        for strat, res in strategy_results.items():
            nav = res.equity_curve
            color = self.COLORS.get(strat, "#000000")
            linewidth = 2.4 if strat == "BLACK_LITTERMAN" else 1.4
            alpha = 1.0 if strat == "BLACK_LITTERMAN" else 0.85
            ax1.plot(nav.index, nav.values, label=labels.get(strat, strat), color=color, linewidth=linewidth, alpha=alpha)

        ax1.set_title("Walk-Forward Out-of-Sample Performance (2017 – 2026)", fontsize=14, fontweight="bold", pad=12, color="#1B365D")
        ax1.set_ylabel("Portfolio Value (Base = $100)", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper left", frameon=True, framealpha=0.9, facecolor="white", edgecolor="#D0D5DD", fontsize=9)
        ax1.grid(True)
        ax1.yaxis.set_major_formatter("${x:,.0f}")

        # 2. Underwater Drawdown
        for strat, res in strategy_results.items():
            nav = res.equity_curve
            peak = nav.cummax()
            dd = (nav - peak) / peak
            color = self.COLORS.get(strat, "#000000")
            
            if strat == "BLACK_LITTERMAN":
                ax2.plot(dd.index, dd.values, color=color, linewidth=1.8, label="Black-Litterman DD")
                ax2.fill_between(dd.index, dd.values, 0, color=color, alpha=0.15)
            elif strat == "BENCHMARK_CAP":
                ax2.plot(dd.index, dd.values, color=color, linewidth=1.2, linestyle="--", label="Benchmark DD")

        ax2.set_ylabel("Drawdown", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
        ax2.yaxis.set_major_formatter("{x:.0%}")
        ax2.grid(True)
        ax2.legend(loc="lower left", fontsize=8, framealpha=0.8)

        # Date formatting
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {filepath}")
        return filepath

    def plot_efficient_frontier_shift(
        self,
        frontier_prior: pd.DataFrame,
        frontier_post: pd.DataFrame,
        opt_prior_pt: Tuple[float, float],
        opt_post_pt: Tuple[float, float],
        asset_stats: pd.DataFrame,
        filename: str = "efficient_frontier.png"
    ) -> Path:
        """Plots the Bayesian shift in the Efficient Frontier before and after tactical views."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot Frontiers
        ax.plot(
            frontier_prior["volatility"], frontier_prior["expected_return"],
            color=self.COLORS["PRIOR_FRONTIER"], linestyle="--", linewidth=2.0,
            label="Prior Frontier (Market Equilibrium)"
        )
        ax.plot(
            frontier_post["volatility"], frontier_post["expected_return"],
            color=self.COLORS["POST_FRONTIER"], linewidth=2.5,
            label="Posterior Frontier (Tactical Views Added)"
        )

        # Highlight Optimal Operating Points
        ax.scatter(opt_prior_pt[0], opt_prior_pt[1], color="#5F6368", s=120, zorder=5, marker="o", edgecolors="black", label="Prior Optimal Portfolio")
        ax.scatter(opt_post_pt[0], opt_post_pt[1], color=self.COLORS["BLACK_LITTERMAN"], s=160, zorder=6, marker="*", edgecolors="black", label="Black-Litterman Optimal Portfolio")

        # Scatter Individual Asset Classes
        for ticker, row in asset_stats.iterrows():
            ax.scatter(row["annualized_volatility"], row["historical_mean_return"], color="#4A90E2", alpha=0.6, s=40)
            ax.annotate(ticker, (row["annualized_volatility"] + 0.003, row["historical_mean_return"]), fontsize=8, color="#333333")

        ax.set_title("Efficient Frontier Expansion via Black-Litterman Views", fontsize=13, fontweight="bold", pad=12, color="#1B365D")
        ax.set_xlabel("Annualized Volatility (Risk)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Expected Return", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter("{x:.1%}")
        ax.yaxis.set_major_formatter("{x:.1%}")
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#D0D5DD", fontsize=9)
        ax.grid(True)

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {filepath}")
        return filepath

    def plot_historical_asset_allocation(
        self,
        weights_history: pd.DataFrame,
        filename: str = "asset_allocation_drift.png"
    ) -> Path:
        """Plots stacked area chart of historical asset allocation drift across rebalance dates."""
        fig, ax = plt.subplots(figsize=(12, 6))

        df_w = weights_history.copy()
        df_w.index = pd.to_datetime(df_w.index)
        
        # Color palette for 10 assets
        palette = sns.color_palette("tab10", n_colors=len(df_w.columns))
        
        ax.stackplot(
            df_w.index, df_w.values.T,
            labels=df_w.columns,
            colors=palette,
            alpha=0.85
        )

        ax.set_title("Systematic Asset Allocation Weight Dynamics (2017 – 2026)", fontsize=13, fontweight="bold", pad=12, color="#1B365D")
        ax.set_ylabel("Portfolio Weight", fontsize=11, fontweight="bold")
        ax.set_xlabel("Rebalance Date", fontsize=11, fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter("{x:.0%}")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=True, edgecolor="#D0D5DD", fontsize=9)
        ax.grid(True, axis="y")

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {filepath}")
        return filepath

    def plot_stress_test_comparison(
        self,
        stress_df: pd.DataFrame,
        filename: str = "stress_test_comparison.png"
    ) -> Path:
        """Plots a grouped horizontal bar chart comparing Black-Litterman vs. Benchmark in crises."""
        fig, ax = plt.subplots(figsize=(10, 5))

        scenarios = stress_df["Crisis Scenario"].tolist()
        y = np.arange(len(scenarios))
        height = 0.35

        port_rets = stress_df["Portfolio Return"].values * 100.0
        bench_rets = stress_df["Benchmark Return"].values * 100.0

        rects1 = ax.barh(y - height/2, port_rets, height, label="Black-Litterman", color="#1B365D")
        rects2 = ax.barh(y + height/2, bench_rets, height, label="Global Benchmark", color="#80868B")

        ax.set_title("Historical Crisis Stress-Test Capital Preservation", fontsize=13, fontweight="bold", pad=12, color="#1B365D")
        ax.set_xlabel("Cumulative Return During Shock Window (%)", fontsize=11, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(scenarios, fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.xaxis.set_major_formatter("{x:.0f}%")
        ax.legend(loc="lower left", frameon=True, edgecolor="#D0D5DD", fontsize=9)
        ax.grid(True, axis="x")

        # Label bars
        for r in rects1:
            w = r.get_width()
            ax.annotate(f"{w:.1f}%", (w - 1.8 if w < 0 else w + 0.5, r.get_y() + r.get_height()/2),
                        ha="center", va="center", fontsize=8, color="white" if w < -5 else "black", fontweight="bold")
        for r in rects2:
            w = r.get_width()
            ax.annotate(f"{w:.1f}%", (w - 1.8 if w < 0 else w + 0.5, r.get_y() + r.get_height()/2),
                        ha="center", va="center", fontsize=8, color="white" if w < -5 else "black", fontweight="bold")

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved figure: {filepath}")
        return filepath