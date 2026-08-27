import logging
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.linalg import cholesky

from src.engine.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Convex Quadratic Programming & SOCP Portfolio Optimizer using CVXPY."""

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02
    ):
        self.tickers = list(covariance_matrix.columns)
        self.mu = expected_returns.reindex(self.tickers).values
        self.Sigma = covariance_matrix.reindex(index=self.tickers, columns=self.tickers).values
        self.risk_free_rate = risk_free_rate
        self.n = len(self.tickers)

        # Ensure covariance matrix is symmetric and strictly positive-definite
        self.Sigma = (self.Sigma + self.Sigma.T) / 2.0
        min_eig = np.min(np.linalg.eigvalsh(self.Sigma))
        if min_eig < 1e-7:
            self.Sigma += np.eye(self.n) * (1e-6 - min_eig)

        # Cholesky decomposition: Sigma = L @ L.T (used for SOCP tracking error constraint)
        self.L = cholesky(self.Sigma, lower=False)

    def _build_cvxpy_constraints(
        self,
        w: cp.Variable,
        constraints: PortfolioConstraints
    ) -> list:
        """Translates PortfolioConstraints into mathematical CVXPY constraint objects."""
        cvx_constraints = [
            cp.sum(w) == 1.0  # 1. Full Investment Budget
        ]

        # 2. Individual Asset Bounds
        lb, ub = constraints.get_individual_bounds()
        cvx_constraints.append(w >= lb)
        cvx_constraints.append(w <= ub)

        # 3. Macro Asset Class Aggregate Bounds
        ac_mapping = constraints.get_asset_class_mapping_matrix()
        for ac, bounds in constraints.asset_class_bounds.items():
            if ac in ac_mapping:
                selection_vec = ac_mapping[ac]
                lower_bound, upper_bound = bounds
                cvx_constraints.append(selection_vec @ w >= lower_bound)
                cvx_constraints.append(selection_vec @ w <= upper_bound)

        # 4. Tracking Error Constraint via Second-Order Cone: || L @ (w - wb) ||_2 <= TE_max
        if constraints.max_tracking_error is not None and constraints.benchmark_weights is not None:
            wb = constraints.benchmark_weights.reindex(self.tickers).fillna(0.0).values
            active_w = w - wb
            # Second-order cone formulation
            cvx_constraints.append(cp.norm(self.L @ active_w, 2) <= constraints.max_tracking_error)

        # 5. Turnover Constraint: sum(|w - w_curr|) <= max_turnover
        if constraints.max_turnover is not None and constraints.current_weights is not None:
            w_curr = constraints.current_weights.reindex(self.tickers).fillna(0.0).values
            turnover = cp.sum(cp.abs(w - w_curr))
            cvx_constraints.append(turnover <= constraints.max_turnover)

        return cvx_constraints

    def _solve_problem(self, problem: cp.Problem) -> None:
        """Attempts to solve using CLARABEL, with fallbacks to ECOS and SCS."""
        solvers = [cp.CLARABEL, cp.ECOS, cp.SCS]
        for solver in solvers:
            try:
                problem.solve(solver=solver)
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    return
            except Exception:
                continue

        # If explicit solvers fail, try default solve
        problem.solve()

    def optimize_utility(
        self,
        risk_aversion: float,
        constraints: PortfolioConstraints
    ) -> pd.Series:
        """
        Solves: max_w [ w^T * mu - (gamma / 2) * w^T * Sigma * w ]
        """
        w = cp.Variable(self.n)
        expected_return = self.mu @ w
        portfolio_variance = cp.quad_form(w, self.Sigma)

        objective = cp.Maximize(expected_return - (risk_aversion / 2.0) * portfolio_variance)
        cvx_constraints = self._build_cvxpy_constraints(w, constraints)

        problem = cp.Problem(objective, cvx_constraints)
        self._solve_problem(problem)

        if problem.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
            logger.error(f"Optimization failed with status: {problem.status}")
            # Fallback to benchmark weights if constraints cannot be satisfied
            if constraints.benchmark_weights is not None:
                return constraints.benchmark_weights.reindex(self.tickers).fillna(0.0)
            return pd.Series(1.0 / self.n, index=self.tickers)

        weights = np.clip(w.value, 0.0, 1.0)
        weights = weights / np.sum(weights)  # Clean normalization to exactly 1.0

        return pd.Series(weights, index=self.tickers, name="optimal_weight")

    def optimize_minimum_variance(
        self,
        constraints: PortfolioConstraints
    ) -> pd.Series:
        """Solves: min_w [ w^T * Sigma * w ]"""
        w = cp.Variable(self.n)
        portfolio_variance = cp.quad_form(w, self.Sigma)

        objective = cp.Minimize(portfolio_variance)
        cvx_constraints = self._build_cvxpy_constraints(w, constraints)

        problem = cp.Problem(objective, cvx_constraints)
        self._solve_problem(problem)

        weights = np.clip(w.value, 0.0, 1.0)
        weights = weights / np.sum(weights)
        return pd.Series(weights, index=self.tickers, name="min_variance_weight")

    def generate_efficient_frontier(
        self,
        constraints: PortfolioConstraints,
        num_points: int = 15
    ) -> pd.DataFrame:
        """Traces the constrained efficient frontier across risk aversion spectrum."""
        gammas = np.logspace(-1, 2, num_points)
        frontier_records = []

        for gamma in gammas:
            try:
                w_opt = self.optimize_utility(risk_aversion=gamma, constraints=constraints)
                ret = float(w_opt.values @ self.mu)
                vol = float(np.sqrt(w_opt.values @ self.Sigma @ w_opt.values))
                sharpe = (ret - self.risk_free_rate) / vol

                frontier_records.append({
                    "risk_aversion": gamma,
                    "expected_return": ret,
                    "volatility": vol,
                    "sharpe_ratio": sharpe
                })
            except Exception:
                continue

        return pd.DataFrame(frontier_records)