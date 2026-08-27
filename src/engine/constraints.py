import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConstraints:
    """Defines institutional mandate and regulatory boundaries for portfolio optimization."""
    tickers: List[str]
    asset_classes: Dict[str, str] = field(default_factory=dict)
    
    # 1. Single Asset Bounds: (min_weight, max_weight)
    min_weight: float = 0.0          # Long-only default
    max_weight: float = 0.40         # Single-asset cap (e.g. 40% for broad US market)
    custom_asset_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # 2. Macro Asset Class Aggregate Bounds: e.g. {"Equity": (0.35, 0.65), "Fixed Income": (0.25, 0.50)}
    asset_class_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # 3. Active Risk & Turnover Limits
    max_tracking_error: Optional[float] = None      # Annualized active risk limit (e.g., 0.035 for 3.5%)
    benchmark_weights: Optional[pd.Series] = None   # Required if tracking error limit is active
    
    max_turnover: Optional[float] = None            # Max one-way turnover (e.g., 0.20 for 20%)
    current_weights: Optional[pd.Series] = None     # Current portfolio state for turnover limit

    def get_individual_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns lower and upper bound arrays for all assets."""
        n = len(self.tickers)
        lb = np.full(n, self.min_weight)
        ub = np.full(n, self.max_weight)

        for i, ticker in enumerate(self.tickers):
            if ticker in self.custom_asset_bounds:
                lb[i], ub[i] = self.custom_asset_bounds[ticker]

        return lb, ub

    def get_asset_class_mapping_matrix(self) -> Dict[str, np.ndarray]:
        """Returns binary selection vectors for each asset class group."""
        mapping = {}
        for ac in set(self.asset_classes.values()):
            vector = np.array([1.0 if self.asset_classes.get(t) == ac else 0.0 for t in self.tickers])
            mapping[ac] = vector
        return mapping