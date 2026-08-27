import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.db.connection import get_db_engine

logger = logging.getLogger(__name__)


@dataclass
class MarketView:
    """Represents a single tactical investor view."""
    view_type: str                  # 'ABSOLUTE' or 'RELATIVE'
    asset_long: str                 # e.g., 'VGK' or 'GLD'
    asset_short: Optional[str]      # e.g., 'SPY' for relative view, None for absolute
    expected_outperformance: float  # Magnitude (e.g., 0.02 for +2.0%)
    confidence: float               # Confidence parameter in range (0.0, 1.0]

    def __post_init__(self):
        if not (0.0 < self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in (0.0, 1.0]. Got: {self.confidence}")
        if self.view_type.upper() not in ["ABSOLUTE", "RELATIVE"]:
            raise ValueError("view_type must be either 'ABSOLUTE' or 'RELATIVE'.")
        if self.view_type.upper() == "RELATIVE" and not self.asset_short:
            raise ValueError("Relative view requires an asset_short.")


class ViewsManager:
    """Constructs pick matrices and saves/loads views from PostgreSQL."""

    def __init__(self, tickers: List[str], engine: Engine = None):
        self.tickers = list(tickers)
        self.engine = engine or get_db_engine()
        self.views: List[MarketView] = []

    def add_absolute_view(self, asset: str, expected_return: float, confidence: float = 0.50):
        """Adds an absolute view: E[R_asset] = expected_return."""
        if asset not in self.tickers:
            raise ValueError(f"Asset '{asset}' is not in universe tickers.")
        self.views.append(MarketView("ABSOLUTE", asset, None, expected_return, confidence))

    def add_relative_view(self, asset_long: str, asset_short: str, expected_outperformance: float, confidence: float = 0.50):
        """Adds a relative view: E[R_long] - E[R_short] = expected_outperformance."""
        if asset_long not in self.tickers or asset_short not in self.tickers:
            raise ValueError(f"Assets must be in universe tickers. Long: {asset_long}, Short: {asset_short}")
        self.views.append(MarketView("RELATIVE", asset_long, asset_short, expected_outperformance, confidence))

    def build_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds P (K x N pick matrix), Q (K x 1 view return vector),
        and confidence vector c (K x 1).
        """
        k = len(self.views)
        n = len(self.tickers)

        if k == 0:
            return np.empty((0, n)), np.empty((0, 1)), np.empty((0, 1))

        P = np.zeros((k, n))
        Q = np.zeros((k, 1))
        confidences = np.zeros((k, 1))

        for i, view in enumerate(self.views):
            long_idx = self.tickers.index(view.asset_long)
            P[i, long_idx] = 1.0
            
            if view.view_type.upper() == "RELATIVE" and view.asset_short:
                short_idx = self.tickers.index(view.asset_short)
                P[i, short_idx] = -1.0

            Q[i, 0] = view.expected_outperformance
            confidences[i, 0] = view.confidence

        return P, Q, confidences

    def sync_to_database(self, as_of_date: str = "2026-08-21"):
        """Saves current views to the bl_views table in PostgreSQL."""
        if not self.views:
            return

        with self.engine.begin() as conn:
            # Clear previous views for this as_of_date to avoid duplicates
            conn.execute(text("DELETE FROM bl_views WHERE view_date = :vdate;"), {"vdate": as_of_date})
            
            insert_query = text("""
                INSERT INTO bl_views (view_date, view_type, asset_long, asset_short, expected_outperformance, confidence)
                VALUES (:view_date, :view_type, :asset_long, :asset_short, :expected_outperformance, :confidence);
            """)

            for v in self.views:
                conn.execute(insert_query, {
                    "view_date": as_of_date,
                    "view_type": v.view_type,
                    "asset_long": v.asset_long,
                    "asset_short": v.asset_short,
                    "expected_outperformance": v.expected_outperformance,
                    "confidence": v.confidence
                })
        logger.info(f"Synchronized {len(self.views)} views to PostgreSQL database.")