#!/usr/bin/env python3
"""
Small API wrapper around the trained ΔΔG model.

Usage (example)
---------------
from model_api import MutantScorer
import pandas as pd

scorer = MutantScorer(
    model_path="results/models_structural/rf_structural_1EY0A.pkl",
    feature_list_path="results/models_structural/feature_list.txt",
)

# df_row: 1-row DataFrame with all engineered features
ddg_pred = scorer.score_single_mut(df_row)

# df_multi: k-row DataFrame (k mutations within same protein)
ddg_combined = scorer.score_multi_mut(df_multi)
"""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class MutantScorer:
    """
    Wraps the trained sklearn Pipeline and exposes simple scoring methods.

    - score_single_mut(df_row): predict ΔΔG for a single mutation
    - score_multi_mut(df_multi): predict combined ΔΔG for multiple mutations
    """

    def __init__(self, model_path: str | Path,
                 feature_list_path: str | Path | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)

        # Load feature list from file if provided, otherwise infer from pipeline
        self.features: list[str]
        if feature_list_path is not None:
            feature_list_path = Path(feature_list_path)
            if not feature_list_path.exists():
                raise FileNotFoundError(
                    f"feature_list.txt not found: {feature_list_path}"
                )
            with open(feature_list_path) as f:
                self.features = [ln.strip() for ln in f if ln.strip()]
        else:
            # Try to infer from ColumnTransformer inside the pipeline
            pre = self.model.named_steps.get("pre", None)
            if pre is None:
                raise ValueError("Pipeline does not contain a 'pre' step.")
            num_cols = None
            for name, trans, cols in pre.transformers_:
                if name == "num":
                    num_cols = cols
                    break
            if num_cols is None:
                raise ValueError("Could not infer numeric feature columns.")
            # cols may be a list or numpy array
            self.features = list(num_cols)

    # -------------------------
    # Public scoring methods
    # -------------------------

    def score_single_mut(self, df_single_row: pd.DataFrame) -> float:
        """
        df_single_row: DataFrame with exactly one row and all engineered
        feature columns. Must include at least self.features.
        """
        if len(df_single_row) != 1:
            raise ValueError("df_single_row must contain exactly one row.")
        self._check_columns(df_single_row)
        X = df_single_row[self.features].values
        pred = self.model.predict(X)[0]
        return float(pred)

    def score_multi_mut(self, df_multi: pd.DataFrame) -> float:
        """
        df_multi: DataFrame with each row representing a single-site mutation
        within the same protein (same wild-type background).

        Returns:
            Combined ΔΔG as a simple sum of per-mutation predictions.
            (This is the approximation we use for GA fitness.)
        """
        if df_multi.empty:
            raise ValueError("df_multi is empty.")
        self._check_columns(df_multi)
        X = df_multi[self.features].values
        preds = self.model.predict(X)
        return float(np.sum(preds))

    # -------------------------
    # Internal helpers
    # -------------------------

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise KeyError(
                f"Input DataFrame is missing required feature columns: {missing}"
            )