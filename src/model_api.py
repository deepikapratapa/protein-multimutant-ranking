#!/usr/bin/env python3
"""
API wrapper for the trained ΔΔG model (1EY0A structural surrogate).

Provides:
    - MutantScorer.score_single_mut(df_row)
    - MutantScorer.score_multi_mut(df_multi)

Assumes the model was trained in 03_ML_Model_1EY0A_structural.ipynb
with the following numeric features:

[
    "pH", "temp_C",
    "delta_hydropathy", "delta_charge", "delta_volume", "delta_polarity",
    "blosum62",
    "is_gly", "is_pro", "is_to_gly", "is_to_pro",
    "ss_idx", "ss_H", "ss_E", "ss_C",
    "asa", "asa_norm", "is_buried", "is_exposed",
]
"""

from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class MutantScorer:
    """
    Wrapper over the trained sklearn Pipeline for ΔΔG prediction.

    Works with:
        rf_structural_1EY0A.pkl

    Methods:
        - score_single_mut(df_row): return ΔΔG prediction for one mutation
        - score_multi_mut(df_multi): sum predictions for multiple mutations
    """

    # This matches EXACTLY the numeric_features list inside the notebook
    FEATURE_ORDER = [
        "pH", "temp_C",
        "delta_hydropathy", "delta_charge",
        "delta_volume", "delta_polarity",
        "blosum62",
        "is_gly", "is_pro", "is_to_gly", "is_to_pro",
        "ss_idx", "ss_H", "ss_E", "ss_C",
        "asa", "asa_norm", "is_buried", "is_exposed",
    ]

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.features = self.FEATURE_ORDER.copy()

    # -----------------------------------------------------
    # Public prediction methods
    # -----------------------------------------------------

    def score_single_mut(self, df_single_row: pd.DataFrame) -> float:
        """
        Predict ΔΔG for a single mutation.
        df_single_row must be a 1-row DataFrame with all required features.
        """
        if len(df_single_row) != 1:
            raise ValueError("df_single_row must contain exactly one row.")

        self._check_columns(df_single_row)

        X = df_single_row[self.features].values
        pred = self.model.predict(X)[0]
        return float(pred)

    def score_multi_mut(self, df_multi: pd.DataFrame) -> float:
        """
        Predict combined ΔΔG for multiple mutations.
        Returns the simple sum of per-site predicted ΔΔG.
        (This approximation is used inside the GA search.)
        """
        if df_multi.empty:
            raise ValueError("df_multi is empty.")

        self._check_columns(df_multi)

        X = df_multi[self.features].values
        preds = self.model.predict(X)
        return float(np.sum(preds))

    # -----------------------------------------------------
    # Internal validation helper
    # -----------------------------------------------------

    def _check_columns(self, df: pd.DataFrame):
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise KeyError(
                f"Input DataFrame is missing required feature columns: {missing}"
            )