"""Feature Engineering Module for E-Commerce Fraud Detection System.

Implements domain-informed, leakage-free feature engineering transformations:
- Log-transformed transaction amount
- Diurnal hour and cyclical sinusoidal time transformations
- Amount-to-median ratio based on training distribution statistics
- Multivariate PCA risk index (combining key discriminative features V4, V11, V12, V14, V17)
- Vector norm of PCA features
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    AMOUNT_COL,
    TIME_COL,
    V_COLS,
)

logger = logging.getLogger(__name__)


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for engineering fraud detection features.

    Stores training statistics (such as median transaction amount) during fit()
    and applies identical transformations during transform(), preventing data leakage.
    """

    def __init__(self, include_cyclical_time: bool = True) -> None:
        self.include_cyclical_time = include_cyclical_time
        self.train_amount_median_ = 25.0
        self.feature_names_out_: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FraudFeatureEngineer":
        """Computes training distribution baselines (e.g., median transaction amount).

        Args:
            X: Training DataFrame.
            y: Ignored (sklearn compatibility).

        Returns:
            self
        """
        if AMOUNT_COL in X.columns:
            median_val = float(X[AMOUNT_COL].median())
            self.train_amount_median_ = median_val if median_val > 0 else 25.0

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies feature transformations to input DataFrame.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame with engineered features appended.
        """
        X_out = X.copy()

        # 1. Log-transformed Amount to tame extreme right skew
        if AMOUNT_COL in X_out.columns:
            # np.log1p handles 0 amounts smoothly
            X_out["log_amount"] = np.log1p(np.maximum(0.0, X_out[AMOUNT_COL].values))

            # 2. Ratio of Amount relative to learned baseline population median
            X_out["amount_to_median_ratio"] = X_out[AMOUNT_COL].values / (self.train_amount_median_ + 1e-5)

        # 3. Time Transformations: Diurnal Hour and Cyclical Sine/Cosine
        if TIME_COL in X_out.columns:
            # Time in dataset represents seconds elapsed over a 48-hour period
            hour_of_day = (X_out[TIME_COL].values / 3600.0) % 24.0
            X_out["hour_of_day"] = hour_of_day

            if self.include_cyclical_time:
                # Continuous cyclical representation so hour 23 and hour 0 are close
                X_out["time_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
                X_out["time_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)

        # 4. Discriminative PCA Composite Risk Index
        # Domain observation: Fraud transactions reliably manifest high V4, V11 and low/negative V12, V14, V17
        required_v = ["V4", "V11", "V12", "V14", "V17"]
        if all(col in X_out.columns for col in required_v):
            pos_risk = X_out["V4"].values + X_out["V11"].values
            neg_risk = X_out["V12"].values + X_out["V14"].values + X_out["V17"].values
            X_out["v_risk_interaction"] = pos_risk - neg_risk

        # 5. Overall PCA Vector Norm (Anomalous deviation from the origin)
        present_v_cols = [c for c in V_COLS if c in X_out.columns]
        if len(present_v_cols) >= 10:
            X_out["v_vector_norm"] = np.linalg.norm(X_out[present_v_cols].values, axis=1)

        self.feature_names_out_ = list(X_out.columns)
        return X_out

    def get_feature_names_out(self) -> List[str]:
        """Returns the list of output feature column names."""
        return self.feature_names_out_
