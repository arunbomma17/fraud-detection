"""Data Preprocessing Module for E-Commerce Fraud Detection System.

Implements clean, reusable, leakage-free preprocessing routines:
- Missing value imputation
- Duplicate removal
- Outlier-resilient scaling (RobustScaler fitted strictly on training data)
- Stratified train-test splitting
- Data schema validation
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import (
    AMOUNT_COL,
    RANDOM_STATE,
    SCALER_PATH,
    TARGET_COL,
    TEST_SIZE,
    TIME_COL,
)

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Performs initial data sanitation, missing value handling, and duplicate removal.

    Args:
        df: Raw input DataFrame.
        drop_duplicates: Whether to drop duplicate transaction records.

    Returns:
        Cleaned pandas DataFrame.
    """
    cleaned_df = df.copy()

    # 1. Missing value handling
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning("Detected %d missing values. Imputing numeric medians...", missing_count)
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())

    # 2. Duplicate handling
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        dropped_rows = initial_rows - len(cleaned_df)
        if dropped_rows > 0:
            logger.info("Removed %d duplicate rows. Remaining: %d rows.", dropped_rows, len(cleaned_df))

    return cleaned_df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset into train and test sets using stratified sampling to preserve fraud ratio.

    Args:
        df: Cleaned input DataFrame.
        test_size: Fraction of samples to allocate to test partition.
        random_state: Seed for reproducible random state.
        stratify: Whether to perform stratified sampling on the target column.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in DataFrame columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify_target,
        random_state=random_state,
    )

    logger.info(
        "Stratified Split: Train set = %d rows (Fraud: %d, %.3f%%) | Test set = %d rows (Fraud: %d, %.3f%%)",
        len(X_train),
        int(y_train.sum()),
        (y_train.sum() / len(y_train)) * 100,
        len(X_test),
        int(y_test.sum()),
        (y_test.sum() / len(y_test)) * 100,
    )
    return X_train, X_test, y_train, y_test


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer for scaling skewed features (Amount, Time) using RobustScaler.

    Strictly guarantees NO data leakage: scalers are fitted ONLY on training data.
    V1-V28 are PCA features that are already centered and normalized.
    """

    def __init__(self, scale_columns: Optional[list] = None) -> None:
        self.scale_columns = scale_columns or [AMOUNT_COL, TIME_COL]
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FraudPreprocessor":
        """Fits the RobustScaler on the training set features.

        Args:
            X: Training DataFrame.
            y: Ignored (for sklearn API compatibility).

        Returns:
            self
        """
        cols_to_scale = [col for col in self.scale_columns if col in X.columns]
        if cols_to_scale:
            self.scaler.fit(X[cols_to_scale])
        self.is_fitted = True
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame by scaling designated columns.

        Args:
            X: Input DataFrame to transform.

        Returns:
            Transformed DataFrame with scaled columns.
        """
        if not self.is_fitted:
            raise RuntimeError("FraudPreprocessor must be fitted before transforming data.")

        X_transformed = X.copy()
        cols_to_scale = [col for col in self.scale_columns if col in X_transformed.columns]
        if cols_to_scale:
            scaled_vals = self.scaler.transform(X_transformed[cols_to_scale])
            for idx, col in enumerate(cols_to_scale):
                X_transformed[f"scaled_{col}"] = scaled_vals[:, idx]

        return X_transformed

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Serializes the fitted preprocessor object to disk using Joblib.

        Args:
            filepath: Destination path. Defaults to SCALER_PATH.

        Returns:
            Path to saved artifact.
        """
        path = filepath or SCALER_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved preprocessor to: %s", path)
        return path

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "FraudPreprocessor":
        """Deserializes a fitted preprocessor object from disk.

        Args:
            filepath: Source path. Defaults to SCALER_PATH.

        Returns:
            Loaded FraudPreprocessor instance.
        """
        path = filepath or SCALER_PATH
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {path}")
        return joblib.load(path)


def validate_processed_data(X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[bool, str]:
    """Validates processed data arrays before feeding to ML models.

    Args:
        X: Feature matrix.
        y: Optional target vector.

    Returns:
        Tuple of (is_valid, validation_message).
    """
    if X.empty:
        return False, "Feature matrix is empty."

    if X.isnull().values.any():
        null_counts = X.isnull().sum()
        return False, f"Feature matrix contains NaNs: {null_counts[null_counts > 0].to_dict()}"

    if np.isinf(X.values).any():
        return False, "Feature matrix contains infinite values."

    if y is not None:
        if len(X) != len(y):
            return False, f"Row count mismatch between X ({len(X)}) and y ({len(y)})."
        if y.isnull().any():
            return False, "Target vector contains null values."

    return True, "Data is valid for model consumption."
