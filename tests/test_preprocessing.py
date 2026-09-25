"""Unit tests for preprocessing and data loading modules."""

import numpy as np
import pandas as pd
import pytest

from src.config import AMOUNT_COL, TARGET_COL, TIME_COL
from src.data_loader import generate_synthetic_benchmark, validate_dataset_schema
from src.feature_engineering import FraudFeatureEngineer
from src.preprocessing import FraudPreprocessor, clean_data, split_data, validate_processed_data


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture providing a miniature realistic fraud dataset."""
    return generate_synthetic_benchmark(n_samples=200, fraud_ratio=0.05, random_state=42)


def test_validate_dataset_schema(sample_raw_df: pd.DataFrame) -> None:
    """Tests that schema validation correctly validates expected columns."""
    is_valid, msg = validate_dataset_schema(sample_raw_df)
    assert is_valid is True
    assert "valid" in msg.lower()

    # Missing column should fail validation
    corrupted_df = sample_raw_df.drop(columns=[AMOUNT_COL])
    is_valid_bad, msg_bad = validate_dataset_schema(corrupted_df)
    assert is_valid_bad is False
    assert "Missing required" in msg_bad


def test_clean_data_handles_missing_and_duplicates() -> None:
    """Tests that clean_data imputes missing values and removes duplicate rows."""
    test_df = pd.DataFrame({
        TIME_COL: [100.0, 100.0, 200.0, np.nan],
        AMOUNT_COL: [50.0, 50.0, np.nan, 20.0],
        TARGET_COL: [0, 0, 1, 0],
    })
    cleaned = clean_data(test_df, drop_duplicates=True)

    # Duplicates should be dropped
    assert len(cleaned) == 3
    # No NaNs should remain
    assert cleaned.isnull().sum().sum() == 0


def test_split_data_preserves_stratification(sample_raw_df: pd.DataFrame) -> None:
    """Tests that train_test_split properly stratifies the target class."""
    X_train, X_test, y_train, y_test = split_data(sample_raw_df, test_size=0.2, random_state=42)

    assert len(X_train) == 160
    assert len(X_test) == 40

    train_fraud_ratio = y_train.sum() / len(y_train)
    test_fraud_ratio = y_test.sum() / len(y_test)

    # Stratification ensures fraud ratios are closely aligned
    assert abs(train_fraud_ratio - test_fraud_ratio) < 0.05


def test_preprocessor_and_feature_engineering_no_leakage(sample_raw_df: pd.DataFrame) -> None:
    """Tests that preprocessor and feature engineer fit properly and transform cleanly."""
    X_train, X_test, y_train, y_test = split_data(sample_raw_df, test_size=0.2, random_state=42)

    # 1. Feature Engineering
    fe = FraudFeatureEngineer()
    fe.fit(X_train)
    X_train_fe = fe.transform(X_train)
    X_test_fe = fe.transform(X_test)

    assert "log_amount" in X_train_fe.columns
    assert "hour_of_day" in X_train_fe.columns
    assert "amount_to_median_ratio" in X_train_fe.columns

    # 2. Scaling
    preproc = FraudPreprocessor()
    preproc.fit(X_train_fe)
    X_train_proc = preproc.transform(X_train_fe)
    X_test_proc = preproc.transform(X_test_fe)

    assert "scaled_Amount" in X_train_proc.columns
    assert "scaled_Time" in X_train_proc.columns

    # 3. Validation
    is_valid, msg = validate_processed_data(X_train_proc, y_train)
    assert is_valid is True
