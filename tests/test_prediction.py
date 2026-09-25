"""Unit tests for inference and prediction engine."""

import pytest

from src.config import AMOUNT_COL, TIME_COL
from src.predict import determine_risk_level, get_predictor, predict_transaction


def test_determine_risk_level() -> None:
    """Tests risk categorization thresholds."""
    # Low: 0% - 30%
    assert determine_risk_level(0.0) == "LOW"
    assert determine_risk_level(0.15) == "LOW"
    assert determine_risk_level(0.29) == "LOW"

    # Medium: 30% - 70%
    assert determine_risk_level(0.30) == "MEDIUM"
    assert determine_risk_level(0.50) == "MEDIUM"
    assert determine_risk_level(0.69) == "MEDIUM"

    # High: 70% - 100%
    assert determine_risk_level(0.70) == "HIGH"
    assert determine_risk_level(0.95) == "HIGH"
    assert determine_risk_level(1.0) == "HIGH"


def test_predict_transaction_structure() -> None:
    """Tests that predict_transaction returns complete and properly typed result dictionary."""
    sample = {AMOUNT_COL: 45.0, TIME_COL: 5000.0, "V1": 0.0, "V14": 0.5}
    res = predict_transaction(sample, log_to_db=False)

    assert "prediction" in res
    assert res["prediction"] in ["Genuine", "Fraud"]

    assert "fraud_probability" in res
    assert 0.0 <= res["fraud_probability"] <= 1.0

    assert "risk_level" in res
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    assert "fraud_percentage" in res
    assert "%" in res["fraud_percentage"]

    assert "risk_factors" in res
    assert isinstance(res["risk_factors"], list)


def test_predict_handles_missing_columns() -> None:
    """Tests that inference handles payloads with missing columns gracefully by using defaults."""
    incomplete_sample = {AMOUNT_COL: 120.0}
    res = predict_transaction(incomplete_sample, log_to_db=False)

    assert "prediction" in res
    assert 0.0 <= res["fraud_probability"] <= 1.0


def test_predict_fraud_sample_triggers_high_risk() -> None:
    """Tests that an anomalous payload triggers high risk level."""
    fraud_sample = {
        AMOUNT_COL: 950.0,
        TIME_COL: 70000.0,
        "V14": -6.5,
        "V12": -4.8,
        "V17": -4.0,
        "V4": 4.5,
        "V11": 3.8,
    }
    res = predict_transaction(fraud_sample, log_to_db=False)

    assert res["fraud_probability"] > 0.70
    assert res["risk_level"] == "HIGH"
    assert res["prediction"] == "Fraud"
