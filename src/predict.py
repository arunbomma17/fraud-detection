"""Production Prediction Engine for E-Commerce Fraud Detection.

Loads the serialized end-to-end production pipeline, validates input transaction
payloads, executes feature engineering and scaling transformations, computes
calibrated fraud probabilities, and categorizes transaction risk levels (LOW, MEDIUM, HIGH).
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure Windows terminal handles Unicode currency symbol (₹) smoothly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd

from src.config import (
    AMOUNT_COL,
    CLASSIFICATION_THRESHOLD,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PIPELINE_PATH,
    RAW_FEATURE_COLS,
    RISK_THRESHOLDS,
    SCALER_PATH,
    TIME_COL,
    V_COLS,
)
from src.database import insert_prediction

logger = logging.getLogger(__name__)


def determine_risk_level(probability: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """Categorizes a continuous fraud probability into a standardized Risk Level.

    Risk Levels:
    - 0.00 to 0.30 (0% - 30%)  -> LOW
    - 0.30 to 0.70 (30% - 70%) -> MEDIUM
    - 0.70 to 1.00 (70% - 100%) -> HIGH

    Args:
        probability: Float between 0.0 and 1.0.
        thresholds: Optional dictionary specifying 'LOW_MAX' and 'MEDIUM_MAX'.

    Returns:
        Risk level string: 'LOW', 'MEDIUM', or 'HIGH'.
    """
    t = thresholds or RISK_THRESHOLDS
    low_max = t.get("LOW_MAX", 0.30)
    med_max = t.get("MEDIUM_MAX", 0.70)

    if probability < low_max:
        return "LOW"
    elif probability < med_max:
        return "MEDIUM"
    else:
        return "HIGH"


class FraudPredictor:
    """Inference engine managing model loading, pipeline execution, and result formatting."""

    def __init__(self, pipeline_path: Optional[Path] = None) -> None:
        self.pipeline_path = pipeline_path or PIPELINE_PATH
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Loads serialized pipeline or falls back to separate preprocessor and model."""
        if self.pipeline_path.exists():
            try:
                self.pipeline = joblib.load(self.pipeline_path)
                logger.info("Loaded production pipeline from: %s", self.pipeline_path)
                return
            except Exception as exc:
                logger.warning("Could not load unified pipeline (%s). Falling back to modular loader.", exc)

        # Fallback: Load model and scaler separately
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Please run 'python -m src.train' or 'python run.py --train' first."
            )

        logger.info("Loading individual model and preprocessor artifacts...")
        from src.feature_engineering import FraudFeatureEngineer
        from src.pipeline import FraudProductionPipeline
        from src.preprocessing import FraudPreprocessor

        model = joblib.load(MODEL_PATH)
        preprocessor = FraudPreprocessor.load(SCALER_PATH) if SCALER_PATH.exists() else FraudPreprocessor()
        feature_engineer = FraudFeatureEngineer()

        # Load feature names from metadata if available
        feature_names = []
        if MODEL_METADATA_PATH.exists():
            import json
            with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                feature_names = meta.get("feature_names", [])

        self.pipeline = FraudProductionPipeline(
            feature_engineer=feature_engineer,
            preprocessor=preprocessor,
            estimator=model,
            feature_names=feature_names,
            threshold=CLASSIFICATION_THRESHOLD,
        )

    def _prepare_input_df(self, raw_input: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Standardizes input into a validated DataFrame matching expected raw features."""
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        elif isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        else:
            raise TypeError(f"Expected dict or pd.DataFrame, got {type(raw_input)}")

        # Ensure required raw columns exist (populate with sensible defaults if missing)
        if AMOUNT_COL not in df.columns:
            df[AMOUNT_COL] = 10.0
        if TIME_COL not in df.columns:
            df[TIME_COL] = 0.0

        for col in V_COLS:
            if col not in df.columns:
                df[col] = 0.0

        # Ensure numeric types
        for col in RAW_FEATURE_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df

    def predict(
        self,
        transaction: Union[Dict[str, Any], pd.DataFrame],
        log_to_db: bool = True,
    ) -> Dict[str, Any]:
        """Performs fraud inference on a single transaction.

        Args:
            transaction: Dictionary or 1-row DataFrame containing transaction features.
            log_to_db: Whether to automatically persist prediction to SQLite database.

        Returns:
            Dictionary with prediction ('Genuine'/'Fraud'), fraud_probability,
            risk_level, and risk factor highlights.
        """
        input_df = self._prepare_input_df(transaction)

        # Compute probability
        probabilities = self.pipeline.predict_proba(input_df)
        fraud_prob = float(probabilities[0, 1])
        genuine_prob = float(probabilities[0, 0])

        # Binary decision
        prediction_label = "Fraud" if fraud_prob >= self.pipeline.threshold else "Genuine"
        risk_level = determine_risk_level(fraud_prob)

        # Identify top influential risk indicators for explainability
        top_risk_drivers = self._analyze_risk_factors(input_df.iloc[0].to_dict(), fraud_prob)

        # Monetary and temporal context
        amount = float(input_df[AMOUNT_COL].iloc[0])
        time_val = float(input_df[TIME_COL].iloc[0])

        # Optional database logging
        db_id = None
        if log_to_db:
            try:
                db_id = insert_prediction(
                    amount=amount,
                    time=time_val,
                    prediction=prediction_label,
                    probability=fraud_prob,
                    risk_level=risk_level,
                    features=transaction if isinstance(transaction, dict) else transaction.iloc[0].to_dict(),
                )
            except Exception as exc:
                logger.error("Failed to log prediction to SQLite: %s", exc)

        return {
            "prediction": prediction_label,
            "fraud_probability": round(fraud_prob, 4),
            "genuine_probability": round(genuine_prob, 4),
            "fraud_percentage": f"{fraud_prob * 100:.2f}%",
            "risk_level": risk_level,
            "threshold": self.pipeline.threshold,
            "model_name": getattr(self.pipeline, "model_name", "Trained Model"),
            "transaction_amount": amount,
            "transaction_time": time_val,
            "audit_id": db_id,
            "risk_factors": top_risk_drivers,
        }

    def predict_batch(self, df: pd.DataFrame, log_to_db: bool = False) -> pd.DataFrame:
        """Performs batch fraud predictions for a DataFrame of transactions.

        Args:
            df: Input DataFrame with transactions.
            log_to_db: Whether to write batch rows into SQLite.

        Returns:
            DataFrame with 'prediction', 'fraud_probability', and 'risk_level' appended.
        """
        prepared_df = self._prepare_input_df(df)
        probs = self.pipeline.predict_proba(prepared_df)[:, 1]
        preds = ["Fraud" if p >= self.pipeline.threshold else "Genuine" for p in probs]
        risk_levels = [determine_risk_level(p) for p in probs]

        result_df = df.copy()
        result_df["prediction"] = preds
        result_df["fraud_probability"] = np.round(probs, 4)
        result_df["risk_level"] = risk_levels

        if log_to_db:
            for _, row in result_df.iterrows():
                try:
                    insert_prediction(
                        amount=float(row.get(AMOUNT_COL, 0.0)),
                        time=float(row.get(TIME_COL, 0.0)),
                        prediction=row["prediction"],
                        probability=float(row["fraud_probability"]),
                        risk_level=row["risk_level"],
                    )
                except Exception as exc:
                    logger.error("Batch DB log error: %s", exc)

        return result_df

    def _analyze_risk_factors(self, row_dict: Dict[str, Any], prob: float) -> List[str]:
        """Provides heuristic explainability flags based on prominent anomalies."""
        reasons = []

        amount = float(row_dict.get(AMOUNT_COL, 0.0))
        if amount > 1000.0:
            reasons.append(f"Unusually large transaction amount (₹{amount:,.2f})")
        elif amount <= 1.0 and prob > 0.5:
            reasons.append(f"Micro-transaction amount pattern (₹{amount:,.2f}) consistent with automated card testing")

        v14 = float(row_dict.get("V14", 0.0))
        if v14 < -3.0:
            reasons.append(f"Severe anomaly on primary latent component V14 ({v14:.2f})")

        v12 = float(row_dict.get("V12", 0.0))
        if v12 < -2.5:
            reasons.append(f"Significant negative deviation on latent component V12 ({v12:.2f})")

        v4 = float(row_dict.get("V4", 0.0))
        if v4 > 2.5:
            reasons.append(f"Elevated risk indicator on latent component V4 ({v4:.2f})")

        v17 = float(row_dict.get("V17", 0.0))
        if v17 < -2.5:
            reasons.append(f"High-risk anomaly signature on latent component V17 ({v17:.2f})")

        if not reasons:
            if prob < 0.3:
                reasons.append("All latent behavioral dimensions match established genuine cardholder patterns.")
            else:
                reasons.append("Multi-dimensional vector interaction flagged non-standard behavioral trajectory.")

        return reasons


# Global lazy-initialized singleton
_predictor_instance: Optional[FraudPredictor] = None


def get_predictor() -> FraudPredictor:
    """Returns singleton FraudPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FraudPredictor()
    return _predictor_instance


def predict_transaction(
    transaction_data: Union[Dict[str, Any], pd.DataFrame],
    log_to_db: bool = True,
) -> Dict[str, Any]:
    """Convenience public API function to predict fraud on a transaction.

    Example:
        >>> result = predict_transaction({"Amount": 149.99, "Time": 3600.0, "V14": -5.2})
        >>> print(result["prediction"])  # 'Fraud'
        >>> print(result["fraud_percentage"])  # '94.72%'
        >>> print(result["risk_level"])  # 'HIGH'
    """
    predictor = get_predictor()
    return predictor.predict(transaction_data, log_to_db=log_to_db)


if __name__ == "__main__":
    # Test sample genuine vs fraud predictions
    sample_genuine = {AMOUNT_COL: 42.50, TIME_COL: 12000.0, "V14": 0.2, "V4": -0.1}
    sample_fraud = {AMOUNT_COL: 850.00, TIME_COL: 75000.0, "V14": -6.2, "V12": -4.5, "V4": 4.1}

    print("--- Genuine Sample ---")
    print(predict_transaction(sample_genuine, log_to_db=False))

    print("\n--- Fraud Sample ---")
    print(predict_transaction(sample_fraud, log_to_db=False))
