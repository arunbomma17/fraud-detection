"""Pipeline module defining the serializable FraudProductionPipeline class."""

from typing import Any, List
import numpy as np
import pandas as pd

from src.config import CLASSIFICATION_THRESHOLD
from src.feature_engineering import FraudFeatureEngineer
from src.preprocessing import FraudPreprocessor


class FraudProductionPipeline:
    """End-to-end production pipeline bundling feature engineering, scaling, and the trained estimator."""

    def __init__(
        self,
        feature_engineer: FraudFeatureEngineer,
        preprocessor: FraudPreprocessor,
        estimator: Any,
        feature_names: List[str],
        model_name: str = "Trained Model",
        threshold: float = CLASSIFICATION_THRESHOLD,
    ) -> None:
        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.feature_names = feature_names
        self.model_name = model_name
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms raw input features and computes class probabilities."""
        X_engineered = self.feature_engineer.transform(X)
        X_scaled = self.preprocessor.transform(X_engineered)
        X_aligned = X_scaled[self.feature_names]
        return self.estimator.predict_proba(X_aligned)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms raw input features and predicts binary class based on calibrated threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)
