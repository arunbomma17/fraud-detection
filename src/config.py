"""Configuration Module for E-Commerce Fraud Detection System.

Defines centralized, cross-platform paths, hyperparameters, and risk thresholds
using pathlib.Path for full compatibility across Windows, Linux, and macOS.
"""

from pathlib import Path
from typing import Dict, List

# Base Project Directories
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
APP_DIR: Path = PROJECT_ROOT / "app"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DATABASE_DIR: Path = PROJECT_ROOT / "database"

# Ensure runtime directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    DATABASE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Paths & Download URLs
DATASET_FILENAME: str = "creditcard.csv"
RAW_DATA_PATH: Path = RAW_DATA_DIR / DATASET_FILENAME
BENCHMARK_DATA_PATH: Path = RAW_DATA_DIR / "synthetic_creditcard_benchmark.csv"
PROCESSED_TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train_processed.parquet"
PROCESSED_TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test_processed.parquet"

# Public download mirror for official Credit Card Fraud Detection dataset (approx 150MB)
DATASET_DOWNLOAD_URL: str = (
    "https://huggingface.co/datasets/jyunyilin/credit-card-fraud-detection/resolve/main/creditcard.csv"
)

# Model Artifact Paths
MODEL_PATH: Path = MODELS_DIR / "fraud_model.pkl"
SCALER_PATH: Path = MODELS_DIR / "scaler.pkl"
PIPELINE_PATH: Path = MODELS_DIR / "pipeline.pkl"
EVALUATION_RESULTS_PATH: Path = MODELS_DIR / "evaluation_results.json"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

# Database Configuration
DATABASE_PATH: Path = DATABASE_DIR / "fraud_detection.db"
DATABASE_URL: str = f"sqlite:///{DATABASE_PATH.as_posix()}"
SCHEMA_PATH: Path = DATABASE_DIR / "schema.sql"

# Reproducibility & Model Training Hyperparameters
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# Schema Definitions
TIME_COL: str = "Time"
AMOUNT_COL: str = "Amount"
TARGET_COL: str = "Class"
V_COLS: List[str] = [f"V{i}" for i in range(1, 29)]
RAW_FEATURE_COLS: List[str] = [TIME_COL] + V_COLS + [AMOUNT_COL]
ENGINEERED_FEATURE_COLS: List[str] = [
    "log_amount",
    "hour_of_day",
    "amount_to_mean_ratio",
    "v_risk_interaction",
]

# Risk Level Classification Thresholds
# Fraud Probability: 0% - 30% -> LOW | 30% - 70% -> MEDIUM | 70% - 100% -> HIGH
RISK_THRESHOLDS: Dict[str, float] = {
    "LOW_MAX": 0.30,
    "MEDIUM_MAX": 0.70,
}

# Decision classification threshold for binary label (0 = Genuine, 1 = Fraud)
CLASSIFICATION_THRESHOLD: float = 0.50
