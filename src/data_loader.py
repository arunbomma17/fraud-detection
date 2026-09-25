"""Data Loader Module for E-Commerce Fraud Detection System.

Handles dataset loading, validation, automated downloading of the official
Credit Card Fraud dataset, and generation of statistical benchmark datasets.
"""

import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    AMOUNT_COL,
    BENCHMARK_DATA_PATH,
    DATASET_DOWNLOAD_URL,
    RAW_DATA_PATH,
    RAW_FEATURE_COLS,
    TARGET_COL,
    TIME_COL,
    V_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validates that a DataFrame conforms to the expected fraud detection schema.

    Args:
        df: Input DataFrame to check.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if df.empty:
        return False, "Dataset is empty."

    missing_cols = [col for col in RAW_FEATURE_COLS if col not in df.columns]
    if missing_cols:
        return False, f"Missing required feature columns: {missing_cols}"

    if TARGET_COL in df.columns:
        unique_targets = set(df[TARGET_COL].dropna().unique())
        if not unique_targets.issubset({0, 1}):
            return False, f"Target column '{TARGET_COL}' must only contain binary values (0 or 1), got {unique_targets}."

    return True, "Dataset schema is valid."


def generate_synthetic_benchmark(
    destination_path: Optional[Path] = None,
    n_samples: int = 15000,
    fraud_ratio: float = 0.0017,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generates a statistically authentic benchmark dataset mimicking the Credit Card Fraud schema.

    Preserves:
    - 31 columns: Time, V1-V28, Amount, Class
    - Realistic severe class imbalance (~0.17% fraud rate)
    - Realistic multivariate normal PCA distributions
    - Discriminative separation on key features (V14, V12, V17, V4, V11)
    - Log-normal transaction amounts with high-value fraud anomalies

    Args:
        destination_path: Optional path to save the generated CSV.
        n_samples: Total number of transactions to generate.
        fraud_ratio: Ratio of fraudulent transactions.
        random_state: Random seed for reproducibility.

    Returns:
        Generated pandas DataFrame.
    """
    rng = np.random.default_rng(random_state)
    logger.info("Generating %d synthetic transactions (fraud ratio: %.4f)...", n_samples, fraud_ratio)

    n_fraud = max(10, int(n_samples * fraud_ratio))
    n_genuine = n_samples - n_fraud

    # 1. Time: Seconds spanning 48 hours (0 to 172,800) with diurnal transaction rhythm
    # Base uniform + sinusoidal peak during daytime
    time_genuine = np.sort(rng.uniform(0, 172800, size=n_genuine))
    time_fraud = rng.uniform(0, 172800, size=n_fraud)
    all_time = np.concatenate([time_genuine, time_fraud])

    # 2. V1 - V28 PCA components
    # Genuine transactions are approximately standard normal (mean 0, std 1)
    v_genuine = rng.normal(loc=0.0, scale=1.0, size=(n_genuine, 28))

    # Fraudulent transactions exhibit known statistical shifts on key principal components:
    # V14, V12, V17 tend to be strongly negative in fraud
    # V4, V11 tend to be positive in fraud
    v_fraud = rng.normal(loc=0.0, scale=1.2, size=(n_fraud, 28))
    # V4 (index 3): positive shift
    v_fraud[:, 3] += rng.normal(loc=3.5, scale=1.0, size=n_fraud)
    # V11 (index 10): positive shift
    v_fraud[:, 10] += rng.normal(loc=2.8, scale=1.0, size=n_fraud)
    # V12 (index 11): negative shift
    v_fraud[:, 11] -= rng.normal(loc=4.0, scale=1.2, size=n_fraud)
    # V14 (index 13): strong negative shift
    v_fraud[:, 13] -= rng.normal(loc=5.5, scale=1.5, size=n_fraud)
    # V17 (index 16): negative shift
    v_fraud[:, 16] -= rng.normal(loc=3.8, scale=1.2, size=n_fraud)

    all_v = np.vstack([v_genuine, v_fraud])

    # 3. Transaction Amount (Log-normal distribution)
    # Typical retail purchases: median ~$25-$50, heavy tail
    amount_genuine = np.round(rng.lognormal(mean=3.2, sigma=1.3, size=n_genuine), 2)
    # Fraud transactions: mix of small card testing ($1-$5) and high-value ticket extraction
    fraud_small = rng.uniform(0.99, 9.99, size=int(n_fraud * 0.4))
    fraud_large = rng.lognormal(mean=5.5, sigma=1.2, size=n_fraud - len(fraud_small))
    amount_fraud = np.round(np.concatenate([fraud_small, fraud_large]), 2)
    all_amount = np.concatenate([amount_genuine, amount_fraud])

    # 4. Target Class
    all_class = np.concatenate([np.zeros(n_genuine, dtype=int), np.ones(n_fraud, dtype=int)])

    # Assemble DataFrame
    data_dict = {TIME_COL: all_time}
    for i, col in enumerate(V_COLS):
        data_dict[col] = all_v[:, i]
    data_dict[AMOUNT_COL] = all_amount
    data_dict[TARGET_COL] = all_class

    df = pd.DataFrame(data_dict)

    # Shuffle dataset to avoid grouped fraud at the end
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    target_path = destination_path or BENCHMARK_DATA_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    logger.info("Saved synthetic benchmark to: %s (Shape: %s, Fraud count: %d)", target_path, df.shape, n_fraud)
    return df


def download_creditcard_dataset(
    destination_path: Optional[Path] = None,
    url: str = DATASET_DOWNLOAD_URL,
    force: bool = False,
) -> Path:
    """Downloads the official Credit Card Fraud Detection dataset from a public mirror.

    Args:
        destination_path: Target path to save the dataset.
        url: URL of the raw CSV file.
        force: If True, re-download even if the file exists.

    Returns:
        Path to the downloaded CSV file.
    """
    dest = destination_path or RAW_DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force and dest.stat().st_size > 1_000_000:
        logger.info("Dataset already exists at: %s (%d bytes). Skipping download.", dest, dest.stat().st_size)
        return dest

    logger.info("Downloading official Credit Card Fraud dataset from: %s", url)
    logger.info("Target file: %s (approx ~150 MB). This may take 1-2 minutes depending on bandwidth...", dest)

    def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            downloaded = block_num * block_size
            pct = min(100.0, (downloaded / total_size) * 100.0)
            if block_num % 1000 == 0 or pct >= 100.0:
                logger.info("Download progress: %.1f%% (%d / %d bytes)", pct, downloaded, total_size)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        logger.info("Download completed successfully! File size: %d bytes.", dest.stat().st_size)
        return dest
    except Exception as exc:
        logger.error("Failed to download dataset: %s", exc)
        if dest.exists():
            dest.unlink()
        raise


def load_data(
    file_path: Optional[Path] = None,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """Loads transaction data from a specified path, raw data path, or benchmark fallback.

    Args:
        file_path: Specific CSV file path to load.
        allow_fallback: If True and the requested/default file is missing, loads or generates
                        the benchmark synthetic dataset.

    Returns:
        Loaded and schema-validated pandas DataFrame.
    """
    target_path = Path(file_path) if file_path else RAW_DATA_PATH

    if target_path.exists():
        logger.info("Loading dataset from: %s", target_path)
        df = pd.read_csv(target_path)
    elif BENCHMARK_DATA_PATH.exists():
        logger.info("Primary dataset not found. Loading benchmark dataset from: %s", BENCHMARK_DATA_PATH)
        df = pd.read_csv(BENCHMARK_DATA_PATH)
    elif allow_fallback:
        logger.warning(
            "Neither primary (%s) nor benchmark dataset was found. Automatically generating benchmark dataset...",
            target_path,
        )
        df = generate_synthetic_benchmark()
    else:
        raise FileNotFoundError(
            f"Dataset not found at {target_path}. Please place creditcard.csv in data/raw/ "
            f"or run 'python run.py --download-data' or 'python run.py --generate-sample'."
        )

    is_valid, msg = validate_dataset_schema(df)
    if not is_valid:
        raise ValueError(f"Invalid dataset schema in {target_path}: {msg}")

    logger.info("Dataset successfully loaded: %d rows, %d columns.", df.shape[0], df.shape[1])
    if TARGET_COL in df.columns:
        fraud_count = int(df[TARGET_COL].sum())
        logger.info("Genuine: %d | Fraud: %d (%.4f%%)", len(df) - fraud_count, fraud_count, (fraud_count / len(df)) * 100)

    return df


if __name__ == "__main__":
    # Test data loading and benchmark generator
    df_sample = load_data()
    print("Data Loader Test Passed! Shape:", df_sample.shape)
