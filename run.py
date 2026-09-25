"""Master CLI Runner for E-Commerce Fraud Detection System.

Provides unified entry points for:
- Training machine learning models
- Launching the Streamlit dashboard
- Executing test suites
- Downloading or generating datasets
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_training() -> None:
    """Executes the machine learning training and evaluation pipeline."""
    print("=" * 60)
    print("LAUNCHING FRAUD DETECTION TRAINING PIPELINE")
    print("=" * 60)
    from src.train import run_pipeline
    run_pipeline()


def run_app() -> None:
    """Launches the Streamlit interactive dashboard."""
    print("=" * 60)
    print("STARTING STREAMLIT FRAUD DETECTION APPLICATION")
    print("=" * 60)
    app_path = PROJECT_ROOT / "app" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd, check=True)


def run_tests() -> None:
    """Executes the complete Pytest test suite."""
    print("=" * 60)
    print("EXECUTING PYTEST TEST SUITE")
    print("=" * 60)
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    subprocess.run(cmd, check=True)


def run_download_data() -> None:
    """Downloads the official Credit Card Fraud Detection dataset."""
    print("=" * 60)
    print("DOWNLOADING OFFICIAL CREDIT CARD FRAUD DATASET (~150 MB)")
    print("=" * 60)
    from src.data_loader import download_creditcard_dataset
    download_creditcard_dataset()


def run_generate_benchmark() -> None:
    """Generates the benchmark synthetic dataset."""
    print("=" * 60)
    print("GENERATING SYNTHETIC BENCHMARK DATASET")
    print("=" * 60)
    from src.data_loader import generate_synthetic_benchmark
    generate_synthetic_benchmark()


def run_eda_figures() -> None:
    """Generates all exploratory data analysis figures."""
    print("=" * 60)
    print("GENERATING EXPLORATORY DATA ANALYSIS (EDA) FIGURES")
    print("=" * 60)
    from src.generate_eda_figures import df
    print("Figures generated in reports/figures/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-Based E-Commerce Fraud Detection System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", "-t", action="store_true", help="Train, tune, and evaluate all ML models")
    parser.add_argument("--app", "-a", action="store_true", help="Launch interactive Streamlit dashboard")
    parser.add_argument("--test", action="store_true", help="Run pytest test suite")
    parser.add_argument("--download-data", action="store_true", help="Download official Credit Card Fraud dataset")
    parser.add_argument("--generate-sample", action="store_true", help="Generate synthetic benchmark dataset")
    parser.add_argument("--eda", action="store_true", help="Generate all EDA figures")

    args = parser.parse_args()

    if args.train:
        run_training()
    elif args.app:
        run_app()
    elif args.test:
        run_tests()
    elif args.download_data:
        run_download_data()
    elif args.generate_sample:
        run_generate_benchmark()
    elif args.eda:
        run_eda_figures()
    else:
        parser.print_help()
        print("\nQuick Start:")
        print("  python run.py --train       # Train and evaluate models")
        print("  python run.py --app         # Start web application")
        print("  python run.py --test        # Run all test cases")


if __name__ == "__main__":
    main()
