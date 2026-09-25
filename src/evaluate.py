"""Model Evaluation Module for E-Commerce Fraud Detection System.

Computes comprehensive classification metrics strictly from holdout test data:
- Precision, Recall, F1-Score, PR-AUC (Average Precision), ROC-AUC, Accuracy
- Generates and persists publication-quality diagnostic plots (Confusion Matrix, ROC, PR Curves)
- Serializes evaluation reports to JSON
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import EVALUATION_RESULTS_PATH, FIGURES_DIR

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Calculates all key fraud detection performance metrics programmatically.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted fraud probabilities (floats between 0.0 and 1.0).
        model_name: Identifier for the model being evaluated.

    Returns:
        Dictionary of calculated metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Programmatic metric calculations
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_test_samples": int(len(y_true)),
        "actual_fraud_count": int(np.sum(y_true)),
        "predicted_fraud_count": int(np.sum(y_pred)),
    }

    logger.info(
        "[%s Metrics] Precision: %.4f | Recall: %.4f | F1: %.4f | PR-AUC: %.4f | ROC-AUC: %.4f",
        model_name,
        prec,
        rec,
        f1,
        pr_auc,
        roc_auc,
    )
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Fraud Detection Model",
    save_path: Optional[Path] = None,
) -> Path:
    """Generates and saves a styled Confusion Matrix heatmap.

    Args:
        y_true: Actual test labels.
        y_pred: Predicted test labels.
        model_name: Title identifier.
        save_path: Destination filepath. Defaults to FIGURES_DIR / 'confusion_matrix.png'.

    Returns:
        Path to saved figure.
    """
    dest = save_path or (FIGURES_DIR / "confusion_matrix.png")
    dest.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Genuine (0)", "Fraud (1)"],
        yticklabels=["Genuine (0)", "Fraud (1)"],
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, weight="bold")
    plt.ylabel("Actual Class", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(dest, dpi=300)
    plt.close()
    return dest


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> Path:
    """Generates and saves the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true: Actual test labels.
        y_prob: Predicted probabilities.
        roc_auc: ROC-AUC score.
        model_name: Title identifier.
        save_path: Destination filepath.

    Returns:
        Path to saved figure.
    """
    dest = save_path or (FIGURES_DIR / "roc_curve.png")
    dest.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontsize=11, weight="bold")
    plt.title("ROC Curve", fontsize=13, weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(dest, dpi=300)
    plt.close()
    return dest


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> Path:
    """Generates and saves the Precision-Recall (PR) curve, crucial for imbalanced data.

    Args:
        y_true: Actual test labels.
        y_prob: Predicted probabilities.
        pr_auc: PR-AUC (Average Precision) score.
        model_name: Title identifier.
        save_path: Destination filepath.

    Returns:
        Path to saved figure.
    """
    dest = save_path or (FIGURES_DIR / "precision_recall_curve.png")
    dest.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = np.sum(y_true) / len(y_true)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#10B981", lw=2.5, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    plt.axhline(y=baseline, color="#EF4444", lw=1.5, linestyle="--", label=f"No-Skill Baseline ({baseline:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11, weight="bold")
    plt.ylabel("Precision", fontsize=11, weight="bold")
    plt.title("Precision-Recall Curve (Imbalanced Benchmark)", fontsize=13, weight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(dest, dpi=300)
    plt.close()
    return dest


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Path:
    """Generates and saves a multi-metric comparison bar chart across models.

    Args:
        comparison_df: DataFrame with 'Model', 'Precision', 'Recall', 'F1 Score', 'PR-AUC', 'ROC-AUC'.
        save_path: Destination filepath.

    Returns:
        Path to saved figure.
    """
    dest = save_path or (FIGURES_DIR / "model_comparison.png")
    dest.parent.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["Precision", "Recall", "F1 Score", "PR-AUC", "ROC-AUC"]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

    melted_df = comparison_df.melt(
        id_vars=["Model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(9, 5))
    palette = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#EF4444"]
    sns.barplot(data=melted_df, x="Metric", y="Score", hue="Model", palette=palette[: len(comparison_df)])
    plt.title("Model Comparison Across Key Evaluation Metrics", fontsize=13, weight="bold", pad=12)
    plt.ylim([0.0, 1.08])
    plt.ylabel("Metric Score (0.0 - 1.0)", fontsize=11, weight="bold")
    plt.xlabel("")
    plt.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(dest, dpi=300)
    plt.close()
    return dest


def save_evaluation_results(
    results: Dict[str, Any],
    filepath: Optional[Path] = None,
) -> Path:
    """Saves evaluation results dictionary to a JSON file.

    Args:
        results: Dictionary containing model performance metrics and comparison table.
        filepath: Destination filepath. Defaults to EVALUATION_RESULTS_PATH.

    Returns:
        Path to saved JSON file.
    """
    dest = filepath or EVALUATION_RESULTS_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info("Saved evaluation results to: %s", dest)
    return dest


def load_evaluation_results(filepath: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Loads evaluation results dictionary from JSON file.

    Args:
        filepath: Source filepath. Defaults to EVALUATION_RESULTS_PATH.

    Returns:
        Loaded results dictionary or None if file doesn't exist.
    """
    dest = filepath or EVALUATION_RESULTS_PATH
    if not dest.exists():
        return None
    with open(dest, "r", encoding="utf-8") as f:
        return json.load(f)
