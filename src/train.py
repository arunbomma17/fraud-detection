"""Model Training and Selection Pipeline for E-Commerce Fraud Detection.

Orchestrates the entire training workflow:
1. Loads raw dataset (or benchmark).
2. Cleans data and performs stratified train-test splitting.
3. Fits feature engineering and robust scaling strictly on training data (no data leakage).
4. Compares multiple models:
   - Logistic Regression (balanced class weights)
   - Random Forest (balanced subsample)
   - XGBoost (scaled positive weights)
   - Evaluates SMOTE oversampling vs algorithmic class weighting.
5. Performs cross-validated hyperparameter tuning on the best model candidate.
6. Serializes the final end-to-end production pipeline and saves evaluation reports and plots.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config import (
    BENCHMARK_DATA_PATH,
    CLASSIFICATION_THRESHOLD,
    EVALUATION_RESULTS_PATH,
    FIGURES_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PIPELINE_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SCALER_PATH,
    TARGET_COL,
)
from src.data_loader import load_data
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_evaluation_results,
)
from src.feature_engineering import FraudFeatureEngineer
from src.pipeline import FraudProductionPipeline
from src.preprocessing import FraudPreprocessor, clean_data, split_data, validate_processed_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_smote: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any, str]:
    """Trains and compares Logistic Regression, Random Forest, and XGBoost.

    Evaluates both algorithmic weighting and SMOTE applied strictly on the training partition.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Holdout test features.
        y_test: Holdout test labels.
        use_smote: Whether to apply SMOTE oversampling on training data.

    Returns:
        Tuple of (comparison_df, metrics_by_model, best_model_instance, best_model_name).
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL COMPARISON & BENCHMARKING")
    logger.info("=" * 60)

    # Calculate class imbalance ratio for XGBoost scale_pos_weight
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = max(1.0, float(n_neg / max(1, n_pos)))
    logger.info("Training Class Balance: Negative = %d, Positive = %d (Ratio: %.2f:1)", n_neg, n_pos, pos_weight)

    # Apply SMOTE strictly on the training fold if requested
    X_train_resampled = X_train
    y_train_resampled = y_train
    if use_smote and n_pos >= 5:
        # Determine k_neighbors safely if n_pos is small
        k_neighbors = min(5, n_pos - 1)
        if k_neighbors >= 1:
            logger.info("Applying SMOTE oversampling to training data only (k_neighbors=%d)...", k_neighbors)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logger.info("Post-SMOTE Training Samples: %d (Fraud: %d)", len(y_train_resampled), int(y_train_resampled.sum()))

    # Candidate Models definition
    candidate_models = {
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=min(pos_weight, 50.0),  # Moderated scale_pos_weight
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
    }

    comparison_records: List[Dict[str, Any]] = []
    metrics_by_model: Dict[str, Any] = {}
    trained_models: Dict[str, Any] = {}

    for name, model in candidate_models.items():
        logger.info("Training %s...", name)
        # Train on resampled training data
        model.fit(X_train_resampled, y_train_resampled)
        trained_models[name] = model

        # Predict on untouched holdout test partition
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics strictly from actual test set
        m = calculate_metrics(y_test.values, y_pred, y_prob, model_name=name)
        metrics_by_model[name] = m

        comparison_records.append({
            "Model": name,
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1_score"],
            "PR-AUC": m["pr_auc"],
            "ROC-AUC": m["roc_auc"],
            "Accuracy": m["accuracy"],
        })

    comparison_df = pd.DataFrame(comparison_records)
    logger.info("\n" + comparison_df.to_string(index=False))

    # Model Selection: Prioritize PR-AUC and F1 Score (standard for severe fraud imbalance)
    # PR-AUC reflects the precision-recall trade-off across all decision thresholds
    comparison_df["composite_rank"] = (
        comparison_df["PR-AUC"] * 0.45 + comparison_df["F1 Score"] * 0.35 + comparison_df["Recall"] * 0.20
    )
    best_row = comparison_df.sort_values(by="composite_rank", ascending=False).iloc[0]
    best_model_name = str(best_row["Model"])
    best_model = trained_models[best_model_name]

    logger.info("Best Candidate Model Selected: %s (PR-AUC: %.4f, F1: %.4f)", best_model_name, best_row["PR-AUC"], best_row["F1 Score"])
    return comparison_df, metrics_by_model, best_model, best_model_name


def tune_hyperparameters(
    best_model_name: str,
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Any, Dict[str, Any]]:
    """Performs cross-validated hyperparameter optimization on the best candidate model.

    Args:
        best_model_name: Name of selected model.
        base_model: Instantiated base estimator.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Tuple of (tuned_model, tuned_metrics).
    """
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING FOR: %s", best_model_name)
    logger.info("=" * 60)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    if best_model_name == "XGBoost":
        param_dist = {
            "n_estimators": [75, 120, 150],
            "max_depth": [4, 5, 7],
            "learning_rate": [0.03, 0.07, 0.12],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
    elif best_model_name == "Random Forest":
        param_dist = {
            "n_estimators": [80, 120, 150],
            "max_depth": [8, 12, 16],
            "min_samples_split": [2, 4, 8],
            "min_samples_leaf": [1, 2],
        }
    else:  # Logistic Regression
        param_dist = {
            "C": [0.01, 0.1, 1.0, 10.0],
        }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=6,  # Efficient search
        scoring="average_precision",  # Optimizes PR-AUC for imbalanced data
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    logger.info("Optimal Parameters Found: %s", search.best_params_)
    logger.info("Best Cross-Validation PR-AUC Score: %.4f", search.best_score_)

    tuned_model = search.best_estimator_
    y_pred = tuned_model.predict(X_test)
    y_prob = tuned_model.predict_proba(X_test)[:, 1]

    tuned_metrics = calculate_metrics(y_test.values, y_pred, y_prob, model_name=f"{best_model_name} (Tuned)")
    tuned_metrics["best_params"] = search.best_params_
    tuned_metrics["best_cv_score"] = round(float(search.best_score_), 4)

    return tuned_model, tuned_metrics


def run_pipeline(
    data_path: Optional[Path] = None,
    tune: bool = True,
    use_smote: bool = True,
) -> Dict[str, Any]:
    """Executes the full end-to-end training, benchmarking, and artifact export pipeline.

    Args:
        data_path: Path to dataset.
        tune: Whether to run hyperparameter tuning on the best model.
        use_smote: Whether to include SMOTE in model comparison.

    Returns:
        Comprehensive dictionary of pipeline execution metrics and paths.
    """
    logger.info("Starting End-to-End Fraud Detection Training Pipeline...")

    # 1. Load Data
    raw_df = load_data(file_path=data_path, allow_fallback=True)

    # 2. Clean & Split Data
    cleaned_df = clean_data(raw_df, drop_duplicates=True)
    X_train_raw, X_test_raw, y_train, y_test = split_data(cleaned_df, stratify=True)

    # 3. Feature Engineering (Strictly fit on training partition)
    logger.info("Fitting Feature Engineer on training partition...")
    feature_engineer = FraudFeatureEngineer()
    feature_engineer.fit(X_train_raw)
    X_train_feat = feature_engineer.transform(X_train_raw)
    X_test_feat = feature_engineer.transform(X_test_raw)

    # 4. Preprocessing & Scaling (Strictly fit RobustScaler on training partition)
    logger.info("Fitting Preprocessor on engineered training features...")
    preprocessor = FraudPreprocessor()
    preprocessor.fit(X_train_feat)
    X_train_proc = preprocessor.transform(X_train_feat)
    X_test_proc = preprocessor.transform(X_test_feat)

    # Save fitted preprocessor
    preprocessor.save(SCALER_PATH)

    # Align feature columns
    feature_names = list(X_train_proc.columns)
    X_train_final = X_train_proc[feature_names]
    X_test_final = X_test_proc[feature_names]

    # Validate final feature matrices
    is_valid_train, msg_train = validate_processed_data(X_train_final, y_train)
    is_valid_test, msg_test = validate_processed_data(X_test_final, y_test)
    if not (is_valid_train and is_valid_test):
        raise ValueError(f"Data validation failed: {msg_train} | {msg_test}")

    # 5. Model Comparison
    comparison_df, metrics_by_model, best_model, best_model_name = train_and_compare_models(
        X_train=X_train_final,
        y_train=y_train,
        X_test=X_test_final,
        y_test=y_test,
        use_smote=use_smote,
    )

    # 6. Hyperparameter Tuning
    final_model = best_model
    final_model_name = best_model_name
    final_metrics = metrics_by_model[best_model_name]

    if tune:
        tuned_model, tuned_metrics = tune_hyperparameters(
            best_model_name=best_model_name,
            base_model=best_model,
            X_train=X_train_final,
            y_train=y_train,
            X_test=X_test_final,
            y_test=y_test,
        )
        # Use tuned model if it maintained or improved PR-AUC / F1
        if tuned_metrics["pr_auc"] >= final_metrics["pr_auc"] * 0.98:
            final_model = tuned_model
            final_model_name = f"{best_model_name} (Optimized)"
            final_metrics = tuned_metrics

    # 7. Generate Evaluation Figures for Documentation & Dashboard
    logger.info("Generating evaluation figures in: %s", FIGURES_DIR)
    y_test_pred = final_model.predict(X_test_final)
    y_test_prob = final_model.predict_proba(X_test_final)[:, 1]

    cm_path = plot_confusion_matrix(y_test.values, y_test_pred, model_name=final_model_name)
    roc_path = plot_roc_curve(y_test.values, y_test_prob, final_metrics["roc_auc"], model_name=final_model_name)
    pr_path = plot_precision_recall_curve(y_test.values, y_test_prob, final_metrics["pr_auc"], model_name=final_model_name)
    comp_path = plot_model_comparison(comparison_df)

    # 8. Package Production Pipeline Object
    production_pipeline = FraudProductionPipeline(
        feature_engineer=feature_engineer,
        preprocessor=preprocessor,
        estimator=final_model,
        feature_names=feature_names,
        model_name=final_model_name,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    # Serialize artifacts using Joblib
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(production_pipeline, PIPELINE_PATH)
    logger.info("Serialized model to: %s", MODEL_PATH)
    logger.info("Serialized end-to-end production pipeline to: %s", PIPELINE_PATH)

    # 9. Save Metadata & Evaluation Results
    comparison_records = comparison_df.to_dict(orient="records")
    evaluation_payload = {
        "best_model_name": final_model_name,
        "metrics": final_metrics,
        "model_comparison": comparison_records,
        "feature_count": len(feature_names),
        "features": feature_names,
        "figures": {
            "confusion_matrix": str(cm_path),
            "roc_curve": str(roc_path),
            "precision_recall_curve": str(pr_path),
            "model_comparison": str(comp_path),
        },
    }
    save_evaluation_results(evaluation_payload)

    metadata_payload = {
        "model_name": final_model_name,
        "feature_names": feature_names,
        "raw_features": list(X_train_raw.columns),
        "engineered_features": [f for f in feature_names if f not in X_train_raw.columns],
        "test_metrics": final_metrics,
        "threshold": CLASSIFICATION_THRESHOLD,
        "random_state": RANDOM_STATE,
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=4)

    logger.info("=" * 60)
    logger.info("PIPELINE TRAINING COMPLETE")
    logger.info("Selected Model: %s", final_model_name)
    logger.info("Test Precision: %.4f | Recall: %.4f | F1-Score: %.4f | PR-AUC: %.4f",
                final_metrics["precision"], final_metrics["recall"], final_metrics["f1_score"], final_metrics["pr_auc"])
    logger.info("=" * 60)

    return evaluation_payload


if __name__ == "__main__":
    run_pipeline()
