"""
Deterministic baseline classifiers for fraud detection.

These serve as the comparison point named in the literature review: standard
binary classifiers (Random Forest, SVM) that produce a hard fraud/not-fraud
label, in contrast to the Bayesian model (added in a later commit) which
produces a full posterior probability distribution. Comparing against these
baselines is what lets us later demonstrate the Bayesian approach's actual
advantage rather than just asserting it.
"""

import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

from src.data.preprocessing import build_feature_matrix
from src.evaluation.report_utils import FIGURES_DIR, REPORTS_DIR, save_metrics

RANDOM_STATE = 42


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with class weighting to partially
    compensate for the extreme fraud/non-fraud imbalance.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_linear_svm(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE
) -> LinearSVC:
    """
    Train a linear SVM classifier (LinearSVC scales far better than kernel
    SVC on datasets this size) with class weighting for the same reason.
    """
    model = LinearSVC(
        class_weight="balanced",
        random_state=random_state,
        max_iter=5000,
        dual="auto",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute the metrics that matter for extreme class imbalance: accuracy
    alone is meaningless when fraud is <1% of transactions (a model that
    predicts "not fraud" every time would score >99% accuracy), so we report
    precision, recall, F1, AUROC/AUPRC, and the confusion matrix instead.
    """
    y_pred = model.predict(X_test)

    # Not every model exposes predict_proba (LinearSVC doesn't); fall back
    # to decision_function for ranking-based metrics where available.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    # AUROC/AUPRC require both classes present and a meaningful score signal.
    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, y_score)
        metrics["average_precision"] = average_precision_score(y_test, y_score)

    return metrics


def print_evaluation(name: str, metrics: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"AUPRC:     {metrics['average_precision']:.4f}")
    print(f"Confusion matrix (rows=actual, cols=predicted):\n{np.array(metrics['confusion_matrix'])}")


if __name__ == "__main__":
    processed_dir = (
        pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
    )
    train_path, test_path = processed_dir / "train.csv", processed_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "No processed train/test data found. Run "
            "`python -m src.data.preprocessing` first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)

    rf = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_classifier(rf, X_test, y_test)
    print_evaluation("Random Forest", rf_metrics)
    save_metrics("Random Forest", rf_metrics)

    svm = train_linear_svm(X_train, y_train)
    svm_metrics = evaluate_classifier(svm, X_test, y_test)
    print_evaluation("Linear SVM", svm_metrics)
    save_metrics("Linear SVM", svm_metrics)

    print(f"\nSaved metrics to {REPORTS_DIR / 'metrics.json'} and confusion "
          f"matrix plots to {FIGURES_DIR}/")
