import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from src.data.generate_sample import generate_sample
from src.data.preprocessing import build_feature_matrix, engineer_features, stratified_split
from src.models.baseline import evaluate_classifier, train_linear_svm, train_random_forest


@pytest.fixture
def train_test_features():
    # Large enough sample that both classes appear in train and test.
    raw = generate_sample(n_rows=20_000, n_steps=20)
    engineered = engineer_features(raw)
    train_df, test_df = stratified_split(engineered, test_size=0.2, random_state=42)
    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)
    return X_train, y_train, X_test, y_test


def test_train_random_forest_returns_fitted_model(train_test_features):
    X_train, y_train, _, _ = train_test_features
    model = train_random_forest(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")


def test_train_linear_svm_returns_fitted_model(train_test_features):
    X_train, y_train, _, _ = train_test_features
    model = train_linear_svm(X_train, y_train)
    assert isinstance(model, LinearSVC)
    assert hasattr(model, "classes_")


def test_evaluate_classifier_returns_expected_metrics(train_test_features):
    X_train, y_train, X_test, y_test = train_test_features
    model = train_random_forest(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)

    for key in ["precision", "recall", "f1", "confusion_matrix", "classification_report"]:
        assert key in metrics
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_evaluate_classifier_includes_ranking_metrics_when_both_classes_present(
    train_test_features,
):
    X_train, y_train, X_test, y_test = train_test_features
    model = train_random_forest(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)

    assert "roc_auc" in metrics
    assert "average_precision" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_classifier_handles_model_without_predict_proba(train_test_features):
    # LinearSVC has no predict_proba; evaluate_classifier should fall back
    # to decision_function without raising.
    X_train, y_train, X_test, y_test = train_test_features
    model = train_linear_svm(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)
    assert "f1" in metrics
