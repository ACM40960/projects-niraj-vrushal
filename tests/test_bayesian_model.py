import numpy as np
import pandas as pd
import pytest

from src.data.generate_sample import generate_sample
from src.data.preprocessing import build_feature_matrix, engineer_features, stratified_split
from src.models.bayesian_model import (
    build_bayesian_logistic_model,
    fit_bayesian_logistic,
    posterior_predictive_probabilities,
    scale_features,
    summarize_posterior_probabilities,
)


@pytest.fixture(scope="module")
def train_test_features():
    raw = generate_sample(n_rows=5000, n_steps=10)
    engineered = engineer_features(raw)
    train_df, test_df = stratified_split(engineered, test_size=0.3, random_state=42)
    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, y_train, X_test_scaled, y_test


def test_scale_features_standardizes_continuous_columns():
    raw = generate_sample(n_rows=500, n_steps=5)
    engineered = engineer_features(raw)
    X, y = build_feature_matrix(engineered)
    X_train, X_test = X.iloc[:400], X.iloc[400:]
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Standardized training columns should have ~zero mean, unit variance.
    assert abs(X_train_scaled["amount"].mean()) < 1e-6
    assert abs(X_train_scaled["amount"].std() - 1.0) < 0.05
    # Binary/one-hot columns should be untouched (still 0/1).
    assert set(X_train_scaled["orig_balance_drained"].unique()).issubset({0, 1})


def test_build_bayesian_logistic_model_returns_valid_model():
    X = np.random.default_rng(0).normal(size=(50, 4))
    y = np.random.default_rng(1).integers(0, 2, size=50).astype(float)
    model = build_bayesian_logistic_model(X, y)
    assert "intercept" in [rv.name for rv in model.free_RVs]
    assert "coefs" in [rv.name for rv in model.free_RVs]


def test_fit_bayesian_logistic_advi_runs_and_returns_trace(train_test_features):
    X_train, y_train, _, _ = train_test_features
    model, trace = fit_bayesian_logistic(
        X_train, y_train, method="advi", advi_iterations=500, draws=200, progressbar=False
    )
    assert "intercept" in trace.posterior
    assert "coefs" in trace.posterior


def test_posterior_predictive_probabilities_shape_and_range(train_test_features):
    X_train, y_train, X_test, y_test = train_test_features
    model, trace = fit_bayesian_logistic(
        X_train, y_train, method="advi", advi_iterations=500, draws=200, progressbar=False
    )
    probs = posterior_predictive_probabilities(trace, X_test)

    n_posterior_draws = 200  # matches draws above (chains folded in)
    assert probs.shape[1] == len(X_test)
    assert probs.shape[0] >= n_posterior_draws
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_summarize_posterior_probabilities_returns_expected_columns(train_test_features):
    X_train, y_train, X_test, y_test = train_test_features
    model, trace = fit_bayesian_logistic(
        X_train, y_train, method="advi", advi_iterations=500, draws=200, progressbar=False
    )
    probs = posterior_predictive_probabilities(trace, X_test)
    summary = summarize_posterior_probabilities(probs)

    for col in ["fraud_prob_mean", "fraud_prob_std", "fraud_prob_hdi_3%", "fraud_prob_hdi_97%"]:
        assert col in summary.columns
    assert len(summary) == len(X_test)
    assert summary["fraud_prob_mean"].between(0, 1).all()
