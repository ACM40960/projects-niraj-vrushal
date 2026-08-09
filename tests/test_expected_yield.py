import numpy as np
import pandas as pd
import pytest

from src.data.generate_sample import generate_sample
from src.data.preprocessing import build_feature_matrix, engineer_features, stratified_split
from src.models.bayesian_model import fit_bayesian_logistic, scale_features
from src.models.expected_yield import (
    build_yield_table,
    capture_curve,
    compute_expected_yield_summary,
)


@pytest.fixture(scope="module")
def fitted_model_and_test_data():
    raw = generate_sample(n_rows=5000, n_steps=10)
    engineered = engineer_features(raw)
    train_df, test_df = stratified_split(engineered, test_size=0.3, random_state=42)
    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model, trace = fit_bayesian_logistic(
        X_train_scaled, y_train, method="advi", advi_iterations=500, draws=200, progressbar=False
    )
    return trace, X_test_scaled, test_df


def test_compute_expected_yield_summary_returns_expected_columns(fitted_model_and_test_data):
    trace, X_test_scaled, test_df = fitted_model_and_test_data
    summary = compute_expected_yield_summary(trace, X_test_scaled, test_df["amount"], max_draws=50)

    for col in [
        "fraud_prob_mean",
        "expected_yield_mean",
        "expected_yield_std",
        "expected_yield_hdi_3%",
        "expected_yield_hdi_97%",
    ]:
        assert col in summary.columns
    assert len(summary) == len(test_df)
    assert (summary["expected_yield_mean"] >= 0).all()


def test_expected_yield_bounded_by_amount(fitted_model_and_test_data):
    # expected_yield = P(fraud) * amount, and P(fraud) in [0, 1], so yield
    # can never exceed the transaction's own amount.
    trace, X_test_scaled, test_df = fitted_model_and_test_data
    summary = compute_expected_yield_summary(trace, X_test_scaled, test_df["amount"], max_draws=50)
    assert (summary["expected_yield_mean"].values <= test_df["amount"].values + 1e-3).all()


def test_build_yield_table_is_sorted_descending(fitted_model_and_test_data):
    trace, X_test_scaled, test_df = fitted_model_and_test_data
    summary = compute_expected_yield_summary(trace, X_test_scaled, test_df["amount"], max_draws=50)
    table = build_yield_table(test_df, summary)

    assert len(table) == len(test_df)
    assert (table["expected_yield_mean"].diff().dropna() <= 1e-6).all()
    assert "isFraud" in table.columns


def test_capture_curve_reaches_100_percent_at_k_equals_n(fitted_model_and_test_data):
    trace, X_test_scaled, test_df = fitted_model_and_test_data
    summary = compute_expected_yield_summary(trace, X_test_scaled, test_df["amount"], max_draws=50)
    table = build_yield_table(test_df, summary)

    curve = capture_curve(table, [len(table)])
    assert curve.iloc[0]["fraud_capture_rate"] == pytest.approx(1.0)
    assert curve.iloc[0]["fraud_amount_capture_rate"] == pytest.approx(1.0)


def test_capture_curve_precision_at_k_matches_manual_count(fitted_model_and_test_data):
    trace, X_test_scaled, test_df = fitted_model_and_test_data
    summary = compute_expected_yield_summary(trace, X_test_scaled, test_df["amount"], max_draws=50)
    table = build_yield_table(test_df, summary)

    k = 10
    curve = capture_curve(table, [k])
    expected_precision = table.head(k)["isFraud"].sum() / k
    assert curve.iloc[0]["precision_at_k"] == pytest.approx(expected_precision)


def test_max_draws_reduces_memory_footprint_without_error():
    # A quick sanity check that subsampling posterior draws doesn't break
    # anything on a small model, independent of the fixture above.
    raw = generate_sample(n_rows=1000, n_steps=5)
    engineered = engineer_features(raw)
    X, y = build_feature_matrix(engineered)
    X_scaled, _, _ = scale_features(X, X.iloc[:10])

    model, trace = fit_bayesian_logistic(
        X_scaled, y, method="advi", advi_iterations=300, draws=500, progressbar=False
    )
    summary = compute_expected_yield_summary(
        trace, X_scaled.iloc[:50], pd.Series(np.full(50, 1000.0)), max_draws=20
    )
    assert len(summary) == 50
