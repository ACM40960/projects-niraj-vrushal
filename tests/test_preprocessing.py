import pandas as pd
import pytest

from src.data.generate_sample import generate_sample
from src.data.preprocessing import (
    TARGET_COLUMN,
    TRANSACTION_TYPES,
    add_balance_discrepancy_features,
    build_feature_matrix,
    encode_transaction_type,
    engineer_features,
    run_preprocessing_pipeline,
    stratified_split,
)


@pytest.fixture
def raw_df():
    # Large enough that stratified splitting has more than one fraud example
    # to work with, given how rare fraud is even in the synthetic sample.
    return generate_sample(n_rows=20_000, n_steps=20)


def test_add_balance_discrepancy_features_adds_expected_columns(raw_df):
    df = add_balance_discrepancy_features(raw_df)
    for col in [
        "orig_balance_discrepancy",
        "dest_balance_discrepancy",
        "orig_balance_drained",
        "amount_to_orig_balance_ratio",
    ]:
        assert col in df.columns
    # no NaNs introduced even for zero-balance origin accounts
    assert df["amount_to_orig_balance_ratio"].isna().sum() == 0


def test_encode_transaction_type_produces_one_hot_columns(raw_df):
    df = encode_transaction_type(raw_df)
    assert "type" not in df.columns
    for t in TRANSACTION_TYPES:
        assert f"type_{t}" in df.columns
    # each row should belong to exactly one type
    type_cols = [f"type_{t}" for t in TRANSACTION_TYPES]
    assert (df[type_cols].sum(axis=1) == 1).all()


def test_encode_transaction_type_normalizes_underscore_variants(raw_df):
    # Regression test: some real PaySim CSV distributions store type
    # values with underscores (CASH_OUT, CASH_IN) rather than the
    # hyphenated form (CASH-OUT, CASH-IN) used elsewhere in this
    # codebase and in most PaySim documentation. Previously this silently
    # zeroed out the one-hot encoding for every affected row instead of
    # erroring, corrupting a feature for two of the five transaction
    # types (including CASH_OUT, one of only two fraud-eligible types)
    # without any visible failure.
    underscored = raw_df.copy()
    underscored["type"] = underscored["type"].str.replace("-", "_", regex=False)

    df = encode_transaction_type(underscored)
    type_cols = [f"type_{t}" for t in TRANSACTION_TYPES]

    # Every row must map to exactly one category - none should have
    # silently fallen through to an all-zero encoding.
    assert (df[type_cols].sum(axis=1) == 1).all()

    # Cross-check against the hyphenated version of the same data: the
    # resulting one-hot columns should be identical either way.
    df_hyphenated = encode_transaction_type(raw_df)
    pd.testing.assert_frame_equal(
        df[type_cols].reset_index(drop=True), df_hyphenated[type_cols].reset_index(drop=True)
    )


def test_encode_transaction_type_raises_on_truly_unrecognized_values(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "type"] = "NOT_A_REAL_TYPE"
    with pytest.raises(ValueError, match="Unrecognized transaction type"):
        encode_transaction_type(bad)


def test_engineer_features_combines_both_steps(raw_df):
    df = engineer_features(raw_df)
    assert "type" not in df.columns
    assert "orig_balance_discrepancy" in df.columns


def test_stratified_split_preserves_target_column(raw_df):
    engineered = engineer_features(raw_df)
    train_df, test_df = stratified_split(engineered, test_size=0.25, random_state=1)
    assert len(train_df) + len(test_df) == len(engineered)
    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN in test_df.columns
    # both partitions should contain at least one fraud example
    assert train_df[TARGET_COLUMN].sum() > 0
    assert test_df[TARGET_COLUMN].sum() > 0


def test_build_feature_matrix_drops_identifier_and_target_columns(raw_df):
    engineered = engineer_features(raw_df)
    X, y = build_feature_matrix(engineered)
    assert "nameOrig" not in X.columns
    assert "nameDest" not in X.columns
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y)
    # all remaining columns should be numeric (safe to feed into a model)
    assert all(pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes)


def test_run_preprocessing_pipeline_end_to_end(raw_df, tmp_path):
    train_df, test_df = run_preprocessing_pipeline(
        raw_df, test_size=0.2, random_state=0, save=True, output_dir=tmp_path
    )
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()
    assert len(train_df) + len(test_df) == len(raw_df)
