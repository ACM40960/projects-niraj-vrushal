"""
Preprocessing and feature engineering for the PaySim dataset.

Turns raw PaySim transactions into a modeling-ready feature matrix:
  - encodes the categorical `type` column
  - engineers balance-discrepancy features (the strongest fraud signal
    identified during EDA — see notebooks/01_eda.ipynb)
  - produces a stratified train/test split that preserves the (very rare)
    fraud rate in both partitions
"""

import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TRANSACTION_TYPES = ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Columns that are pure identifiers / leak information and should not be
# fed into a model as features.
NON_FEATURE_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud"]

TARGET_COLUMN = "isFraud"

DEFAULT_PROCESSED_DIR = (
    pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
)


def add_balance_discrepancy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features that flag inconsistencies between a transaction's
    stated amount and the resulting account balances.

    PaySim fraud very often (a) fully drains the origin account and/or
    (b) leaves a nonzero discrepancy between the expected and reported
    new balance — both are strong, cheap-to-compute fraud signals.
    """
    df = df.copy()

    df["orig_balance_discrepancy"] = (
        df["newbalanceOrig"] - (df["oldbalanceOrg"] - df["amount"])
    )
    df["dest_balance_discrepancy"] = (
        df["newbalanceDest"] - (df["oldbalanceDest"] + df["amount"])
    )

    df["orig_balance_drained"] = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype(int)

    # Guard against division by zero for accounts that started at 0 balance.
    df["amount_to_orig_balance_ratio"] = df["amount"] / df["oldbalanceOrg"].replace(0, np.nan)
    df["amount_to_orig_balance_ratio"] = df["amount_to_orig_balance_ratio"].fillna(0)

    return df


def encode_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the `type` column into fixed, known categories.

    Normalizes underscores to hyphens first: PaySim documentation almost
    universally writes transaction types with hyphens (CASH-OUT), but some
    raw CSV distributions of the dataset actually store them with
    underscores (CASH_OUT) instead. Since pd.Categorical only matches
    exact strings, a silent mismatch here would zero out the *entire*
    one-hot encoding for every affected row rather than error - and
    CASH_OUT is one of only two transaction types PaySim fraud ever
    occurs in, so this is a correctness-critical normalization, not
    cosmetic. Raises a clear error if any value still doesn't match a
    known category after normalizing, rather than silently producing an
    all-zero row.
    """
    df = df.copy()
    normalized = df["type"].astype(str).str.replace("_", "-", regex=False)

    unrecognized = set(normalized.unique()) - set(TRANSACTION_TYPES)
    if unrecognized:
        raise ValueError(
            f"Unrecognized transaction type(s) after normalization: {sorted(unrecognized)}. "
            f"Expected one of {TRANSACTION_TYPES}. Check the raw 'type' column values."
        )

    df["type"] = pd.Categorical(normalized, categories=TRANSACTION_TYPES)
    dummies = pd.get_dummies(df["type"], prefix="type", dtype=int)
    df = pd.concat([df.drop(columns=["type"]), dummies], axis=1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline: balance features + type encoding."""
    df = add_balance_discrepancy_features(df)
    df = encode_transaction_type(df)
    return df


def stratified_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train/test, preserving the fraud rate in both partitions.
    Critical given how rare fraud is — a plain random split risks a test
    set with too few (or zero) positive examples.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_feature_matrix(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a processed DataFrame into (X, y), dropping non-feature columns."""
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns] + [target_col]
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y


def run_preprocessing_pipeline(
    raw_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save: bool = True,
    output_dir: pathlib.Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end: raw PaySim DataFrame -> engineered, split, (optionally saved)
    train/test DataFrames. Returns (train_df, test_df) with all engineered
    feature columns and the original target column intact.
    """
    engineered = engineer_features(raw_df)
    train_df, test_df = stratified_split(
        engineered, test_size=test_size, random_state=random_state
    )

    if save:
        output_dir = pathlib.Path(output_dir or DEFAULT_PROCESSED_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

    return train_df, test_df


if __name__ == "__main__":
    from src.data.load_data import load_paysim

    raw = load_paysim()
    train_df, test_df = run_preprocessing_pipeline(raw)
    print(f"Train: {len(train_df):,} rows, fraud rate {train_df[TARGET_COLUMN].mean():.4%}")
    print(f"Test:  {len(test_df):,} rows, fraud rate {test_df[TARGET_COLUMN].mean():.4%}")
