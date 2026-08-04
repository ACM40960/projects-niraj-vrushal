import pandas as pd
import pytest

from src.data.generate_sample import generate_sample
from src.data.load_data import REQUIRED_COLUMNS, load_paysim, summarize


@pytest.fixture
def sample_csv(tmp_path):
    df = generate_sample(n_rows=500, n_steps=5)
    path = tmp_path / "sample_paysim.csv"
    df.to_csv(path, index=False)
    return path


def test_load_paysim_reads_valid_file(sample_csv):
    df = load_paysim(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert set(REQUIRED_COLUMNS).issubset(df.columns)
    assert len(df) == 500


def test_load_paysim_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_paysim(missing_path)


def test_load_paysim_missing_columns_raises(tmp_path):
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2, 3]}).to_csv(bad_path, index=False)
    with pytest.raises(ValueError):
        load_paysim(bad_path)


def test_summarize_returns_expected_keys(sample_csv):
    df = load_paysim(sample_csv)
    stats = summarize(df)
    expected_keys = {"n_rows", "n_steps", "fraud_rate", "n_fraud", "transaction_types"}
    assert expected_keys.issubset(stats.keys())
    assert stats["n_rows"] == len(df)
    assert 0.0 <= stats["fraud_rate"] <= 1.0


def test_generate_sample_schema_matches_required_columns():
    df = generate_sample(n_rows=100, n_steps=3)
    assert set(REQUIRED_COLUMNS).issubset(df.columns)
    assert df["isFraud"].isin([0, 1]).all()
