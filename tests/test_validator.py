# tests/test_validator.py

from pathlib import Path

import pandas as pd

from app.ingestion.csv_loader import load_imc_csv
from app.ingestion.normalizer import FileMetadata, normalize_df
from app.ingestion.validator import validate_df


def test_validate_prices_df():
    df = load_imc_csv(Path("data/sample/prices_round_1_day_-1.csv"))
    metadata = FileMetadata(
        dataset_type="prices",
        round=1,
        day=-1,
        source_file="prices_round_1_day_-1.csv",
    )
    out = normalize_df(df, metadata)

    result = validate_df(out, "prices")

    assert result.is_valid is True
    assert result.dataset_type == "prices"
    assert result.missing_required_columns == []
    assert result.errors == []


def test_validate_trades_df():
    df = load_imc_csv(Path("data/sample/trades_round_1_day_-1.csv"))
    metadata = FileMetadata(
        dataset_type="trades",
        round=1,
        day=-1,
        source_file="trades_round_1_day_-1.csv",
    )
    out = normalize_df(df, metadata)

    result = validate_df(out, "trades")

    assert result.is_valid is True
    assert result.dataset_type == "trades"
    assert result.missing_required_columns == []
    assert result.errors == []


def test_validate_invalid_prices_df_missing_required_columns():
    df = pd.DataFrame(
        {
            "timestamp": [0, 100],
            "product": ["KELP", "KELP"],
            "bid_price_1": [2028, 2025],
        }
    )

    result = validate_df(df, "prices")

    assert result.is_valid is False
    assert "ask_price_1" in result.missing_required_columns
    assert len(result.errors) == 1


def test_validate_invalid_dataset_type():
    df = pd.DataFrame({"a": [1]})

    result = validate_df(df, "unknown")

    assert result.is_valid is False
    assert len(result.errors) == 1