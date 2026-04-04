# tests/test_normalizer.py

from pathlib import Path

from app.ingestion.csv_loader import load_imc_csv
from app.ingestion.normalizer import FileMetadata, normalize_df


def test_normalize_prices_df():
    df = load_imc_csv(Path("data/sample/prices_round_1_day_-1.csv"))
    metadata = FileMetadata(
        dataset_type="prices",
        round=1,
        day=-1,
        source_file="prices_round_1_day_-1.csv",
    )

    out = normalize_df(df, metadata)

    assert "dataset_type" in out.columns
    assert "round" in out.columns
    assert "day" in out.columns
    assert "source_file" in out.columns
    assert "product" in out.columns
    assert "bid_price_1" in out.columns
    assert "ask_price_1" in out.columns

    assert out["dataset_type"].iloc[0] == "prices"
    assert out["round"].iloc[0] == 1
    assert out["day"].iloc[0] == -1
    assert out["source_file"].iloc[0] == "prices_round_1_day_-1.csv"


def test_normalize_trades_df_renames_symbol_to_product():
    df = load_imc_csv(Path("data/sample/trades_round_1_day_-1.csv"))
    metadata = FileMetadata(
        dataset_type="trades",
        round=1,
        day=-1,
        source_file="trades_round_1_day_-1.csv",
    )

    out = normalize_df(df, metadata)

    assert "product" in out.columns
    assert "symbol" not in out.columns
    assert "price" in out.columns
    assert "quantity" in out.columns

    assert out["dataset_type"].iloc[0] == "trades"
    assert out["round"].iloc[0] == 1
    assert out["day"].iloc[0] == -1
    assert out["source_file"].iloc[0] == "trades_round_1_day_-1.csv"


def test_normalized_output_uses_canonical_column_order():
    df = load_imc_csv(Path("data/sample/trades_round_1_day_-1.csv"))
    metadata = FileMetadata(
        dataset_type="trades",
        round=1,
        day=-1,
        source_file="trades_round_1_day_-1.csv",
    )

    out = normalize_df(df, metadata)

    expected_start = ["dataset_type", "round", "day", "timestamp", "product"]
    assert list(out.columns[:5]) == expected_start