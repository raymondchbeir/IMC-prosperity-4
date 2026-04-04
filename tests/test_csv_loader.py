# tests/test_csv_loader.py

from pathlib import Path

from app.ingestion.csv_loader import load_imc_csv


def test_load_prices_csv():
    path = Path("data/sample/prices_round_1_day_-1.csv")
    df = load_imc_csv(path)

    assert not df.empty
    assert "product" in df.columns
    assert "bid_price_1" in df.columns
    assert "ask_price_1" in df.columns
    assert "timestamp" in df.columns


def test_load_trades_csv():
    path = Path("data/sample/trades_round_1_day_-1.csv")
    df = load_imc_csv(path)

    assert not df.empty
    assert "symbol" in df.columns
    assert "price" in df.columns
    assert "quantity" in df.columns
    assert "timestamp" in df.columns


def test_columns_are_normalized():
    path = Path("data/sample/trades_round_1_day_-1.csv")
    df = load_imc_csv(path)

    for col in df.columns:
        assert col == col.strip().lower()
        