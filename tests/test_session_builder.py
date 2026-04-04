# tests/test_session_builder.py

from pathlib import Path

from app.ingestion.session_builder import build_session


def test_build_session_with_valid_prices_and_trades():
    files = [
        Path("data/sample/prices_round_1_day_-1.csv"),
        Path("data/sample/trades_round_1_day_-1.csv"),
    ]

    result = build_session(files)

    assert len(result.files) == 2
    assert result.valid_file_count == 2
    assert result.invalid_file_count == 0

    assert not result.prices_df.empty
    assert not result.trades_df.empty

    assert result.available_rounds == [1]
    assert result.available_days == [-1]

    assert "KELP" in result.available_products
    assert "RAINFOREST_RESIN" in result.available_products
    assert "SQUID_INK" in result.available_products


def test_build_session_with_invalid_file():
    files = [
        Path("data/sample/prices_round_1_day_-1.csv"),
        Path("data/sample/not_a_real_file.csv"),
    ]

    result = build_session(files)

    assert len(result.files) == 2
    assert result.valid_file_count == 1
    assert result.invalid_file_count == 1
    assert not result.prices_df.empty


def test_build_session_summary_contains_per_file_status():
    files = [
        Path("data/sample/prices_round_1_day_-1.csv"),
        Path("data/sample/trades_round_1_day_-1.csv"),
    ]

    result = build_session(files)

    filenames = [f.source_file for f in result.files]
    statuses = [f.status for f in result.files]

    assert "prices_round_1_day_-1.csv" in filenames
    assert "trades_round_1_day_-1.csv" in filenames
    assert statuses == ["valid", "valid"]