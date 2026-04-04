# tests/test_file_detector.py

from pathlib import Path

from app.ingestion.csv_loader import load_imc_csv
from app.ingestion.file_detector import detect_file_type


def test_detect_prices_file():
    df = load_imc_csv(Path("data/sample/prices_round_1_day_-1.csv"))
    result = detect_file_type(df)

    assert result.detected_type == "prices"
    assert result.confidence == "high"
    assert result.missing_columns == []


def test_detect_trades_file():
    df = load_imc_csv(Path("data/sample/trades_round_1_day_-1.csv"))
    result = detect_file_type(df)

    assert result.detected_type == "trades"
    assert result.confidence == "high"
    assert result.missing_columns == []


def test_detect_unknown_file():
    import pandas as pd

    df = pd.DataFrame(
        {
            "foo": [1, 2],
            "bar": [3, 4],
        }
    )

    result = detect_file_type(df)

    assert result.detected_type == "unknown"
    assert result.confidence == "low"
    assert len(result.warnings) >= 1