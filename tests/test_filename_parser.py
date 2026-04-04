# tests/test_filename_parser.py

from app.ingestion.filename_parser import parse_filename


def test_parse_valid_prices_filename():
    result = parse_filename("prices_round_1_day_-1.csv")

    assert result.original_filename == "prices_round_1_day_-1.csv"
    assert result.stem == "prices_round_1_day_-1"
    assert result.dataset_type == "prices"
    assert result.round == 1
    assert result.day == -1
    assert result.matched_pattern is True
    assert result.warnings == []


def test_parse_valid_trades_filename():
    result = parse_filename("trades_round_4_day_0.csv")

    assert result.dataset_type == "trades"
    assert result.round == 4
    assert result.day == 0
    assert result.matched_pattern is True
    assert result.warnings == []


def test_parse_invalid_filename():
    result = parse_filename("weird_file.csv")

    assert result.dataset_type is None
    assert result.round is None
    assert result.day is None
    assert result.matched_pattern is False
    assert len(result.warnings) == 1