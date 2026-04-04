# app/ingestion/file_detector.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


PRICE_REQUIRED_COLUMNS = {
    "timestamp",
    "product",
    "bid_price_1",
    "ask_price_1",
}

TRADE_REQUIRED_COLUMNS = {
    "timestamp",
    "price",
    "quantity",
}

TRADE_PRODUCT_ALIASES = {"symbol", "product"}


@dataclass
class DetectedFileType:
    detected_type: str
    confidence: str
    matched_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_file_type(df: pd.DataFrame) -> DetectedFileType:
    """
    Detect whether a dataframe is:
    - prices
    - trades
    - unknown

    Detection is based on columns only.
    """
    cols = {str(col).strip().lower() for col in df.columns}

    price_matches = sorted(PRICE_REQUIRED_COLUMNS.intersection(cols))
    price_missing = sorted(PRICE_REQUIRED_COLUMNS.difference(cols))

    trade_core_matches = sorted(TRADE_REQUIRED_COLUMNS.intersection(cols))
    has_trade_product_col = any(alias in cols for alias in TRADE_PRODUCT_ALIASES)

    trade_matches = trade_core_matches.copy()
    if has_trade_product_col:
        trade_matches.append("symbol_or_product")

    trade_missing = sorted(TRADE_REQUIRED_COLUMNS.difference(cols))
    if not has_trade_product_col:
        trade_missing.append("symbol_or_product")

    is_prices = PRICE_REQUIRED_COLUMNS.issubset(cols)
    is_trades = TRADE_REQUIRED_COLUMNS.issubset(cols) and has_trade_product_col

    if is_prices and not is_trades:
        return DetectedFileType(
            detected_type="prices",
            confidence="high",
            matched_columns=price_matches,
            missing_columns=[],
            warnings=[],
        )

    if is_trades and not is_prices:
        return DetectedFileType(
            detected_type="trades",
            confidence="high",
            matched_columns=trade_matches,
            missing_columns=[],
            warnings=[],
        )

    if is_prices and is_trades:
        return DetectedFileType(
            detected_type="unknown",
            confidence="low",
            matched_columns=sorted(set(price_matches + trade_matches)),
            missing_columns=[],
            warnings=["Columns match both prices and trades patterns."],
        )

    return DetectedFileType(
        detected_type="unknown",
        confidence="low",
        matched_columns=sorted(set(price_matches + trade_matches)),
        missing_columns=sorted(set(price_missing + trade_missing)),
        warnings=["Could not confidently detect file type from columns."],
    )