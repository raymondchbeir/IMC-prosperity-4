# app/ingestion/validator.py

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


PRICE_REQUIRED_COLUMNS = {
    "timestamp",
    "product",
    "bid_price_1",
    "ask_price_1",
}

TRADE_REQUIRED_COLUMNS = {
    "timestamp",
    "product",
    "price",
    "quantity",
}


@dataclass
class ValidationResult:
    is_valid: bool
    dataset_type: str
    missing_required_columns: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_df(df: pd.DataFrame, dataset_type: str) -> ValidationResult:
    cols = {str(col).strip().lower() for col in df.columns}

    if dataset_type == "prices":
        required = PRICE_REQUIRED_COLUMNS
    elif dataset_type == "trades":
        required = TRADE_REQUIRED_COLUMNS
    else:
        return ValidationResult(
            is_valid=False,
            dataset_type=dataset_type,
            missing_required_columns=[],
            errors=[f"Unsupported dataset_type: {dataset_type}"],
            warnings=[],
        )

    missing = sorted(required.difference(cols))

    if missing:
        return ValidationResult(
            is_valid=False,
            dataset_type=dataset_type,
            missing_required_columns=missing,
            errors=[f"Missing required columns for {dataset_type}: {', '.join(missing)}"],
            warnings=[],
        )

    return ValidationResult(
        is_valid=True,
        dataset_type=dataset_type,
        missing_required_columns=[],
        errors=[],
        warnings=[],
    )