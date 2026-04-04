# app/ingestion/normalizer.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


PRICE_CANONICAL_COLUMNS = [
    "dataset_type",
    "round",
    "day",
    "timestamp",
    "product",
    "bid_price_1",
    "bid_volume_1",
    "bid_price_2",
    "bid_volume_2",
    "bid_price_3",
    "bid_volume_3",
    "ask_price_1",
    "ask_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "ask_price_3",
    "ask_volume_3",
    "mid_price",
    "profit_and_loss",
    "source_file",
]

TRADE_CANONICAL_COLUMNS = [
    "dataset_type",
    "round",
    "day",
    "timestamp",
    "product",
    "buyer",
    "seller",
    "currency",
    "price",
    "quantity",
    "source_file",
]


@dataclass
class FileMetadata:
    dataset_type: str
    round: Optional[int]
    day: Optional[int]
    source_file: str


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def normalize_prices_df(df: pd.DataFrame, metadata: FileMetadata) -> pd.DataFrame:
    out = df.copy()

    out.columns = [str(col).strip().lower() for col in out.columns]

    out["dataset_type"] = metadata.dataset_type
    out["round"] = metadata.round
    out["day"] = metadata.day
    out["source_file"] = metadata.source_file

    out = _ensure_columns(out, PRICE_CANONICAL_COLUMNS)
    out = out[PRICE_CANONICAL_COLUMNS]

    return out


def normalize_trades_df(df: pd.DataFrame, metadata: FileMetadata) -> pd.DataFrame:
    out = df.copy()

    out.columns = [str(col).strip().lower() for col in out.columns]

    if "symbol" in out.columns and "product" not in out.columns:
        out = out.rename(columns={"symbol": "product"})

    out["dataset_type"] = metadata.dataset_type
    out["round"] = metadata.round
    out["day"] = metadata.day
    out["source_file"] = metadata.source_file

    out = _ensure_columns(out, TRADE_CANONICAL_COLUMNS)
    out = out[TRADE_CANONICAL_COLUMNS]

    return out


def normalize_df(df: pd.DataFrame, metadata: FileMetadata) -> pd.DataFrame:
    if metadata.dataset_type == "prices":
        return normalize_prices_df(df, metadata)
    if metadata.dataset_type == "trades":
        return normalize_trades_df(df, metadata)
    raise ValueError(f"Unsupported dataset_type: {metadata.dataset_type}")