# app/ingestion/csv_loader.py

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd


_NUMERIC_COLS = [
    "day",
    "timestamp",
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
    "buyer",
    "seller",
    "price",
    "quantity",
]


def load_imc_csv(file: str | Path | BinaryIO) -> pd.DataFrame:
    """
    Fast IMC Prosperity CSV loader.

    Key speedups:
    - use pandas' default C parser instead of engine="python"
    - normalize columns once
    - downcast numeric columns to reduce memory
    - use category dtype for product-like string columns
    """
    df = pd.read_csv(
        file,
        sep=";",
        low_memory=False,
    )

    df.columns = [str(col).strip().lower() for col in df.columns]

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

    for col in ["product", "symbol", "currency"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df
