# app/ingestion/csv_loader.py

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd


def load_imc_csv(file: str | Path | BinaryIO) -> pd.DataFrame:
    """
    Load an IMC Prosperity CSV file.

    Supports:
    - file path strings
    - pathlib.Path objects
    - uploaded file-like binary objects

    Assumes:
    - semicolon-separated values
    - header row present
    - empty cells should remain NaN
    """
    df = pd.read_csv(
        file,
        sep=";",
        engine="python",
    )

    df.columns = [str(col).strip().lower() for col in df.columns]
    return df