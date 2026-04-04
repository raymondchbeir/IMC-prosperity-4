# app/ingestion/session_builder.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from app.ingestion.csv_loader import load_imc_csv
from app.ingestion.file_detector import detect_file_type
from app.ingestion.filename_parser import parse_filename
from app.ingestion.normalizer import FileMetadata, normalize_df
from app.ingestion.validator import validate_df


@dataclass
class FileProcessingResult:
    source_file: str
    dataset_type: str | None
    round: int | None
    day: int | None
    row_count: int
    products: list[str] = field(default_factory=list)
    status: str = "invalid"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class SessionBuildResult:
    files: list[FileProcessingResult]
    prices_df: pd.DataFrame
    trades_df: pd.DataFrame
    available_rounds: list[int]
    available_days: list[int]
    available_products: list[str]

    @property
    def valid_file_count(self) -> int:
        return sum(1 for f in self.files if f.status == "valid")

    @property
    def invalid_file_count(self) -> int:
        return sum(1 for f in self.files if f.status == "invalid")


def _extract_products(df: pd.DataFrame) -> list[str]:
    if "product" not in df.columns:
        return []
    return sorted(df["product"].dropna().astype(str).unique().tolist())


def build_session(file_paths: Iterable[str | Path]) -> SessionBuildResult:
    file_results: list[FileProcessingResult] = []
    prices_frames: list[pd.DataFrame] = []
    trades_frames: list[pd.DataFrame] = []

    for file_path in file_paths:
        path = Path(file_path)
        parsed = parse_filename(path.name)

        try:
            raw_df = load_imc_csv(path)
        except Exception as e:
            file_results.append(
                FileProcessingResult(
                    source_file=path.name,
                    dataset_type=parsed.dataset_type,
                    round=parsed.round,
                    day=parsed.day,
                    row_count=0,
                    status="invalid",
                    warnings=parsed.warnings.copy(),
                    errors=[f"Failed to load CSV: {e}"],
                )
            )
            continue

        detected = detect_file_type(raw_df)
        dataset_type = detected.detected_type if detected.detected_type != "unknown" else parsed.dataset_type

        if dataset_type is None or dataset_type == "unknown":
            file_results.append(
                FileProcessingResult(
                    source_file=path.name,
                    dataset_type=None,
                    round=parsed.round,
                    day=parsed.day,
                    row_count=len(raw_df),
                    status="invalid",
                    warnings=parsed.warnings + detected.warnings,
                    errors=["Could not determine dataset_type from filename or columns."],
                )
            )
            continue

        metadata = FileMetadata(
            dataset_type=dataset_type,
            round=parsed.round,
            day=parsed.day,
            source_file=path.name,
        )

        normalized_df = normalize_df(raw_df, metadata)
        validation = validate_df(normalized_df, dataset_type)

        status = "valid" if validation.is_valid else "invalid"
        warnings = parsed.warnings + detected.warnings + validation.warnings
        errors = validation.errors.copy()

        products = _extract_products(normalized_df)

        result = FileProcessingResult(
            source_file=path.name,
            dataset_type=dataset_type,
            round=parsed.round,
            day=parsed.day,
            row_count=len(normalized_df),
            products=products,
            status=status,
            warnings=warnings,
            errors=errors,
        )
        file_results.append(result)

        if validation.is_valid:
            if dataset_type == "prices":
                prices_frames.append(normalized_df)
            elif dataset_type == "trades":
                trades_frames.append(normalized_df)

    prices_df = pd.concat(prices_frames, ignore_index=True) if prices_frames else pd.DataFrame()
    trades_df = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()

    rounds = sorted(
        {
            int(x)
            for x in (
                list(prices_df["round"].dropna()) if "round" in prices_df.columns else []
            ) + (
                list(trades_df["round"].dropna()) if "round" in trades_df.columns else []
            )
        }
    )

    days = sorted(
        {
            int(x)
            for x in (
                list(prices_df["day"].dropna()) if "day" in prices_df.columns else []
            ) + (
                list(trades_df["day"].dropna()) if "day" in trades_df.columns else []
            )
        }
    )

    products = sorted(
        set(
            (prices_df["product"].dropna().astype(str).tolist() if "product" in prices_df.columns else [])
            + (trades_df["product"].dropna().astype(str).tolist() if "product" in trades_df.columns else [])
        )
    )

    return SessionBuildResult(
        files=file_results,
        prices_df=prices_df,
        trades_df=trades_df,
        available_rounds=rounds,
        available_days=days,
        available_products=products,
    )