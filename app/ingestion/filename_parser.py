# app/ingestion/filename_parser.py

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


FILENAME_PATTERN = re.compile(
    r"^(?P<dataset_type>prices|trades)_round_(?P<round>\d+)_day_(?P<day>-?\d+)\.csv$",
    re.IGNORECASE,
)


@dataclass
class ParsedFilename:
    original_filename: str
    stem: str
    dataset_type: Optional[str]
    round: Optional[int]
    day: Optional[int]
    matched_pattern: bool
    warnings: list[str] = field(default_factory=list)


def parse_filename(filename: str) -> ParsedFilename:
    """
    Parse IMC Prosperity filenames like:
      - prices_round_1_day_-1.csv
      - trades_round_1_day_0.csv

    Returns a ParsedFilename object. If the filename does not match the expected
    pattern, the fields dataset_type / round / day are set to None and a warning
    is attached.
    """
    path = Path(filename)
    original_filename = path.name
    stem = path.stem

    match = FILENAME_PATTERN.match(original_filename)

    if not match:
        return ParsedFilename(
            original_filename=original_filename,
            stem=stem,
            dataset_type=None,
            round=None,
            day=None,
            matched_pattern=False,
            warnings=[
                "Filename does not match expected pattern: "
                "'prices_round_<n>_day_<d>.csv' or 'trades_round_<n>_day_<d>.csv'."
            ],
        )

    dataset_type = match.group("dataset_type").lower()
    round_num = int(match.group("round"))
    day_num = int(match.group("day"))

    return ParsedFilename(
        original_filename=original_filename,
        stem=stem,
        dataset_type=dataset_type,
        round=round_num,
        day=day_num,
        matched_pattern=True,
        warnings=[],
    )
