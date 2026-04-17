from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BacktestTarget:
    round_num: int
    day_num: int

    @property
    def label(self) -> str:
        return f"{self.round_num}-{self.day_num}"


@dataclass(slots=True)
class BacktestRequest:
    preset: str
    match_trades: str
    targets_text: str = ""
    selected_round: int | None = None
    selected_day: int | None = None
    limit_overrides: dict[str, int] = field(default_factory=dict)
    use_custom_data: bool = False
    extra_volume_pct: float = 0.0


@dataclass(slots=True)
class BacktestPayload:
    targets: list[BacktestTarget]
    activity_rows: list[dict[str, Any]]
    submission_trade_rows: list[dict[str, Any]]
    realized_trade_rows: list[dict[str, Any]]
    sandbox_rows: list[dict[str, Any]]
    per_run_rows: list[dict[str, Any]]
    per_product_rows: list[dict[str, Any]]
    summary: dict[str, Any]
