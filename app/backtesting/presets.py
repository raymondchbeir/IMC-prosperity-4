from __future__ import annotations

from app.config import DEFAULT_BACKTEST_PRESET, ROUND_0_LIMITS


PRESET_OPTIONS = [
    {"label": "Tutorial / Round 0 preset", "value": "tutorial_round_0"},
    {"label": "Manual targets", "value": "manual"},
    {"label": "Current dashboard round", "value": "selected_dashboard_round"},
    {"label": "Current dashboard round + day", "value": "selected_dashboard_round_day"},
]


def get_preset_targets(preset: str, selected_round: int | None, selected_day: int | None) -> str:
    if preset == "tutorial_round_0":
        return "0"
    if preset == "selected_dashboard_round":
        if selected_round is None:
            raise ValueError("Choose a dashboard round first.")
        return str(selected_round)
    if preset == "selected_dashboard_round_day":
        if selected_round is None or selected_day is None:
            raise ValueError("Choose a dashboard round and day first.")
        return f"{selected_round}-{selected_day}"
    return ""


def get_default_limit_overrides(preset: str) -> dict[str, int]:
    if preset == DEFAULT_BACKTEST_PRESET:
        return ROUND_0_LIMITS.copy()
    return {}
