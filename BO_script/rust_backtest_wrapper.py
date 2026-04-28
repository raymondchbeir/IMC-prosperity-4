"""Adapter: runs the Rust backtester for one (round, day) with overridden Trader params.

Drops into Optuna's `run_backtest_one_day` stub.  Sentinel (round=0, day=0)
routes to the held-out eval directory pointed at by EVAL_DATA_PATH.

Env vars:
    BASE_STRATEGY_PATH   Path to the patched Trader file.
    EVAL_DATA_PATH       Directory containing prices_round_0_day_0.csv +
                         trades_round_0_day_0_nn.csv (held-out 100k).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from app.backtesting.rust_runner import run_rust_backtests
from app.models.schemas import BacktestRequest


BASE_STRATEGY_PATH = Path(
    os.environ.get("BASE_STRATEGY_PATH", "498727_v17_patched.py")
)
EVAL_DATA_PATH = os.environ.get("EVAL_DATA_PATH", "")


def _emit_strategy_with_overrides(params: Dict[str, Any]) -> Path:
    """Write base trader + `Trader.NAME = value` lines to a fresh .py file."""
    if not BASE_STRATEGY_PATH.exists():
        raise FileNotFoundError(
            f"Base strategy not found at {BASE_STRATEGY_PATH}. "
            f"Set BASE_STRATEGY_PATH env var."
        )
    base = BASE_STRATEGY_PATH.read_text()
    parts = [base, "\n\n# === optuna parameter overrides ===\n"]
    for k, v in params.items():
        parts.append(f"Trader.{k} = {v!r}\n")
    fd, name = tempfile.mkstemp(prefix="trader_trial_", suffix=".py", text=True)
    os.close(fd)
    Path(name).write_text("".join(parts))
    return Path(name)


def run_one_day(params: Dict[str, Any], round_num: int, day_num: int) -> float:
    """Run one (round, day) backtest under `params`; return total PnL.

    Special case: (round_num=0, day_num=0) routes to EVAL_DATA_PATH for the
    held-out 100k.  All other (round, day) pairs go through the regular
    dashboard alias resolution.
    """
    strategy = _emit_strategy_with_overrides(params)
    try:
        if int(round_num) == 0 and int(day_num) == 0:
            if not EVAL_DATA_PATH:
                raise RuntimeError(
                    "Eval (0,0) requested but EVAL_DATA_PATH env var is empty. "
                    "Set EVAL_DATA_PATH=data/round4_eval_100k (or wherever)."
                )
            eval_dir = Path(EVAL_DATA_PATH).resolve()
            if not eval_dir.exists():
                raise RuntimeError(f"EVAL_DATA_PATH does not exist: {eval_dir}")
            request = BacktestRequest(
                preset="manual",
                selected_round=0,
                selected_day=0,
                targets_text="0-0",
                extra_volume_pct=0.0,
                match_trades=True,
            )
            payload = run_rust_backtests(strategy, request,
                                          custom_data_root=eval_dir)
        else:
            request = BacktestRequest(
                preset="selected_dashboard_round_day",
                selected_round=int(round_num),
                selected_day=int(day_num),
                targets_text=f"{int(round_num)}-{int(day_num)}",
                extra_volume_pct=0.0,
                match_trades=True,
            )
            payload = run_rust_backtests(strategy, request)

        pnl = payload.summary.get("total_pnl")
        if pnl is None:
            rows = payload.per_run_rows or []
            pnl = rows[0].get("final_pnl") if rows else 0.0
        if pnl is None:
            raise RuntimeError(
                f"rust_backtester returned no total_pnl for {round_num}-{day_num}"
            )
        return float(pnl)
    finally:
        try: strategy.unlink()
        except OSError: pass

