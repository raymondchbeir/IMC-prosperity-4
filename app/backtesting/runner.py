from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from app.config import BACKTEST_DAY_SCAN_RANGE, BACKTESTER_INSTALL_HINT
from app.backtesting.presets import get_default_limit_overrides, get_preset_targets
from app.models.schemas import BacktestPayload, BacktestRequest, BacktestTarget

ACTIVITY_COLUMNS = [
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
    "profit_loss",
]

ROUND_DAY_TOKEN_RE = re.compile(r"^(\d+)(?:-(-?\d+))?$")
DATA_FILE_RE = re.compile(r"^(prices|trades|observations)_round_(\d+)_day_(-?\d+)\.csv$")


def _load_backtester_api():
    try:
        from prosperity4bt.data import has_day_data
        from prosperity4bt.file_reader import FileSystemReader, PackageResourcesReader
        from prosperity4bt.models import TradeMatchingMode
        from prosperity4bt.runner import run_backtest
    except ImportError as exc:
        raise RuntimeError(
            "The Prosperity 4 backtester is not installed in this environment. "
            f"Run `{BACKTESTER_INSTALL_HINT}` inside your trading_venv first."
        ) from exc

    return {
        "has_day_data": has_day_data,
        "FileSystemReader": FileSystemReader,
        "PackageResourcesReader": PackageResourcesReader,
        "TradeMatchingMode": TradeMatchingMode,
        "run_backtest": run_backtest,
    }


def _load_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _try_register_local_datamodel(strategy_path: Path) -> None:
    if "datamodel" in sys.modules:
        return

    strategy_dir = strategy_path.parent.resolve()

    preferred_names = [
        "datamodel.py",
        "data_model.py",
        "imc_datamodel.py",
        "model.py",
    ]

    for name in preferred_names:
        candidate = strategy_dir / name
        if candidate.exists() and candidate.is_file():
            _load_module_from_path("datamodel", candidate)
            return

    for candidate in strategy_dir.glob("*.py"):
        if candidate.name == strategy_path.name:
            continue

        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        looks_like_datamodel = (
            "class TradingState" in text
            or "class OrderDepth" in text
            or "class Order" in text
            or "class Trade" in text
        )

        if looks_like_datamodel:
            _load_module_from_path("datamodel", candidate)
            return


def _load_strategy_module(strategy_path: Path) -> ModuleType:
    module_name = f"imc_strategy_{uuid4().hex}"
    strategy_dir = str(strategy_path.parent.resolve())

    inserted = False
    if strategy_dir not in sys.path:
        sys.path.insert(0, strategy_dir)
        inserted = True

    try:
        _try_register_local_datamodel(strategy_path)
        return _load_module_from_path(module_name, strategy_path)
    finally:
        if inserted and sys.path and sys.path[0] == strategy_dir:
            sys.path.pop(0)


def _instantiate_trader(strategy_path: Path):
    module = _load_strategy_module(strategy_path)
    trader_cls = getattr(module, "Trader", None)
    if trader_cls is None:
        raise ValueError("Your strategy file must expose a `Trader` class.")
    return trader_cls()


def parse_limit_overrides(limit_text: str | None, preset: str) -> dict[str, int]:
    overrides = get_default_limit_overrides(preset)
    if not limit_text or not limit_text.strip():
        return overrides

    for raw_line in re.split(r"[\n,]+", limit_text.strip()):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid limit override '{line}'. Use PRODUCT:LIMIT.")
        product, value = line.split(":", 1)
        product = product.strip().upper()
        value = value.strip()
        if not value.lstrip("-").isdigit():
            raise ValueError(f"Invalid limit for '{product}': '{value}'")
        overrides[product] = int(value)
    return overrides


def _dedupe_targets(targets: list[BacktestTarget]) -> list[BacktestTarget]:
    seen: set[tuple[int, int]] = set()
    ordered: list[BacktestTarget] = []
    for target in targets:
        key = (target.round_num, target.day_num)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(target)
    return ordered


def resolve_targets(request: BacktestRequest, file_reader: Any, has_day_data_func) -> list[BacktestTarget]:
    expr = (request.targets_text or "").strip()
    if request.preset != "manual":
        expr = get_preset_targets(request.preset, request.selected_round, request.selected_day)

    if not expr:
        raise ValueError("Enter at least one target. Examples: 0, 1, 1-0, 1--1 1-0")

    targets: list[BacktestTarget] = []
    for token in re.split(r"[\s,]+", expr):
        if not token:
            continue
        match = ROUND_DAY_TOKEN_RE.match(token)
        if not match:
            raise ValueError(f"Invalid target token '{token}'.")

        round_num = int(match.group(1))
        day_group = match.group(2)

        if day_group is None:
            available_days = [
                day for day in range(-10, 10)
                if has_day_data_func(file_reader, round_num, day)
            ]

            if not available_days:
                raise ValueError(f"No data is available for round {round_num} in the selected data source.")

            expanded = [
                BacktestTarget(round_num=round_num, day_num=day)
                for day in sorted(available_days)
            ]
            targets.extend(expanded)
        else:
            day_num = int(day_group)
            if not has_day_data_func(file_reader, round_num, day_num):
                raise ValueError(f"No data is available for round {round_num} day {day_num} in the selected data source.")
            targets.append(BacktestTarget(round_num=round_num, day_num=day_num))

    targets = sorted(_dedupe_targets(targets), key=lambda t: (t.round_num, t.day_num))
    if not targets:
        raise ValueError("No valid backtest targets were resolved.")
    return targets


def build_custom_data_root(upload_dir: Path, filenames: list[str]) -> Path:
    data_root = upload_dir / "custom_backtest_data"
    data_root.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        match = DATA_FILE_RE.match(Path(filename).name)
        if match is None:
            raise ValueError(
                f"Custom backtest data file '{filename}' must follow the backtester naming pattern, "
                "for example prices_round_1_day_0.csv or trades_round_0_day_-1.csv."
            )

        _, round_num, _ = match.groups()
        round_dir = data_root / f"round{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)
        source = upload_dir / Path(filename).name
        target = round_dir / Path(filename).name
        target.write_bytes(source.read_bytes())

    return data_root


class _PatchedTextFile:
    def __init__(self, original_file, patched_text: str):
        self._original_file = original_file
        self._patched_text = patched_text

    def read_text(self, encoding: str = "utf-8", *args, **kwargs):
        return self._patched_text

    def __getattr__(self, name):
        return getattr(self._original_file, name)


def _scale_prices_csv_text(raw_text: str, factor: float) -> str:
    if factor == 1.0 or not raw_text:
        return raw_text

    lines = raw_text.splitlines()
    if len(lines) <= 1:
        return raw_text

    header = lines[0]
    out_lines = [header]

    volume_indices = [4, 6, 8, 10, 12, 14]

    for line in lines[1:]:
        if not line:
            out_lines.append(line)
            continue

        cols = line.split(";")
        for idx in volume_indices:
            if idx >= len(cols):
                continue
            value = cols[idx].strip()
            if value == "":
                continue
            try:
                original = int(float(value))
            except Exception:
                continue

            if original > 0:
                scaled = max(original, int(round(original * factor)))
                cols[idx] = str(scaled)

        out_lines.append(";".join(cols))

    return "\n".join(out_lines)


def _patch_file_reader_for_extra_volume(file_reader: Any, factor: float) -> Any:
    if factor == 1.0 or not hasattr(file_reader, "file"):
        return file_reader

    original_file_method = file_reader.file

    class _PatchedFileContext:
        def __init__(self, path_parts):
            self._path_parts = path_parts
            self._inner_cm = None
            self._entered_obj = None

        def __enter__(self):
            self._inner_cm = original_file_method(self._path_parts)
            self._entered_obj = self._inner_cm.__enter__()

            if self._entered_obj is None:
                return None

            filename = ""
            try:
                filename = str(self._path_parts[-1])
            except Exception:
                filename = ""

            if not filename.startswith("prices_round_") or not filename.endswith(".csv"):
                return self._entered_obj

            try:
                raw_text = self._entered_obj.read_text(encoding="utf-8")
                patched_text = _scale_prices_csv_text(raw_text, factor)
                return _PatchedTextFile(self._entered_obj, patched_text)
            except Exception:
                return self._entered_obj

        def __exit__(self, exc_type, exc, tb):
            if self._inner_cm is None:
                return False
            return self._inner_cm.__exit__(exc_type, exc, tb)

    def _wrapped_file(path_parts):
        return _PatchedFileContext(path_parts)

    file_reader.file = _wrapped_file
    setattr(file_reader, "_extra_volume_factor", factor)
    setattr(file_reader, "_extra_volume_patched", True)
    return file_reader



def _activity_logs_to_df(results: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    run_index = 0
    global_step_offset = 0

    for result in results:
        timestamps = sorted({row.timestamp for row in result.activity_logs})
        timestamp_to_step = {timestamp: global_step_offset + idx for idx, timestamp in enumerate(timestamps)}

        for row in result.activity_logs:
            data = dict(zip(ACTIVITY_COLUMNS, row.columns))
            data["round"] = result.round_num
            data["run_day"] = result.day_num
            data["run_label"] = f"R{result.round_num} D{result.day_num}"
            data["run_index"] = run_index
            data["global_step"] = timestamp_to_step[row.timestamp]

            numeric_cols = [
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
                "profit_loss",
            ]
            for col in numeric_cols:
                if col in data:
                    data[col] = pd.to_numeric(pd.Series([data[col]]), errors="coerce").iloc[0]

            bid = data.get("bid_price_1")
            ask = data.get("ask_price_1")

            if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
                data["spread"] = float(ask) - float(bid)

                mid = data.get("mid_price")
                if pd.isna(mid) or mid <= 0:
                    data["mid_price"] = (float(bid) + float(ask)) / 2.0
            else:
                data["bid_price_1"] = np.nan
                data["ask_price_1"] = np.nan
                data["mid_price"] = np.nan
                data["spread"] = np.nan
                data["profit_loss"] = np.nan

            rows.append(data)

        global_step_offset += len(timestamps)
        run_index += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in [
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
        "profit_loss",
        "spread",
        "round",
        "run_day",
        "run_index",
        "global_step",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["run_index", "product", "timestamp"]).reset_index(drop=True)

    fill_cols = [c for c in ["bid_price_1", "ask_price_1", "mid_price", "spread", "profit_loss"] if c in df.columns]
    if fill_cols:
        df[fill_cols] = (
            df.groupby(["run_index", "product"], sort=False)[fill_cols]
            .ffill()
        )

    if "mid_price" in df.columns:
        df.loc[df["mid_price"] <= 0, "mid_price"] = np.nan

    df = df.sort_values(["run_index", "timestamp", "product"]).reset_index(drop=True)
    return df


def _submission_trades_to_df(results: list[Any], activity_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    step_lookup: dict[tuple[int, int, int], int] = {}
    snapshot_lookup: dict[tuple[int, int, int, str], dict[str, Any]] = {}

    if not activity_df.empty:
        step_lookup = {
            (int(row["round"]), int(row["run_day"]), int(row["timestamp"])): int(row["global_step"])
            for _, row in activity_df[["round", "run_day", "timestamp", "global_step"]].drop_duplicates().iterrows()
        }

        snapshot_cols = [
            "round",
            "run_day",
            "timestamp",
            "product",
            "bid_price_1",
            "ask_price_1",
            "mid_price",
            "spread",
            "profit_loss",
        ]
        for _, row in activity_df[snapshot_cols].drop_duplicates(subset=["round", "run_day", "timestamp", "product"]).iterrows():
            snapshot_lookup[
                (int(row["round"]), int(row["run_day"]), int(row["timestamp"]), str(row["product"]))
            ] = {
                "bid_price_1": row.get("bid_price_1"),
                "ask_price_1": row.get("ask_price_1"),
                "mid_price": row.get("mid_price"),
                "spread": row.get("spread"),
                "profit_loss": row.get("profit_loss"),
            }

    fill_index = 0
    for result in results:
        for trade_row in result.trades:
            trade = trade_row.trade
            if trade.buyer == "SUBMISSION":
                side = "Buy"
                signed_qty = int(trade.quantity)
            elif trade.seller == "SUBMISSION":
                side = "Sell"
                signed_qty = -int(trade.quantity)
            else:
                continue

            snapshot = snapshot_lookup.get(
                (result.round_num, result.day_num, int(trade.timestamp), str(trade.symbol)),
                {},
            )

            rows.append(
                {
                    "fill_id": fill_index,
                    "round": result.round_num,
                    "day": result.day_num,
                    "run_label": f"R{result.round_num} D{result.day_num}",
                    "timestamp": int(trade.timestamp),
                    "product": trade.symbol,
                    "side": side,
                    "price": int(trade.price),
                    "quantity": int(trade.quantity),
                    "signed_quantity": signed_qty,
                    "cash_flow": -int(trade.price) * signed_qty,
                    "buyer": trade.buyer,
                    "seller": trade.seller,
                    "global_step": step_lookup.get((result.round_num, result.day_num, int(trade.timestamp))),
                    "bid_price_1": snapshot.get("bid_price_1"),
                    "ask_price_1": snapshot.get("ask_price_1"),
                    "mid_price": snapshot.get("mid_price"),
                    "spread": snapshot.get("spread"),
                    "mtm_profit_loss": snapshot.get("profit_loss"),
                }
            )
            fill_index += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in [
        "fill_id",
        "round",
        "day",
        "timestamp",
        "price",
        "quantity",
        "signed_quantity",
        "cash_flow",
        "global_step",
        "bid_price_1",
        "ask_price_1",
        "mid_price",
        "spread",
        "mtm_profit_loss",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["round", "day", "timestamp", "product", "fill_id"]).reset_index(drop=True)
    df["position_after_trade"] = df.groupby(["round", "day", "product"])["signed_quantity"].cumsum()
    df["gross_notional"] = df["price"] * df["quantity"]
    return df


def _sandbox_logs_to_df(results: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        for row in result.sandbox_logs:
            rows.append(
                {
                    "round": result.round_num,
                    "day": result.day_num,
                    "run_label": f"R{result.round_num} D{result.day_num}",
                    "timestamp": row.timestamp,
                    "sandbox_log": row.sandbox_log,
                    "lambda_log": row.lambda_log,
                }
            )
    return pd.DataFrame(rows)


def _realized_trades_to_df(submission_trades_df: pd.DataFrame) -> pd.DataFrame:
    if submission_trades_df.empty:
        return pd.DataFrame()

    df = submission_trades_df.copy()
    df = df.sort_values(["round", "day", "product", "timestamp", "fill_id"]).reset_index(drop=True)

    realized_rows: list[dict[str, Any]] = []
    realized_index = 0

    for (round_num, day_num, product), group in df.groupby(["round", "day", "product"], sort=False):
        open_longs: list[dict[str, Any]] = []
        open_shorts: list[dict[str, Any]] = []

        for _, row in group.iterrows():
            fill = row.to_dict()
            signed_qty = int(fill["signed_quantity"])
            remaining = abs(signed_qty)

            if signed_qty > 0:
                while remaining > 0 and open_shorts:
                    entry_lot = open_shorts[0]
                    matched_qty = min(remaining, int(entry_lot["remaining_qty"]))
                    realized_rows.append(
                        _build_realized_trade_row(
                            realized_index=realized_index,
                            product=product,
                            round_num=round_num,
                            day_num=day_num,
                            entry_fill=entry_lot["fill"],
                            exit_fill=fill,
                            quantity=matched_qty,
                            direction="Short",
                        )
                    )
                    realized_index += 1
                    remaining -= matched_qty
                    entry_lot["remaining_qty"] -= matched_qty
                    if entry_lot["remaining_qty"] <= 0:
                        open_shorts.pop(0)

                if remaining > 0:
                    long_fill = fill.copy()
                    long_fill["quantity"] = remaining
                    long_fill["signed_quantity"] = remaining
                    open_longs.append({"remaining_qty": remaining, "fill": long_fill})

            elif signed_qty < 0:
                while remaining > 0 and open_longs:
                    entry_lot = open_longs[0]
                    matched_qty = min(remaining, int(entry_lot["remaining_qty"]))
                    realized_rows.append(
                        _build_realized_trade_row(
                            realized_index=realized_index,
                            product=product,
                            round_num=round_num,
                            day_num=day_num,
                            entry_fill=entry_lot["fill"],
                            exit_fill=fill,
                            quantity=matched_qty,
                            direction="Long",
                        )
                    )
                    realized_index += 1
                    remaining -= matched_qty
                    entry_lot["remaining_qty"] -= matched_qty
                    if entry_lot["remaining_qty"] <= 0:
                        open_longs.pop(0)

                if remaining > 0:
                    short_fill = fill.copy()
                    short_fill["quantity"] = remaining
                    short_fill["signed_quantity"] = -remaining
                    open_shorts.append({"remaining_qty": remaining, "fill": short_fill})

    realized_df = pd.DataFrame(realized_rows)
    if realized_df.empty:
        return realized_df

    realized_df = realized_df.sort_values(["round", "day", "entry_timestamp", "exit_timestamp", "trade_id"]).reset_index(drop=True)
    return realized_df


def _build_realized_trade_row(
    realized_index: int,
    product: str,
    round_num: int,
    day_num: int,
    entry_fill: dict[str, Any],
    exit_fill: dict[str, Any],
    quantity: int,
    direction: str,
) -> dict[str, Any]:
    entry_price = float(entry_fill["price"])
    exit_price = float(exit_fill["price"])
    qty = int(quantity)

    if direction == "Long":
        pnl = (exit_price - entry_price) * qty
    else:
        pnl = (entry_price - exit_price) * qty

    entry_notional = abs(entry_price * qty)
    return_pct = (100.0 * pnl / entry_notional) if entry_notional > 0 else np.nan

    if pnl > 0:
        outcome = "Win"
    elif pnl < 0:
        outcome = "Loss"
    else:
        outcome = "Flat"

    return {
        "trade_id": f"RT-{realized_index}",
        "round": round_num,
        "day": day_num,
        "run_label": entry_fill.get("run_label") or exit_fill.get("run_label") or f"R{round_num} D{day_num}",
        "product": product,
        "direction": direction,
        "entry_side": entry_fill.get("side"),
        "exit_side": exit_fill.get("side"),
        "quantity": qty,
        "entry_timestamp": entry_fill.get("timestamp"),
        "exit_timestamp": exit_fill.get("timestamp"),
        "hold_ticks": (
            int(exit_fill["timestamp"]) - int(entry_fill["timestamp"])
            if pd.notna(entry_fill.get("timestamp")) and pd.notna(exit_fill.get("timestamp"))
            else np.nan
        ),
        "entry_global_step": entry_fill.get("global_step"),
        "exit_global_step": exit_fill.get("global_step"),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "return_pct": return_pct,
        "outcome": outcome,
        "entry_bid_price_1": entry_fill.get("bid_price_1"),
        "entry_ask_price_1": entry_fill.get("ask_price_1"),
        "entry_mid_price": entry_fill.get("mid_price"),
        "entry_spread": entry_fill.get("spread"),
        "exit_bid_price_1": exit_fill.get("bid_price_1"),
        "exit_ask_price_1": exit_fill.get("ask_price_1"),
        "exit_mid_price": exit_fill.get("mid_price"),
        "exit_spread": exit_fill.get("spread"),
        "entry_fill_id": entry_fill.get("fill_id"),
        "exit_fill_id": exit_fill.get("fill_id"),
    }


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return None
    return float(cleaned.mean())


def _build_summary(
    activity_df: pd.DataFrame,
    submission_trades_df: pd.DataFrame,
    realized_trades_df: pd.DataFrame,
    targets: list[BacktestTarget],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if activity_df.empty:
        empty_df = pd.DataFrame()
        return {
            "num_runs": len(targets),
            "targets": [t.label for t in targets],
            "total_pnl": 0.0,
            "submission_trade_count": 0,
            "products_traded": [],
            "realized_trade_count": 0,
            "winning_trade_count": 0,
            "losing_trade_count": 0,
            "win_rate": None,
            "average_win_per_trade": None,
            "average_win_pct": None,
        }, empty_df, empty_df

    final_rows = (
        activity_df.sort_values(["round", "run_day", "product", "timestamp"])
        .groupby(["round", "run_day", "product"], as_index=False)
        .tail(1)
    )

    per_run = (
        final_rows.groupby(["round", "run_day", "run_label"], as_index=False)["profit_loss"]
        .sum()
        .rename(columns={"profit_loss": "final_pnl"})
    )

    if not submission_trades_df.empty:
        trade_counts = submission_trades_df.groupby(["round", "day"], as_index=False).size().rename(columns={"size": "submission_trades"})
        per_run = per_run.merge(
            trade_counts,
            how="left",
            left_on=["round", "run_day"],
            right_on=["round", "day"],
        ).drop(columns=["day"], errors="ignore")
    else:
        per_run["submission_trades"] = 0

    if not realized_trades_df.empty:
        realized_counts = (
            realized_trades_df.groupby(["round", "day"], as_index=False)
            .agg(
                realized_trade_count=("trade_id", "size"),
                winning_trade_count=("pnl", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
                losing_trade_count=("pnl", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
                average_win_per_trade=("pnl", lambda s: _safe_mean(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") > 0])),
                average_win_pct=("return_pct", lambda s: _safe_mean(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") > 0])),
            )
        )
        per_run = per_run.merge(
            realized_counts,
            how="left",
            left_on=["round", "run_day"],
            right_on=["round", "day"],
        ).drop(columns=["day"], errors="ignore")
    else:
        per_run["realized_trade_count"] = 0
        per_run["winning_trade_count"] = 0
        per_run["losing_trade_count"] = 0
        per_run["average_win_per_trade"] = np.nan
        per_run["average_win_pct"] = np.nan

    per_run["submission_trades"] = per_run["submission_trades"].fillna(0).astype(int)
    per_run["realized_trade_count"] = per_run["realized_trade_count"].fillna(0).astype(int)
    per_run["winning_trade_count"] = per_run["winning_trade_count"].fillna(0).astype(int)
    per_run["losing_trade_count"] = per_run["losing_trade_count"].fillna(0).astype(int)

    per_product = (
        final_rows.groupby(["product"], as_index=False)["profit_loss"]
        .sum()
        .rename(columns={"profit_loss": "total_final_pnl"})
    )

    if not submission_trades_df.empty:
        qty = submission_trades_df.groupby("product", as_index=False)["quantity"].sum().rename(columns={"quantity": "filled_quantity"})
        trades = submission_trades_df.groupby("product", as_index=False).size().rename(columns={"size": "submission_trades"})
        per_product = per_product.merge(qty, how="left", on="product").merge(trades, how="left", on="product")
    else:
        per_product["filled_quantity"] = 0
        per_product["submission_trades"] = 0

    if not realized_trades_df.empty:
        realized_product = (
            realized_trades_df.groupby("product", as_index=False)
            .agg(
                realized_trade_count=("trade_id", "size"),
                winning_trade_count=("pnl", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
                losing_trade_count=("pnl", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
                average_win_per_trade=("pnl", lambda s: _safe_mean(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") > 0])),
                average_win_pct=("return_pct", lambda s: _safe_mean(pd.to_numeric(s, errors="coerce")[pd.to_numeric(s, errors="coerce") > 0])),
            )
        )
        per_product = per_product.merge(realized_product, how="left", on="product")
    else:
        per_product["realized_trade_count"] = 0
        per_product["winning_trade_count"] = 0
        per_product["losing_trade_count"] = 0
        per_product["average_win_per_trade"] = np.nan
        per_product["average_win_pct"] = np.nan

    int_cols = ["filled_quantity", "submission_trades", "realized_trade_count", "winning_trade_count", "losing_trade_count"]
    per_product[int_cols] = per_product[int_cols].fillna(0).astype(int)
    per_product = per_product.sort_values("total_final_pnl", ascending=False).reset_index(drop=True)

    winning_trades = realized_trades_df[realized_trades_df["pnl"] > 0].copy() if not realized_trades_df.empty else pd.DataFrame()
    losing_trades = realized_trades_df[realized_trades_df["pnl"] < 0].copy() if not realized_trades_df.empty else pd.DataFrame()

    realized_count = int(len(realized_trades_df))
    winning_count = int(len(winning_trades))
    losing_count = int(len(losing_trades))
    non_flat_count = winning_count + losing_count
    win_rate = (100.0 * winning_count / non_flat_count) if non_flat_count > 0 else None

    summary = {
        "num_runs": len(targets),
        "targets": [t.label for t in targets],
        "total_pnl": float(per_run["final_pnl"].sum()),
        "submission_trade_count": int(len(submission_trades_df)),
        "products_traded": sorted(per_product["product"].astype(str).tolist()),
        "realized_trade_count": realized_count,
        "winning_trade_count": winning_count,
        "losing_trade_count": losing_count,
        "win_rate": win_rate,
        "average_win_per_trade": _safe_mean(winning_trades["pnl"]) if not winning_trades.empty else None,
        "average_win_pct": _safe_mean(winning_trades["return_pct"]) if not winning_trades.empty else None,
        "extra_volume_pct": 0.0,
    }
    return summary, per_run, per_product


def run_backtests(
    strategy_path: Path,
    request: BacktestRequest,
    custom_data_root: Path | None = None,
) -> BacktestPayload:
    api = _load_backtester_api()

    if custom_data_root is not None:
        data_root = custom_data_root
    else:
        project_root = Path(__file__).resolve().parents[2]
        data_root = project_root / "data"

    file_reader = api["FileSystemReader"](data_root)

    extra_volume_pct = float(request.extra_volume_pct or 0.0)
    if extra_volume_pct < 0:
        raise ValueError("Extra volume percentage cannot be negative.")

    extra_volume_factor = 1.0 + extra_volume_pct
    if extra_volume_factor != 1.0:
        file_reader = _patch_file_reader_for_extra_volume(file_reader, extra_volume_factor)

    previous_env_value = os.environ.get("PROSPERITY_EXTRA_VOLUME_PCT")
    os.environ["PROSPERITY_EXTRA_VOLUME_PCT"] = str(extra_volume_pct)

    try:
        targets = resolve_targets(request, file_reader, api["has_day_data"])
        results = []

        for target in targets:
            trader = _instantiate_trader(strategy_path)
            print(
                "RUN_BACKTEST",
                {
                    "target": target.label,
                    "extra_volume_pct": extra_volume_pct,
                    "extra_volume_factor": extra_volume_factor,
                    "reader_patched": bool(getattr(file_reader, "_extra_volume_patched", False)),
                },
            )
            result = api["run_backtest"](
                trader=trader,
                file_reader=file_reader,
                round_num=target.round_num,
                day_num=target.day_num,
                print_output=False,
                trade_matching_mode=api["TradeMatchingMode"](request.match_trades),
                no_names=False,
                show_progress_bar=False,
                limits_override=request.limit_overrides or None,
            )
            results.append(result)
    finally:
        if previous_env_value is None:
            os.environ.pop("PROSPERITY_EXTRA_VOLUME_PCT", None)
        else:
            os.environ["PROSPERITY_EXTRA_VOLUME_PCT"] = previous_env_value

    activity_df = _activity_logs_to_df(results)
    submission_trades_df = _submission_trades_to_df(results, activity_df)
    realized_trades_df = _realized_trades_to_df(submission_trades_df)
    sandbox_df = _sandbox_logs_to_df(results)
    summary, per_run_df, per_product_df = _build_summary(
        activity_df,
        submission_trades_df,
        realized_trades_df,
        targets,
    )
    summary["extra_volume_pct"] = extra_volume_pct

    return BacktestPayload(
        targets=targets,
        activity_rows=activity_df.to_dict("records") if not activity_df.empty else [],
        submission_trade_rows=submission_trades_df.to_dict("records") if not submission_trades_df.empty else [],
        realized_trade_rows=realized_trades_df.to_dict("records") if not realized_trades_df.empty else [],
        sandbox_rows=sandbox_df.to_dict("records") if not sandbox_df.empty else [],
        per_run_rows=per_run_df.to_dict("records") if not per_run_df.empty else [],
        per_product_rows=per_product_df.to_dict("records") if not per_product_df.empty else [],
        summary=summary,
    )
