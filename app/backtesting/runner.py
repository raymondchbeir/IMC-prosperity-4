from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

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
            expanded = [
                BacktestTarget(round_num=round_num, day_num=day)
                for day in BACKTEST_DAY_SCAN_RANGE
                if has_day_data_func(file_reader, round_num, day)
            ]
            if not expanded:
                raise ValueError(f"No data is available for round {round_num} in the selected data source.")
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
            rows.append(data)

        global_step_offset += len(timestamps)
        run_index += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["run_index", "timestamp", "product"]).reset_index(drop=True)
    return df


def _submission_trades_to_df(results: list[Any], activity_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    step_lookup: dict[tuple[int, int, int], int] = {}
    if not activity_df.empty:
        step_lookup = {
            (int(row["round"]), int(row["run_day"]), int(row["timestamp"])): int(row["global_step"])
            for _, row in activity_df[["round", "run_day", "timestamp", "global_step"]].drop_duplicates().iterrows()
        }

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

            rows.append(
                {
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
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["round", "day", "timestamp", "product", "side"]).reset_index(drop=True)
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


def _build_summary(activity_df: pd.DataFrame, submission_trades_df: pd.DataFrame, targets: list[BacktestTarget]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if activity_df.empty:
        empty_df = pd.DataFrame()
        return {
            "num_runs": len(targets),
            "targets": [t.label for t in targets],
            "total_pnl": 0.0,
            "submission_trade_count": 0,
            "products_traded": [],
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

    per_run["submission_trades"] = per_run["submission_trades"].fillna(0).astype(int)

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

    per_product[["filled_quantity", "submission_trades"]] = (
        per_product[["filled_quantity", "submission_trades"]].fillna(0).astype(int)
    )
    per_product = per_product.sort_values("total_final_pnl", ascending=False).reset_index(drop=True)

    summary = {
        "num_runs": len(targets),
        "targets": [t.label for t in targets],
        "total_pnl": float(per_run["final_pnl"].sum()),
        "submission_trade_count": int(len(submission_trades_df)),
        "products_traded": sorted(per_product["product"].astype(str).tolist()),
    }
    return summary, per_run, per_product


def run_backtests(
    strategy_path: Path,
    request: BacktestRequest,
    custom_data_root: Path | None = None,
) -> BacktestPayload:
    api = _load_backtester_api()

    if custom_data_root is not None:
        file_reader = api["FileSystemReader"](custom_data_root)
    else:
        file_reader = api["PackageResourcesReader"]()

    targets = resolve_targets(request, file_reader, api["has_day_data"])
    results = []

    for target in targets:
        trader = _instantiate_trader(strategy_path)
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

    activity_df = _activity_logs_to_df(results)
    submission_trades_df = _submission_trades_to_df(results, activity_df)
    sandbox_df = _sandbox_logs_to_df(results)
    summary, per_run_df, per_product_df = _build_summary(activity_df, submission_trades_df, targets)

    return BacktestPayload(
        targets=targets,
        activity_rows=activity_df.to_dict("records") if not activity_df.empty else [],
        submission_trade_rows=submission_trades_df.to_dict("records") if not submission_trades_df.empty else [],
        sandbox_rows=sandbox_df.to_dict("records") if not sandbox_df.empty else [],
        per_run_rows=per_run_df.to_dict("records") if not per_run_df.empty else [],
        per_product_rows=per_product_df.to_dict("records") if not per_product_df.empty else [],
        summary=summary,
    )