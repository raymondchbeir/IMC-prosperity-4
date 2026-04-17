#!/usr/bin/env python3
"""
run_backtest_csv.py

Prosperity-style local backtest runner for the IMC CSV files you uploaded.

Supported inputs
----------------
- prices_round_0_day_-2.csv
- prices_round_0_day_-1.csv
- trades_round_0_day_-2.csv
- trades_round_0_day_-1.csv

These files are semicolon-separated.

What this runner does
---------------------
- Loads one or more price CSV files
- Optionally loads one or more market-trade CSV files
- Reconstructs top-of-book order depths per timestamp/product
- Builds a Prosperity-style TradingState
- Calls Trader.run(state) each tick
- Simulates fills against visible best bid / best ask only
- Tracks cash, positions, PnL, trade counts, and simple Sharpe

Notes
-----
- This is a practical tuning runner, not a perfect exchange simulator.
- Passive orders only fill if they cross the current best price.
- Market trades are passed through as market_trades for strategies that inspect them.

Example:
python run_backtest_csv.py \
  --strategy 57265_merged_best_emeralds_tomatoes.py \
  --prices prices_round_0_day_-2.csv prices_round_0_day_-1.csv \
  --trades trades_round_0_day_-2.csv trades_round_0_day_-1.csv \
  --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


# -------------------------------------------------------------------
# Minimal Prosperity-style datamodel fallbacks
# -------------------------------------------------------------------

@dataclass
class Order:
    symbol: str
    price: int
    quantity: int


@dataclass
class OrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)
    sell_orders: Dict[int, int] = field(default_factory=dict)


@dataclass
class Trade:
    symbol: str
    price: int
    quantity: int
    buyer: str
    seller: str
    timestamp: int


@dataclass
class Listing:
    symbol: str
    product: str
    denomination: str


@dataclass
class Observation:
    plainValueObservations: Dict[str, Any] = field(default_factory=dict)
    conversionObservations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingState:
    traderData: str
    timestamp: int
    listings: Dict[str, Listing]
    order_depths: Dict[str, OrderDepth]
    own_trades: Dict[str, List[Trade]]
    market_trades: Dict[str, List[Trade]]
    position: Dict[str, int]
    observations: Any


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def best_prices(depth: OrderDepth) -> Tuple[int | None, int | None]:
    best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
    best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
    return best_bid, best_ask


def mid_price(depth: OrderDepth) -> float | None:
    best_bid, best_ask = best_prices(depth)
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2.0


def load_strategy(strategy_path: Path):
    spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy from {strategy_path}")
    module = importlib.util.module_from_spec(spec)

    module.Order = Order
    module.OrderDepth = OrderDepth
    module.Trade = Trade
    module.Listing = Listing
    module.Observation = Observation
    module.TradingState = TradingState

    import types
    dm = types.ModuleType("datamodel")
    dm.Order = Order
    dm.OrderDepth = OrderDepth
    dm.Trade = Trade
    dm.Listing = Listing
    dm.Observation = Observation
    dm.TradingState = TradingState
    sys.modules["datamodel"] = dm

    spec.loader.exec_module(module)
    if not hasattr(module, "Trader"):
        raise AttributeError("Strategy file must define class Trader")
    return module.Trader()


def _normalize_price_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    df.columns = [c.strip() for c in df.columns]

    numeric_cols = [
        "day", "timestamp",
        "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
        "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
        "mid_price", "profit_and_loss"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["product"] = df["product"].astype(str)
    return df


def _normalize_trade_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    df.columns = [c.strip() for c in df.columns]

    for col in ["timestamp", "price", "quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)
    for col in ["buyer", "seller", "currency"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df


def load_price_rows(price_paths: List[Path]) -> pd.DataFrame:
    dfs = [_normalize_price_df(p) for p in price_paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)
    return df


def load_trade_rows(trade_paths: List[Path]) -> pd.DataFrame:
    if not trade_paths:
        return pd.DataFrame(columns=["day", "timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"])

    dfs = []
    for p in trade_paths:
        df = _normalize_trade_df(p)

        # Infer day from filename like trades_round_0_day_-2.csv
        name = p.stem
        inferred_day = None
        if "_day_" in name:
            try:
                inferred_day = int(name.split("_day_")[-1])
            except Exception:
                inferred_day = None

        if "day" not in df.columns:
            df["day"] = inferred_day

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["day", "timestamp", "symbol"]).reset_index(drop=True)
    return df


def build_order_depth(row: pd.Series) -> OrderDepth:
    buy_orders: Dict[int, int] = {}
    sell_orders: Dict[int, int] = {}

    for i in [1, 2, 3]:
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")

        if pd.notna(bp) and pd.notna(bv):
            buy_orders[int(bp)] = int(bv)

        if pd.notna(ap) and pd.notna(av):
            # Prosperity convention: sell book quantities are negative
            sell_orders[int(ap)] = -int(av)

    return OrderDepth(buy_orders=buy_orders, sell_orders=sell_orders)


def build_market_trade_map(trades_df: pd.DataFrame, day: int, timestamp: int) -> Dict[str, List[Trade]]:
    if trades_df.empty:
        return {}

    subset = trades_df[(trades_df["day"] == day) & (trades_df["timestamp"] == timestamp)]
    out: Dict[str, List[Trade]] = defaultdict(list)

    for _, row in subset.iterrows():
        symbol = str(row["symbol"])
        out[symbol].append(
            Trade(
                symbol=symbol,
                price=int(row["price"]),
                quantity=int(row["quantity"]),
                buyer=str(row.get("buyer", "")),
                seller=str(row.get("seller", "")),
                timestamp=int(row["timestamp"]),
            )
        )
    return dict(out)


def simulate_fills_for_product(
    product: str,
    orders: List[Order],
    depth: OrderDepth,
    timestamp: int,
    global_timestamp: int,
) -> Tuple[List[Trade], int]:
    fills: List[Trade] = []
    cash_delta = 0

    best_bid, best_ask = best_prices(depth)
    remaining_best_bid = depth.buy_orders.get(best_bid, 0) if best_bid is not None else 0
    remaining_best_ask = -depth.sell_orders.get(best_ask, 0) if best_ask is not None else 0

    for order in orders:
        if order.quantity == 0:
            continue

        if order.quantity > 0:
            if best_ask is None:
                continue
            if order.price >= best_ask and remaining_best_ask > 0:
                fill_qty = min(order.quantity, remaining_best_ask)
                remaining_best_ask -= fill_qty
                cash_delta -= fill_qty * best_ask
                fills.append(
                    Trade(
                        symbol=product,
                        price=best_ask,
                        quantity=fill_qty,
                        buyer="SUBMISSION",
                        seller="MARKET",
                        timestamp=global_timestamp,
                    )
                )
        else:
            sell_qty = -order.quantity
            if best_bid is None:
                continue
            if order.price <= best_bid and remaining_best_bid > 0:
                fill_qty = min(sell_qty, remaining_best_bid)
                remaining_best_bid -= fill_qty
                cash_delta += fill_qty * best_bid
                fills.append(
                    Trade(
                        symbol=product,
                        price=best_bid,
                        quantity=fill_qty,
                        buyer="MARKET",
                        seller="SUBMISSION",
                        timestamp=global_timestamp,
                    )
                )

    return fills, cash_delta


def run_backtest(strategy_path: Path, price_paths: List[Path], trade_paths: List[Path]) -> Dict[str, Any]:
    trader = load_strategy(strategy_path)
    prices_df = load_price_rows(price_paths)
    trades_df = load_trade_rows(trade_paths)

    grouped = prices_df.groupby(["day", "timestamp"], sort=True)

    trader_data = ""
    positions: Dict[str, int] = defaultdict(int)
    cash: float = 0.0
    own_trades_prev: Dict[str, List[Trade]] = defaultdict(list)

    pnl_series: List[float] = []
    trade_count_by_product: Dict[str, int] = defaultdict(int)
    filled_volume_by_product: Dict[str, int] = defaultdict(int)
    last_depths: Dict[str, OrderDepth] = {}

    listings = {
        "EMERALDS": Listing("EMERALDS", "EMERALDS", "SEASHELLS"),
        "TOMATOES": Listing("TOMATOES", "TOMATOES", "SEASHELLS"),
    }

    for (day, timestamp), group in grouped:
        order_depths: Dict[str, OrderDepth] = {}
        observations: Dict[str, Any] = {}

        for _, row in group.iterrows():
            product = str(row["product"])
            order_depths[product] = build_order_depth(row)
            observations[f"{product}_mid_price"] = float(row["mid_price"]) if pd.notna(row["mid_price"]) else None
            observations[f"{product}_day"] = int(day)

        # Use unique global timestamp across multiple days
        global_timestamp = int((int(day) + 10) * 1_000_000 + int(timestamp))

        state = TradingState(
            traderData=trader_data,
            timestamp=global_timestamp,
            listings=listings,
            order_depths=order_depths,
            own_trades=dict(own_trades_prev),
            market_trades=build_market_trade_map(trades_df, int(day), int(timestamp)),
            position=dict(positions),
            observations=observations,
        )

        result = trader.run(state)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("Trader.run must return (orders_dict, conversions, traderData)")

        orders_by_product, _conversions, trader_data = result

        own_trades_curr: Dict[str, List[Trade]] = defaultdict(list)

        for product, orders in (orders_by_product or {}).items():
            if product not in order_depths:
                continue

            fills, cash_delta = simulate_fills_for_product(
                product=product,
                orders=orders,
                depth=order_depths[product],
                timestamp=int(timestamp),
                global_timestamp=global_timestamp,
            )
            cash += cash_delta

            for fill in fills:
                qty = int(fill.quantity)
                if fill.buyer == "SUBMISSION":
                    positions[product] += qty
                    filled_volume_by_product[product] += qty
                elif fill.seller == "SUBMISSION":
                    positions[product] -= qty
                    filled_volume_by_product[product] += qty

                trade_count_by_product[product] += 1
                own_trades_curr[product].append(fill)

        own_trades_prev = own_trades_curr
        last_depths = order_depths

        mtm = cash
        for product, pos in positions.items():
            depth = order_depths.get(product, last_depths.get(product))
            if depth is None:
                continue
            m = mid_price(depth)
            if m is not None:
                mtm += pos * m

        pnl_series.append(mtm)

    sharpe = 0.0
    if len(pnl_series) >= 2:
        diffs = [pnl_series[i] - pnl_series[i - 1] for i in range(1, len(pnl_series))]
        mean_diff = sum(diffs) / len(diffs)
        var = sum((x - mean_diff) ** 2 for x in diffs) / max(1, len(diffs) - 1)
        std = math.sqrt(var)
        if std > 1e-9:
            sharpe = mean_diff / std

    return {
        "pnl": pnl_series[-1] if pnl_series else 0.0,
        "sharpe": sharpe,
        "final_positions": dict(positions),
        "trade_count_by_product": dict(trade_count_by_product),
        "filled_volume_by_product": dict(filled_volume_by_product),
        "ticks": len(pnl_series),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prosperity CSV backtest runner.")
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--prices", type=Path, nargs="+", required=True, help="One or more prices CSV files")
    parser.add_argument("--trades", type=Path, nargs="*", default=[], help="Optional trade CSV files")
    parser.add_argument("--json", action="store_true", help="Print JSON metrics")
    args = parser.parse_args()

    metrics = run_backtest(args.strategy, args.prices, args.trades)

    if args.json:
        print(json.dumps(metrics))
    else:
        print(f"PnL: {metrics['pnl']:.2f}")
        print(f"Sharpe: {metrics['sharpe']:.4f}")
        print(f"Final positions: {metrics['final_positions']}")
        print(f"Trade count by product: {metrics['trade_count_by_product']}")
        print(f"Filled volume by product: {metrics['filled_volume_by_product']}")
        print(f"Ticks: {metrics['ticks']}")


if __name__ == "__main__":
    main()

