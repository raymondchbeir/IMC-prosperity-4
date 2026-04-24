from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PortalLogPayload:
    metadata: dict[str, Any] = field(default_factory=dict)
    activity_rows: list[dict[str, Any]] = field(default_factory=list)
    submission_trade_rows: list[dict[str, Any]] = field(default_factory=list)
    realized_trade_rows: list[dict[str, Any]] = field(default_factory=list)
    bot_trade_rows: list[dict[str, Any]] = field(default_factory=list)
    sandbox_rows: list[dict[str, Any]] = field(default_factory=list)
    per_product_rows: list[dict[str, Any]] = field(default_factory=list)
    bot_distribution_rows: list[dict[str, Any]] = field(default_factory=list)
    bot_normalized_distribution_rows: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


PRICE_COLS = [
    "day", "timestamp", "product",
    "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
    "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
    "mid_price", "profit_and_loss",
]


def parse_portal_files(paths: list[str | Path]) -> PortalLogPayload:
    raw_docs: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if p.suffix.lower() not in {".log", ".json"}:
            continue
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            raw_docs.append(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name} is not valid JSON/portal log text: {exc}") from exc

    if not raw_docs:
        raise ValueError("Upload at least one IMC portal .log or .json file.")

    merged: dict[str, Any] = {}
    for doc in raw_docs:
        merged.update({k: v for k, v in doc.items() if v not in [None, "", [], {}]})

    activity_df = _parse_activities_log(merged.get("activitiesLog") or merged.get("activities_log"))
    trade_df = _parse_trade_history(merged.get("tradeHistory") or merged.get("trades") or [])
    sandbox_df = _parse_sandbox_logs(merged.get("logs") or merged.get("sandboxLogs") or [])

    submission_df = _build_submission_trades(trade_df, activity_df)
    bot_df = _build_bot_trades(trade_df, activity_df)
    realized_df = _realized_trades_to_df(submission_df)
    per_product_df = _build_per_product(activity_df, submission_df, bot_df, realized_df)
    bot_dist_df = _build_bot_distribution(bot_df)
    bot_norm_df = _build_bot_normalized_distribution(bot_df)
    summary = _build_summary(merged, activity_df, submission_df, bot_df, realized_df)

    return PortalLogPayload(
        metadata={
            "submission_id": merged.get("submissionId") or merged.get("submission_id"),
            "round": merged.get("round"),
            "status": merged.get("status"),
            "profit": merged.get("profit"),
        },
        activity_rows=_clean_records(activity_df),
        submission_trade_rows=_clean_records(submission_df),
        realized_trade_rows=_clean_records(realized_df),
        bot_trade_rows=_clean_records(bot_df),
        sandbox_rows=_clean_records(sandbox_df),
        per_product_rows=_clean_records(per_product_df),
        bot_distribution_rows=_clean_records(bot_dist_df),
        bot_normalized_distribution_rows=_clean_records(bot_norm_df),
        summary=_clean_dict(summary),
    )


def _clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    safe = df.replace({np.nan: None, pd.NaT: None})
    return safe.to_dict("records")


def _clean_dict(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, float) and np.isnan(value):
            out[key] = None
        else:
            out[key] = value
    return out


def _parse_activities_log(raw: str | None) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=PRICE_COLS)
    df = pd.read_csv(StringIO(raw), sep=";")
    df.columns = [str(c).strip() for c in df.columns]
    if "profit_and_loss" in df.columns and "profit_loss" not in df.columns:
        df = df.rename(columns={"profit_and_loss": "profit_loss"})
    for col in df.columns:
        if col != "product":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "round" not in df.columns:
        df["round"] = np.nan
    if "day" not in df.columns:
        df["day"] = 0
    if "run_day" not in df.columns:
        df["run_day"] = df["day"]
    if "run_label" not in df.columns:
        df["run_label"] = df.apply(lambda r: f"D{int(r['day'])}" if pd.notna(r.get("day")) else "Portal", axis=1)
    if "run_index" not in df.columns:
        df["run_index"] = 0
    if "global_step" not in df.columns:
        ts = sorted(df["timestamp"].dropna().unique().tolist()) if "timestamp" in df.columns else []
        step_map = {t: i for i, t in enumerate(ts)}
        df["global_step"] = df["timestamp"].map(step_map) if "timestamp" in df.columns else np.nan
    if {"bid_price_1", "ask_price_1"}.issubset(df.columns):
        valid = (df["bid_price_1"] > 0) & (df["ask_price_1"] > 0)
        df["spread"] = np.nan
        df.loc[valid, "spread"] = df.loc[valid, "ask_price_1"] - df.loc[valid, "bid_price_1"]
        if "mid_price" not in df.columns:
            df["mid_price"] = np.nan
        bad_mid = df["mid_price"].isna() | (df["mid_price"] <= 0)
        df.loc[valid & bad_mid, "mid_price"] = (df.loc[valid & bad_mid, "bid_price_1"] + df.loc[valid & bad_mid, "ask_price_1"]) / 2.0
    return df.sort_values([c for c in ["day", "timestamp", "product"] if c in df.columns]).reset_index(drop=True)


def _parse_trade_history(raw: Any) -> pd.DataFrame:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    if not raw:
        return pd.DataFrame()
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rows.append({
            "timestamp": item.get("timestamp"),
            "buyer": item.get("buyer", ""),
            "seller": item.get("seller", ""),
            "product": item.get("symbol") or item.get("product"),
            "currency": item.get("currency"),
            "price": item.get("price"),
            "quantity": item.get("quantity"),
        })
    df = pd.DataFrame(rows)
    for col in ["timestamp", "price", "quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp", "product", "price", "quantity"]).reset_index(drop=True)


def _parse_sandbox_logs(raw: Any) -> pd.DataFrame:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    rows = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        rows.append({
            "timestamp": item.get("timestamp"),
            "sandbox_log": item.get("sandboxLog") or item.get("sandbox_log") or "",
            "lambda_log": item.get("lambdaLog") or item.get("lambda_log") or "",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["round"] = np.nan
        df["day"] = 0
        df["run_label"] = "Portal"
    return df


def _snapshot_lookup(activity_df: pd.DataFrame) -> pd.DataFrame:
    if activity_df.empty:
        return pd.DataFrame()
    cols = [
        c for c in [
            "timestamp", "product", "round", "day", "run_day", "run_label", "run_index", "global_step",
            "bid_price_1", "ask_price_1", "mid_price", "spread", "profit_loss",
        ]
        if c in activity_df.columns
    ]
    return activity_df[cols].drop_duplicates(subset=["timestamp", "product"]).copy()


def _enrich_trades_with_book(trades_df: pd.DataFrame, activity_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()
    out = trades_df.copy()
    snap = _snapshot_lookup(activity_df)
    if not snap.empty:
        out = out.merge(snap, on=["timestamp", "product"], how="left")

    ts_step_map = {}
    if not activity_df.empty and {"timestamp", "global_step"}.issubset(activity_df.columns):
        ts_step_map = dict(activity_df[["timestamp", "global_step"]].drop_duplicates().itertuples(index=False, name=None))

    default_day = _first_value(activity_df, "day", 0)
    default_run_day = _first_value(activity_df, "run_day", default_day)
    default_round = _first_value(activity_df, "round", np.nan)
    default_run_label = _first_value(activity_df, "run_label", "Portal")

    for col in ["bid_price_1", "ask_price_1", "mid_price", "spread"]:
        if col not in out.columns:
            out[col] = np.nan
    if "day" not in out.columns:
        out["day"] = default_day
    out["day"] = out["day"].fillna(default_day)
    if "run_day" not in out.columns:
        out["run_day"] = default_run_day
    out["run_day"] = out["run_day"].fillna(default_run_day)
    if "round" not in out.columns:
        out["round"] = default_round
    if "run_label" not in out.columns:
        out["run_label"] = default_run_label
    out["run_label"] = out["run_label"].fillna(default_run_label)
    if "run_index" not in out.columns:
        out["run_index"] = 0
    if "global_step" not in out.columns:
        out["global_step"] = out["timestamp"].map(ts_step_map) if ts_step_map else np.nan
    if out["global_step"].isna().any():
        fallback_map = {t: i for i, t in enumerate(sorted(out["timestamp"].dropna().unique().tolist()))}
        out["global_step"] = out["global_step"].fillna(out["timestamp"].map(fallback_map))

    out["price_minus_mid"] = out["price"] - out["mid_price"]
    out["price_minus_mid_bps"] = np.where(out["mid_price"] > 0, 10000 * out["price_minus_mid"] / out["mid_price"], np.nan)
    out["spread_units_from_mid"] = np.where(out["spread"].abs() > 0, out["price_minus_mid"] / out["spread"], np.nan)
    out["distance_from_best_bid"] = out["price"] - out["bid_price_1"]
    out["distance_from_best_ask"] = out["price"] - out["ask_price_1"]
    out["inferred_side"] = "Inside/Unknown"
    out.loc[pd.notna(out["ask_price_1"]) & (out["price"] >= out["ask_price_1"]), "inferred_side"] = "Lifted Ask"
    out.loc[pd.notna(out["bid_price_1"]) & (out["price"] <= out["bid_price_1"]), "inferred_side"] = "Hit Bid"
    return out


def _first_value(df: pd.DataFrame, col: str, default: Any) -> Any:
    if df.empty or col not in df.columns or df[col].dropna().empty:
        return default
    return df[col].dropna().iloc[0]


def _build_submission_trades(trade_df: pd.DataFrame, activity_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()
    mask = (trade_df["buyer"] == "SUBMISSION") | (trade_df["seller"] == "SUBMISSION")
    out = _enrich_trades_with_book(trade_df[mask].copy(), activity_df)
    if out.empty:
        return out
    out["side"] = np.where(out["buyer"] == "SUBMISSION", "Buy", "Sell")
    out["signed_quantity"] = np.where(out["side"] == "Buy", out["quantity"], -out["quantity"]).astype(int)
    out["cash_flow"] = -out["price"] * out["signed_quantity"]
    out["fill_id"] = range(len(out))
    out["position_after_trade"] = out.groupby(["product"], sort=False)["signed_quantity"].cumsum()
    out["gross_notional"] = out["price"] * out["quantity"]
    return out.sort_values(["timestamp", "product", "fill_id"]).reset_index(drop=True)


def _build_bot_trades(trade_df: pd.DataFrame, activity_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()
    mask = (trade_df["buyer"] != "SUBMISSION") & (trade_df["seller"] != "SUBMISSION")
    out = _enrich_trades_with_book(trade_df[mask].copy(), activity_df)
    if out.empty:
        return out
    out["gross_notional"] = out["price"] * out["quantity"]
    out["bot_trade_id"] = range(len(out))
    return out.sort_values(["timestamp", "product", "bot_trade_id"]).reset_index(drop=True)


def _realized_trades_to_df(submission_df: pd.DataFrame) -> pd.DataFrame:
    if submission_df.empty or "signed_quantity" not in submission_df.columns:
        return pd.DataFrame()
    rows = []
    trade_id = 0
    for product, group in submission_df.sort_values(["product", "timestamp", "fill_id"]).groupby("product", sort=False):
        longs: list[dict[str, Any]] = []
        shorts: list[dict[str, Any]] = []
        for _, row in group.iterrows():
            fill = row.to_dict()
            signed = int(fill["signed_quantity"])
            remaining = abs(signed)
            if signed > 0:
                while remaining and shorts:
                    lot = shorts[0]
                    qty = min(remaining, lot["qty"])
                    rows.append(_realized_row(trade_id, product, lot["fill"], fill, qty, "Short"))
                    trade_id += 1
                    remaining -= qty
                    lot["qty"] -= qty
                    if lot["qty"] <= 0:
                        shorts.pop(0)
                if remaining:
                    longs.append({"qty": remaining, "fill": fill})
            elif signed < 0:
                while remaining and longs:
                    lot = longs[0]
                    qty = min(remaining, lot["qty"])
                    rows.append(_realized_row(trade_id, product, lot["fill"], fill, qty, "Long"))
                    trade_id += 1
                    remaining -= qty
                    lot["qty"] -= qty
                    if lot["qty"] <= 0:
                        longs.pop(0)
                if remaining:
                    shorts.append({"qty": remaining, "fill": fill})
    return pd.DataFrame(rows)


def _realized_row(trade_id: int, product: str, entry: dict[str, Any], exit_: dict[str, Any], qty: int, direction: str) -> dict[str, Any]:
    entry_price = float(entry["price"])
    exit_price = float(exit_["price"])
    pnl = (exit_price - entry_price) * qty if direction == "Long" else (entry_price - exit_price) * qty
    return {
        "trade_id": f"PRT-{trade_id}",
        "round": entry.get("round"),
        "day": entry.get("day"),
        "run_label": entry.get("run_label", "Portal"),
        "product": product,
        "direction": direction,
        "entry_side": entry.get("side"),
        "exit_side": exit_.get("side"),
        "quantity": int(qty),
        "entry_timestamp": entry.get("timestamp"),
        "exit_timestamp": exit_.get("timestamp"),
        "hold_ticks": (exit_.get("timestamp") or 0) - (entry.get("timestamp") or 0),
        "entry_global_step": entry.get("global_step"),
        "exit_global_step": exit_.get("global_step"),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "return_pct": 100 * pnl / abs(entry_price * qty) if entry_price and qty else np.nan,
        "outcome": "Win" if pnl > 0 else "Loss" if pnl < 0 else "Flat",
        "entry_bid_price_1": entry.get("bid_price_1"),
        "entry_ask_price_1": entry.get("ask_price_1"),
        "entry_mid_price": entry.get("mid_price"),
        "entry_spread": entry.get("spread"),
        "exit_bid_price_1": exit_.get("bid_price_1"),
        "exit_ask_price_1": exit_.get("ask_price_1"),
        "exit_mid_price": exit_.get("mid_price"),
        "exit_spread": exit_.get("spread"),
        "entry_fill_id": entry.get("fill_id"),
        "exit_fill_id": exit_.get("fill_id"),
    }


def _build_per_product(activity_df: pd.DataFrame, sub_df: pd.DataFrame, bot_df: pd.DataFrame, realized_df: pd.DataFrame) -> pd.DataFrame:
    products = sorted(set(activity_df.get("product", pd.Series(dtype=str)).dropna().astype(str).tolist()) |
                      set(sub_df.get("product", pd.Series(dtype=str)).dropna().astype(str).tolist()) |
                      set(bot_df.get("product", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    rows = []
    for product in products:
        final_pnl = np.nan
        if not activity_df.empty and "profit_loss" in activity_df.columns:
            part = activity_df[activity_df["product"] == product].sort_values("timestamp")
            if not part.empty:
                final_pnl = part["profit_loss"].dropna().iloc[-1] if part["profit_loss"].notna().any() else np.nan
        s = sub_df[sub_df["product"] == product] if not sub_df.empty else pd.DataFrame()
        b = bot_df[bot_df["product"] == product] if not bot_df.empty else pd.DataFrame()
        r = realized_df[realized_df["product"] == product] if not realized_df.empty else pd.DataFrame()
        wins = r[pd.to_numeric(r.get("pnl", pd.Series(dtype=float)), errors="coerce") > 0] if not r.empty else pd.DataFrame()
        rows.append({
            "product": product,
            "total_final_pnl": final_pnl,
            "submission_trades": int(len(s)),
            "filled_quantity": int(s["quantity"].sum()) if not s.empty else 0,
            "bot_trades": int(len(b)),
            "bot_quantity": int(b["quantity"].sum()) if not b.empty else 0,
            "realized_trade_count": int(len(r)),
            "winning_trade_count": int((r["pnl"] > 0).sum()) if not r.empty else 0,
            "losing_trade_count": int((r["pnl"] < 0).sum()) if not r.empty else 0,
            "average_win_per_trade": float(wins["pnl"].mean()) if not wins.empty else np.nan,
            "average_win_pct": float(wins["return_pct"].mean()) if not wins.empty and "return_pct" in wins.columns else np.nan,
        })
    return pd.DataFrame(rows)


def _build_bot_distribution(bot_df: pd.DataFrame) -> pd.DataFrame:
    if bot_df.empty:
        return pd.DataFrame()
    return (bot_df.groupby(["product", "price", "inferred_side"], as_index=False)
            .agg(trade_count=("quantity", "size"), total_quantity=("quantity", "sum"), max_quantity=("quantity", "max"), avg_quantity=("quantity", "mean"), first_timestamp=("timestamp", "min"), last_timestamp=("timestamp", "max"))
            .sort_values(["product", "total_quantity"], ascending=[True, False]))


def _build_bot_normalized_distribution(bot_df: pd.DataFrame) -> pd.DataFrame:
    if bot_df.empty or "price_minus_mid_bps" not in bot_df.columns:
        return pd.DataFrame()
    out = bot_df.dropna(subset=["price_minus_mid_bps"]).copy()
    if out.empty:
        return pd.DataFrame()
    out["bps_bucket"] = pd.cut(out["price_minus_mid_bps"], bins=[-1000, -100, -50, -25, -10, -5, 0, 5, 10, 25, 50, 100, 1000], include_lowest=True).astype(str)
    return (out.groupby(["product", "bps_bucket", "inferred_side"], as_index=False)
            .agg(trade_count=("quantity", "size"), total_quantity=("quantity", "sum"), max_quantity=("quantity", "max"), avg_bps_from_mid=("price_minus_mid_bps", "mean"), avg_spread_units=("spread_units_from_mid", "mean")))


def _build_summary(raw: dict[str, Any], activity_df: pd.DataFrame, sub_df: pd.DataFrame, bot_df: pd.DataFrame, realized_df: pd.DataFrame) -> dict[str, Any]:
    pnl = raw.get("profit")
    if pnl is None and not activity_df.empty and "profit_loss" in activity_df.columns:
        last = activity_df.sort_values("timestamp").groupby("product")["profit_loss"].last()
        pnl = float(last.sum()) if len(last) else 0.0
    wins = int((realized_df["pnl"] > 0).sum()) if not realized_df.empty else 0
    losses = int((realized_df["pnl"] < 0).sum()) if not realized_df.empty else 0
    winning_rows = realized_df[realized_df["pnl"] > 0] if not realized_df.empty else pd.DataFrame()
    return {
        "total_pnl": pnl,
        "num_runs": 1,
        "targets": ["Portal"],
        "portal_status": raw.get("status"),
        "submission_id": raw.get("submissionId"),
        "submission_trade_count": int(len(sub_df)),
        "bot_trade_count": int(len(bot_df)),
        "bot_quantity": int(bot_df["quantity"].sum()) if not bot_df.empty else 0,
        "realized_trade_count": int(len(realized_df)),
        "winning_trade_count": wins,
        "losing_trade_count": losses,
        "win_rate": wins / (wins + losses) if wins + losses else None,
        "average_win_per_trade": float(winning_rows["pnl"].mean()) if not winning_rows.empty else None,
        "average_win_pct": float(winning_rows["return_pct"].mean()) if not winning_rows.empty and "return_pct" in winning_rows.columns else None,
        "products_traded": sorted(set(sub_df.get("product", pd.Series(dtype=str)).dropna().astype(str).tolist())),
    }
