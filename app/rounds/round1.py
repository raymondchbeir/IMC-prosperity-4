from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from app.views.market_overview import _apply_common_layout


def build_round_analysis_layout():
    return html.Div(
        [
            html.H3("Round Analysis"),
            html.Div(id="round-summary-cards"),
            dcc.Graph(id="round-edge-histogram-graph"),
            dcc.Graph(id="round-extrema-signal-graph"),
        ]
    )


def classify_product_profile(prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> str:
    if prices_df.empty:
        return "Unknown"

    mid_std = prices_df["mid_price"].std() if "mid_price" in prices_df.columns else np.nan
    spread_median = prices_df["spread"].median() if "spread" in prices_df.columns else np.nan
    q15_share = (trades_df["quantity"] == 15).mean() if (not trades_df.empty and "quantity" in trades_df.columns) else 0.0

    if pd.notna(mid_std) and mid_std < 1.0:
        return "Fixed-anchor / stable fair"
    if pd.notna(mid_std) and mid_std < 10.0 and (pd.isna(spread_median) or spread_median <= 3):
        return "Slow-drift market-making"
    if q15_share > 0.10:
        return "Signal-driven / anomaly candidate"
    if pd.notna(mid_std) and mid_std >= 10.0:
        return "Jumpy / higher-variance"
    return "General"


def _safe_metric(value, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    return str(value)


def _card_style():
    return {
        "border": "1px solid #ddd",
        "borderRadius": "10px",
        "padding": "14px",
        "backgroundColor": "#fafafa",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
    }


def build_round_summary_cards(prices_df: pd.DataFrame, trades_df: pd.DataFrame):
    profile = classify_product_profile(prices_df, trades_df)
    avg_spread = prices_df["spread"].mean() if (not prices_df.empty and "spread" in prices_df.columns) else None
    med_spread = prices_df["spread"].median() if (not prices_df.empty and "spread" in prices_df.columns) else None
    avg_imb = prices_df["imbalance_top3"].mean() if (not prices_df.empty and "imbalance_top3" in prices_df.columns) else None
    realized_vol = prices_df["rolling_vol"].mean() if (not prices_df.empty and "rolling_vol" in prices_df.columns) else None
    trade_count = len(trades_df)
    q15_count = int((trades_df["quantity"] == 15).sum()) if (not trades_df.empty and "quantity" in trades_df.columns) else 0
    take_buy_count = int((prices_df["buy_take_edge"] > 0).sum()) if (not prices_df.empty and "buy_take_edge" in prices_df.columns) else 0
    take_sell_count = int((prices_df["sell_take_edge"] > 0).sum()) if (not prices_df.empty and "sell_take_edge" in prices_df.columns) else 0

    cards = [
        ("Profile", profile),
        ("Avg Spread", _safe_metric(avg_spread)),
        ("Median Spread", _safe_metric(med_spread)),
        ("Avg Top-3 Imbalance", _safe_metric(avg_imb)),
        ("Avg Rolling Vol", _safe_metric(realized_vol)),
        ("Trade Count", _safe_metric(trade_count, 0)),
        ("Qty=15 Trades", _safe_metric(q15_count, 0)),
        ("Buy Take Opportunities", _safe_metric(take_buy_count, 0)),
        ("Sell Take Opportunities", _safe_metric(take_sell_count, 0)),
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.H4(label),
                    html.P(value),
                ],
                style=_card_style(),
            )
            for label, value in cards
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, minmax(220px, 1fr))",
            "gap": "16px",
            "marginBottom": "20px",
        },
    )


def _estimate_anchor_fair_value(prices_df: pd.DataFrame) -> float | None:
    if prices_df.empty or "mid_price" not in prices_df.columns or not prices_df["mid_price"].notna().any():
        return None

    mid_std = prices_df["mid_price"].std()
    if pd.notna(mid_std) and mid_std < 1.0:
        return float(round(prices_df["mid_price"].median()))
    return None


def make_edge_histogram_figure(prices_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if prices_df.empty:
        return _apply_common_layout(fig, "Taker Edge Histogram", 320)

    df = prices_df.copy()
    anchor_fair = _estimate_anchor_fair_value(df)

    if anchor_fair is not None:
        fair = pd.Series(anchor_fair, index=df.index)
        xlabel = f"Edge vs anchor fair ({anchor_fair:.0f})"
    elif "mid_price" in df.columns and df["mid_price"].notna().any():
        fair = df["mid_price"]
        xlabel = "Edge vs current mid estimate"
    else:
        return _apply_common_layout(fig, "Taker Edge Histogram", 320)

    df["buy_take_edge"] = fair - df["ask_price_1"]
    df["sell_take_edge"] = df["bid_price_1"] - fair

    fig.add_trace(go.Histogram(x=df["buy_take_edge"].dropna(), nbinsx=40, name="Buy Take Edge"))
    fig.add_trace(go.Histogram(x=df["sell_take_edge"].dropna(), nbinsx=40, name="Sell Take Edge"))

    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="Count")
    return _apply_common_layout(fig, "Taker Edge Histogram", 320)


def make_extrema_signal_figure(prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if prices_df.empty:
        return _apply_common_layout(fig, "Extrema Signal View", 380)

    p = prices_df.sort_values("timestamp").copy()
    if "mid_price" in p.columns and p["mid_price"].notna().any():
        fig.add_trace(go.Scatter(x=p["timestamp"], y=p["mid_price"], mode="lines", name="Mid Price"))
        p["running_low"] = p["mid_price"].cummin()
        p["running_high"] = p["mid_price"].cummax()

        fig.add_trace(go.Scatter(x=p["timestamp"], y=p["running_low"], mode="lines", name="Running Low"))
        fig.add_trace(go.Scatter(x=p["timestamp"], y=p["running_high"], mode="lines", name="Running High"))

    if not trades_df.empty:
        t = trades_df.sort_values("timestamp").copy()
        if "quantity" in t.columns:
            flagged = t[t["quantity"] == 15]
            if not flagged.empty:
                fig.add_trace(
                    go.Scatter(
                        x=flagged["timestamp"],
                        y=flagged["price"],
                        mode="markers",
                        name="Qty 15 Trades",
                        marker={"size": 10, "symbol": "diamond"},
                    )
                )

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Price")
    return _apply_common_layout(fig, "Extrema Signal View", 380)