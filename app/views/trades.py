from __future__ import annotations

import math
from typing import Iterable

import dash_ag_grid as dag
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from app.views.market_overview import _apply_common_layout


def build_nancy_pelosi_identifier_layout():
    return html.Div(
        [
            dcc.Store(id="nancy-extra-events-store", data=[]),
            html.H3("Nancy Pelosi Identifier"),
            html.P(
                "Uses the same uploaded data and shared controls above to surface anonymous fingerprints that repeatedly trade near highs and lows."
            ),
            html.Div(id="nancy-summary-cards"),
            dcc.Graph(id="nancy-extrema-graph"),
            dcc.Graph(id="nancy-fingerprint-graph"),
            dcc.Graph(id="nancy-forward-graph"),
            html.Div(id="nancy-fingerprint-table-container", style={"marginTop": "12px"}),
            html.Div(
                [
                    html.H4("Flagged Trades / Event Clusters"),
                    html.P("Select rows to plot them below. Use the quantity buttons to add/remove every event with the same quantity."),
                    html.Div(
                        [
                            html.Button(
                                "Add all same quantity as selected",
                                id="nancy-add-same-qty-btn",
                                n_clicks=0,
                                style={"marginRight": "8px"},
                            ),
                            html.Button(
                                "Remove same quantity as selected",
                                id="nancy-remove-same-qty-btn",
                                n_clicks=0,
                                style={"marginRight": "8px"},
                            ),
                            html.Button(
                                "Clear added groups",
                                id="nancy-clear-extra-events-btn",
                                n_clicks=0,
                            ),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    html.Div(id="nancy-overlay-status", style={"marginBottom": "10px"}),
                    html.Div(id="nancy-flagged-grid-help", style={"marginBottom": "8px"}),
                    dag.AgGrid(
                        id="nancy-flagged-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={
                            "pagination": True,
                            "paginationPageSize": 15,
                            "rowSelection": "multiple",
                            "rowMultiSelectWithClick": True,
                            "suppressRowClickSelection": False,
                        },
                        style={"height": "440px", "width": "100%"},
                        className="ag-theme-alpine",
                    ),
                ],
                style={"marginTop": "12px"},
            ),
            html.Div(
                [
                    html.H4("Selected Events"),
                    html.P("This chart shows your currently selected rows plus any same-quantity groups you added."),
                    dcc.Graph(id="nancy-selected-events-graph"),
                ],
                style={"marginTop": "16px"},
            ),
        ]
    )


def build_nancy_pelosi_identifier_components(
    prices_df: pd.DataFrame,
    trades_df: pd.DataFrame,
):
    analysis = analyze_nancy_pelosi_identifier(prices_df, trades_df)

    flagged_defs, flagged_rows, flagged_help = build_flagged_grid_payload(analysis)

    return (
        build_nancy_summary_cards(analysis),
        make_nancy_extrema_figure(analysis),
        make_nancy_fingerprint_figure(analysis),
        make_nancy_forward_figure(analysis),
        build_fingerprint_table(analysis),
        flagged_defs,
        flagged_rows,
        flagged_help,
    )


def analyze_nancy_pelosi_identifier(
    prices_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    local_window: int = 31,
) -> dict:
    empty = {
        "events": pd.DataFrame(),
        "fingerprints": pd.DataFrame(),
        "raw_trades": pd.DataFrame(),
        "prices": pd.DataFrame(),
        "tick_size": np.nan,
        "median_trade_qty": np.nan,
        "verdict": "No data",
    }

    if prices_df.empty or trades_df.empty:
        empty["verdict"] = "Need both price and trade data for this symbol."
        return empty

    required_price_cols = {"timestamp", "mid_price"}
    required_trade_cols = {"timestamp", "price", "quantity"}
    if not required_price_cols.issubset(prices_df.columns) or not required_trade_cols.issubset(trades_df.columns):
        empty["verdict"] = "Missing required columns."
        return empty

    prices = prices_df.copy()
    trades = trades_df.copy()

    price_sort_cols = [c for c in ["day", "timestamp"] if c in prices.columns]
    trade_sort_cols = [c for c in ["day", "timestamp"] if c in trades.columns]

    if price_sort_cols:
        prices = prices.sort_values(price_sort_cols).reset_index(drop=True)
    if trade_sort_cols:
        trades = trades.sort_values(trade_sort_cols).reset_index(drop=True)

    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")
    median_trade_qty = float(trades["quantity"].dropna().median()) if trades["quantity"].notna().any() else np.nan

    tick_size = _estimate_tick_size(prices, trades)
    tol = max(float(tick_size) if pd.notna(tick_size) else 1.0, 0.5)

    prices = _add_extrema_features(prices, local_window=local_window)
    prices = _add_forward_mid_features(prices)

    merge_cols = [
        c
        for c in [
            "day",
            "timestamp",
            "mid_price",
            "bid_price_1",
            "ask_price_1",
            "running_high",
            "running_low",
            "day_high",
            "day_low",
            "local_high",
            "local_low",
            "future_mid_1",
            "future_mid_5",
            "future_mid_10",
        ]
        if c in prices.columns
    ]

    price_snapshots = prices[merge_cols].copy()

    if "day" in trades.columns and "day" in price_snapshots.columns:
        merged = pd.merge_asof(
            trades.sort_values(["day", "timestamp"]),
            price_snapshots.sort_values(["day", "timestamp"]),
            on="timestamp",
            by="day",
            direction="backward",
        )
    else:
        merged = pd.merge_asof(
            trades.sort_values("timestamp"),
            price_snapshots.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    if "side" not in merged.columns:
        merged["side"] = "Unknown"

    if pd.notna(median_trade_qty):
        merged = merged[merged["quantity"] >= median_trade_qty].copy()

    if merged.empty:
        empty["median_trade_qty"] = median_trade_qty
        empty["verdict"] = "No trades remain after filtering out quantities below the median trade size."
        return empty

    merged["execution_style"] = merged["side"].map(
        {
            "Buy": "Lifted ask",
            "Sell": "Hit bid",
            "Unknown": "Unknown",
        }
    )

    merged["near_daily_high"] = _is_near(merged["price"], merged.get("day_high"), tol)
    merged["near_daily_low"] = _is_near(merged["price"], merged.get("day_low"), tol)
    merged["near_local_high"] = _is_near(merged["price"], merged.get("local_high"), tol)
    merged["near_local_low"] = _is_near(merged["price"], merged.get("local_low"), tol)

    merged["pattern_bucket"] = merged.apply(_classify_pattern_bucket, axis=1)
    merged["is_suspicious"] = merged["pattern_bucket"].isin(["Buys highs", "Sells lows"])
    merged["is_opportunistic"] = merged["pattern_bucket"].isin(["Buys lows", "Sells highs"])

    merged["signed_move_5"] = merged.apply(lambda row: _signed_move(row, "future_mid_5"), axis=1)
    merged["signed_move_10"] = merged.apply(lambda row: _signed_move(row, "future_mid_10"), axis=1)

    merged["flags"] = merged.apply(_build_flags_text, axis=1)
    merged["suspicion_score"] = merged.apply(_calc_suspicion_score, axis=1)

    cluster_cols = [c for c in ["day", "timestamp", "price", "side", "execution_style", "pattern_bucket"] if c in merged.columns]
    events = (
        merged.groupby(cluster_cols, dropna=False, as_index=False)
        .agg(
            quantity=("quantity", "sum"),
            fills=("quantity", "size"),
            near_daily_high=("near_daily_high", "max"),
            near_daily_low=("near_daily_low", "max"),
            near_local_high=("near_local_high", "max"),
            near_local_low=("near_local_low", "max"),
            is_suspicious=("is_suspicious", "max"),
            is_opportunistic=("is_opportunistic", "max"),
            suspicion_score=("suspicion_score", "max"),
            signed_move_5=("signed_move_5", "mean"),
            signed_move_10=("signed_move_10", "mean"),
            flags=("flags", _join_unique_text),
        )
        .copy()
    )

    events["repeated_event_count"] = events.groupby(["pattern_bucket", "side", "quantity"], dropna=False)["timestamp"].transform("size")
    events["day_count_for_pattern"] = (
        events.groupby(["pattern_bucket", "side", "quantity"], dropna=False)["day"].transform("nunique")
        if "day" in events.columns
        else 1
    )
    events["suspicion_score"] = (
        events["suspicion_score"].fillna(0)
        + np.where(events["repeated_event_count"] >= 3, 1, 0)
        + np.where(events["day_count_for_pattern"] >= 2, 1, 0)
        + np.where(events["fills"] >= 2, 1, 0)
    )

    events["event_key"] = events.apply(
        lambda row: f"{row.get('day', '')}|{row.get('timestamp', '')}|{row.get('price', '')}|{row.get('side', '')}|{row.get('pattern_bucket', '')}|{row.get('quantity', '')}",
        axis=1,
    )

    fingerprints = (
        events.groupby(["pattern_bucket", "side", "execution_style", "quantity"], dropna=False, as_index=False)
        .agg(
            event_count=("timestamp", "size"),
            day_count=("day", "nunique") if "day" in events.columns else ("timestamp", "size"),
            avg_suspicion_score=("suspicion_score", "mean"),
            total_quantity=("quantity", "sum"),
            avg_signed_move_5=("signed_move_5", "mean"),
            avg_signed_move_10=("signed_move_10", "mean"),
            daily_high_hits=("near_daily_high", "sum"),
            daily_low_hits=("near_daily_low", "sum"),
            local_high_hits=("near_local_high", "sum"),
            local_low_hits=("near_local_low", "sum"),
            suspicious_hits=("is_suspicious", "sum"),
            opportunistic_hits=("is_opportunistic", "sum"),
            fills_avg=("fills", "mean"),
        )
        .copy()
    )

    if not fingerprints.empty:
        denom = fingerprints["event_count"].replace(0, np.nan)
        fingerprints["suspicious_share"] = fingerprints["suspicious_hits"] / denom
        fingerprints["opportunistic_share"] = fingerprints["opportunistic_hits"] / denom
        fingerprints["fingerprint_score"] = (
            fingerprints["avg_suspicion_score"].fillna(0) * fingerprints["event_count"].fillna(0)
            + 1.5 * fingerprints["day_count"].fillna(0)
        )
        fingerprints["confidence"] = fingerprints.apply(_confidence_label, axis=1)
        fingerprints["behavior_label"] = fingerprints.apply(_behavior_label, axis=1)
        fingerprints = fingerprints.sort_values(
            ["fingerprint_score", "event_count", "day_count", "quantity"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    events = events.sort_values(
        ["suspicion_score", "timestamp", "price"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    suspicious_fingerprints = (
        fingerprints[
            (fingerprints["behavior_label"].isin(["Suspicious high-chaser", "Suspicious low-dumper"]))
            & (fingerprints["event_count"] >= 2)
        ].copy()
        if not fingerprints.empty
        else pd.DataFrame()
    )

    suspicious_days = (
        int(events.loc[events["is_suspicious"], "day"].nunique())
        if (not events.empty and "day" in events.columns)
        else int(events["is_suspicious"].sum()) if not events.empty else 0
    )

    verdict = "No suspicious anonymous fingerprint detected."
    if not suspicious_fingerprints.empty and suspicious_days >= 2:
        verdict = "Suspicious repeated anonymous fingerprint detected across multiple days."
    elif not suspicious_fingerprints.empty:
        verdict = "Suspicious repeated anonymous fingerprint detected."
    elif not events.empty and int(events["is_suspicious"].sum()) >= 3:
        verdict = "Some suspicious extrema chasing is present, but the fingerprint is weaker."

    return {
        "events": events,
        "fingerprints": fingerprints,
        "raw_trades": merged,
        "prices": prices,
        "tick_size": tick_size,
        "median_trade_qty": median_trade_qty,
        "verdict": verdict,
    }


def build_nancy_summary_cards(analysis: dict):
    events = analysis["events"]
    fingerprints = analysis["fingerprints"]
    tick_size = analysis.get("tick_size")
    median_trade_qty = analysis.get("median_trade_qty")

    if events.empty:
        return html.Div(
            [
                html.Div(
                    [
                        html.H4("Verdict"),
                        html.P(analysis.get("verdict", "No data")),
                    ],
                    style=_card_style(),
                )
            ],
            style={"marginBottom": "18px"},
        )

    suspicious_events = int(events["is_suspicious"].sum()) if "is_suspicious" in events.columns else 0
    opportunistic_events = int(events["is_opportunistic"].sum()) if "is_opportunistic" in events.columns else 0
    suspicious_days = int(events.loc[events["is_suspicious"], "day"].nunique()) if "day" in events.columns and suspicious_events > 0 else 0
    top_score = float(fingerprints["fingerprint_score"].max()) if not fingerprints.empty else 0.0
    top_qty = int(fingerprints.iloc[0]["quantity"]) if not fingerprints.empty and pd.notna(fingerprints.iloc[0]["quantity"]) else "N/A"
    top_behavior = str(fingerprints.iloc[0]["behavior_label"]) if not fingerprints.empty else "N/A"
    repeated_fps = int((fingerprints["event_count"] >= 2).sum()) if not fingerprints.empty else 0

    cards = [
        ("Verdict", analysis.get("verdict", "No data")),
        ("Median Qty Filter", _fmt_number(median_trade_qty, 2)),
        ("Tick Size", _fmt_number(tick_size, 3)),
        ("Suspicious Events", suspicious_events),
        ("Suspicious Days", suspicious_days),
        ("Opportunistic Events", opportunistic_events),
        ("Repeated Fingerprints", repeated_fps),
        ("Top Quantity", top_qty),
        ("Top Behavior", top_behavior),
        ("Top Fingerprint Score", _fmt_number(top_score, 2)),
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.H4(label),
                    html.P(str(value)),
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


def make_nancy_extrema_figure(analysis: dict) -> go.Figure:
    prices = analysis["prices"]
    events = analysis["events"]

    fig = go.Figure()
    if prices.empty:
        return _apply_common_layout(fig, "Extrema + Flagged Trades", 420)

    p = prices.sort_values([c for c in ["day", "timestamp"] if c in prices.columns]).copy()
    p["plot_x"] = _make_plot_x(p)

    fig.add_trace(
        go.Scatter(
            x=p["plot_x"],
            y=p["mid_price"],
            mode="lines",
            name="Mid Price",
        )
    )

    if "running_low" in p.columns:
        fig.add_trace(
            go.Scatter(
                x=p["plot_x"],
                y=p["running_low"],
                mode="lines",
                name="Running Low",
            )
        )

    if "running_high" in p.columns:
        fig.add_trace(
            go.Scatter(
                x=p["plot_x"],
                y=p["running_high"],
                mode="lines",
                name="Running High",
            )
        )

    if not events.empty:
        plot_events = events.copy()
        plot_events["plot_x"] = _make_plot_x(plot_events)

        event_style = {
            "Buys highs": {"symbol": "triangle-up", "size": 11, "color": "#d62728"},
            "Sells lows": {"symbol": "triangle-down", "size": 11, "color": "#d62728"},
            "Buys lows": {"symbol": "circle", "size": 9, "color": "#2ca02c"},
            "Sells highs": {"symbol": "diamond", "size": 9, "color": "#2ca02c"},
            "Other / unclear": {"symbol": "x", "size": 8, "color": "#7f7f7f"},
        }

        for bucket, bucket_df in plot_events.groupby("pattern_bucket"):
            style = event_style.get(bucket, {"symbol": "circle", "size": 8, "color": "#7f7f7f"})
            fig.add_trace(
                go.Scatter(
                    x=bucket_df["plot_x"],
                    y=bucket_df["price"],
                    mode="markers",
                    name=bucket,
                    marker={
                        "symbol": style["symbol"],
                        "size": style["size"],
                        "opacity": 0.85,
                        "color": style["color"],
                    },
                    customdata=np.stack(
                        [
                            bucket_df["day"].fillna("").astype(str) if "day" in bucket_df.columns else pd.Series([""] * len(bucket_df)),
                            bucket_df["timestamp"].astype(str),
                            bucket_df["quantity"].astype(str),
                            bucket_df["fills"].astype(str),
                            bucket_df["suspicion_score"].astype(str),
                            bucket_df["flags"].fillna("").astype(str),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "Day=%{customdata[0]}<br>Timestamp=%{customdata[1]}<br>Price=%{y}"
                        "<br>Qty=%{customdata[2]}<br>Fills=%{customdata[3]}"
                        "<br>Score=%{customdata[4]}<br>Flags=%{customdata[5]}<extra></extra>"
                    ),
                )
            )

    fig.update_xaxes(title_text="Session Time")
    fig.update_yaxes(title_text="Price")
    return _apply_common_layout(fig, "Extrema + Flagged Trades", 420)


def make_nancy_fingerprint_figure(analysis: dict) -> go.Figure:
    fingerprints = analysis["fingerprints"]
    fig = go.Figure()

    if fingerprints.empty:
        return _apply_common_layout(fig, "Top Anonymous Fingerprints", 380)

    top = fingerprints.head(15).copy()
    top["label"] = top.apply(
        lambda row: f"qty={int(row['quantity'])} | {row['behavior_label']}",
        axis=1,
    )

    fig.add_trace(
        go.Bar(
            x=top["label"],
            y=top["fingerprint_score"],
            name="Fingerprint Score",
            customdata=np.stack(
                [
                    top["event_count"],
                    top["day_count"],
                    top["suspicious_share"].fillna(0.0),
                    top["avg_signed_move_5"].fillna(0.0),
                    top["confidence"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{x}<br>Score=%{y}<br>Events=%{customdata[0]}"
                "<br>Days=%{customdata[1]}<br>Suspicious Share=%{customdata[2]:.2f}"
                "<br>Avg Signed Move +5=%{customdata[3]:.4f}<br>Confidence=%{customdata[4]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(title_text="Fingerprint")
    fig.update_yaxes(title_text="Score")
    return _apply_common_layout(fig, "Top Anonymous Fingerprints", 380)


def make_nancy_forward_figure(analysis: dict) -> go.Figure:
    fingerprints = analysis["fingerprints"]
    fig = go.Figure()

    if fingerprints.empty:
        return _apply_common_layout(fig, "Forward Outcome by Fingerprint", 360)

    top = fingerprints.head(15).copy()
    top["label"] = top.apply(
        lambda row: f"qty={int(row['quantity'])} | {row['behavior_label']}",
        axis=1,
    )

    fig.add_trace(
        go.Bar(
            x=top["label"],
            y=top["avg_signed_move_5"],
            name="Avg Signed Move +5",
        )
    )
    fig.add_trace(
        go.Bar(
            x=top["label"],
            y=top["avg_signed_move_10"],
            name="Avg Signed Move +10",
        )
    )

    fig.update_xaxes(title_text="Fingerprint")
    fig.update_yaxes(title_text="Signed Forward Move")
    return _apply_common_layout(fig, "Forward Outcome by Fingerprint", 360)


def make_selected_events_figure(prices_df: pd.DataFrame, selected_rows: list[dict] | None) -> go.Figure:
    fig = go.Figure()

    if prices_df.empty:
        return _apply_common_layout(fig, "Selected Events", 380)

    p = prices_df.copy()
    p = p.sort_values([c for c in ["day", "timestamp"] if c in p.columns]).reset_index(drop=True)
    p["plot_x"] = _make_plot_x(p)

    fig.add_trace(
        go.Scatter(
            x=p["plot_x"],
            y=p["mid_price"],
            mode="lines",
            name="Mid Price",
        )
    )

    if "running_high" in p.columns:
        fig.add_trace(
            go.Scatter(
                x=p["plot_x"],
                y=p["running_high"],
                mode="lines",
                name="Running High",
                opacity=0.45,
            )
        )
    if "running_low" in p.columns:
        fig.add_trace(
            go.Scatter(
                x=p["plot_x"],
                y=p["running_low"],
                mode="lines",
                name="Running Low",
                opacity=0.45,
            )
        )

    if not selected_rows:
        fig.add_annotation(
            text="Select rows or add same-quantity groups to plot them here.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14},
        )
        return _apply_common_layout(fig, "Selected Events", 380)

    s = pd.DataFrame(selected_rows).copy()
    if s.empty or "timestamp" not in s.columns or "price" not in s.columns:
        return _apply_common_layout(fig, "Selected Events", 380)

    s["plot_x"] = _make_plot_x(s)
    s["quantity"] = pd.to_numeric(s["quantity"], errors="coerce")

    buys = s[s["side"] == "Buy"].copy()
    sells = s[s["side"] == "Sell"].copy()
    other = s[~s["side"].isin(["Buy", "Sell"])].copy()

    def add_trace(df: pd.DataFrame, name: str, color: str, symbol: str):
        if df.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=df["plot_x"],
                y=df["price"],
                mode="markers+text",
                name=name,
                text=df["quantity"].fillna("").astype(str),
                textposition="top center",
                marker={
                    "size": np.clip(np.sqrt(df["quantity"].fillna(0)) * 3.0, 10, 24),
                    "color": color,
                    "symbol": symbol,
                    "opacity": 0.9,
                },
                customdata=np.stack(
                    [
                        df["day"].fillna("").astype(str) if "day" in df.columns else pd.Series([""] * len(df)),
                        df["timestamp"].astype(str),
                        df["pattern_bucket"].fillna("").astype(str) if "pattern_bucket" in df.columns else pd.Series([""] * len(df)),
                        df["flags"].fillna("").astype(str) if "flags" in df.columns else pd.Series([""] * len(df)),
                        df["suspicion_score"].astype(str) if "suspicion_score" in df.columns else pd.Series([""] * len(df)),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Day=%{customdata[0]}<br>Timestamp=%{customdata[1]}<br>Price=%{y}"
                    "<br>Pattern=%{customdata[2]}<br>Flags=%{customdata[3]}"
                    "<br>Score=%{customdata[4]}<extra></extra>"
                ),
            )
        )

    add_trace(buys, "Selected Buys", "#1f77b4", "triangle-up")
    add_trace(sells, "Selected Sells", "#ff7f0e", "triangle-down")
    add_trace(other, "Selected Other", "#7f7f7f", "diamond")

    fig.update_xaxes(title_text="Session Time")
    fig.update_yaxes(title_text="Price")
    return _apply_common_layout(fig, "Selected Events", 380)


def build_fingerprint_table(analysis: dict):
    fingerprints = analysis["fingerprints"]

    if fingerprints.empty:
        return html.Div("No fingerprint summary available.")

    table_df = fingerprints.copy()
    keep_cols = [
        "behavior_label",
        "pattern_bucket",
        "side",
        "execution_style",
        "quantity",
        "event_count",
        "day_count",
        "suspicious_hits",
        "opportunistic_hits",
        "suspicious_share",
        "avg_signed_move_5",
        "avg_signed_move_10",
        "fingerprint_score",
        "confidence",
    ]
    keep_cols = [c for c in keep_cols if c in table_df.columns]
    table_df = table_df[keep_cols].copy()

    for col in ["suspicious_share", "avg_signed_move_5", "avg_signed_move_10", "fingerprint_score"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(lambda x: None if pd.isna(x) else float(x))

    grid = dag.AgGrid(
        id="nancy-fingerprint-grid",
        columnDefs=[
            {"field": col, "headerName": col.replace("_", " ").title(), "flex": 1}
            for col in table_df.columns
        ],
        rowData=table_df.to_dict("records"),
        defaultColDef={"resizable": True, "sortable": True, "filter": True},
        dashGridOptions={"pagination": True, "paginationPageSize": 12},
        style={"height": "420px", "width": "100%"},
        className="ag-theme-alpine",
    )

    return html.Div(
        [
            html.H4("Anonymous Fingerprint Summary"),
            grid,
        ]
    )


def build_flagged_grid_payload(analysis: dict):
    events = analysis["events"]

    if events.empty:
        cols = [{"field": "message", "headerName": "Message", "flex": 1}]
        rows = [{"message": "No flagged trades available."}]
        return cols, rows, "No flagged trades available."

    table_df = events.copy()
    keep_cols = [
        "event_key",
        "day",
        "timestamp",
        "price",
        "quantity",
        "fills",
        "side",
        "execution_style",
        "pattern_bucket",
        "near_daily_high",
        "near_daily_low",
        "near_local_high",
        "near_local_low",
        "repeated_event_count",
        "day_count_for_pattern",
        "signed_move_5",
        "signed_move_10",
        "suspicion_score",
        "flags",
    ]
    keep_cols = [c for c in keep_cols if c in table_df.columns]
    table_df = table_df[keep_cols].copy()

    for col in ["signed_move_5", "signed_move_10", "suspicion_score"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(lambda x: None if pd.isna(x) else float(x))

    column_defs = []
    for col in table_df.columns:
        if col == "event_key":
            column_defs.append({"field": col, "hide": True})
        else:
            column_defs.append({"field": col, "headerName": col.replace("_", " ").title(), "flex": 1})

    help_text = f"Showing flagged events only. Trades below the median quantity filter were removed."

    return column_defs, table_df.to_dict("records"), help_text


def _add_extrema_features(prices: pd.DataFrame, local_window: int = 31) -> pd.DataFrame:
    out = prices.copy()

    if "mid_price" not in out.columns:
        return out

    if "day" in out.columns:
        out["running_high"] = out.groupby("day")["mid_price"].cummax()
        out["running_low"] = out.groupby("day")["mid_price"].cummin()
        out["day_high"] = out.groupby("day")["mid_price"].transform("max")
        out["day_low"] = out.groupby("day")["mid_price"].transform("min")
        out["local_high"] = out.groupby("day")["mid_price"].transform(
            lambda s: s.rolling(local_window, center=True, min_periods=max(3, local_window // 4)).max()
        )
        out["local_low"] = out.groupby("day")["mid_price"].transform(
            lambda s: s.rolling(local_window, center=True, min_periods=max(3, local_window // 4)).min()
        )
    else:
        out["running_high"] = out["mid_price"].cummax()
        out["running_low"] = out["mid_price"].cummin()
        out["day_high"] = out["mid_price"].max()
        out["day_low"] = out["mid_price"].min()
        out["local_high"] = out["mid_price"].rolling(local_window, center=True, min_periods=max(3, local_window // 4)).max()
        out["local_low"] = out["mid_price"].rolling(local_window, center=True, min_periods=max(3, local_window // 4)).min()

    return out


def _add_forward_mid_features(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    if "mid_price" not in out.columns:
        return out

    if "day" in out.columns:
        out["future_mid_1"] = out.groupby("day")["mid_price"].shift(-1)
        out["future_mid_5"] = out.groupby("day")["mid_price"].shift(-5)
        out["future_mid_10"] = out.groupby("day")["mid_price"].shift(-10)
    else:
        out["future_mid_1"] = out["mid_price"].shift(-1)
        out["future_mid_5"] = out["mid_price"].shift(-5)
        out["future_mid_10"] = out["mid_price"].shift(-10)
    return out


def _estimate_tick_size(prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> float:
    candidates: list[float] = []

    for col in ["bid_price_1", "ask_price_1", "mid_price"]:
        if col in prices_df.columns:
            vals = pd.Series(prices_df[col]).dropna().astype(float).sort_values().unique()
            if len(vals) >= 2:
                diffs = np.diff(vals)
                diffs = diffs[diffs > 0]
                if len(diffs) > 0:
                    candidates.extend(diffs.tolist())

    if "price" in trades_df.columns:
        vals = pd.Series(trades_df["price"]).dropna().astype(float).sort_values().unique()
        if len(vals) >= 2:
            diffs = np.diff(vals)
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                candidates.extend(diffs.tolist())

    if not candidates:
        return float("nan")

    arr = np.array(candidates, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return float("nan")

    return float(np.quantile(arr, 0.05))


def _is_near(series_a, series_b, tol: float):
    if series_b is None:
        return pd.Series(False, index=series_a.index)
    a = pd.Series(series_a).astype(float)
    b = pd.Series(series_b, index=series_a.index).astype(float)
    return (a - b).abs() <= tol


def _classify_pattern_bucket(row) -> str:
    side = row.get("side", "Unknown")
    near_high = bool(row.get("near_daily_high", False) or row.get("near_local_high", False))
    near_low = bool(row.get("near_daily_low", False) or row.get("near_local_low", False))

    if side == "Buy" and near_high:
        return "Buys highs"
    if side == "Sell" and near_low:
        return "Sells lows"
    if side == "Buy" and near_low:
        return "Buys lows"
    if side == "Sell" and near_high:
        return "Sells highs"
    return "Other / unclear"


def _signed_move(row, future_col: str) -> float:
    future_mid = row.get(future_col)
    trade_price = row.get("price")
    side = row.get("side", "Unknown")

    if pd.isna(future_mid) or pd.isna(trade_price):
        return np.nan
    if side == "Buy":
        return float(future_mid - trade_price)
    if side == "Sell":
        return float(trade_price - future_mid)
    return np.nan


def _build_flags_text(row) -> str:
    flags: list[str] = []
    if bool(row.get("near_daily_high", False)):
        flags.append("daily high")
    if bool(row.get("near_daily_low", False)):
        flags.append("daily low")
    if bool(row.get("near_local_high", False)):
        flags.append("local high")
    if bool(row.get("near_local_low", False)):
        flags.append("local low")
    if row.get("side") == "Buy":
        flags.append("aggressive buy")
    elif row.get("side") == "Sell":
        flags.append("aggressive sell")
    return ", ".join(flags)


def _calc_suspicion_score(row) -> float:
    score = 0.0
    if row.get("pattern_bucket") in ["Buys highs", "Sells lows"]:
        score += 2.0
    if bool(row.get("near_daily_high", False)) or bool(row.get("near_daily_low", False)):
        score += 1.5
    if bool(row.get("near_local_high", False)) or bool(row.get("near_local_low", False)):
        score += 1.0
    qty = row.get("quantity")
    if pd.notna(qty) and float(qty) >= 10:
        score += 1.0
    return score


def _confidence_label(row) -> str:
    score = float(row.get("fingerprint_score", 0.0))
    event_count = int(row.get("event_count", 0))
    day_count = int(row.get("day_count", 0))

    if score >= 18 and event_count >= 4 and day_count >= 2:
        return "High"
    if score >= 9 and event_count >= 2:
        return "Medium"
    return "Low"


def _behavior_label(row) -> str:
    pattern = row.get("pattern_bucket", "Other / unclear")
    if pattern == "Buys highs":
        return "Suspicious high-chaser"
    if pattern == "Sells lows":
        return "Suspicious low-dumper"
    if pattern == "Buys lows":
        return "Opportunistic low-buyer"
    if pattern == "Sells highs":
        return "Opportunistic high-seller"
    return "Other / unclear"


def _join_unique_text(values: Iterable[str]) -> str:
    clean = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s not in clean:
            clean.append(s)
    return " | ".join(clean)


def _fmt_number(value, decimals: int = 2):
    if value is None or (isinstance(value, float) and (math.isnan(value) or np.isnan(value))):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    return value


def _make_plot_x(df: pd.DataFrame) -> pd.Series:
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)
    if "day" not in df.columns:
        return timestamps

    day_vals = pd.to_numeric(df["day"], errors="coerce").fillna(0)
    day_min = day_vals.min()
    ts_span = max(float(timestamps.max() - timestamps.min()), 1.0)
    return timestamps + (day_vals - day_min) * (ts_span + 1000.0)


def _card_style():
    return {
        "border": "1px solid #ddd",
        "borderRadius": "10px",
        "padding": "14px",
        "backgroundColor": "#fafafa",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
    }