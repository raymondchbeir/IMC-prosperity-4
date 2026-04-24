from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots


def build_overview_bot_heatmap_layout():
    return html.Div(
        [
            html.H3("Uploaded CSV Bot Trade Heatmaps"),
            html.P(
                "These heatmaps use the uploaded trades CSV as bot/market trades and enrich them with the uploaded prices CSV book snapshot. Each selected symbol gets its own panel. Color is total quantity traded."
            ),
            dcc.Graph(id="overview-bot-price-heatmap", figure=_empty_figure("Upload prices/trades CSVs to begin.")),
            dcc.Graph(id="overview-bot-normalized-heatmap", figure=_empty_figure("Upload prices/trades CSVs to begin.")),
            dcc.Graph(id="overview-bot-spread-heatmap", figure=_empty_figure("Upload prices/trades CSVs to begin.")),
        ],
        style={"marginTop": "24px"},
    )


def append_overview_bot_heatmap_layout(layout):
    children = list(getattr(layout, "children", []) or [])
    if not any(getattr(child, "id", None) == "overview-bot-heatmap-section" for child in children):
        children.append(html.Div(build_overview_bot_heatmap_layout(), id="overview-bot-heatmap-section"))
    layout.children = children
    return layout


def register_overview_bot_heatmap_callbacks(app):
    @app.callback(
        Output("overview-bot-price-heatmap", "figure"),
        Output("overview-bot-normalized-heatmap", "figure"),
        Output("overview-bot-spread-heatmap", "figure"),
        Input("session-data-store", "data"),
        Input("product-dropdown", "value"),
        Input("round-dropdown", "value"),
        Input("day-dropdown", "value"),
        Input("compare-days-toggle", "value"),
        Input("compare-product-dropdown", "value"),
        Input("timestamp-range-slider", "value"),
        prevent_initial_call=True,
    )
    def update_overview_bot_heatmaps(
        store_data,
        selected_product,
        selected_round,
        selected_day,
        compare_days_value,
        compare_products,
        timestamp_range,
    ):
        if not store_data or not selected_product or selected_round is None:
            empty = _empty_figure("Upload prices/trades CSVs and select a product.")
            return empty, empty, empty

        prices_df, trades_df = _filter_uploaded_market_data(
            store_data=store_data,
            selected_product=selected_product,
            selected_round=selected_round,
            selected_day=selected_day,
            compare_days_value=compare_days_value,
            compare_products=compare_products,
            timestamp_range=timestamp_range,
        )
        bot_df = _build_uploaded_bot_trades(prices_df, trades_df)
        return (
            _make_price_level_heatmap(bot_df),
            _make_normalized_level_heatmap(bot_df),
            _make_spread_execution_heatmap(bot_df),
        )


def _filter_uploaded_market_data(
    store_data: dict,
    selected_product: str,
    selected_round: int,
    selected_day: int | None,
    compare_days_value,
    compare_products,
    timestamp_range,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices_df = pd.DataFrame(store_data.get("prices", []))
    trades_df = pd.DataFrame(store_data.get("trades", []))
    compare_days = "compare_days" in (compare_days_value or [])
    products = [selected_product] + [p for p in (compare_products or []) if p != selected_product]

    for df_name, df in (("prices", prices_df), ("trades", trades_df)):
        if not df.empty:
            for col in ["round", "day", "timestamp", "price", "quantity", "mid_price", "bid_price_1", "ask_price_1"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

    if not prices_df.empty and {"product", "round"}.issubset(prices_df.columns):
        prices_df = prices_df[(prices_df["product"].isin(products)) & (prices_df["round"] == selected_round)].copy()
        if not compare_days and selected_day is not None and "day" in prices_df.columns:
            prices_df = prices_df[prices_df["day"] == selected_day].copy()

    if not trades_df.empty and {"product", "round"}.issubset(trades_df.columns):
        trades_df = trades_df[(trades_df["product"].isin(products)) & (trades_df["round"] == selected_round)].copy()
        if not compare_days and selected_day is not None and "day" in trades_df.columns:
            trades_df = trades_df[trades_df["day"] == selected_day].copy()

    if timestamp_range and len(timestamp_range) == 2:
        t0, t1 = timestamp_range
        if not prices_df.empty and "timestamp" in prices_df.columns:
            prices_df = prices_df[(prices_df["timestamp"] >= t0) & (prices_df["timestamp"] <= t1)].copy()
        if not trades_df.empty and "timestamp" in trades_df.columns:
            trades_df = trades_df[(trades_df["timestamp"] >= t0) & (trades_df["timestamp"] <= t1)].copy()

    return prices_df, trades_df


def _build_uploaded_bot_trades(prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    trades = trades_df.copy()
    if {"buyer", "seller"}.issubset(trades.columns):
        buyer = trades["buyer"].fillna("").astype(str)
        seller = trades["seller"].fillna("").astype(str)
        if ((buyer == "SUBMISSION") | (seller == "SUBMISSION")).any():
            trades = trades[(buyer != "SUBMISSION") & (seller != "SUBMISSION")].copy()

    if trades.empty:
        return pd.DataFrame()

    if prices_df.empty:
        trades["mid_price"] = np.nan
        trades["spread"] = np.nan
        trades["price_minus_mid"] = np.nan
        trades["spread_units_from_mid"] = np.nan
        return trades

    prices = prices_df.copy()
    if {"bid_price_1", "ask_price_1"}.issubset(prices.columns):
        valid = (prices["bid_price_1"] > 0) & (prices["ask_price_1"] > 0)
        prices["spread"] = np.nan
        prices.loc[valid, "spread"] = prices.loc[valid, "ask_price_1"] - prices.loc[valid, "bid_price_1"]
        if "mid_price" not in prices.columns:
            prices["mid_price"] = np.nan
        bad_mid = prices["mid_price"].isna() | (prices["mid_price"] <= 0)
        prices.loc[valid & bad_mid, "mid_price"] = (prices.loc[valid & bad_mid, "bid_price_1"] + prices.loc[valid & bad_mid, "ask_price_1"]) / 2.0

    book_cols = [c for c in ["product", "timestamp", "mid_price", "spread", "bid_price_1", "ask_price_1"] if c in prices.columns]
    if not {"product", "timestamp"}.issubset(book_cols):
        return trades

    book = prices[book_cols].dropna(subset=["product", "timestamp"]).sort_values(["product", "timestamp"]).copy()
    trades = trades.dropna(subset=["product", "timestamp", "price", "quantity"]).sort_values(["product", "timestamp"]).copy()
    if trades.empty:
        return trades

    enriched_parts = []
    for product, part in trades.groupby("product", sort=False):
        book_part = book[book["product"] == product].drop(columns=["product"], errors="ignore").sort_values("timestamp")
        trade_part = part.sort_values("timestamp")
        if book_part.empty:
            enriched = trade_part.copy()
        else:
            enriched = pd.merge_asof(trade_part, book_part, on="timestamp", direction="nearest")
        enriched_parts.append(enriched)

    out = pd.concat(enriched_parts, ignore_index=True) if enriched_parts else pd.DataFrame()
    if out.empty:
        return out

    out["price_minus_mid"] = out["price"] - out.get("mid_price", np.nan)
    out["price_minus_mid_bps"] = np.where(out.get("mid_price", np.nan) > 0, 10000 * out["price_minus_mid"] / out["mid_price"], np.nan)
    out["spread_units_from_mid"] = np.where(out.get("spread", np.nan).abs() > 0, out["price_minus_mid"] / out["spread"], np.nan)
    out["inferred_side"] = "Inside/Unknown"
    if "ask_price_1" in out.columns:
        out.loc[pd.notna(out["ask_price_1"]) & (out["price"] >= out["ask_price_1"]), "inferred_side"] = "Lifted Ask"
    if "bid_price_1" in out.columns:
        out.loc[pd.notna(out["bid_price_1"]) & (out["price"] <= out["bid_price_1"]), "inferred_side"] = "Hit Bid"
    return out


def _time_bucket_series(series: pd.Series, max_bins: int = 80) -> pd.Series:
    ts = pd.to_numeric(series, errors="coerce")
    if ts.dropna().nunique() <= max_bins:
        return ts.astype("Int64").astype(str)
    try:
        return pd.cut(ts, bins=max_bins, duplicates="drop").astype(str)
    except ValueError:
        return ts.astype(str)


def _price_level_series(series: pd.Series, max_levels: int = 140) -> pd.Series:
    price = pd.to_numeric(series, errors="coerce")
    if price.dropna().nunique() <= max_levels:
        return price.astype("Int64").astype(str)
    try:
        return pd.cut(price, bins=max_levels, duplicates="drop").astype(str)
    except ValueError:
        return price.astype(str)


def _normalized_level_series(series: pd.Series, step: float = 1.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    bucketed = (values / step).round() * step

    def fmt(v):
        if pd.isna(v):
            return "unknown"
        if abs(v) < 1e-9:
            return "0 at mid"
        if float(v).is_integer():
            value = str(int(v))
        else:
            value = f"{v:.2f}".rstrip("0").rstrip(".")
        return f"+{value} above mid" if v > 0 else f"{value} below mid"

    return bucketed.map(fmt)


def _faceted_heatmap(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    x_title: str,
    y_title: str,
    color_title: str = "Total Qty",
) -> go.Figure:
    products = sorted(df["product"].dropna().astype(str).unique().tolist()) if "product" in df.columns else []
    if not products:
        return _empty_figure(title)

    fig = make_subplots(
        rows=len(products),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=min(0.08, 0.3 / max(len(products), 1)),
        subplot_titles=products,
    )
    for row_idx, product in enumerate(products, start=1):
        part = df[df["product"].astype(str) == product].copy()
        pivot = part.pivot_table(index=y_col, columns="time_bucket", values="quantity", aggfunc="sum", fill_value=0)
        pivot = pivot.sort_index()
        fig.add_trace(
            go.Heatmap(
                x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                z=pivot.values,
                coloraxis="coloraxis",
                hovertemplate=f"Time=%{{x}}<br>{y_title}=%{{y}}<br>{color_title}=%{{z}}<extra>{product}</extra>",
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text=y_title, autorange="reversed", row=row_idx, col=1)
        fig.update_xaxes(title_text=x_title, row=row_idx, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(460, 310 * len(products)),
        coloraxis={"colorbar": {"title": color_title}},
        showlegend=False,
        hovermode="closest",
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    return fig


def _make_price_level_heatmap(bot_df: pd.DataFrame) -> go.Figure:
    if bot_df.empty or not {"timestamp", "price", "quantity", "product"}.issubset(bot_df.columns):
        return _empty_figure("No uploaded trade rows available for price-level heatmap.")
    df = bot_df.copy()
    df["time_bucket"] = _time_bucket_series(df["timestamp"], max_bins=80)
    df["price_bucket"] = _price_level_series(df["price"], max_levels=120)
    return _faceted_heatmap(
        df,
        y_col="price_bucket",
        title="Uploaded CSV Bot Trades: Price Level × Time by Symbol",
        x_title="Timestamp bucket",
        y_title="Price level",
    )


def _make_normalized_level_heatmap(bot_df: pd.DataFrame) -> go.Figure:
    if bot_df.empty or not {"timestamp", "quantity", "product", "price_minus_mid"}.issubset(bot_df.columns):
        return _empty_figure("No uploaded trade rows available for normalized heatmap.")
    df = bot_df.dropna(subset=["price_minus_mid"]).copy()
    if df.empty:
        return _empty_figure("No uploaded trade rows available for normalized heatmap.")
    df["time_bucket"] = _time_bucket_series(df["timestamp"], max_bins=80)
    df["normalized_level"] = _normalized_level_series(df["price_minus_mid"], step=1.0)
    return _faceted_heatmap(
        df,
        y_col="normalized_level",
        title="Uploaded CSV Bot Trades: Price-Minus-Mid Level × Time by Symbol",
        x_title="Timestamp bucket",
        y_title="Distance from mid",
    )


def _make_spread_execution_heatmap(bot_df: pd.DataFrame) -> go.Figure:
    needed = {"spread", "quantity", "product", "spread_units_from_mid"}
    if bot_df.empty or not needed.issubset(bot_df.columns):
        return _empty_figure("No uploaded trade rows available for spread heatmap.")
    df = bot_df.dropna(subset=["spread", "spread_units_from_mid"]).copy()
    if df.empty:
        return _empty_figure("No uploaded trade rows available for spread heatmap.")

    spread_num = pd.to_numeric(df["spread"], errors="coerce")
    if spread_num.dropna().nunique() <= 30:
        df["spread_bucket"] = spread_num.astype("Int64").astype(str)
    else:
        df["spread_bucket"] = pd.cut(spread_num, bins=30, duplicates="drop").astype(str)

    bins = [-5, -2, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 2, 5]
    df["execution_level"] = pd.cut(df["spread_units_from_mid"], bins=bins, include_lowest=True, duplicates="drop").astype(str)

    products = sorted(df["product"].dropna().astype(str).unique().tolist())
    if not products:
        return _empty_figure("No uploaded trade rows available for spread heatmap.")

    fig = make_subplots(
        rows=len(products),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=min(0.08, 0.3 / max(len(products), 1)),
        subplot_titles=products,
    )
    for row_idx, product in enumerate(products, start=1):
        part = df[df["product"].astype(str) == product]
        pivot = part.pivot_table(index="execution_level", columns="spread_bucket", values="quantity", aggfunc="sum", fill_value=0)
        fig.add_trace(
            go.Heatmap(
                x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                z=pivot.values,
                coloraxis="coloraxis",
                hovertemplate="Spread=%{x}<br>Execution level=%{y}<br>Total Qty=%{z}<extra>" + product + "</extra>",
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text="(price-mid)/spread", autorange="reversed", row=row_idx, col=1)
        fig.update_xaxes(title_text="Quoted spread at trade", row=row_idx, col=1)

    fig.update_layout(
        title="Uploaded CSV Bot Trades: Execution Level × Spread by Symbol",
        template="plotly_white",
        height=max(460, 310 * len(products)),
        coloraxis={"colorbar": {"title": "Total Qty"}},
        showlegend=False,
        hovermode="closest",
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
    fig.update_layout(template="plotly_white", height=360)
    return fig
