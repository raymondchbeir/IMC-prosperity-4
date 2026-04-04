from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots


def build_market_overview_layout():
    return html.Div(
        [
            html.H3("Market Overview"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Product"),
                            dcc.Dropdown(
                                id="product-dropdown",
                                options=[],
                                value=None,
                                placeholder="Select product",
                                clearable=False,
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                    html.Div(
                        [
                            html.Label("Round"),
                            dcc.Dropdown(
                                id="round-dropdown",
                                options=[],
                                value=None,
                                placeholder="Select round",
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Label("Day"),
                            dcc.Dropdown(
                                id="day-dropdown",
                                options=[],
                                value=None,
                                placeholder="Select day",
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Label("Compare Mode"),
                            dcc.Checklist(
                                id="compare-days-toggle",
                                options=[{"label": " Compare all available days", "value": "compare_days"}],
                                value=[],
                                style={"marginTop": "8px"},
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "marginBottom": "16px",
                    "alignItems": "end",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Cross-Product Compare"),
                            dcc.Dropdown(
                                id="compare-product-dropdown",
                                options=[],
                                value=[],
                                multi=True,
                                placeholder="Optional: compare against other products",
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                    html.Div(
                        [
                            html.Label("Time Range"),
                            dcc.RangeSlider(
                                id="timestamp-range-slider",
                                min=0,
                                max=1,
                                step=1,
                                value=[0, 1],
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        style={"flex": "3", "padding": "0 12px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "marginBottom": "20px",
                    "alignItems": "end",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(
                [
                    html.Button("Download Filtered Prices CSV", id="download-prices-btn", n_clicks=0, style={"marginRight": "10px"}),
                    html.Button("Download Filtered Trades CSV", id="download-trades-btn", n_clicks=0),
                    dcc.Download(id="download-prices"),
                    dcc.Download(id="download-trades"),
                ],
                style={"marginBottom": "20px"},
            ),
            dcc.Graph(id="price-book-graph"),
            dcc.Graph(id="spread-graph"),
            dcc.Graph(id="depth-graph"),
            dcc.Graph(id="trade-volume-graph"),
            dcc.Graph(id="imbalance-graph"),
            dcc.Graph(id="returns-volatility-graph"),
            dcc.Graph(id="book-heatmap-graph"),
            dcc.Graph(id="cross-product-graph"),
        ]
    )


def _apply_common_layout(fig: go.Figure, title: str, height: int = 350) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        hovermode="x unified",
        legend={"orientation": "h"},
    )
    return fig


def infer_trade_side(prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    if prices_df.empty or trades_df.empty:
        out = trades_df.copy()
        if not out.empty and "side" not in out.columns:
            out["side"] = "Unknown"
        return out

    book = prices_df[["timestamp", "bid_price_1", "ask_price_1"]].dropna(subset=["timestamp"]).sort_values("timestamp").copy()
    trades = trades_df.sort_values("timestamp").copy()

    merged = pd.merge_asof(
        trades,
        book,
        on="timestamp",
        direction="backward",
    )

    def classify(row):
        price = row.get("price")
        bid = row.get("bid_price_1")
        ask = row.get("ask_price_1")
        if pd.isna(price) or (pd.isna(bid) and pd.isna(ask)):
            return "Unknown"
        if pd.notna(ask) and price >= ask:
            return "Buy"
        if pd.notna(bid) and price <= bid:
            return "Sell"
        return "Unknown"

    merged["side"] = merged.apply(classify, axis=1)
    return merged


def make_price_book_figure(prices_df: pd.DataFrame, trades_df: pd.DataFrame, compare_days: bool = False) -> go.Figure:
    fig = go.Figure()

    if not prices_df.empty:
        prices_df = prices_df.sort_values(["day", "timestamp"]).copy()

        if compare_days and "day" in prices_df.columns:
            for day, day_df in prices_df.groupby("day"):
                fig.add_trace(go.Scatter(x=day_df["timestamp"], y=day_df["bid_price_1"], mode="lines", name=f"Bid 1 (day {day})"))
                fig.add_trace(go.Scatter(x=day_df["timestamp"], y=day_df["ask_price_1"], mode="lines", name=f"Ask 1 (day {day})"))
                if "mid_price" in day_df.columns and day_df["mid_price"].notna().any():
                    fig.add_trace(go.Scatter(x=day_df["timestamp"], y=day_df["mid_price"], mode="lines", name=f"Mid Price (day {day})"))
        else:
            fig.add_trace(go.Scatter(x=prices_df["timestamp"], y=prices_df["bid_price_1"], mode="lines", name="Bid 1"))
            fig.add_trace(go.Scatter(x=prices_df["timestamp"], y=prices_df["ask_price_1"], mode="lines", name="Ask 1"))
            if "mid_price" in prices_df.columns and prices_df["mid_price"].notna().any():
                fig.add_trace(go.Scatter(x=prices_df["timestamp"], y=prices_df["mid_price"], mode="lines", name="Mid Price"))

    if not trades_df.empty and not compare_days:
        trades_df = infer_trade_side(prices_df, trades_df)

        side_to_color = {"Buy": "green", "Sell": "red", "Unknown": "gray"}

        for side, side_df in trades_df.groupby("side"):
            sizes = side_df["quantity"].fillna(0)
            marker_sizes = np.clip(np.sqrt(sizes) * 2.5, 5, 18)

            fig.add_trace(
                go.Scatter(
                    x=side_df["timestamp"],
                    y=side_df["price"],
                    mode="markers",
                    name=f"Trades - {side}",
                    marker={
                        "size": marker_sizes,
                        "color": side_to_color.get(side, "gray"),
                        "opacity": 0.72,
                    },
                    customdata=np.stack([side_df["quantity"]], axis=-1),
                    hovertemplate="Timestamp=%{x}<br>Price=%{y}<br>Qty=%{customdata[0]}<br>Side="
                    + side
                    + "<extra></extra>",
                )
            )

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Price")
    return _apply_common_layout(fig, "Top of Book + Trades", 460)


def make_spread_figure(prices_df: pd.DataFrame, compare_days: bool = False) -> go.Figure:
    fig = go.Figure()

    if not prices_df.empty:
        prices_df = prices_df.sort_values(["day", "timestamp"]).copy()
        prices_df["spread"] = prices_df["ask_price_1"] - prices_df["bid_price_1"]

        if compare_days and "day" in prices_df.columns:
            for day, day_df in prices_df.groupby("day"):
                fig.add_trace(go.Scatter(x=day_df["timestamp"], y=day_df["spread"], mode="lines", name=f"Spread (day {day})"))
        else:
            fig.add_trace(go.Scatter(x=prices_df["timestamp"], y=prices_df["spread"], mode="lines", name="Spread"))

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Spread")
    return _apply_common_layout(fig, "Spread Over Time", 300)


def make_depth_figure(prices_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)

    if not prices_df.empty:
        prices_df = prices_df.sort_values("timestamp").copy()

        bid_volume_cols = [c for c in ["bid_volume_1", "bid_volume_2", "bid_volume_3"] if c in prices_df.columns]
        ask_volume_cols = [c for c in ["ask_volume_1", "ask_volume_2", "ask_volume_3"] if c in prices_df.columns]

        if "bid_volume_1" in prices_df.columns:
            fig.add_trace(go.Scatter(x=prices_df["timestamp"], y=prices_df["bid_volume_1"], mode="lines", name="Bid Vol 1"), row=1, col=1)
        if "ask_volume_1" in prices_df.columns:
            fig.add_trace(go.Scatter(x=prices_df["timestamp"], y=prices_df["ask_volume_1"], mode="lines", name="Ask Vol 1"), row=1, col=1)

        if bid_volume_cols:
            prices_df["total_bid_depth"] = prices_df[bid_volume_cols].fillna(0).sum(axis=1)
            fig.add_trace(go.Scatter(x=prices_df["timestamp"], y=prices_df["total_bid_depth"], mode="lines", name="Total Bid Depth"), row=2, col=1)
        if ask_volume_cols:
            prices_df["total_ask_depth"] = prices_df[ask_volume_cols].fillna(0).sum(axis=1)
            fig.add_trace(go.Scatter(x=prices_df["timestamp"], y=prices_df["total_ask_depth"], mode="lines", name="Total Ask Depth"), row=2, col=1)

    fig.update_layout(
        title="Book Volumes and Total Depth",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend={"orientation": "h"},
    )
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Level 1 Volume", row=1, col=1)
    fig.update_yaxes(title_text="Total Depth", row=2, col=1)
    return fig


def make_trade_volume_figure(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if not trades_df.empty:
        classified = infer_trade_side(prices_df, trades_df)
        volume_df = classified.groupby(["timestamp", "side"], as_index=False)["quantity"].sum()

        for side in ["Buy", "Sell", "Unknown"]:
            side_df = volume_df[volume_df["side"] == side]
            if not side_df.empty:
                fig.add_trace(
                    go.Bar(
                        x=side_df["timestamp"],
                        y=side_df["quantity"],
                        name=f"{side} Volume",
                    )
                )

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Quantity")
    return _apply_common_layout(fig, "Trade Volume by Side", 320)


def make_imbalance_figure(prices_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if not prices_df.empty and "bid_volume_1" in prices_df.columns and "ask_volume_1" in prices_df.columns:
        prices_df = prices_df.sort_values("timestamp").copy()
        denom = prices_df["bid_volume_1"].fillna(0) + prices_df["ask_volume_1"].fillna(0)
        denom = denom.replace(0, pd.NA)
        prices_df["imbalance_l1"] = ((prices_df["bid_volume_1"].fillna(0) - prices_df["ask_volume_1"].fillna(0)) / denom)

        fig.add_trace(go.Scatter(x=prices_df["timestamp"], y=prices_df["imbalance_l1"], mode="lines", name="L1 Imbalance"))

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Imbalance")
    return _apply_common_layout(fig, "Level 1 Book Imbalance", 300)


def make_returns_volatility_figure(prices_df: pd.DataFrame, window: int = 25) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)

    if not prices_df.empty and "mid_price" in prices_df.columns:
        prices_df = prices_df.sort_values("timestamp").copy()
        prices_df["mid_return"] = prices_df["mid_price"].pct_change()
        prices_df["rolling_vol"] = prices_df["mid_return"].rolling(window).std()

        fig.add_trace(go.Scatter(x=prices_df["timestamp"], y=prices_df["mid_return"], mode="lines", name="Mid Return"), row=1, col=1)
        fig.add_trace(go.Scatter(x=prices_df["timestamp"], y=prices_df["rolling_vol"], mode="lines", name=f"Rolling Vol ({window})"), row=2, col=1)

    fig.update_layout(
        title="Midprice Returns and Rolling Volatility",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend={"orientation": "h"},
    )
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Return", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", row=2, col=1)
    return fig


def make_book_heatmap_figure(prices_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if not prices_df.empty:
        prices_df = prices_df.sort_values("timestamp").copy()

        price_cols = [
            ("bid_price_3", "bid_volume_3"),
            ("bid_price_2", "bid_volume_2"),
            ("bid_price_1", "bid_volume_1"),
            ("ask_price_1", "ask_volume_1"),
            ("ask_price_2", "ask_volume_2"),
            ("ask_price_3", "ask_volume_3"),
        ]

        records = []
        for _, row in prices_df.iterrows():
            ts = row["timestamp"]
            for pcol, vcol in price_cols:
                price = row.get(pcol)
                vol = row.get(vcol)
                if pd.notna(price) and pd.notna(vol):
                    records.append((ts, price, vol))

        if records:
            heat_df = pd.DataFrame(records, columns=["timestamp", "price", "volume"])
            pivot = heat_df.pivot_table(index="price", columns="timestamp", values="volume", aggfunc="sum", fill_value=0)

            fig.add_trace(
                go.Heatmap(
                    x=pivot.columns,
                    y=pivot.index,
                    z=pivot.values,
                    colorbar={"title": "Volume"},
                )
            )

    fig.update_layout(
        title="Book Heatmap",
        template="plotly_white",
        height=500,
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Price Level")
    return fig


def make_cross_product_figure(prices_df: pd.DataFrame, selected_product: str, compare_products: list[str]) -> go.Figure:
    fig = go.Figure()

    if prices_df.empty:
        return _apply_common_layout(fig, "Cross-Product Midprice Comparison", 380)

    compare_set = [selected_product] + [p for p in compare_products if p != selected_product]

    for product in compare_set:
        dfp = prices_df[prices_df["product"] == product].sort_values("timestamp").copy()
        if dfp.empty:
            continue
        if "mid_price" in dfp.columns and dfp["mid_price"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=dfp["timestamp"],
                    y=dfp["mid_price"],
                    mode="lines",
                    name=product,
                )
            )

    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Mid Price")
    return _apply_common_layout(fig, "Cross-Product Midprice Comparison", 380)