from __future__ import annotations

import base64
import io
import math

import dash_ag_grid as dag
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate


MARKET_REQUIRED_COLUMNS = [
    "timestamp",
    "product",
    "mid_price",
    "bid_price_1",
    "ask_price_1",
]

REALIZED_REQUIRED_COLUMNS = [
    "product",
    "direction",
    "quantity",
    "entry_timestamp",
    "exit_timestamp",
    "hold_ticks",
    "entry_price",
    "exit_price",
    "pnl",
    "return_pct",
    "outcome",
    "entry_mid_price",
    "entry_spread",
    "exit_mid_price",
    "exit_spread",
]

MARKET_RENAME_MAP = {
    "symbol": "product",
    "Symbol": "product",
    "PRODUCT": "product",
    "Product": "product",
    "Timestamp": "timestamp",
    "TimeStamp": "timestamp",
    "MID_PRICE": "mid_price",
    "MidPrice": "mid_price",
    "BID_PRICE_1": "bid_price_1",
    "ASK_PRICE_1": "ask_price_1",
    "BID_PRICE_2": "bid_price_2",
    "ASK_PRICE_2": "ask_price_2",
    "BID_PRICE_3": "bid_price_3",
    "ASK_PRICE_3": "ask_price_3",
    "BID_VOLUME_1": "bid_volume_1",
    "ASK_VOLUME_1": "ask_volume_1",
    "BID_VOLUME_2": "bid_volume_2",
    "ASK_VOLUME_2": "ask_volume_2",
    "BID_VOLUME_3": "bid_volume_3",
    "ASK_VOLUME_3": "ask_volume_3",
}


def build_market_microstructure_layout():
    return html.Div(
        [
            dcc.Store(id="micro-market-store", data=[]),
            dcc.Store(id="micro-trades-store", data=[]),
            html.H3("Market / Microstructure"),
            html.P(
                "Upload market-state CSV and optionally realized_trades.csv to analyze spread, depth, imbalance, execution patterns, and single-name behavior."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Market-state CSV"),
                            dcc.Upload(
                                id="micro-market-upload",
                                children=html.Div(
                                    [
                                        "Drop market-state CSV here or ",
                                        html.Span("click to select", style={"fontWeight": "bold"}),
                                    ]
                                ),
                                multiple=False,
                                style=_upload_style(),
                            ),
                            html.Div(id="micro-market-upload-status", style={"marginTop": "8px"}),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Label("Realized trades CSV"),
                            dcc.Upload(
                                id="micro-trades-upload",
                                children=html.Div(
                                    [
                                        "Drop realized_trades.csv here or ",
                                        html.Span("click to select", style={"fontWeight": "bold"}),
                                    ]
                                ),
                                multiple=False,
                                style=_upload_style(),
                            ),
                            html.Div(id="micro-trades-upload-status", style={"marginTop": "8px"}),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "flexWrap": "wrap",
                    "marginBottom": "20px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Product"),
                            dcc.Dropdown(
                                id="micro-product-filter",
                                options=[],
                                value=None,
                                multi=False,
                                placeholder="Select a product",
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                    html.Div(
                        [
                            html.Label("Time bucket"),
                            dcc.Dropdown(
                                id="micro-time-bucket",
                                options=[
                                    {"label": "100 ticks", "value": 100},
                                    {"label": "250 ticks", "value": 250},
                                    {"label": "500 ticks", "value": 500},
                                    {"label": "1000 ticks", "value": 1000},
                                    {"label": "2500 ticks", "value": 2500},
                                    {"label": "5000 ticks", "value": 5000},
                                ],
                                value=1000,
                                multi=False,
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "flexWrap": "wrap",
                    "marginBottom": "20px",
                },
            ),
            html.Div(id="micro-summary-cards", style={"marginBottom": "20px"}),
            html.Div(id="micro-product-detail-summary", style={"marginBottom": "20px"}),
            dcc.Graph(id="micro-mid-price-graph"),
            dcc.Graph(id="micro-bid-ask-graph"),
            dcc.Graph(id="micro-spread-time-graph"),
            dcc.Graph(id="micro-depth-time-graph"),
            dcc.Graph(id="micro-imbalance-time-graph"),
            dcc.Graph(id="micro-spread-histogram-graph"),
            dcc.Graph(id="micro-imbalance-histogram-graph"),
            html.H3("Execution / Trade Flow"),
            dcc.Graph(id="micro-trade-count-graph"),
            dcc.Graph(id="micro-traded-quantity-graph"),
            dcc.Graph(id="micro-direction-count-graph"),
            dcc.Graph(id="micro-entry-vs-mid-graph"),
            dcc.Graph(id="micro-exit-vs-mid-graph"),
            html.Div(
                [
                    html.H4("Selected Product Rows"),
                    dag.AgGrid(
                        id="micro-market-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={"pagination": True, "paginationPageSize": 25},
                        style={"height": "500px", "width": "100%"},
                        className="ag-theme-alpine",
                    ),
                ]
            ),
        ]
    )


def register_market_microstructure_callbacks(app):
    @app.callback(
        Output("micro-market-upload-status", "children"),
        Output("micro-market-store", "data"),
        Output("micro-product-filter", "options"),
        Output("micro-product-filter", "value"),
        Input("micro-market-upload", "contents"),
        State("micro-market-upload", "filename"),
        prevent_initial_call=True,
    )
    def handle_market_upload(contents, filename):
        if not contents or not filename:
            raise PreventUpdate

        try:
            df = _decode_csv_upload(contents)
            df = _normalize_market_df(df)

            missing = [c for c in MARKET_REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                return _error_box(
                    f"Missing required market-state columns: {', '.join(missing)}"
                ), [], [], None

            df = _prepare_market_df(df)

            products = sorted(df["product"].dropna().astype(str).unique().tolist())
            product_options = [{"label": p, "value": p} for p in products]
            default_value = products[0] if products else None

            status = html.Div(
                [
                    html.Strong("Market-state upload complete. "),
                    html.Span(f"Loaded {len(df):,} rows from {filename}."),
                ],
                style=_success_style(),
            )

            return status, df.to_dict("records"), product_options, default_value
        except Exception as exc:
            return _error_box(str(exc)), [], [], None

    @app.callback(
        Output("micro-trades-upload-status", "children"),
        Output("micro-trades-store", "data"),
        Input("micro-trades-upload", "contents"),
        State("micro-trades-upload", "filename"),
        prevent_initial_call=True,
    )
    def handle_trades_upload(contents, filename):
        if not contents or not filename:
            raise PreventUpdate

        try:
            df = _decode_csv_upload(contents)
            missing = [c for c in REALIZED_REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                return _error_box(
                    f"Missing required realized-trades columns: {', '.join(missing)}"
                ), []

            df = _prepare_realized_trades_df(df)

            status = html.Div(
                [
                    html.Strong("Realized trades upload complete. "),
                    html.Span(f"Loaded {len(df):,} rows from {filename}."),
                ],
                style=_success_style(),
            )

            return status, df.to_dict("records")
        except Exception as exc:
            return _error_box(str(exc)), []

    @app.callback(
        Output("micro-summary-cards", "children"),
        Output("micro-product-detail-summary", "children"),
        Output("micro-mid-price-graph", "figure"),
        Output("micro-bid-ask-graph", "figure"),
        Output("micro-spread-time-graph", "figure"),
        Output("micro-depth-time-graph", "figure"),
        Output("micro-imbalance-time-graph", "figure"),
        Output("micro-spread-histogram-graph", "figure"),
        Output("micro-imbalance-histogram-graph", "figure"),
        Output("micro-trade-count-graph", "figure"),
        Output("micro-traded-quantity-graph", "figure"),
        Output("micro-direction-count-graph", "figure"),
        Output("micro-entry-vs-mid-graph", "figure"),
        Output("micro-exit-vs-mid-graph", "figure"),
        Output("micro-market-grid", "columnDefs"),
        Output("micro-market-grid", "rowData"),
        Input("micro-market-store", "data"),
        Input("micro-trades-store", "data"),
        Input("micro-product-filter", "value"),
        Input("micro-time-bucket", "value"),
        prevent_initial_call=True,
    )
    def render_microstructure_dashboard(market_rows, trade_rows, selected_product, bucket_size):
        empty_fig = _empty_figure("Upload market-state CSV to begin.")

        if not market_rows:
            return (
                html.Div(),
                html.Div(),
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                _empty_figure("Upload realized_trades.csv to see execution charts."),
                _empty_figure("Upload realized_trades.csv to see execution charts."),
                _empty_figure("Upload realized_trades.csv to see execution charts."),
                _empty_figure("Upload realized_trades.csv to see execution charts."),
                _empty_figure("Upload realized_trades.csv to see execution charts."),
                [],
                [],
            )

        market_df = _prepare_market_df(pd.DataFrame(market_rows))
        if selected_product:
            market_df = market_df[market_df["product"] == selected_product].copy()

        trades_df = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame()
        if not trades_df.empty:
            trades_df = _prepare_realized_trades_df(trades_df)
            if selected_product:
                trades_df = trades_df[trades_df["product"] == selected_product].copy()

        if market_df.empty:
            no_data_fig = _empty_figure("No market rows match the selected product.")
            return (
                html.Div("No market rows match the selected product."),
                html.Div(),
                no_data_fig,
                no_data_fig,
                no_data_fig,
                no_data_fig,
                no_data_fig,
                no_data_fig,
                no_data_fig,
                _empty_figure("No trade rows match the selected product."),
                _empty_figure("No trade rows match the selected product."),
                _empty_figure("No trade rows match the selected product."),
                _empty_figure("No trade rows match the selected product."),
                _empty_figure("No trade rows match the selected product."),
                _market_table_columns(market_df),
                _prepare_market_grid_rows(market_df),
            )

        summary_cards = _safe_call(
            lambda: _build_market_summary_cards(market_df, trades_df),
            html.Div("Could not render summary cards."),
        )
        product_summary = _safe_call(
            lambda: _build_product_detail_summary(market_df, trades_df, selected_product),
            html.Div("Could not render selected product summary."),
        )

        mid_fig = _safe_fig(
            lambda: _make_mid_price_figure(market_df),
            "Could not render mid-price chart.",
        )
        bid_ask_fig = _safe_fig(
            lambda: _make_bid_ask_figure(market_df),
            "Could not render bid/ask chart.",
        )
        spread_time_fig = _safe_fig(
            lambda: _make_spread_time_figure(market_df),
            "Could not render spread-over-time chart.",
        )
        depth_time_fig = _safe_fig(
            lambda: _make_depth_time_figure(market_df),
            "Could not render depth-over-time chart.",
        )
        imbalance_time_fig = _safe_fig(
            lambda: _make_imbalance_time_figure(market_df),
            "Could not render imbalance-over-time chart.",
        )
        spread_hist_fig = _safe_fig(
            lambda: _make_spread_histogram_figure(market_df),
            "Could not render spread histogram.",
        )
        imbalance_hist_fig = _safe_fig(
            lambda: _make_imbalance_histogram_figure(market_df),
            "Could not render imbalance histogram.",
        )

        trade_count_fig = _safe_fig(
            lambda: _make_trade_count_figure(trades_df, bucket_size),
            "Could not render trade-count chart.",
        )
        traded_qty_fig = _safe_fig(
            lambda: _make_trade_quantity_figure(trades_df, bucket_size),
            "Could not render traded-quantity chart.",
        )
        direction_count_fig = _safe_fig(
            lambda: _make_direction_count_figure(trades_df, bucket_size),
            "Could not render direction-count chart.",
        )
        entry_vs_mid_fig = _safe_fig(
            lambda: _make_entry_vs_mid_figure(trades_df, bucket_size),
            "Could not render entry-vs-mid chart.",
        )
        exit_vs_mid_fig = _safe_fig(
            lambda: _make_exit_vs_mid_figure(trades_df, bucket_size),
            "Could not render exit-vs-mid chart.",
        )

        grid_cols = _safe_call(lambda: _market_table_columns(market_df), [])
        grid_rows = _safe_call(lambda: _prepare_market_grid_rows(market_df), [])

        return (
            summary_cards,
            product_summary,
            mid_fig,
            bid_ask_fig,
            spread_time_fig,
            depth_time_fig,
            imbalance_time_fig,
            spread_hist_fig,
            imbalance_hist_fig,
            trade_count_fig,
            traded_qty_fig,
            direction_count_fig,
            entry_vs_mid_fig,
            exit_vs_mid_fig,
            grid_cols,
            grid_rows,
        )


def _decode_csv_upload(contents: str) -> pd.DataFrame:
    _, content_string = contents.split(",", 1)
    decoded_text = base64.b64decode(content_string).decode("utf-8-sig", errors="replace")

    try:
        df = pd.read_csv(io.StringIO(decoded_text), sep=None, engine="python")
    except Exception:
        df = pd.read_csv(io.StringIO(decoded_text))

    if len(df.columns) == 1:
        first_col = str(df.columns[0])
        if ";" in first_col:
            df = pd.read_csv(io.StringIO(decoded_text), sep=";")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalize_market_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    rename = {k: v for k, v in MARKET_RENAME_MAP.items() if k in out.columns and v not in out.columns}
    if rename:
        out = out.rename(columns=rename)
    return out


def _prepare_market_df(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_market_df(df)

    numeric_cols = [
        "timestamp",
        "mid_price",
        "bid_price_1",
        "ask_price_1",
        "bid_price_2",
        "ask_price_2",
        "bid_price_3",
        "ask_price_3",
        "bid_volume_1",
        "ask_volume_1",
        "bid_volume_2",
        "ask_volume_2",
        "bid_volume_3",
        "ask_volume_3",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "product" in out.columns:
        out["product"] = out["product"].astype(str).str.strip()

    required_present = [c for c in MARKET_REQUIRED_COLUMNS if c in out.columns]
    out = out.dropna(subset=required_present).copy()

    out["spread"] = out["ask_price_1"] - out["bid_price_1"]

    bid_cols = [c for c in ["bid_volume_1", "bid_volume_2", "bid_volume_3"] if c in out.columns]
    ask_cols = [c for c in ["ask_volume_1", "ask_volume_2", "ask_volume_3"] if c in out.columns]

    out["total_bid_depth"] = out[bid_cols].fillna(0).sum(axis=1) if bid_cols else 0.0
    out["total_ask_depth"] = out[ask_cols].fillna(0).sum(axis=1) if ask_cols else 0.0

    denom = out["total_bid_depth"].fillna(0) + out["total_ask_depth"].fillna(0)
    denom = denom.replace(0, np.nan)
    out["imbalance_top3"] = (
        (out["total_bid_depth"].fillna(0) - out["total_ask_depth"].fillna(0)) / denom
    )

    out = out.sort_values(["product", "timestamp"]).reset_index(drop=True)
    return out


def _prepare_realized_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    numeric_cols = [
        "quantity",
        "entry_timestamp",
        "exit_timestamp",
        "hold_ticks",
        "entry_price",
        "exit_price",
        "pnl",
        "return_pct",
        "entry_mid_price",
        "entry_spread",
        "exit_mid_price",
        "exit_spread",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["product", "direction", "outcome"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    if "trade_id" not in out.columns:
        out["trade_id"] = np.arange(1, len(out) + 1)

    if "outcome" not in out.columns or out["outcome"].isna().all():
        if "pnl" in out.columns:
            out["outcome"] = np.where(
                out["pnl"] > 0, "win",
                np.where(out["pnl"] < 0, "loss", "breakeven")
            )

    if "outcome" in out.columns:
        lower_outcome = out["outcome"].astype(str).str.lower()
        out["win_flag"] = lower_outcome.isin(["win", "winner", "profit"]).astype(float)
    else:
        out["win_flag"] = (out["pnl"] > 0).astype(float)

    if {"entry_price", "entry_mid_price"}.issubset(out.columns):
        out["entry_vs_mid"] = out["entry_price"] - out["entry_mid_price"]
    else:
        out["entry_vs_mid"] = np.nan

    if {"exit_price", "exit_mid_price"}.issubset(out.columns):
        out["exit_vs_mid"] = out["exit_price"] - out["exit_mid_price"]
    else:
        out["exit_vs_mid"] = np.nan

    out["abs_pnl"] = pd.to_numeric(out.get("pnl", np.nan), errors="coerce").abs()

    return out


def _build_market_summary_cards(market_df: pd.DataFrame, trades_df: pd.DataFrame):
    total_rows = len(market_df)
    products = market_df["product"].nunique() if "product" in market_df.columns else 0
    avg_spread = pd.to_numeric(market_df["spread"], errors="coerce").mean() if "spread" in market_df.columns else np.nan
    median_spread = pd.to_numeric(market_df["spread"], errors="coerce").median() if "spread" in market_df.columns else np.nan
    avg_imb = pd.to_numeric(market_df["imbalance_top3"], errors="coerce").mean() if "imbalance_top3" in market_df.columns else np.nan
    avg_bid_depth = pd.to_numeric(market_df["total_bid_depth"], errors="coerce").mean() if "total_bid_depth" in market_df.columns else np.nan
    avg_ask_depth = pd.to_numeric(market_df["total_ask_depth"], errors="coerce").mean() if "total_ask_depth" in market_df.columns else np.nan
    total_trades = len(trades_df) if not trades_df.empty else 0
    total_pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").sum() if not trades_df.empty and "pnl" in trades_df.columns else np.nan

    cards = [
        ("Market Rows", f"{total_rows:,}"),
        ("Products", f"{products:,}"),
        ("Avg Spread", _fmt_number(avg_spread)),
        ("Median Spread", _fmt_number(median_spread)),
        ("Avg Imbalance", _fmt_number(avg_imb)),
        ("Avg Bid Depth", _fmt_number(avg_bid_depth)),
        ("Avg Ask Depth", _fmt_number(avg_ask_depth)),
        ("Trade Rows", f"{total_trades:,}"),
        ("Trade PnL", _fmt_number(total_pnl)),
    ]

    return html.Div(
        [html.Div([html.H4(label), html.P(value)], style=_card_style()) for label, value in cards],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
            "gap": "16px",
        },
    )


def _build_product_detail_summary(market_df: pd.DataFrame, trades_df: pd.DataFrame, selected_product):
    label = selected_product if selected_product else "All Products"

    rows = [
        f"Selected Product: {label}",
        f"Market Rows: {len(market_df):,}",
        f"Time Range: {_fmt_number(market_df['timestamp'].min(), 0)} to {_fmt_number(market_df['timestamp'].max(), 0)}",
        f"Mid Price Range: {_fmt_number(market_df['mid_price'].min())} to {_fmt_number(market_df['mid_price'].max())}",
        f"Average Mid Price: {_fmt_number(market_df['mid_price'].mean())}",
        f"Average Spread: {_fmt_number(market_df['spread'].mean())}",
        f"Average Bid Depth: {_fmt_number(market_df['total_bid_depth'].mean())}",
        f"Average Ask Depth: {_fmt_number(market_df['total_ask_depth'].mean())}",
        f"Average Imbalance: {_fmt_number(market_df['imbalance_top3'].mean())}",
    ]

    if not trades_df.empty:
        rows.extend(
            [
                f"Trade Count: {len(trades_df):,}",
                f"Average PnL: {_fmt_number(pd.to_numeric(trades_df['pnl'], errors='coerce').mean())}",
                f"Win Rate: {_fmt_pct(100.0 * pd.to_numeric(trades_df['win_flag'], errors='coerce').mean())}",
                f"Average Hold Ticks: {_fmt_number(pd.to_numeric(trades_df['hold_ticks'], errors='coerce').mean())}",
                f"Average Entry vs Mid: {_fmt_number(pd.to_numeric(trades_df['entry_vs_mid'], errors='coerce').mean())}",
                f"Average Exit vs Mid: {_fmt_number(pd.to_numeric(trades_df['exit_vs_mid'], errors='coerce').mean())}",
            ]
        )
    else:
        rows.append("Trade Count: 0")

    return html.Div(
        [
            html.H3("Single-Name Detail Panel"),
            html.Div(
                [html.Div([html.P(r) for r in rows], style=_card_style())],
                style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"},
            ),
        ]
    )


def _make_mid_price_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["mid_price"], mode="lines", name="Mid Price"))
    fig.update_layout(
        title="Mid Price Over Time",
        template="plotly_white",
        height=380,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Mid Price")
    return fig


def _make_bid_ask_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bid_price_1"], mode="lines", name="Best Bid"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ask_price_1"], mode="lines", name="Best Ask"))
    if "mid_price" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["mid_price"], mode="lines", name="Mid Price"))
    fig.update_layout(
        title="Best Bid / Ask Over Time",
        template="plotly_white",
        height=380,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Price")
    return fig


def _make_spread_time_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["spread"], mode="lines", name="Spread"))
    fig.update_layout(
        title="Spread Over Time",
        template="plotly_white",
        height=360,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Spread")
    return fig


def _make_depth_time_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["total_bid_depth"], mode="lines", name="Total Bid Depth"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["total_ask_depth"], mode="lines", name="Total Ask Depth"))
    fig.update_layout(
        title="Bid vs Ask Depth Over Time",
        template="plotly_white",
        height=360,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Depth")
    return fig


def _make_imbalance_time_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["imbalance_top3"], mode="lines", name="Imbalance"))
    fig.update_layout(
        title="Imbalance Over Time",
        template="plotly_white",
        height=360,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Imbalance")
    return fig


def _make_spread_histogram_figure(df: pd.DataFrame) -> go.Figure:
    work = df.dropna(subset=["spread"]).copy()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=work["spread"], nbinsx=40, name="Spread"))
    fig.update_layout(
        title="Spread Distribution",
        template="plotly_white",
        height=360,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Spread")
    fig.update_yaxes(title_text="Count")
    return fig


def _make_imbalance_histogram_figure(df: pd.DataFrame) -> go.Figure:
    work = df.dropna(subset=["imbalance_top3"]).copy()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=work["imbalance_top3"], nbinsx=40, name="Imbalance"))
    fig.update_layout(
        title="Imbalance Distribution",
        template="plotly_white",
        height=360,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Imbalance")
    fig.update_yaxes(title_text="Count")
    return fig


def _bucket_trades(trades_df: pd.DataFrame, bucket_size: int) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()

    work = trades_df.copy()
    work["entry_timestamp"] = pd.to_numeric(work["entry_timestamp"], errors="coerce")
    work = work.dropna(subset=["entry_timestamp"]).copy()

    if work.empty:
        return work

    bucket = max(int(bucket_size or 1000), 1)
    work["time_bucket"] = (np.floor(work["entry_timestamp"] / bucket) * bucket).astype(int)
    return work


def _make_trade_count_figure(trades_df: pd.DataFrame, bucket_size: int) -> go.Figure:
    if trades_df.empty:
        return _empty_figure("Upload realized_trades.csv to see trade count by time bucket.")

    work = _bucket_trades(trades_df, bucket_size)
    if work.empty:
        return _empty_figure("No valid trade timestamps available.")

    grouped = work.groupby("time_bucket", as_index=False).size()
    grouped = grouped.rename(columns={"size": "trade_count"})

    fig = go.Figure()
    fig.add_trace(go.Bar(x=grouped["time_bucket"], y=grouped["trade_count"], name="Trade Count"))
    fig.update_layout(
        title="Trade Count by Time Bucket",
        template="plotly_white",
        height=360,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Time Bucket")
    fig.update_yaxes(title_text="Trade Count")
    return fig


def _make_trade_quantity_figure(trades_df: pd.DataFrame, bucket_size: int) -> go.Figure:
    if trades_df.empty:
        return _empty_figure("Upload realized_trades.csv to see traded quantity by time bucket.")

    work = _bucket_trades(trades_df, bucket_size)
    if work.empty or "quantity" not in work.columns:
        return _empty_figure("No valid traded quantity data available.")

    work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce")
    grouped = work.groupby("time_bucket", as_index=False)["quantity"].sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=grouped["time_bucket"], y=grouped["quantity"], name="Traded Quantity"))
    fig.update_layout(
        title="Traded Quantity by Time Bucket",
        template="plotly_white",
        height=360,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Time Bucket")
    fig.update_yaxes(title_text="Quantity")
    return fig


def _make_direction_count_figure(trades_df: pd.DataFrame, bucket_size: int) -> go.Figure:
    if trades_df.empty:
        return _empty_figure("Upload realized_trades.csv to see long vs short counts over time.")

    work = _bucket_trades(trades_df, bucket_size)
    if work.empty or "direction" not in work.columns:
        return _empty_figure("No valid trade direction data available.")

    grouped = work.groupby(["time_bucket", "direction"], as_index=False).size()
    grouped = grouped.rename(columns={"size": "trade_count"})

    fig = go.Figure()
    for direction, sub in grouped.groupby("direction"):
        fig.add_trace(go.Bar(x=sub["time_bucket"], y=sub["trade_count"], name=str(direction)))

    fig.update_layout(
        title="Long vs Short Trade Count by Time Bucket",
        template="plotly_white",
        height=380,
        barmode="group",
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Time Bucket")
    fig.update_yaxes(title_text="Trade Count")
    return fig


def _make_entry_vs_mid_figure(trades_df: pd.DataFrame, bucket_size: int) -> go.Figure:
    if trades_df.empty:
        return _empty_figure("Upload realized_trades.csv to see entry-vs-mid by time bucket.")

    work = _bucket_trades(trades_df, bucket_size)
    if work.empty or "entry_vs_mid" not in work.columns:
        return _empty_figure("No valid entry-vs-mid data available.")

    work["entry_vs_mid"] = pd.to_numeric(work["entry_vs_mid"], errors="coerce")
    grouped = work.groupby("time_bucket", as_index=False)["entry_vs_mid"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["time_bucket"], y=grouped["entry_vs_mid"], mode="lines+markers", name="Entry vs Mid"))
    fig.update_layout(
        title="Average Entry vs Mid by Time Bucket",
        template="plotly_white",
        height=360,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Time Bucket")
    fig.update_yaxes(title_text="Average Entry vs Mid")
    return fig


def _make_exit_vs_mid_figure(trades_df: pd.DataFrame, bucket_size: int) -> go.Figure:
    if trades_df.empty:
        return _empty_figure("Upload realized_trades.csv to see exit-vs-mid by time bucket.")

    work = _bucket_trades(trades_df, bucket_size)
    if work.empty or "exit_vs_mid" not in work.columns:
        return _empty_figure("No valid exit-vs-mid data available.")

    work["exit_vs_mid"] = pd.to_numeric(work["exit_vs_mid"], errors="coerce")
    grouped = work.groupby("time_bucket", as_index=False)["exit_vs_mid"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["time_bucket"], y=grouped["exit_vs_mid"], mode="lines+markers", name="Exit vs Mid"))
    fig.update_layout(
        title="Average Exit vs Mid by Time Bucket",
        template="plotly_white",
        height=360,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Time Bucket")
    fig.update_yaxes(title_text="Average Exit vs Mid")
    return fig


def _market_table_columns(df: pd.DataFrame) -> list[dict]:
    preferred = [
        "timestamp",
        "product",
        "bid_price_1",
        "ask_price_1",
        "mid_price",
        "spread",
        "bid_volume_1",
        "ask_volume_1",
        "bid_volume_2",
        "ask_volume_2",
        "bid_volume_3",
        "ask_volume_3",
        "total_bid_depth",
        "total_ask_depth",
        "imbalance_top3",
    ]
    cols = [c for c in preferred if c in df.columns]
    return [{"field": col, "headerName": col.replace("_", " ").title(), "flex": 1} for col in cols]


def _prepare_market_grid_rows(df: pd.DataFrame) -> list[dict]:
    preferred = [
        "timestamp",
        "product",
        "bid_price_1",
        "ask_price_1",
        "mid_price",
        "spread",
        "bid_volume_1",
        "ask_volume_1",
        "bid_volume_2",
        "ask_volume_2",
        "bid_volume_3",
        "ask_volume_3",
        "total_bid_depth",
        "total_ask_depth",
        "imbalance_top3",
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        return []

    out = df[cols].copy()
    if "timestamp" in out.columns:
        out = out.sort_values("timestamp", ascending=True)
    return out.to_dict("records")


def _safe_fig(fn, fallback_title: str) -> go.Figure:
    try:
        return fn()
    except Exception:
        return _empty_figure(fallback_title)


def _safe_call(fn, fallback):
    try:
        return fn()
    except Exception:
        return fallback


def _upload_style():
    return {
        "width": "100%",
        "height": "90px",
        "lineHeight": "90px",
        "borderWidth": "2px",
        "borderStyle": "dashed",
        "borderRadius": "12px",
        "textAlign": "center",
        "backgroundColor": "#f8f9fb",
    }


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=320,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    return fig


def _card_style() -> dict:
    return {
        "border": "1px solid #ddd",
        "borderRadius": "10px",
        "padding": "14px",
        "backgroundColor": "#fafafa",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
    }


def _success_style() -> dict:
    return {
        "padding": "10px 14px",
        "borderRadius": "10px",
        "backgroundColor": "#eef6ff",
        "border": "1px solid #cfe3ff",
    }


def _error_box(message: str):
    return html.Div(
        f"Upload failed: {message}",
        style={
            "padding": "10px 14px",
            "borderRadius": "10px",
            "backgroundColor": "#fff1f0",
            "border": "1px solid #ffccc7",
            "color": "#a8071a",
        },
    )


def _fmt_number(value, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def _fmt_pct(value, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{decimals}f}%"
