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


def build_research_layout():
    return html.Div(
        [
            dcc.Store(id="research-realized-trades-store", data=[]),
            dcc.Store(id="research-market-store", data=[]),
            dcc.Store(id="research-merged-store", data=[]),
            html.H3("Research"),
            html.P(
                "Upload realized_trades.csv from the Backtester tab, and optionally upload a market-state CSV with timestamp, product, bid/ask, and mid-price to merge z-scores onto each entry."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Realized trades CSV"),
                            dcc.Upload(
                                id="research-realized-trades-upload",
                                children=html.Div(
                                    [
                                        "Drop realized_trades.csv here or ",
                                        html.Span("click to select", style={"fontWeight": "bold"}),
                                    ]
                                ),
                                multiple=False,
                                style=_upload_style(),
                            ),
                            html.Div(id="research-upload-status", style={"marginTop": "8px"}),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Label("Market-state CSV"),
                            dcc.Upload(
                                id="research-market-upload",
                                children=html.Div(
                                    [
                                        "Drop market-state CSV here or ",
                                        html.Span("click to select", style={"fontWeight": "bold"}),
                                    ]
                                ),
                                multiple=False,
                                style=_upload_style(),
                            ),
                            html.Div(id="research-market-upload-status", style={"marginTop": "8px"}),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "20px"},
            ),
            html.Div(id="research-summary-cards", style={"marginBottom": "20px"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Product filter"),
                            dcc.Dropdown(
                                id="research-product-filter",
                                options=[],
                                value=[],
                                multi=True,
                                placeholder="Optional: filter by product",
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                    html.Div(
                        [
                            html.Label("Direction filter"),
                            dcc.Dropdown(
                                id="research-direction-filter",
                                options=[],
                                value=[],
                                multi=True,
                                placeholder="Optional: filter by direction",
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                    html.Div(
                        [
                            html.Label("Outcome filter"),
                            dcc.Dropdown(
                                id="research-outcome-filter",
                                options=[],
                                value=[],
                                multi=True,
                                placeholder="Optional: filter by outcome",
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "flexWrap": "wrap",
                    "marginBottom": "20px",
                },
            ),
            dcc.Graph(id="research-pnl-by-outcome-graph"),
            dcc.Graph(id="research-hold-vs-pnl-graph"),
            dcc.Graph(id="research-winrate-heatmap-graph"),
            dcc.Graph(id="research-pnl-heatmap-graph"),
            html.Div(id="research-winner-loser-summary", style={"marginBottom": "20px"}),
            html.H3("Research V2: Entry Signal Diagnostics"),
            dcc.Graph(id="research-pnl-by-imbalance-z-graph"),
            dcc.Graph(id="research-winrate-z-heatmap-graph"),
            dcc.Graph(id="research-entry-z-boxplot-graph"),
            dcc.Graph(id="research-entry-z-scatter-graph"),
            html.Div(
                [
                    html.H4("Merged Trade Diagnostics"),
                    dag.AgGrid(
                        id="research-trades-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={"pagination": True, "paginationPageSize": 25},
                        style={"height": "560px", "width": "100%"},
                        className="ag-theme-alpine",
                    ),
                ]
            ),
        ]
    )


def register_research_callbacks(app):
    @app.callback(
        Output("research-upload-status", "children"),
        Output("research-realized-trades-store", "data"),
        Output("research-product-filter", "options"),
        Output("research-direction-filter", "options"),
        Output("research-outcome-filter", "options"),
        Input("research-realized-trades-upload", "contents"),
        State("research-realized-trades-upload", "filename"),
        prevent_initial_call=True,
    )
    def handle_realized_trades_upload(contents, filename):
        if not contents or not filename:
            raise PreventUpdate

        try:
            df = _decode_csv_upload(contents)
            missing = [c for c in REALIZED_REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                return _error_box(
                    f"Missing required realized-trades columns: {', '.join(missing)}"
                ), [], [], [], []

            df = _prepare_realized_trades_df(df)

            product_options = [
                {"label": p, "value": p}
                for p in sorted(df["product"].dropna().astype(str).unique().tolist())
            ]
            direction_options = [
                {"label": d, "value": d}
                for d in sorted(df["direction"].dropna().astype(str).unique().tolist())
            ]
            outcome_options = [
                {"label": o, "value": o}
                for o in sorted(df["outcome"].dropna().astype(str).unique().tolist())
            ]

            status = html.Div(
                [
                    html.Strong("Realized trades upload complete. "),
                    html.Span(f"Loaded {len(df):,} rows from {filename}."),
                ],
                style=_success_style(),
            )

            return status, df.to_dict("records"), product_options, direction_options, outcome_options
        except Exception as exc:
            return _error_box(str(exc)), [], [], [], []

    @app.callback(
        Output("research-market-upload-status", "children"),
        Output("research-market-store", "data"),
        Input("research-market-upload", "contents"),
        State("research-market-upload", "filename"),
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
                ), []

            df = _prepare_market_df(df)

            status = html.Div(
                [
                    html.Strong("Market-state upload complete. "),
                    html.Span(f"Loaded {len(df):,} rows from {filename}."),
                ],
                style=_success_style(),
            )

            return status, df.to_dict("records")
        except Exception as exc:
            return _error_box(str(exc)), []

    @app.callback(
        Output("research-merged-store", "data"),
        Input("research-realized-trades-store", "data"),
        Input("research-market-store", "data"),
        prevent_initial_call=True,
    )
    def build_merged_dataset(realized_rows, market_rows):
        if not realized_rows:
            return []

        trades_df = _prepare_realized_trades_df(pd.DataFrame(realized_rows))

        if not market_rows:
            return trades_df.to_dict("records")

        try:
            market_df = _prepare_market_df(pd.DataFrame(market_rows))
            merged = _merge_trades_with_market(trades_df, market_df)
            merged = _prepare_realized_trades_df(merged)
            return merged.to_dict("records")
        except Exception:
            return trades_df.to_dict("records")

    @app.callback(
        Output("research-summary-cards", "children"),
        Output("research-pnl-by-outcome-graph", "figure"),
        Output("research-hold-vs-pnl-graph", "figure"),
        Output("research-winrate-heatmap-graph", "figure"),
        Output("research-pnl-heatmap-graph", "figure"),
        Output("research-winner-loser-summary", "children"),
        Output("research-pnl-by-imbalance-z-graph", "figure"),
        Output("research-winrate-z-heatmap-graph", "figure"),
        Output("research-entry-z-boxplot-graph", "figure"),
        Output("research-entry-z-scatter-graph", "figure"),
        Output("research-trades-grid", "columnDefs"),
        Output("research-trades-grid", "rowData"),
        Input("research-merged-store", "data"),
        Input("research-product-filter", "value"),
        Input("research-direction-filter", "value"),
        Input("research-outcome-filter", "value"),
        prevent_initial_call=True,
    )
    def render_research_dashboard(store_rows, selected_products, selected_directions, selected_outcomes):
        if not store_rows:
            empty_fig = _empty_figure("Upload realized_trades.csv to begin.")
            return html.Div(), empty_fig, empty_fig, empty_fig, empty_fig, html.Div(), empty_fig, empty_fig, empty_fig, empty_fig, [], []

        df = _prepare_realized_trades_df(pd.DataFrame(store_rows))
        df = _filter_research_df(df, selected_products, selected_directions, selected_outcomes)

        if df.empty:
            empty_fig = _empty_figure("No trades match the current filters.")
            return (
                html.Div("No trades match the current filters."),
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                html.Div(),
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                _research_table_columns(df),
                [],
            )

        summary_cards = _safe_call(
            lambda: _build_summary_cards(df),
            html.Div("Could not render summary cards."),
        )
        pnl_by_outcome_fig = _safe_fig(
            lambda: _make_pnl_by_outcome_figure(df),
            "Could not render PnL by outcome.",
        )
        hold_vs_pnl_fig = _safe_fig(
            lambda: _make_hold_vs_pnl_figure(df),
            "Could not render hold time vs PnL.",
        )
        winrate_heatmap_fig = _safe_fig(
            lambda: _make_heatmap_figure(
                df,
                value_col="win_flag",
                title="Win Rate by Hold Time × Entry Spread",
                value_title="Win Rate",
            ),
            "Could not render win-rate heatmap.",
        )
        pnl_heatmap_fig = _safe_fig(
            lambda: _make_heatmap_figure(
                df,
                value_col="pnl",
                title="Average PnL by Hold Time × Entry Spread",
                value_title="Avg PnL",
            ),
            "Could not render PnL heatmap.",
        )
        winner_loser_summary = _safe_call(
            lambda: _build_winner_loser_summary(df),
            html.Div("Could not render winner vs loser summary."),
        )
        pnl_by_imbalance_z_fig = _safe_fig(
            lambda: _make_pnl_by_imbalance_z_figure(df),
            "Could not render imbalance-z chart.",
        )
        winrate_z_heatmap_fig = _safe_fig(
            lambda: _make_z_heatmap_figure(df),
            "Could not render z-score heatmap.",
        )
        entry_z_boxplot_fig = _safe_fig(
            lambda: _make_entry_z_boxplot_figure(df),
            "Could not render entry z-score boxplot.",
        )
        entry_z_scatter_fig = _safe_fig(
            lambda: _make_entry_z_scatter_figure(df),
            "Could not render entry z-score scatter.",
        )
        grid_cols = _safe_call(lambda: _research_table_columns(df), [])
        grid_rows = _safe_call(lambda: _prepare_grid_rows(df), [])

        return (
            summary_cards,
            pnl_by_outcome_fig,
            hold_vs_pnl_fig,
            winrate_heatmap_fig,
            pnl_heatmap_fig,
            winner_loser_summary,
            pnl_by_imbalance_z_fig,
            winrate_z_heatmap_fig,
            entry_z_boxplot_fig,
            entry_z_scatter_fig,
            grid_cols,
            grid_rows,
        )


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

MARKET_REQUIRED_COLUMNS = [
    "timestamp",
    "product",
    "mid_price",
    "bid_price_1",
    "ask_price_1",
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
}


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
        "imbalance_top3",
        "imbalance_z",
        "spread_z",
        "mid_z",
        "market_timestamp_at_entry",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Guarantee these columns exist so downstream filters never KeyError
    for col in ["product", "direction", "outcome"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = out[col].astype("string").str.strip()

    if "trade_id" not in out.columns:
        out["trade_id"] = np.arange(1, len(out) + 1)

    if "win_flag" not in out.columns and "pnl" in out.columns:
        pnl_num = pd.to_numeric(out["pnl"], errors="coerce")
        out["win_flag"] = (pnl_num > 0).astype(float)

    return out

def _prepare_market_df(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_market_df(df)

    numeric_cols = [
        "timestamp",
        "mid_price",
        "bid_price_1",
        "ask_price_1",
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

    if {"ask_price_1", "bid_price_1"}.issubset(out.columns):
        out["spread"] = out["ask_price_1"] - out["bid_price_1"]

    bid_cols = [c for c in ["bid_volume_1", "bid_volume_2", "bid_volume_3"] if c in out.columns]
    ask_cols = [c for c in ["ask_volume_1", "ask_volume_2", "ask_volume_3"] if c in out.columns]

    if bid_cols:
        out["total_bid_depth"] = out[bid_cols].fillna(0).sum(axis=1)
    else:
        out["total_bid_depth"] = 0.0

    if ask_cols:
        out["total_ask_depth"] = out[ask_cols].fillna(0).sum(axis=1)
    else:
        out["total_ask_depth"] = 0.0

    denom = out["total_bid_depth"].fillna(0) + out["total_ask_depth"].fillna(0)
    denom = denom.replace(0, np.nan)
    out["imbalance_top3"] = (
        (out["total_bid_depth"].fillna(0) - out["total_ask_depth"].fillna(0)) / denom
    )

    out = out.sort_values(["product", "timestamp"]).copy()

    grouped = []
    for _, g in out.groupby("product", dropna=False):
        gg = g.copy()

        if "mid_price" in gg.columns:
            gg["mid_return"] = gg["mid_price"].pct_change()
            gg["mid_z"] = _rolling_zscore(gg["mid_price"], window=50)

        if "spread" in gg.columns:
            gg["spread_z"] = _rolling_zscore(gg["spread"], window=50)

        if "imbalance_top3" in gg.columns:
            gg["imbalance_z"] = _rolling_zscore(gg["imbalance_top3"], window=50)

        grouped.append(gg)

    out = pd.concat(grouped, ignore_index=True) if grouped else out
    return out


def _rolling_zscore(series: pd.Series, window: int = 50) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.rolling(window=window, min_periods=max(10, window // 5)).mean()
    std = s.rolling(window=window, min_periods=max(10, window // 5)).std()
    std = std.replace(0, np.nan)
    return (s - mean) / std


def _merge_trades_with_market(trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or market_df.empty:
        return trades_df.copy()

    left = trades_df.copy()
    right = market_df.copy()

    left["entry_timestamp"] = pd.to_numeric(left["entry_timestamp"], errors="coerce")
    right["timestamp"] = pd.to_numeric(right["timestamp"], errors="coerce")

    left["product"] = left["product"].astype(str).str.strip()
    right["product"] = right["product"].astype(str).str.strip()

    left = left.dropna(subset=["entry_timestamp", "product"]).copy()
    right = right.dropna(subset=["timestamp", "product"]).copy()

    left["entry_timestamp"] = left["entry_timestamp"].astype(float)
    right["timestamp"] = right["timestamp"].astype(float)

    keep_cols = [
        "product",
        "timestamp",
        "mid_price",
        "spread",
        "imbalance_top3",
        "mid_z",
        "spread_z",
        "imbalance_z",
    ]
    keep_cols = [c for c in keep_cols if c in right.columns]
    right = right[keep_cols].copy()

    merged_parts = []
    common_products = sorted(set(left["product"]).intersection(set(right["product"])))

    for product in common_products:
        lsub = left[left["product"] == product].sort_values("entry_timestamp").copy()
        rsub = right[right["product"] == product].sort_values("timestamp").copy()

        if lsub.empty or rsub.empty:
            continue

        merged_sub = pd.merge_asof(
            lsub,
            rsub,
            left_on="entry_timestamp",
            right_on="timestamp",
            direction="backward",
        )
        merged_parts.append(merged_sub)

    remaining_products = set(left["product"]) - set(common_products)
    if remaining_products:
        merged_parts.append(left[left["product"].isin(remaining_products)].copy())

    if not merged_parts:
        return trades_df.copy()

    merged = pd.concat(merged_parts, ignore_index=True)

    if "timestamp" in merged.columns:
        merged = merged.rename(columns={"timestamp": "market_timestamp_at_entry"})

    return merged



def _filter_research_df(df: pd.DataFrame, selected_products, selected_directions, selected_outcomes) -> pd.DataFrame:
    out = df.copy()

    if selected_products and "product" in out.columns:
        out = out[out["product"].isin(selected_products)]

    if selected_directions and "direction" in out.columns:
        out = out[out["direction"].isin(selected_directions)]

    if selected_outcomes and "outcome" in out.columns:
        out = out[out["outcome"].isin(selected_outcomes)]

    return out

def _build_summary_cards(df: pd.DataFrame):
    total_trades = len(df)
    total_pnl = pd.to_numeric(df["pnl"], errors="coerce").sum()
    avg_pnl = pd.to_numeric(df["pnl"], errors="coerce").mean()
    median_pnl = pd.to_numeric(df["pnl"], errors="coerce").median()
    win_rate = 100.0 * pd.to_numeric(df["win_flag"], errors="coerce").mean() if len(df) else np.nan
    avg_hold = pd.to_numeric(df["hold_ticks"], errors="coerce").mean()
    avg_entry_spread = pd.to_numeric(df["entry_spread"], errors="coerce").mean()
    avg_exit_spread = pd.to_numeric(df["exit_spread"], errors="coerce").mean()
    z_coverage = 100.0 * df["imbalance_z"].notna().mean() if "imbalance_z" in df.columns else 0.0

    cards = [
        ("Trades", f"{total_trades:,}"),
        ("Total PnL", _fmt_number(total_pnl)),
        ("Average PnL", _fmt_number(avg_pnl)),
        ("Median PnL", _fmt_number(median_pnl)),
        ("Win Rate", _fmt_pct(win_rate)),
        ("Average Hold Ticks", _fmt_number(avg_hold)),
        ("Average Entry Spread", _fmt_number(avg_entry_spread)),
        ("Average Exit Spread", _fmt_number(avg_exit_spread)),
        ("Z-Score Coverage", _fmt_pct(z_coverage)),
    ]

    return html.Div(
        [html.Div([html.H4(label), html.P(value)], style=_card_style()) for label, value in cards],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
            "gap": "16px",
        },
    )


def _build_winner_loser_summary(df: pd.DataFrame):
    winners = df[df["win_flag"] == 1].copy()
    losers = df[df["win_flag"] == 0].copy()

    def summarize(frame: pd.DataFrame, label: str):
        if frame.empty:
            return html.Div([html.H4(label), html.P("No trades")], style=_card_style())

        rows = [
            f"Trades: {len(frame):,}",
            f"Avg PnL: {_fmt_number(pd.to_numeric(frame['pnl'], errors='coerce').mean())}",
            f"Avg Hold: {_fmt_number(pd.to_numeric(frame['hold_ticks'], errors='coerce').mean())}",
            f"Avg Entry Spread: {_fmt_number(pd.to_numeric(frame['entry_spread'], errors='coerce').mean())}",
            f"Avg Exit Spread: {_fmt_number(pd.to_numeric(frame['exit_spread'], errors='coerce').mean())}",
            f"Avg Entry vs Mid: {_fmt_number(pd.to_numeric(frame['entry_vs_mid'], errors='coerce').mean())}",
            f"Avg Exit vs Mid: {_fmt_number(pd.to_numeric(frame['exit_vs_mid'], errors='coerce').mean())}",
        ]

        if "imbalance_z" in frame.columns:
            rows.append(f"Avg Imbalance Z: {_fmt_number(pd.to_numeric(frame['imbalance_z'], errors='coerce').mean())}")
        if "spread_z" in frame.columns:
            rows.append(f"Avg Spread Z: {_fmt_number(pd.to_numeric(frame['spread_z'], errors='coerce').mean())}")
        if "mid_z" in frame.columns:
            rows.append(f"Avg Mid Z: {_fmt_number(pd.to_numeric(frame['mid_z'], errors='coerce').mean())}")

        return html.Div(
            [html.H4(label)] + [html.P(r) for r in rows],
            style=_card_style(),
        )

    return html.Div(
        [
            html.H3("Winner vs Loser Comparison"),
            html.Div(
                [summarize(winners, "Winners"), summarize(losers, "Losers")],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                    "gap": "16px",
                },
            ),
        ]
    )


def _make_pnl_by_outcome_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()

    if "pnl" not in work.columns:
        return _empty_figure("Missing pnl column.")

    work["pnl"] = pd.to_numeric(work["pnl"], errors="coerce")

    if "outcome" not in work.columns or work["outcome"].isna().all():
        work["outcome"] = np.where(
            work["pnl"] > 0, "win",
            np.where(work["pnl"] < 0, "loss", "breakeven")
        )

    work["product"] = work["product"].astype(str)
    work["outcome"] = work["outcome"].astype(str)
    work = work.dropna(subset=["pnl", "product", "outcome"])

    if work.empty:
        return _empty_figure("No valid outcome data.")

    grouped = work.groupby(["product", "outcome"], as_index=False)["pnl"].mean().sort_values(["product", "outcome"])

    fig = go.Figure()
    for outcome, sub in grouped.groupby("outcome"):
        fig.add_trace(go.Bar(x=sub["product"], y=sub["pnl"], name=str(outcome)))

    fig.update_layout(
        title="Average PnL by Product and Outcome",
        template="plotly_white",
        height=380,
        barmode="group",
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Product")
    fig.update_yaxes(title_text="Average PnL")
    return fig


def _make_hold_vs_pnl_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()

    required = ["hold_ticks", "pnl", "direction"]
    for col in required:
        if col not in work.columns:
            return _empty_figure("Missing hold time or pnl data.")

    work["hold_ticks"] = pd.to_numeric(work["hold_ticks"], errors="coerce")
    work["pnl"] = pd.to_numeric(work["pnl"], errors="coerce")
    work["quantity"] = pd.to_numeric(work["quantity"], errors="coerce") if "quantity" in work.columns else np.nan
    work["entry_spread"] = pd.to_numeric(work["entry_spread"], errors="coerce") if "entry_spread" in work.columns else np.nan
    work["return_pct"] = pd.to_numeric(work["return_pct"], errors="coerce") if "return_pct" in work.columns else np.nan

    work = work.dropna(subset=["hold_ticks", "pnl", "direction"])
    if work.empty:
        return _empty_figure("No valid hold time vs pnl data.")

    fig = go.Figure()

    for direction, sub in work.groupby("direction"):
        custom = np.column_stack(
            [
                sub["product"].astype(str).to_numpy() if "product" in sub.columns else np.array([""] * len(sub)),
                sub["quantity"].fillna(np.nan).astype(str).to_numpy(),
                sub["entry_spread"].fillna(np.nan).astype(str).to_numpy(),
                sub["return_pct"].round(4).fillna(np.nan).astype(str).to_numpy(),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=sub["hold_ticks"],
                y=sub["pnl"],
                mode="markers",
                name=str(direction),
                customdata=custom,
                hovertemplate=(
                    "Hold=%{x}<br>"
                    "PnL=%{y}<br>"
                    "Product=%{customdata[0]}<br>"
                    "Qty=%{customdata[1]}<br>"
                    "Entry Spread=%{customdata[2]}<br>"
                    "Return %=%{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Hold Time vs PnL",
        template="plotly_white",
        height=420,
        hovermode="closest",
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Hold Ticks")
    fig.update_yaxes(title_text="PnL")
    return fig


def _make_heatmap_figure(df: pd.DataFrame, value_col: str, title: str, value_title: str) -> go.Figure:
    work = df.copy()

    for col in ["hold_ticks", "entry_spread", value_col]:
        if col not in work.columns:
            return _empty_figure(title)

    work["hold_ticks"] = pd.to_numeric(work["hold_ticks"], errors="coerce")
    work["entry_spread"] = pd.to_numeric(work["entry_spread"], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    work = work.dropna(subset=["hold_ticks", "entry_spread", value_col]).copy()
    if work.empty:
        return _empty_figure(title)

    try:
        work["hold_bucket"] = pd.qcut(work["hold_ticks"], q=6, duplicates="drop")
    except Exception:
        work["hold_bucket"] = pd.cut(work["hold_ticks"], bins=6)

    try:
        work["spread_bucket"] = pd.qcut(work["entry_spread"], q=6, duplicates="drop")
    except Exception:
        work["spread_bucket"] = pd.cut(work["entry_spread"], bins=6)

    heat = (
        work.groupby(["hold_bucket", "spread_bucket"], observed=False)[value_col]
        .mean()
        .reset_index()
        .pivot(index="spread_bucket", columns="hold_bucket", values=value_col)
    )

    if heat.empty:
        return _empty_figure(title)

    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=[str(c) for c in heat.columns],
            y=[str(i) for i in heat.index],
            colorbar={"title": value_title},
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Hold Time Bucket")
    fig.update_yaxes(title_text="Entry Spread Bucket")
    return fig


def _make_pnl_by_imbalance_z_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    if "imbalance_z" not in work.columns or "pnl" not in work.columns:
        return _empty_figure("Upload market-state data to see PnL by imbalance z-score.")

    work["imbalance_z"] = pd.to_numeric(work["imbalance_z"], errors="coerce")
    work["pnl"] = pd.to_numeric(work["pnl"], errors="coerce")
    work = work.dropna(subset=["imbalance_z", "pnl"]).copy()

    if len(work) < 20:
        return _empty_figure("Not enough merged z-score rows yet.")

    try:
        work["imb_bucket"] = pd.qcut(work["imbalance_z"], q=8, duplicates="drop")
    except Exception:
        work["imb_bucket"] = pd.cut(work["imbalance_z"], bins=8)

    grouped = work.groupby("imb_bucket", observed=False)["pnl"].mean().reset_index()

    if grouped.empty:
        return _empty_figure("Not enough merged z-score rows yet.")

    fig = go.Figure(
        data=[go.Bar(x=grouped["imb_bucket"].astype(str), y=grouped["pnl"], name="Avg PnL")]
    )
    fig.update_layout(
        title="Average PnL by Entry Imbalance Z-Score Bucket",
        template="plotly_white",
        height=380,
        margin={"l": 50, "r": 20, "t": 60, "b": 80},
    )
    fig.update_xaxes(title_text="Imbalance Z Bucket")
    fig.update_yaxes(title_text="Average PnL")
    return fig


def _make_z_heatmap_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    needed = ["imbalance_z", "spread_z", "win_flag"]
    if not all(c in work.columns for c in needed):
        return _empty_figure("Upload market-state data to see win rate by spread z-score × imbalance z-score.")

    for col in needed:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=needed).copy()
    if len(work) < 20:
        return _empty_figure("Not enough merged z-score rows yet.")

    try:
        work["imb_bucket"] = pd.qcut(work["imbalance_z"], q=6, duplicates="drop")
    except Exception:
        work["imb_bucket"] = pd.cut(work["imbalance_z"], bins=6)

    try:
        work["spr_bucket"] = pd.qcut(work["spread_z"], q=6, duplicates="drop")
    except Exception:
        work["spr_bucket"] = pd.cut(work["spread_z"], bins=6)

    heat = (
        work.groupby(["spr_bucket", "imb_bucket"], observed=False)["win_flag"]
        .mean()
        .reset_index()
        .pivot(index="spr_bucket", columns="imb_bucket", values="win_flag")
    )

    if heat.empty:
        return _empty_figure("Not enough merged z-score rows yet.")

    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values,
            x=[str(c) for c in heat.columns],
            y=[str(i) for i in heat.index],
            colorbar={"title": "Win Rate"},
        )
    )
    fig.update_layout(
        title="Win Rate by Spread Z-Score × Imbalance Z-Score",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 80},
    )
    fig.update_xaxes(title_text="Imbalance Z Bucket")
    fig.update_yaxes(title_text="Spread Z Bucket")
    return fig


def _make_entry_z_boxplot_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    if "imbalance_z" not in work.columns:
        return _empty_figure("Upload market-state data to compare entry z-scores for winners vs losers.")

    work["imbalance_z"] = pd.to_numeric(work["imbalance_z"], errors="coerce")
    if "spread_z" in work.columns:
        work["spread_z"] = pd.to_numeric(work["spread_z"], errors="coerce")

    if "outcome" not in work.columns or work["outcome"].isna().all():
        if "pnl" in work.columns:
            pnl_numeric = pd.to_numeric(work["pnl"], errors="coerce")
            work["outcome"] = np.where(
                pnl_numeric > 0, "win",
                np.where(pnl_numeric < 0, "loss", "breakeven")
            )

    work = work.dropna(subset=["imbalance_z"]).copy()
    if len(work) < 20:
        return _empty_figure("Not enough merged z-score rows yet.")

    fig = go.Figure()
    for outcome, sub in work.groupby("outcome"):
        fig.add_trace(go.Box(y=sub["imbalance_z"], name=f"{outcome} | imbalance_z", boxmean=True))

    if "spread_z" in work.columns and work["spread_z"].notna().any():
        for outcome, sub in work.groupby("outcome"):
            fig.add_trace(go.Box(y=sub["spread_z"], name=f"{outcome} | spread_z", boxmean=True))

    fig.update_layout(
        title="Entry Z-Score Distribution by Outcome",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_yaxes(title_text="Z-Score")
    return fig


def _make_entry_z_scatter_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()

    if "imbalance_z" not in work.columns:
        return _empty_figure("Upload market-state data to see PnL across entry z-score regimes.")

    x_col = "imbalance_z"
    y_col = None

    if "spread_z" in work.columns and work["spread_z"].notna().any():
        y_col = "spread_z"
    elif "mid_z" in work.columns and work["mid_z"].notna().any():
        y_col = "mid_z"

    if y_col is None or "pnl" not in work.columns:
        return _empty_figure("Upload market-state data to see PnL across entry z-score regimes.")

    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work["pnl"] = pd.to_numeric(work["pnl"], errors="coerce")

    if "hold_ticks" in work.columns:
        work["hold_ticks"] = pd.to_numeric(work["hold_ticks"], errors="coerce")
    else:
        work["hold_ticks"] = np.nan

    if "abs_pnl" not in work.columns:
        work["abs_pnl"] = work["pnl"].abs()
    else:
        work["abs_pnl"] = pd.to_numeric(work["abs_pnl"], errors="coerce")

    if "outcome" not in work.columns or work["outcome"].isna().all():
        work["outcome"] = np.where(
            work["pnl"] > 0, "win",
            np.where(work["pnl"] < 0, "loss", "breakeven")
        )

    work = work.dropna(subset=[x_col, y_col, "pnl"]).copy()
    if len(work) < 20:
        return _empty_figure("Not enough merged z-score rows yet.")

    fig = go.Figure()
    for outcome, sub in work.groupby("outcome"):
        custom = np.column_stack(
            [
                sub["product"].astype(str).to_numpy() if "product" in sub.columns else np.array([""] * len(sub)),
                sub["direction"].astype(str).to_numpy() if "direction" in sub.columns else np.array([""] * len(sub)),
                sub["pnl"].round(4).astype(str).to_numpy(),
                sub["hold_ticks"].round(2).astype(str).to_numpy(),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[y_col],
                mode="markers",
                name=str(outcome),
                marker={"size": np.clip(np.sqrt(sub["abs_pnl"].fillna(0).to_numpy() + 1) * 2.0, 6, 22)},
                customdata=custom,
                hovertemplate=(
                    f"{x_col}=%{{x:.3f}}<br>"
                    f"{y_col}=%{{y:.3f}}<br>"
                    "Product=%{customdata[0]}<br>"
                    "Direction=%{customdata[1]}<br>"
                    "PnL=%{customdata[2]}<br>"
                    "Hold=%{customdata[3]}<extra></extra>"
                ),
            )
        )

    y_title = "Spread Z at Entry" if y_col == "spread_z" else "Mid Z at Entry"

    fig.update_layout(
        title="PnL Across Entry Z-Score Regimes",
        template="plotly_white",
        height=430,
        hovermode="closest",
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Imbalance Z at Entry")
    fig.update_yaxes(title_text=y_title)
    return fig


def _research_table_columns(df: pd.DataFrame) -> list[dict]:
    preferred = [
        "trade_id",
        "product",
        "direction",
        "quantity",
        "entry_timestamp",
        "exit_timestamp",
        "hold_ticks",
        "entry_price",
        "exit_price",
        "entry_mid_price",
        "exit_mid_price",
        "entry_spread",
        "exit_spread",
        "entry_vs_mid",
        "exit_vs_mid",
        "imbalance_top3",
        "imbalance_z",
        "spread_z",
        "mid_z",
        "market_timestamp_at_entry",
        "pnl",
        "return_pct",
        "outcome",
    ]
    cols = [c for c in preferred if c in df.columns]
    return [{"field": col, "headerName": col.replace("_", " ").title(), "flex": 1} for col in cols]


def _prepare_grid_rows(df: pd.DataFrame) -> list[dict]:
    preferred = [
        "trade_id",
        "product",
        "direction",
        "quantity",
        "entry_timestamp",
        "exit_timestamp",
        "hold_ticks",
        "entry_price",
        "exit_price",
        "entry_mid_price",
        "exit_mid_price",
        "entry_spread",
        "exit_spread",
        "entry_vs_mid",
        "exit_vs_mid",
        "imbalance_top3",
        "imbalance_z",
        "spread_z",
        "mid_z",
        "market_timestamp_at_entry",
        "pnl",
        "return_pct",
        "outcome",
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        return []

    out = df[cols].copy()

    if "pnl" in out.columns:
        out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    if "hold_ticks" in out.columns:
        out["hold_ticks"] = pd.to_numeric(out["hold_ticks"], errors="coerce")

    sort_cols = [c for c in ["pnl", "hold_ticks"] if c in out.columns]
    ascending = [False, True][: len(sort_cols)]

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending)

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
        f"Research upload failed: {message}",
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