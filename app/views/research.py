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
            dcc.Graph(id="research-pnl-by-liquidity-graph"),
            dcc.Graph(id="research-pnl-by-inventory-bucket-graph"),
            dcc.Graph(id="research-winrate-heatmap-graph"),
            dcc.Graph(id="research-pnl-heatmap-graph"),
            dcc.Graph(id="research-pnl-by-exact-entry-spread-graph"),
            html.Div(
                [
                    html.H4("Heatmap Bucket Diagnostics"),
                    dag.AgGrid(
                        id="research-heatmap-bucket-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={"pagination": True, "paginationPageSize": 50},
                        style={"height": "420px", "width": "100%", "marginBottom": "20px"},
                        className="ag-theme-alpine",
                    ),
                ]
            ),
            html.Div(id="research-winner-loser-summary", style={"marginBottom": "20px"}),
            html.H3("Research V2: Entry Signal Diagnostics"),
            dcc.Graph(id="research-pnl-by-imbalance-z-graph"),
            dcc.Graph(id="research-winrate-z-heatmap-graph"),
            dcc.Graph(id="research-entry-z-boxplot-graph"),
            dcc.Graph(id="research-entry-z-scatter-graph"),
            html.H3("Research V3: Trade Excursion Diagnostics"),
            dcc.Graph(id="research-realized-vs-mfe-graph"),
            dcc.Graph(id="research-realized-vs-mfe-by-spread-graph"),
            dcc.Graph(id="research-capture-ratio-heatmap-graph"),
            dcc.Graph(id="research-efficiency-gap-boxplot-graph"),
            dcc.Graph(id="research-mfe-vs-hold-graph"),
            dcc.Graph(id="research-mfe-vs-hold-core-regime-graph"),
            dcc.Graph(id="research-mfe-vs-time-to-mfe-core-regime-graph"),
            html.Div(
                [
                    html.H4("Early MFE Bucket Diagnostics"),
                    dag.AgGrid(
                        id="research-early-mfe-bucket-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={"pagination": True, "paginationPageSize": 25},
                        style={"height": "320px", "width": "100%", "marginBottom": "20px"},
                        className="ag-theme-alpine",
                    ),
                ]
            ),
            dcc.Graph(id="research-early-mfe-vs-capture-runner-graph"),
            html.Div(
                [
                    html.H4("Top 20 MFE Trades in Spread Bucket 16 to 21"),
                    dag.AgGrid(
                        id="research-top-mfe-core-regime-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={"pagination": True, "paginationPageSize": 20},
                        style={"height": "420px", "width": "100%", "marginBottom": "20px"},
                        className="ag-theme-alpine",
                    ),
                ]
            ),
            dcc.Graph(id="research-mae-vs-realized-graph"),
            dcc.Graph(id="research-capture-count-heatmap-graph"),
            html.Div(
                [
                    html.H4("Top Spread Bucket Hold-Time Diagnostics"),
                    dag.AgGrid(
                        id="research-top-spread-hold-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={"pagination": True, "paginationPageSize": 25},
                        style={"height": "340px", "width": "100%", "marginBottom": "20px"},
                        className="ag-theme-alpine",
                    ),
                ]
            ),
            html.Div(
                [
                    html.H4("Largest Efficiency Gaps in Spread Bucket 16 to 21"),
                    dag.AgGrid(
                        id="research-efficiency-gap-examples-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={"pagination": True, "paginationPageSize": 20},
                        style={"height": "420px", "width": "100%", "marginBottom": "20px"},
                        className="ag-theme-alpine",
                    ),
                ]
            ),
            dcc.Graph(id="research-realized-vs-mfe-spread-bucket-graph"),
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
        Output("research-pnl-by-liquidity-graph", "figure"),
        Output("research-pnl-by-inventory-bucket-graph", "figure"),
        Output("research-winrate-heatmap-graph", "figure"),
        Output("research-pnl-heatmap-graph", "figure"),
        Output("research-pnl-by-exact-entry-spread-graph", "figure"),
        Output("research-heatmap-bucket-grid", "columnDefs"),
        Output("research-heatmap-bucket-grid", "rowData"),
        Output("research-winner-loser-summary", "children"),
        Output("research-pnl-by-imbalance-z-graph", "figure"),
        Output("research-winrate-z-heatmap-graph", "figure"),
        Output("research-entry-z-boxplot-graph", "figure"),
        Output("research-entry-z-scatter-graph", "figure"),
        Output("research-realized-vs-mfe-graph", "figure"),
        Output("research-realized-vs-mfe-by-spread-graph", "figure"),
        Output("research-capture-ratio-heatmap-graph", "figure"),
        Output("research-efficiency-gap-boxplot-graph", "figure"),
        Output("research-mfe-vs-hold-graph", "figure"),
        Output("research-mfe-vs-hold-core-regime-graph", "figure"),
        Output("research-mfe-vs-time-to-mfe-core-regime-graph", "figure"),
        Output("research-early-mfe-bucket-grid", "columnDefs"),
        Output("research-early-mfe-bucket-grid", "rowData"),
        Output("research-early-mfe-vs-capture-runner-graph", "figure"),
        Output("research-top-mfe-core-regime-grid", "columnDefs"),
        Output("research-top-mfe-core-regime-grid", "rowData"),
        Output("research-mae-vs-realized-graph", "figure"),
        Output("research-capture-count-heatmap-graph", "figure"),
        Output("research-top-spread-hold-grid", "columnDefs"),
        Output("research-top-spread-hold-grid", "rowData"),
        Output("research-efficiency-gap-examples-grid", "columnDefs"),
        Output("research-efficiency-gap-examples-grid", "rowData"),
        Output("research-realized-vs-mfe-spread-bucket-graph", "figure"),
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
            return (
                html.Div(),
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                [],
                [],
                html.Div(),
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                [],
                [],
                empty_fig,
                empty_fig,
                _early_mfe_bucket_table_columns(),
                [],
                empty_fig,
                _top_mfe_core_regime_table_columns(),
                [],
                empty_fig,
                _top_spread_hold_table_columns(),
                [],
                _efficiency_gap_examples_table_columns(),
                [],
                empty_fig,
                [],
                [],
            )

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
                empty_fig,
                empty_fig,
                empty_fig,
                _heatmap_bucket_table_columns(),
                [],
                html.Div(),
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                _early_mfe_bucket_table_columns(),
                [],
                empty_fig,
                _top_mfe_core_regime_table_columns(),
                [],
                empty_fig,
                empty_fig,
                _top_spread_hold_table_columns(),
                [],
                _efficiency_gap_examples_table_columns(),
                [],
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
        pnl_by_liquidity_fig = _safe_fig(
            lambda: _make_pnl_by_liquidity_figure(df),
            "Could not render PnL by passive vs taker.",
        )
        pnl_by_inventory_bucket_fig = _safe_fig(
            lambda: _make_pnl_by_inventory_bucket_figure(df),
            "Could not render PnL by inventory bucket.",
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
        pnl_by_exact_entry_spread_fig = _safe_fig(
            lambda: _make_pnl_by_exact_entry_spread_figure(df),
            "Could not render PnL by exact entry spread.",
        )
        heatmap_bucket_cols = _safe_call(lambda: _heatmap_bucket_table_columns(), [])
        heatmap_bucket_rows = _safe_call(lambda: _prepare_heatmap_bucket_rows(df), [])
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
        realized_vs_mfe_fig = _safe_fig(
            lambda: _make_realized_vs_mfe_figure(df),
            "Could not render realized vs MFE scatter.",
        )
        realized_vs_mfe_by_spread_fig = _safe_fig(
            lambda: _make_realized_vs_mfe_by_spread_figure(df),
            "Could not render realized vs MFE by spread.",
        )
        capture_ratio_heatmap_fig = _safe_fig(
            lambda: _make_capture_ratio_heatmap_figure(df),
            "Could not render capture ratio heatmap.",
        )
        efficiency_gap_boxplot_fig = _safe_fig(
            lambda: _make_efficiency_gap_boxplot_figure(df),
            "Could not render efficiency gap boxplot.",
        )
        mfe_vs_hold_fig = _safe_fig(
            lambda: _make_mfe_vs_hold_figure(df),
            "Could not render MFE vs hold scatter.",
        )
        mfe_vs_hold_core_regime_fig = _safe_fig(
            lambda: _make_mfe_vs_hold_core_regime_figure(df),
            "Could not render MFE vs hold for spread 16 to 21.",
        )
        mfe_vs_time_to_mfe_core_regime_fig = _safe_fig(
            lambda: _make_mfe_vs_time_to_mfe_core_regime_figure(df),
            "Could not render MFE vs time-to-MFE for spread 16 to 21.",
        )
        early_mfe_bucket_cols = _safe_call(lambda: _early_mfe_bucket_table_columns(), [])
        early_mfe_bucket_rows = _safe_call(lambda: _prepare_early_mfe_bucket_rows(df), [])
        early_mfe_vs_capture_runner_fig = _safe_fig(
            lambda: _make_early_mfe_vs_capture_runner_figure(df),
            "Could not render early MFE vs capture ratio for runners.",
        )
        top_mfe_core_regime_cols = _safe_call(lambda: _top_mfe_core_regime_table_columns(), [])
        top_mfe_core_regime_rows = _safe_call(lambda: _prepare_top_mfe_core_regime_rows(df), [])
        mae_vs_realized_fig = _safe_fig(
            lambda: _make_mae_vs_realized_figure(df),
            "Could not render MAE vs realized scatter.",
        )
        capture_count_heatmap_fig = _safe_fig(
            lambda: _make_capture_count_heatmap_figure(df),
            "Could not render capture-ratio count heatmap.",
        )
        top_spread_hold_cols = _safe_call(lambda: _top_spread_hold_table_columns(), [])
        top_spread_hold_rows = _safe_call(lambda: _prepare_top_spread_hold_rows(df), [])
        efficiency_gap_example_cols = _safe_call(lambda: _efficiency_gap_examples_table_columns(), [])
        efficiency_gap_example_rows = _safe_call(lambda: _prepare_efficiency_gap_example_rows(df), [])
        realized_vs_mfe_spread_bucket_fig = _safe_fig(
            lambda: _make_realized_vs_mfe_spread_bucket_figure(df),
            "Could not render realized vs MFE by spread bucket.",
        )
        grid_cols = _safe_call(lambda: _research_table_columns(df), [])
        grid_rows = _safe_call(lambda: _prepare_grid_rows(df), [])

        return (
            summary_cards,
            pnl_by_outcome_fig,
            hold_vs_pnl_fig,
            pnl_by_liquidity_fig,
            pnl_by_inventory_bucket_fig,
            winrate_heatmap_fig,
            pnl_heatmap_fig,
            pnl_by_exact_entry_spread_fig,
            heatmap_bucket_cols,
            heatmap_bucket_rows,
            winner_loser_summary,
            pnl_by_imbalance_z_fig,
            winrate_z_heatmap_fig,
            entry_z_boxplot_fig,
            entry_z_scatter_fig,
            realized_vs_mfe_fig,
            realized_vs_mfe_by_spread_fig,
            capture_ratio_heatmap_fig,
            efficiency_gap_boxplot_fig,
            mfe_vs_hold_fig,
            mfe_vs_hold_core_regime_fig,
            mfe_vs_time_to_mfe_core_regime_fig,
            early_mfe_bucket_cols,
            early_mfe_bucket_rows,
            early_mfe_vs_capture_runner_fig,
            top_mfe_core_regime_cols,
            top_mfe_core_regime_rows,
            mae_vs_realized_fig,
            capture_count_heatmap_fig,
            top_spread_hold_cols,
            top_spread_hold_rows,
            efficiency_gap_example_cols,
            efficiency_gap_example_rows,
            realized_vs_mfe_spread_bucket_fig,
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
        "mfe_pnl",
        "mae_pnl",
        "capture_ratio",
        "efficiency_gap",
        "mfe_price_move",
        "mae_price_move",
        "mfe_timestamp",
        "mae_timestamp",
        "time_to_mfe_ticks",
        "time_to_mae_ticks",
        "market_rows_in_trade",
        "early_mfe_pnl_10000",
        "early_mfe_price_move_10000",
        "time_to_runner",
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

    if "entry_vs_mid" not in out.columns and {"entry_price", "entry_mid_price"}.issubset(out.columns):
        out["entry_vs_mid"] = pd.to_numeric(out["entry_price"], errors="coerce") - pd.to_numeric(out["entry_mid_price"], errors="coerce")

    if "exit_vs_mid" not in out.columns and {"exit_price", "exit_mid_price"}.issubset(out.columns):
        out["exit_vs_mid"] = pd.to_numeric(out["exit_price"], errors="coerce") - pd.to_numeric(out["exit_mid_price"], errors="coerce")

    if "win_flag" not in out.columns and "pnl" in out.columns:
        pnl_num = pd.to_numeric(out["pnl"], errors="coerce")
        out["win_flag"] = (pnl_num > 0).astype(float)

    out = _add_liquidity_tags(out)
    out = _add_inventory_context(out)

    if {"pnl", "mfe_pnl"}.issubset(out.columns):
        pnl_num = pd.to_numeric(out["pnl"], errors="coerce")
        mfe_num = pd.to_numeric(out["mfe_pnl"], errors="coerce")
        if "capture_ratio" not in out.columns or out["capture_ratio"].isna().all():
            out["capture_ratio"] = pnl_num / mfe_num.replace(0, np.nan)
        if "efficiency_gap" not in out.columns or out["efficiency_gap"].isna().all():
            out["efficiency_gap"] = mfe_num - pnl_num

    runner_mfe_threshold = 35.0
    runner_min_hold = 12000.0
    if {"mfe_pnl", "hold_ticks"}.issubset(out.columns):
        mfe_num = pd.to_numeric(out["mfe_pnl"], errors="coerce")
        hold_num = pd.to_numeric(out["hold_ticks"], errors="coerce")
        out["runner_flag"] = ((mfe_num >= runner_mfe_threshold) & (hold_num >= runner_min_hold)).fillna(False)
    elif "runner_flag" not in out.columns:
        out["runner_flag"] = False

    if "runner_flag" in out.columns:
        out["runner_flag"] = out["runner_flag"].fillna(False).astype(bool)

    if "time_to_runner" not in out.columns:
        if "time_to_mfe_ticks" in out.columns and "runner_flag" in out.columns:
            ttm_num = pd.to_numeric(out["time_to_mfe_ticks"], errors="coerce")
            hold_num = pd.to_numeric(out["hold_ticks"], errors="coerce") if "hold_ticks" in out.columns else pd.Series(np.nan, index=out.index)
            approx = np.where(pd.notna(ttm_num), np.minimum(ttm_num, hold_num), hold_num * 0.4)
            out["time_to_runner"] = np.where(out["runner_flag"], approx, np.nan)
        elif "hold_ticks" in out.columns:
            hold_num = pd.to_numeric(out["hold_ticks"], errors="coerce")
            out["time_to_runner"] = np.where(out["runner_flag"], hold_num * 0.4, np.nan)

    if "entry_spread" in out.columns and "entry_spread_bucket" not in out.columns:
        out["entry_spread_bucket"] = _spread_bucket_label(out["entry_spread"])

    return out



def _normalize_liquidity_label(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    taker_tokens = ["taker", "take", "aggressive", "cross", "crossed", "market", "remove", "removed"]
    passive_tokens = ["maker", "passive", "rest", "resting", "posted", "provide", "provided", "limit"]
    if any(tok in s for tok in taker_tokens):
        return "taker"
    if any(tok in s for tok in passive_tokens):
        return "passive"
    return None


def _infer_leg_liquidity(row: pd.Series, leg: str) -> str | None:
    candidate_cols = [
        f"{leg}_liquidity",
        f"{leg}_execution_type",
        f"{leg}_order_type",
        f"{leg}_maker_taker",
        f"{leg}_passive_taker",
    ]
    for col in candidate_cols:
        if col in row.index:
            label = _normalize_liquidity_label(row.get(col))
            if label:
                return label

    price_col = f"{leg}_price"
    mid_col = f"{leg}_mid_price"
    spread_col = f"{leg}_spread"

    if not {price_col, mid_col}.issubset(row.index):
        return None

    price = pd.to_numeric(row.get(price_col), errors="coerce")
    mid = pd.to_numeric(row.get(mid_col), errors="coerce")
    spread = pd.to_numeric(row.get(spread_col), errors="coerce") if spread_col in row.index else np.nan
    direction = str(row.get("direction", "")).strip().lower()

    if not (pd.notna(price) and pd.notna(mid) and direction):
        return None

    tol = 1e-9
    half_spread = spread / 2.0 if pd.notna(spread) else np.nan

    if leg == "entry":
        if direction == "buy":
            if pd.notna(half_spread) and price >= mid + half_spread - tol:
                return "taker"
            if price <= mid + tol:
                return "passive"
        if direction == "sell":
            if pd.notna(half_spread) and price <= mid - half_spread + tol:
                return "taker"
            if price >= mid - tol:
                return "passive"
    else:
        if direction == "buy":
            if pd.notna(half_spread) and price <= mid - half_spread + tol:
                return "taker"
            if price >= mid - tol:
                return "passive"
        if direction == "sell":
            if pd.notna(half_spread) and price >= mid + half_spread - tol:
                return "taker"
            if price <= mid + tol:
                return "passive"

    if pd.notna(half_spread):
        if abs(price - mid) >= max(abs(half_spread) * 0.75, tol):
            return "taker"
        return "passive"

    return None


def _add_liquidity_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "entry_liquidity_bucket" not in out.columns:
        out["entry_liquidity_bucket"] = out.apply(lambda row: _infer_leg_liquidity(row, "entry"), axis=1)

    if "exit_liquidity_bucket" not in out.columns:
        out["exit_liquidity_bucket"] = out.apply(lambda row: _infer_leg_liquidity(row, "exit"), axis=1)

    if "round_trip_liquidity_bucket" not in out.columns:
        def _combine(row):
            entry = row.get("entry_liquidity_bucket")
            exit_ = row.get("exit_liquidity_bucket")
            if entry and exit_:
                if entry == "passive" and exit_ == "passive":
                    return "passive / passive"
                if entry == "taker" and exit_ == "taker":
                    return "taker / taker"
                if entry == "passive" and exit_ == "taker":
                    return "passive entry / taker exit"
                if entry == "taker" and exit_ == "passive":
                    return "taker entry / passive exit"
            if entry:
                return f"entry: {entry}"
            if exit_:
                return f"exit: {exit_}"
            return "unknown"
        out["round_trip_liquidity_bucket"] = out.apply(_combine, axis=1)

    return out


def _add_inventory_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    inventory_candidates = [
        "inventory_before_entry",
        "entry_inventory",
        "inventory_at_entry",
        "position_before",
        "start_inventory",
        "starting_inventory",
        "inventory_before",
        "inventory",
    ]

    inventory_col = next((col for col in inventory_candidates if col in out.columns), None)

    if inventory_col:
        out["inventory_reference"] = pd.to_numeric(out[inventory_col], errors="coerce")
    else:
        sort_cols = [c for c in ["product", "entry_timestamp", "exit_timestamp", "trade_id"] if c in out.columns]
        if sort_cols:
            ordered = out.sort_values(sort_cols).copy()
            ordered_signed = pd.to_numeric(ordered.get("quantity"), errors="coerce").fillna(0.0)
            ordered_direction = ordered.get("direction", pd.Series(index=ordered.index, dtype="object")).astype("string").str.lower()
            ordered_sign = np.where(ordered_direction.eq("sell"), -1.0, 1.0)
            product_group = ordered["product"] if "product" in ordered.columns else pd.Series("__all__", index=ordered.index)
            ordered["inventory_reference"] = (ordered_signed * ordered_sign).groupby(product_group).cumsum().shift(fill_value=0.0)
            out = ordered.sort_index()
        else:
            signed_qty = pd.to_numeric(out.get("quantity"), errors="coerce").fillna(0.0)
            direction = out.get("direction", pd.Series(index=out.index, dtype="object")).astype("string").str.lower()
            sign = np.where(direction.eq("sell"), -1.0, 1.0)
            out["inventory_reference"] = (signed_qty * sign).cumsum().shift(fill_value=0.0)

    out["inventory_reference"] = pd.to_numeric(out["inventory_reference"], errors="coerce")
    out["inventory_abs"] = out["inventory_reference"].abs()

    if out["inventory_abs"].notna().sum() >= 5:
        bins = [-np.inf, 0.5, 10, 25, 50, 80, np.inf]
        labels = ["0", "1-10", "11-25", "26-50", "51-80", "80+"]
        out["inventory_bucket"] = pd.cut(out["inventory_abs"], bins=bins, labels=labels, include_lowest=True)
    else:
        out["inventory_bucket"] = pd.Series(["unknown"] * len(out), index=out.index, dtype="string")

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

    try:
        excursions = _compute_trade_excursions(left, market_df)
        if not excursions.empty and "trade_id" in merged.columns:
            merged = merged.merge(excursions, on="trade_id", how="left")
    except Exception:
        pass

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

    avg_mfe = pd.to_numeric(df["mfe_pnl"], errors="coerce").mean() if "mfe_pnl" in df.columns else np.nan
    avg_mae = pd.to_numeric(df["mae_pnl"], errors="coerce").mean() if "mae_pnl" in df.columns else np.nan
    avg_capture_ratio = pd.to_numeric(df["capture_ratio"], errors="coerce").mean() if "capture_ratio" in df.columns else np.nan
    avg_efficiency_gap = pd.to_numeric(df["efficiency_gap"], errors="coerce").mean() if "efficiency_gap" in df.columns else np.nan

    cards = [
        ("Trades", f"{total_trades:,}"),
        ("Total PnL", _fmt_number(total_pnl)),
        ("Average PnL", _fmt_number(avg_pnl)),
        ("Median PnL", _fmt_number(median_pnl)),
        ("Win Rate", _fmt_pct(win_rate)),
        ("Average Hold Ticks", _fmt_number(avg_hold)),
        ("Average Entry Spread", _fmt_number(avg_entry_spread)),
        ("Average Exit Spread", _fmt_number(avg_exit_spread)),
        ("Average MFE", _fmt_number(avg_mfe)),
        ("Average MAE", _fmt_number(avg_mae)),
        ("Average Capture Ratio", _fmt_pct(100.0 * avg_capture_ratio if pd.notna(avg_capture_ratio) else np.nan)),
        ("Average Efficiency Gap", _fmt_number(avg_efficiency_gap)),
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
        if "mfe_pnl" in frame.columns:
            rows.append(f"Avg MFE: {_fmt_number(pd.to_numeric(frame['mfe_pnl'], errors='coerce').mean())}")
        if "mae_pnl" in frame.columns:
            rows.append(f"Avg MAE: {_fmt_number(pd.to_numeric(frame['mae_pnl'], errors='coerce').mean())}")
        if "capture_ratio" in frame.columns:
            rows.append(f"Avg Capture Ratio: {_fmt_pct(100.0 * pd.to_numeric(frame['capture_ratio'], errors='coerce').mean())}")
        if "efficiency_gap" in frame.columns:
            rows.append(f"Avg Efficiency Gap: {_fmt_number(pd.to_numeric(frame['efficiency_gap'], errors='coerce').mean())}")

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




def _make_pnl_by_liquidity_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    if "round_trip_liquidity_bucket" not in work.columns or "pnl" not in work.columns:
        return _empty_figure("No passive/taker data available.")

    work["pnl"] = pd.to_numeric(work["pnl"], errors="coerce")
    work = work.dropna(subset=["pnl"]).copy()
    work["round_trip_liquidity_bucket"] = work["round_trip_liquidity_bucket"].fillna("unknown").astype(str)

    if work.empty:
        return _empty_figure("No passive/taker data available.")

    order = [
        "passive / passive",
        "passive entry / taker exit",
        "taker entry / passive exit",
        "taker / taker",
        "entry: passive",
        "entry: taker",
        "exit: passive",
        "exit: taker",
        "unknown",
    ]

    grouped = (
        work.groupby("round_trip_liquidity_bucket", as_index=False)
        .agg(
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            trades=("pnl", "size"),
        )
    )
    grouped["sort_key"] = grouped["round_trip_liquidity_bucket"].apply(lambda x: order.index(x) if x in order else 999)
    grouped = grouped.sort_values(["sort_key", "round_trip_liquidity_bucket"]).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["round_trip_liquidity_bucket"],
            y=grouped["total_pnl"],
            name="Total PnL",
            customdata=np.column_stack([grouped["avg_pnl"], grouped["trades"]]),
            hovertemplate="Bucket=%{x}<br>Total PnL=%{y:.2f}<br>Avg PnL=%{customdata[0]:.2f}<br>Trades=%{customdata[1]}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["round_trip_liquidity_bucket"],
            y=grouped["avg_pnl"],
            mode="lines+markers",
            name="Avg PnL / trade",
            yaxis="y2",
            customdata=np.column_stack([grouped["total_pnl"], grouped["trades"]]),
            hovertemplate="Bucket=%{x}<br>Avg PnL=%{y:.2f}<br>Total PnL=%{customdata[0]:.2f}<br>Trades=%{customdata[1]}<extra></extra>",
        )
    )

    fig.update_layout(
        title="PnL by Passive vs Taker",
        template="plotly_white",
        height=420,
        margin={"l": 50, "r": 50, "t": 60, "b": 90},
        yaxis={"title": "Total PnL"},
        yaxis2={"title": "Average PnL / trade", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(title_text="Liquidity Bucket")
    return fig


def _make_pnl_by_inventory_bucket_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    if "inventory_bucket" not in work.columns or "pnl" not in work.columns:
        return _empty_figure("No inventory data available.")

    work["pnl"] = pd.to_numeric(work["pnl"], errors="coerce")
    work["inventory_abs"] = pd.to_numeric(work.get("inventory_abs"), errors="coerce")
    work = work.dropna(subset=["pnl"]).copy()

    if work.empty:
        return _empty_figure("No inventory data available.")

    grouped = (
        work.groupby("inventory_bucket", observed=False, as_index=False)
        .agg(
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            trades=("pnl", "size"),
            avg_inventory=("inventory_abs", "mean"),
        )
    )
    grouped["inventory_bucket"] = grouped["inventory_bucket"].astype(str)
    order = ["0", "1-10", "11-25", "26-50", "51-80", "80+", "unknown"]
    grouped["sort_key"] = grouped["inventory_bucket"].apply(lambda x: order.index(x) if x in order else 999)
    grouped = grouped.sort_values(["sort_key", "inventory_bucket"]).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["inventory_bucket"],
            y=grouped["total_pnl"],
            name="Total PnL",
            customdata=np.column_stack([grouped["avg_pnl"], grouped["trades"], grouped["avg_inventory"]]),
            hovertemplate="Bucket=%{x}<br>Total PnL=%{y:.2f}<br>Avg PnL=%{customdata[0]:.2f}<br>Trades=%{customdata[1]}<br>Avg |inventory|=%{customdata[2]:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["inventory_bucket"],
            y=grouped["avg_pnl"],
            mode="lines+markers",
            name="Avg PnL / trade",
            yaxis="y2",
            customdata=np.column_stack([grouped["total_pnl"], grouped["trades"], grouped["avg_inventory"]]),
            hovertemplate="Bucket=%{x}<br>Avg PnL=%{y:.2f}<br>Total PnL=%{customdata[0]:.2f}<br>Trades=%{customdata[1]}<br>Avg |inventory|=%{customdata[2]:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="PnL by Inventory Bucket",
        template="plotly_white",
        height=420,
        margin={"l": 50, "r": 50, "t": 60, "b": 70},
        yaxis={"title": "Total PnL"},
        yaxis2={"title": "Average PnL / trade", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(title_text="Absolute Inventory Bucket")
    return fig


def _make_heatmap_figure(df: pd.DataFrame, value_col: str, title: str, value_title: str) -> go.Figure:
    work = df.copy()

    required = ["hold_ticks", "entry_spread", "pnl", "win_flag", value_col]
    for col in required:
        if col not in work.columns:
            return _empty_figure(title)

    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["hold_ticks", "entry_spread", "pnl", "win_flag", value_col]).copy()
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

    bucket_stats = (
        work.groupby(["spread_bucket", "hold_bucket"], observed=False)
        .agg(
            trade_count=("pnl", "size"),
            avg_pnl=("pnl", "mean"),
            median_pnl=("pnl", "median"),
            win_rate=("win_flag", "mean"),
            total_pnl=("pnl", "sum"),
            heat_value=(value_col, "mean"),
        )
        .reset_index()
    )

    if bucket_stats.empty:
        return _empty_figure(title)

    z_pivot = bucket_stats.pivot(
        index="spread_bucket",
        columns="hold_bucket",
        values="heat_value",
    )
    count_pivot = bucket_stats.pivot(
        index="spread_bucket",
        columns="hold_bucket",
        values="trade_count",
    )
    avg_pivot = bucket_stats.pivot(
        index="spread_bucket",
        columns="hold_bucket",
        values="avg_pnl",
    )
    median_pivot = bucket_stats.pivot(
        index="spread_bucket",
        columns="hold_bucket",
        values="median_pnl",
    )
    win_pivot = bucket_stats.pivot(
        index="spread_bucket",
        columns="hold_bucket",
        values="win_rate",
    )
    total_pivot = bucket_stats.pivot(
        index="spread_bucket",
        columns="hold_bucket",
        values="total_pnl",
    )

    count_pivot = count_pivot.reindex(index=z_pivot.index, columns=z_pivot.columns)
    avg_pivot = avg_pivot.reindex(index=z_pivot.index, columns=z_pivot.columns)
    median_pivot = median_pivot.reindex(index=z_pivot.index, columns=z_pivot.columns)
    win_pivot = win_pivot.reindex(index=z_pivot.index, columns=z_pivot.columns)
    total_pivot = total_pivot.reindex(index=z_pivot.index, columns=z_pivot.columns)

    customdata = np.dstack(
        [
            count_pivot.fillna(0).to_numpy(),
            avg_pivot.to_numpy(),
            median_pivot.to_numpy(),
            win_pivot.to_numpy(),
            total_pivot.to_numpy(),
        ]
    )

    hovertemplate = (
        "Hold Bucket=%{x}<br>"
        "Entry Spread Bucket=%{y}<br>"
        f"{value_title}=%{{z:.4f}}<br>"
        "Trade Count=%{customdata[0]:,.0f}<br>"
        "Avg PnL=%{customdata[1]:.4f}<br>"
        "Median PnL=%{customdata[2]:.4f}<br>"
        "Win Rate=%{customdata[3]:.2%}<br>"
        "Total PnL=%{customdata[4]:.4f}<extra></extra>"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z_pivot.to_numpy(),
            x=[str(c) for c in z_pivot.columns],
            y=[str(i) for i in z_pivot.index],
            customdata=customdata,
            colorbar={"title": value_title},
            hovertemplate=hovertemplate,
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


def _build_heatmap_bucket_stats(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    required = ["hold_ticks", "entry_spread", "pnl", "win_flag"]
    for col in required:
        if col not in work.columns:
            return pd.DataFrame()

    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=required).copy()
    if work.empty:
        return pd.DataFrame()

    try:
        work["hold_bucket"] = pd.qcut(work["hold_ticks"], q=6, duplicates="drop")
    except Exception:
        work["hold_bucket"] = pd.cut(work["hold_ticks"], bins=6)

    try:
        work["spread_bucket"] = pd.qcut(work["entry_spread"], q=6, duplicates="drop")
    except Exception:
        work["spread_bucket"] = pd.cut(work["entry_spread"], bins=6)

    stats = (
        work.groupby(["spread_bucket", "hold_bucket"], observed=False)
        .agg(
            trade_count=("pnl", "size"),
            avg_pnl=("pnl", "mean"),
            median_pnl=("pnl", "median"),
            win_rate=("win_flag", "mean"),
            total_pnl=("pnl", "sum"),
        )
        .reset_index()
    )

    if stats.empty:
        return pd.DataFrame()

    stats["spread_bucket"] = stats["spread_bucket"].astype(str)
    stats["hold_bucket"] = stats["hold_bucket"].astype(str)
    stats["win_rate_pct"] = 100.0 * stats["win_rate"]

    return stats


def _heatmap_bucket_table_columns() -> list[dict]:
    return [
        {"field": "spread_bucket", "headerName": "Entry Spread Bucket", "flex": 2},
        {"field": "hold_bucket", "headerName": "Hold Time Bucket", "flex": 2},
        {"field": "trade_count", "headerName": "Trade Count", "flex": 1},
        {"field": "avg_pnl", "headerName": "Average PnL", "flex": 1},
        {"field": "median_pnl", "headerName": "Median PnL", "flex": 1},
        {"field": "win_rate_pct", "headerName": "Win Rate %", "flex": 1},
        {"field": "total_pnl", "headerName": "Total PnL", "flex": 1},
    ]


def _prepare_heatmap_bucket_rows(df: pd.DataFrame) -> list[dict]:
    stats = _build_heatmap_bucket_stats(df)
    if stats.empty:
        return []

    stats = stats.sort_values(
        by=["spread_bucket", "hold_bucket"],
        ascending=[True, True],
    ).copy()

    for col in ["avg_pnl", "median_pnl", "win_rate_pct", "total_pnl"]:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce").round(4)

    return stats[
        [
            "spread_bucket",
            "hold_bucket",
            "trade_count",
            "avg_pnl",
            "median_pnl",
            "win_rate_pct",
            "total_pnl",
        ]
    ].to_dict("records")


def _make_pnl_by_exact_entry_spread_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()

    required = ["entry_spread", "pnl", "win_flag"]
    for col in required:
        if col not in work.columns:
            return _empty_figure("PnL by exact entry spread")

    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["entry_spread", "pnl", "win_flag"]).copy()
    if work.empty:
        return _empty_figure("PnL by exact entry spread")

    rounded = work["entry_spread"].round()
    if np.all(np.isclose(work["entry_spread"], rounded, atol=1e-9)):
        work["entry_spread_exact"] = rounded.astype(int)
    else:
        work["entry_spread_exact"] = work["entry_spread"].round(4)

    grouped = (
        work.groupby("entry_spread_exact", as_index=False)
        .agg(
            trade_count=("pnl", "size"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            median_pnl=("pnl", "median"),
            win_rate=("win_flag", "mean"),
        )
        .sort_values("entry_spread_exact")
    )

    if grouped.empty:
        return _empty_figure("PnL by exact entry spread")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["entry_spread_exact"],
            y=grouped["total_pnl"],
            name="Total PnL",
            customdata=np.column_stack(
                [
                    grouped["trade_count"],
                    grouped["avg_pnl"],
                    grouped["median_pnl"],
                    grouped["win_rate"],
                ]
            ),
            hovertemplate=(
                "Entry Spread=%{x}<br>"
                "Total PnL=%{y:.4f}<br>"
                "Trade Count=%{customdata[0]:,.0f}<br>"
                "Avg PnL=%{customdata[1]:.4f}<br>"
                "Median PnL=%{customdata[2]:.4f}<br>"
                "Win Rate=%{customdata[3]:.2%}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["entry_spread_exact"],
            y=grouped["avg_pnl"],
            mode="lines+markers",
            name="Avg PnL / trade",
            yaxis="y2",
            customdata=np.column_stack(
                [
                    grouped["trade_count"],
                    grouped["total_pnl"],
                    grouped["median_pnl"],
                    grouped["win_rate"],
                ]
            ),
            hovertemplate=(
                "Entry Spread=%{x}<br>"
                "Avg PnL=%{y:.4f}<br>"
                "Trade Count=%{customdata[0]:,.0f}<br>"
                "Total PnL=%{customdata[1]:.4f}<br>"
                "Median PnL=%{customdata[2]:.4f}<br>"
                "Win Rate=%{customdata[3]:.2%}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="PnL by Exact Entry Spread",
        template="plotly_white",
        height=420,
        margin={"l": 50, "r": 50, "t": 60, "b": 70},
        yaxis={"title": "Total PnL"},
        yaxis2={"title": "Average PnL / trade", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(title_text="Exact Entry Spread")
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



def _compute_trade_excursions(trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or market_df.empty:
        return pd.DataFrame()

    if "trade_id" not in trades_df.columns:
        working = trades_df.copy()
        working["trade_id"] = np.arange(1, len(working) + 1)
    else:
        working = trades_df.copy()

    market = _prepare_market_df(market_df.copy())
    results = []

    for product, tsub in working.groupby("product", dropna=False):
        msub = market[market["product"] == product].sort_values("timestamp").copy()
        if msub.empty:
            continue

        if "mid_price" in msub.columns and msub["mid_price"].notna().any():
            mark_series = pd.to_numeric(msub["mid_price"], errors="coerce")
        else:
            mark_series = (
                pd.to_numeric(msub.get("bid_price_1"), errors="coerce")
                + pd.to_numeric(msub.get("ask_price_1"), errors="coerce")
            ) / 2.0
        msub = msub.assign(mark_price=mark_series).dropna(subset=["timestamp", "mark_price"])

        if msub.empty:
            continue

        ts_values = msub["timestamp"].to_numpy()

        for _, row in tsub.iterrows():
            trade_id = row.get("trade_id")
            entry_ts = pd.to_numeric(row.get("entry_timestamp"), errors="coerce")
            exit_ts = pd.to_numeric(row.get("exit_timestamp"), errors="coerce")
            entry_price = pd.to_numeric(row.get("entry_price"), errors="coerce")
            qty = abs(pd.to_numeric(row.get("quantity"), errors="coerce"))
            direction = str(row.get("direction", "")).strip().lower()

            if not (pd.notna(entry_ts) and pd.notna(exit_ts) and pd.notna(entry_price) and pd.notna(qty)):
                continue
            if qty <= 0:
                continue

            lo = min(entry_ts, exit_ts)
            hi = max(entry_ts, exit_ts)
            mask = (ts_values >= lo) & (ts_values <= hi)
            path = msub.loc[mask, ["timestamp", "mark_price"]].copy()

            if path.empty:
                continue

            if direction == "sell":
                path_pnl = (entry_price - path["mark_price"]) * qty
                price_move = entry_price - path["mark_price"]
            else:
                path_pnl = (path["mark_price"] - entry_price) * qty
                price_move = path["mark_price"] - entry_price

            pnl_numeric = pd.to_numeric(path_pnl, errors="coerce")
            price_numeric = pd.to_numeric(price_move, errors="coerce")

            early_window_mask = pd.to_numeric(path["timestamp"], errors="coerce") <= (entry_ts + 10000)
            early_path = path.loc[early_window_mask].copy()
            if early_path.empty:
                early_path = path.iloc[[0]].copy()
            early_pnl_numeric = pd.to_numeric(
                ((entry_price - early_path["mark_price"]) * qty) if direction == "sell"
                else ((early_path["mark_price"] - entry_price) * qty),
                errors="coerce",
            )
            early_price_numeric = pd.to_numeric(
                (entry_price - early_path["mark_price"]) if direction == "sell"
                else (early_path["mark_price"] - entry_price),
                errors="coerce",
            )
            early_mfe_pnl_10000 = early_pnl_numeric.max() if len(early_pnl_numeric) else np.nan
            early_mfe_price_move_10000 = early_price_numeric.max() if len(early_price_numeric) else np.nan
            mfe_idx = pnl_numeric.idxmax() if len(pnl_numeric) else None
            mae_idx = pnl_numeric.idxmin() if len(pnl_numeric) else None
            mfe_pnl = pnl_numeric.max()
            mae_pnl = pnl_numeric.min()
            mfe_price_move = price_numeric.max()
            mae_price_move = price_numeric.min()
            mfe_timestamp = pd.to_numeric(path.loc[mfe_idx, "timestamp"], errors="coerce") if mfe_idx is not None else np.nan
            mae_timestamp = pd.to_numeric(path.loc[mae_idx, "timestamp"], errors="coerce") if mae_idx is not None else np.nan
            time_to_mfe_ticks = mfe_timestamp - entry_ts if pd.notna(mfe_timestamp) else np.nan
            time_to_mae_ticks = mae_timestamp - entry_ts if pd.notna(mae_timestamp) else np.nan
            realized_pnl = pd.to_numeric(row.get("pnl"), errors="coerce")
            capture_ratio = realized_pnl / mfe_pnl if pd.notna(realized_pnl) and pd.notna(mfe_pnl) and mfe_pnl > 0 else np.nan
            efficiency_gap = mfe_pnl - realized_pnl if pd.notna(realized_pnl) and pd.notna(mfe_pnl) else np.nan

            results.append(
                {
                    "trade_id": trade_id,
                    "mfe_pnl": mfe_pnl,
                    "mae_pnl": mae_pnl,
                    "capture_ratio": capture_ratio,
                    "efficiency_gap": efficiency_gap,
                    "mfe_price_move": mfe_price_move,
                    "mae_price_move": mae_price_move,
                    "mfe_timestamp": mfe_timestamp,
                    "mae_timestamp": mae_timestamp,
                    "time_to_mfe_ticks": time_to_mfe_ticks,
                    "time_to_mae_ticks": time_to_mae_ticks,
                    "early_mfe_pnl_10000": early_mfe_pnl_10000,
                    "early_mfe_price_move_10000": early_mfe_price_move_10000,
                    "market_rows_in_trade": int(len(path)),
                }
            )

    return pd.DataFrame(results)


def _make_realized_vs_mfe_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    needed = ["mfe_pnl", "pnl"]
    if not all(c in work.columns for c in needed):
        return _empty_figure("Upload market-state data to see realized PnL vs MFE.")

    for col in ["mfe_pnl", "pnl", "entry_spread", "hold_ticks"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["mfe_pnl", "pnl"]).copy()
    if work.empty:
        return _empty_figure("Not enough excursion rows yet.")

    fig = go.Figure()
    spread_vals = work["entry_spread"] if "entry_spread" in work.columns else pd.Series(np.nan, index=work.index)
    hold_vals = work["hold_ticks"] if "hold_ticks" in work.columns else pd.Series(np.nan, index=work.index)
    custom = np.column_stack([
        work["product"].astype(str).to_numpy() if "product" in work.columns else np.array([""] * len(work)),
        work["direction"].astype(str).to_numpy() if "direction" in work.columns else np.array([""] * len(work)),
        np.nan_to_num(spread_vals.to_numpy(dtype=float), nan=np.nan),
        np.nan_to_num(hold_vals.to_numpy(dtype=float), nan=np.nan),
    ])

    fig.add_trace(
        go.Scatter(
            x=work["mfe_pnl"],
            y=work["pnl"],
            mode="markers",
            marker={"size": 8},
            customdata=custom,
            hovertemplate=(
                "MFE=%{x:.4f}<br>"
                "Realized PnL=%{y:.4f}<br>"
                "Product=%{customdata[0]}<br>"
                "Direction=%{customdata[1]}<br>"
                "Entry Spread=%{customdata[2]:.4f}<br>"
                "Hold=%{customdata[3]:.2f}<extra></extra>"
            ),
            name="Trades",
        )
    )

    lo = float(min(work["mfe_pnl"].min(), work["pnl"].min()))
    hi = float(max(work["mfe_pnl"].max(), work["pnl"].max()))
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", hoverinfo="skip")
    )

    fig.update_layout(
        title="Realized PnL vs MFE",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="MFE PnL")
    fig.update_yaxes(title_text="Realized PnL")
    return fig


def _make_realized_vs_mfe_by_spread_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    needed = ["entry_spread", "pnl", "mfe_pnl"]
    if not all(c in work.columns for c in needed):
        return _empty_figure("Upload market-state data to compare realized PnL and MFE by exact spread.")

    for col in needed + ["win_flag"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=needed).copy()
    if work.empty:
        return _empty_figure("Not enough excursion rows yet.")

    rounded = work["entry_spread"].round()
    if np.all(np.isclose(work["entry_spread"], rounded, atol=1e-9)):
        work["entry_spread_exact"] = rounded.astype(int)
    else:
        work["entry_spread_exact"] = work["entry_spread"].round(4)

    grouped = (
        work.groupby("entry_spread_exact", as_index=False)
        .agg(
            avg_realized_pnl=("pnl", "mean"),
            avg_mfe_pnl=("mfe_pnl", "mean"),
            trade_count=("pnl", "size"),
            avg_capture_ratio=("capture_ratio", "mean"),
        )
        .sort_values("entry_spread_exact")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["entry_spread_exact"],
            y=grouped["avg_realized_pnl"],
            name="Avg realized PnL",
            customdata=np.column_stack([grouped["trade_count"], grouped["avg_mfe_pnl"], grouped["avg_capture_ratio"]]),
            hovertemplate=(
                "Entry Spread=%{x}<br>"
                "Avg Realized PnL=%{y:.4f}<br>"
                "Trades=%{customdata[0]:,.0f}<br>"
                "Avg MFE=%{customdata[1]:.4f}<br>"
                "Avg Capture Ratio=%{customdata[2]:.2%}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["entry_spread_exact"],
            y=grouped["avg_mfe_pnl"],
            mode="lines+markers",
            name="Avg MFE",
            customdata=np.column_stack([grouped["trade_count"], grouped["avg_realized_pnl"], grouped["avg_capture_ratio"]]),
            hovertemplate=(
                "Entry Spread=%{x}<br>"
                "Avg MFE=%{y:.4f}<br>"
                "Trades=%{customdata[0]:,.0f}<br>"
                "Avg Realized PnL=%{customdata[1]:.4f}<br>"
                "Avg Capture Ratio=%{customdata[2]:.2%}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Average Realized PnL vs Average MFE by Exact Entry Spread",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 60},
        barmode="group",
    )
    fig.update_xaxes(title_text="Exact Entry Spread")
    fig.update_yaxes(title_text="PnL")
    return fig


def _make_capture_ratio_heatmap_figure(df: pd.DataFrame) -> go.Figure:
    return _make_heatmap_figure(
        df,
        value_col="capture_ratio",
        title="Average Capture Ratio by Hold Time × Entry Spread",
        value_title="Capture Ratio",
    )


def _make_efficiency_gap_boxplot_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    needed = ["entry_spread", "efficiency_gap"]
    if not all(c in work.columns for c in needed):
        return _empty_figure("Upload market-state data to see efficiency gap by spread.")

    for col in needed:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=needed).copy()
    if work.empty:
        return _empty_figure("Not enough excursion rows yet.")

    try:
        work["spread_bucket"] = pd.qcut(work["entry_spread"], q=6, duplicates="drop")
    except Exception:
        work["spread_bucket"] = pd.cut(work["entry_spread"], bins=6)

    fig = go.Figure()
    for bucket, sub in work.groupby("spread_bucket", observed=False):
        if len(sub) == 0:
            continue
        fig.add_trace(go.Box(y=sub["efficiency_gap"], name=str(bucket), boxmean=True))

    fig.update_layout(
        title="Efficiency Gap by Entry Spread Bucket",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 80},
    )
    fig.update_xaxes(title_text="Entry Spread Bucket")
    fig.update_yaxes(title_text="Efficiency Gap (MFE - Realized)")
    return fig


def _make_mfe_vs_hold_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    needed = ["hold_ticks", "mfe_pnl"]
    if not all(c in work.columns for c in needed):
        return _empty_figure("Upload market-state data to see MFE vs hold time.")

    for col in ["hold_ticks", "mfe_pnl", "entry_spread"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["hold_ticks", "mfe_pnl"]).copy()
    if work.empty:
        return _empty_figure("Not enough excursion rows yet.")

    fig = go.Figure()
    if "entry_spread" in work.columns:
        work["spread_regime"] = np.where(work["entry_spread"] >= 16, "spread ≥ 16", "spread < 16")
    else:
        work["spread_regime"] = "all trades"

    for regime, sub in work.groupby("spread_regime"):
        fig.add_trace(
            go.Scatter(
                x=sub["hold_ticks"],
                y=sub["mfe_pnl"],
                mode="markers",
                name=str(regime),
                customdata=np.column_stack([
                    sub["product"].astype(str).to_numpy() if "product" in sub.columns else np.array([""] * len(sub)),
                    sub["direction"].astype(str).to_numpy() if "direction" in sub.columns else np.array([""] * len(sub)),
                ]),
                hovertemplate=(
                    "Hold=%{x:.2f}<br>"
                    "MFE=%{y:.4f}<br>"
                    "Product=%{customdata[0]}<br>"
                    "Direction=%{customdata[1]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="MFE vs Hold Time",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Hold Ticks")
    fig.update_yaxes(title_text="MFE PnL")
    return fig


def _make_mfe_vs_hold_core_regime_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    needed = ["hold_ticks", "mfe_pnl", "entry_spread"]
    if not all(c in work.columns for c in needed):
        return _empty_figure("MFE vs Hold Time for Spread Bucket 16 to 21")

    for col in ["hold_ticks", "mfe_pnl", "entry_spread", "time_to_mfe_ticks", "pnl"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work[(work["entry_spread"] >= 16) & (work["entry_spread"] <= 21)].copy()
    work = work.dropna(subset=["hold_ticks", "mfe_pnl"]).copy()
    if work.empty:
        return _empty_figure("No spread 16 to 21 trades with MFE data.")

    custom = np.column_stack([
        work["product"].astype(str).to_numpy() if "product" in work.columns else np.array([""] * len(work)),
        work["direction"].astype(str).to_numpy() if "direction" in work.columns else np.array([""] * len(work)),
        work["time_to_mfe_ticks"].to_numpy() if "time_to_mfe_ticks" in work.columns else np.array([np.nan] * len(work)),
        work["pnl"].to_numpy() if "pnl" in work.columns else np.array([np.nan] * len(work)),
    ])

    fig = go.Figure(data=[go.Scatter(
        x=work["hold_ticks"],
        y=work["mfe_pnl"],
        mode="markers",
        marker={"size": 8},
        customdata=custom,
        hovertemplate=(
            "Hold=%{x:.2f}<br>"
            "MFE=%{y:.4f}<br>"
            "Product=%{customdata[0]}<br>"
            "Direction=%{customdata[1]}<br>"
            "Time to MFE=%{customdata[2]:.2f}<br>"
            "Realized PnL=%{customdata[3]:.4f}<extra></extra>"
        ),
        name="Spread 16-21 trades",
    )])
    fig.update_layout(
        title="MFE vs Hold Time for Spread Bucket 16 to 21",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Hold Time")
    fig.update_yaxes(title_text="MFE PnL")
    return fig


def _make_mfe_vs_time_to_mfe_core_regime_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    needed = ["mfe_pnl", "time_to_mfe_ticks", "entry_spread"]
    if not all(c in work.columns for c in needed):
        return _empty_figure("MFE vs Time to MFE for Spread Bucket 16 to 21")

    for col in ["mfe_pnl", "time_to_mfe_ticks", "entry_spread", "hold_ticks", "pnl"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work[(work["entry_spread"] >= 16) & (work["entry_spread"] <= 21)].copy()
    work = work.dropna(subset=["mfe_pnl", "time_to_mfe_ticks"]).copy()
    if work.empty:
        return _empty_figure("No spread 16 to 21 trades with time-to-MFE data.")

    custom = np.column_stack([
        work["product"].astype(str).to_numpy() if "product" in work.columns else np.array([""] * len(work)),
        work["direction"].astype(str).to_numpy() if "direction" in work.columns else np.array([""] * len(work)),
        work["hold_ticks"].to_numpy() if "hold_ticks" in work.columns else np.array([np.nan] * len(work)),
        work["pnl"].to_numpy() if "pnl" in work.columns else np.array([np.nan] * len(work)),
    ])

    fig = go.Figure(data=[go.Scatter(
        x=work["time_to_mfe_ticks"],
        y=work["mfe_pnl"],
        mode="markers",
        marker={"size": 8},
        customdata=custom,
        hovertemplate=(
            "Time to MFE=%{x:.2f}<br>"
            "MFE=%{y:.4f}<br>"
            "Product=%{customdata[0]}<br>"
            "Direction=%{customdata[1]}<br>"
            "Hold Time=%{customdata[2]:.2f}<br>"
            "Realized PnL=%{customdata[3]:.4f}<extra></extra>"
        ),
        name="Spread 16-21 trades",
    )])
    fig.update_layout(
        title="MFE vs Time to MFE for Spread Bucket 16 to 21",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Time to MFE")
    fig.update_yaxes(title_text="MFE PnL")
    return fig


def _top_mfe_core_regime_table_columns() -> list[dict]:
    return [
        {"field": "trade_id", "headerName": "Trade ID", "flex": 1},
        {"field": "product", "headerName": "Product", "flex": 1},
        {"field": "direction", "headerName": "Direction", "flex": 1},
        {"field": "entry_spread", "headerName": "Entry Spread", "flex": 1},
        {"field": "mfe_pnl", "headerName": "MFE", "flex": 1},
        {"field": "pnl", "headerName": "Realized PnL", "flex": 1},
        {"field": "time_to_mfe_ticks", "headerName": "Time to MFE", "flex": 1},
        {"field": "hold_ticks", "headerName": "Total Hold Time", "flex": 1},
        {"field": "capture_ratio", "headerName": "Capture Ratio", "flex": 1},
        {"field": "efficiency_gap", "headerName": "Efficiency Gap", "flex": 1},
        {"field": "entry_timestamp", "headerName": "Entry Ts", "flex": 1},
        {"field": "exit_timestamp", "headerName": "Exit Ts", "flex": 1},
    ]


def _prepare_top_mfe_core_regime_rows(df: pd.DataFrame, limit: int = 20) -> list[dict]:
    work = df.copy()
    needed = ["entry_spread", "mfe_pnl", "time_to_mfe_ticks", "hold_ticks"]
    if not all(c in work.columns for c in needed):
        return []

    for col in ["entry_spread", "mfe_pnl", "pnl", "time_to_mfe_ticks", "hold_ticks", "capture_ratio", "efficiency_gap"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work[(work["entry_spread"] >= 16) & (work["entry_spread"] <= 21)].copy()
    work = work.dropna(subset=["mfe_pnl", "time_to_mfe_ticks", "hold_ticks"]).copy()
    if work.empty:
        return []

    cols = [c for c in [
        "trade_id", "product", "direction", "entry_spread", "mfe_pnl", "pnl",
        "time_to_mfe_ticks", "hold_ticks", "capture_ratio", "efficiency_gap",
        "entry_timestamp", "exit_timestamp"
    ] if c in work.columns]

    out = work[cols].sort_values("mfe_pnl", ascending=False).head(limit).copy()
    for col in ["entry_spread", "mfe_pnl", "pnl", "time_to_mfe_ticks", "hold_ticks", "capture_ratio", "efficiency_gap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out.to_dict("records")


def _make_mae_vs_realized_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    needed = ["mae_pnl", "pnl"]
    if not all(c in work.columns for c in needed):
        return _empty_figure("Upload market-state data to see MAE vs realized PnL.")

    for col in ["mae_pnl", "pnl", "entry_spread"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["mae_pnl", "pnl"]).copy()
    if work.empty:
        return _empty_figure("Not enough excursion rows yet.")

    fig = go.Figure()
    custom = np.column_stack([
        work["product"].astype(str).to_numpy() if "product" in work.columns else np.array([""] * len(work)),
        work["direction"].astype(str).to_numpy() if "direction" in work.columns else np.array([""] * len(work)),
        work["entry_spread"].to_numpy() if "entry_spread" in work.columns else np.array([np.nan] * len(work)),
    ])
    fig.add_trace(
        go.Scatter(
            x=work["mae_pnl"],
            y=work["pnl"],
            mode="markers",
            marker={"size": 8},
            customdata=custom,
            hovertemplate=(
                "MAE=%{x:.4f}<br>"
                "Realized PnL=%{y:.4f}<br>"
                "Product=%{customdata[0]}<br>"
                "Direction=%{customdata[1]}<br>"
                "Entry Spread=%{customdata[2]:.4f}<extra></extra>"
            ),
            name="Trades",
        )
    )
    fig.update_layout(
        title="MAE vs Realized PnL",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="MAE PnL")
    fig.update_yaxes(title_text="Realized PnL")
    return fig


def _assign_heatmap_buckets(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    required = ["hold_ticks", "entry_spread"]
    if not all(c in work.columns for c in required):
        return work

    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=required).copy()
    if work.empty:
        return work

    try:
        work["hold_bucket"] = pd.qcut(work["hold_ticks"], q=6, duplicates="drop")
    except Exception:
        work["hold_bucket"] = pd.cut(work["hold_ticks"], bins=6)

    try:
        work["spread_bucket"] = pd.qcut(work["entry_spread"], q=6, duplicates="drop")
    except Exception:
        work["spread_bucket"] = pd.cut(work["entry_spread"], bins=6)

    return work


def _spread_bucket_label(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    bins = [-np.inf, 10, 16, 21, np.inf]
    labels = ["≤10", "11-15", "16-21", "22+"]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True, right=True)


def _make_capture_count_heatmap_figure(df: pd.DataFrame) -> go.Figure:
    work = _assign_heatmap_buckets(df)
    if work.empty or "capture_ratio" not in work.columns:
        return _empty_figure("Trade Count for Capture-Ratio Heatmap Buckets")

    grouped = (
        work.groupby(["spread_bucket", "hold_bucket"], observed=False)
        .agg(trade_count=("capture_ratio", "size"))
        .reset_index()
    )
    if grouped.empty:
        return _empty_figure("Trade Count for Capture-Ratio Heatmap Buckets")

    pivot = grouped.pivot(index="spread_bucket", columns="hold_bucket", values="trade_count")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorbar={"title": "Trade Count"},
            hovertemplate=(
                "Hold Bucket=%{x}<br>"
                "Entry Spread Bucket=%{y}<br>"
                "Trade Count=%{z:,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Trade Count for Capture-Ratio Heatmap Buckets",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Hold Time Bucket")
    fig.update_yaxes(title_text="Entry Spread Bucket")
    return fig


def _top_spread_hold_table_columns() -> list[dict]:
    return [
        {"field": "top_spread_bucket", "headerName": "Top Spread Bucket", "flex": 2},
        {"field": "hold_bucket", "headerName": "Hold Time Bucket", "flex": 2},
        {"field": "trade_count", "headerName": "Trade Count", "flex": 1},
        {"field": "avg_realized_pnl", "headerName": "Avg Realized PnL", "flex": 1},
        {"field": "avg_mfe", "headerName": "Avg MFE", "flex": 1},
        {"field": "avg_mae", "headerName": "Avg MAE", "flex": 1},
        {"field": "avg_capture_ratio", "headerName": "Avg Capture Ratio", "flex": 1},
        {"field": "median_realized_pnl", "headerName": "Median Realized PnL", "flex": 1},
    ]


def _prepare_top_spread_hold_rows(df: pd.DataFrame) -> list[dict]:
    work = _assign_heatmap_buckets(df)
    needed = ["hold_bucket", "spread_bucket", "pnl", "mfe_pnl", "mae_pnl", "capture_ratio"]
    if work.empty or not all(c in work.columns for c in needed):
        return []

    top_bucket = None
    spread_order = work["spread_bucket"].dropna().astype(str).unique().tolist()
    if spread_order:
        top_bucket = sorted(spread_order)[-1]
    if top_bucket is None:
        return []

    filtered = work[work["spread_bucket"].astype(str) == top_bucket].copy()
    if filtered.empty:
        return []

    grouped = (
        filtered.groupby("hold_bucket", observed=False)
        .agg(
            trade_count=("pnl", "size"),
            avg_realized_pnl=("pnl", "mean"),
            avg_mfe=("mfe_pnl", "mean"),
            avg_mae=("mae_pnl", "mean"),
            avg_capture_ratio=("capture_ratio", "mean"),
            median_realized_pnl=("pnl", "median"),
        )
        .reset_index()
    )
    if grouped.empty:
        return []

    grouped["top_spread_bucket"] = top_bucket
    grouped["hold_bucket"] = grouped["hold_bucket"].astype(str)
    for col in ["avg_realized_pnl", "avg_mfe", "avg_mae", "avg_capture_ratio", "median_realized_pnl"]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").round(4)
    grouped = grouped.sort_values("hold_bucket")
    return grouped[
        [
            "top_spread_bucket",
            "hold_bucket",
            "trade_count",
            "avg_realized_pnl",
            "avg_mfe",
            "avg_mae",
            "avg_capture_ratio",
            "median_realized_pnl",
        ]
    ].to_dict("records")


def _efficiency_gap_examples_table_columns() -> list[dict]:
    return [
        {"field": "trade_id", "headerName": "Trade ID", "flex": 1},
        {"field": "product", "headerName": "Product", "flex": 1},
        {"field": "direction", "headerName": "Direction", "flex": 1},
        {"field": "entry_spread", "headerName": "Entry Spread", "flex": 1},
        {"field": "hold_ticks", "headerName": "Hold Ticks", "flex": 1},
        {"field": "pnl", "headerName": "Realized PnL", "flex": 1},
        {"field": "mfe_pnl", "headerName": "MFE", "flex": 1},
        {"field": "mae_pnl", "headerName": "MAE", "flex": 1},
        {"field": "capture_ratio", "headerName": "Capture Ratio", "flex": 1},
        {"field": "efficiency_gap", "headerName": "Efficiency Gap", "flex": 1},
        {"field": "entry_timestamp", "headerName": "Entry Ts", "flex": 1},
        {"field": "exit_timestamp", "headerName": "Exit Ts", "flex": 1},
    ]


def _prepare_efficiency_gap_example_rows(df: pd.DataFrame, limit: int = 20) -> list[dict]:
    work = df.copy()
    needed = ["entry_spread", "efficiency_gap", "mfe_pnl", "pnl"]
    if not all(c in work.columns for c in needed):
        return []

    for col in ["entry_spread", "efficiency_gap", "mfe_pnl", "mae_pnl", "pnl", "capture_ratio", "hold_ticks"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work[(work["entry_spread"] >= 16) & (work["entry_spread"] <= 21)].copy()
    work = work.dropna(subset=["efficiency_gap", "mfe_pnl", "pnl"]).copy()
    if work.empty:
        return []

    cols = [
        c for c in [
            "trade_id", "product", "direction", "entry_spread", "hold_ticks", "pnl", "mfe_pnl",
            "mae_pnl", "capture_ratio", "efficiency_gap", "entry_timestamp", "exit_timestamp"
        ] if c in work.columns
    ]
    out = work[cols].sort_values("efficiency_gap", ascending=False).head(limit).copy()
    for col in ["entry_spread", "hold_ticks", "pnl", "mfe_pnl", "mae_pnl", "capture_ratio", "efficiency_gap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out.to_dict("records")


def _make_realized_vs_mfe_spread_bucket_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    needed = ["pnl", "mfe_pnl", "entry_spread"]
    if not all(c in work.columns for c in needed):
        return _empty_figure("Upload market-state data to compare realized PnL and MFE by spread bucket.")

    for col in needed + ["hold_ticks"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=needed).copy()
    if work.empty:
        return _empty_figure("Not enough excursion rows yet.")

    work["spread_bucket_label"] = _spread_bucket_label(work["entry_spread"]).astype(str)
    order = ["≤10", "11-15", "16-21", "22+"]

    fig = go.Figure()
    for label in order:
        sub = work[work["spread_bucket_label"] == label].copy()
        if sub.empty:
            continue
        custom = np.column_stack([
            sub["product"].astype(str).to_numpy() if "product" in sub.columns else np.array([""] * len(sub)),
            sub["direction"].astype(str).to_numpy() if "direction" in sub.columns else np.array([""] * len(sub)),
            sub["entry_spread"].round(4).to_numpy(),
        ])
        fig.add_trace(
            go.Scatter(
                x=sub["mfe_pnl"],
                y=sub["pnl"],
                mode="markers",
                name=label,
                customdata=custom,
                hovertemplate=(
                    "MFE=%{x:.4f}<br>"
                    "Realized PnL=%{y:.4f}<br>"
                    "Product=%{customdata[0]}<br>"
                    "Direction=%{customdata[1]}<br>"
                    "Entry Spread=%{customdata[2]:.4f}<extra></extra>"
                ),
            )
        )

    if not work.empty:
        lo = min(work["mfe_pnl"].min(), work["pnl"].min())
        hi = max(work["mfe_pnl"].max(), work["pnl"].max())
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x", hoverinfo="skip")
        )

    fig.update_layout(
        title="Realized PnL vs MFE by Spread Bucket",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="MFE PnL")
    fig.update_yaxes(title_text="Realized PnL")
    return fig



def _early_mfe_bucket_table_columns() -> list[dict]:
    return [
        {"field": "early_mfe_bucket", "headerName": "Early MFE Bucket", "flex": 2},
        {"field": "trade_count", "headerName": "Count", "flex": 1},
        {"field": "avg_final_mfe", "headerName": "Avg Final MFE", "flex": 1},
        {"field": "avg_pnl", "headerName": "Avg PnL", "flex": 1},
        {"field": "runner_pct", "headerName": "Runner %", "flex": 1},
    ]


def _prepare_early_mfe_bucket_rows(df: pd.DataFrame) -> list[dict]:
    work = df.copy()
    needed = ["early_mfe_pnl_10000", "mfe_pnl", "pnl", "runner_flag"]
    if not all(c in work.columns for c in needed):
        return []

    for col in ["early_mfe_pnl_10000", "mfe_pnl", "pnl"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["runner_flag"] = work["runner_flag"].fillna(False).astype(bool)
    work = work.dropna(subset=["early_mfe_pnl_10000", "mfe_pnl", "pnl"]).copy()
    if work.empty:
        return []

    bins = [-np.inf, 0, 10, 20, 30, np.inf]
    labels = ["< 0", "0–10", "10–20", "20–30", "30+"]
    work["early_mfe_bucket"] = pd.cut(
        work["early_mfe_pnl_10000"], bins=bins, labels=labels, include_lowest=True, right=False
    )

    grouped = (
        work.groupby("early_mfe_bucket", observed=False)
        .agg(
            trade_count=("pnl", "size"),
            avg_final_mfe=("mfe_pnl", "mean"),
            avg_pnl=("pnl", "mean"),
            runner_pct=("runner_flag", "mean"),
        )
        .reset_index()
    )
    if grouped.empty:
        return []

    grouped["early_mfe_bucket"] = grouped["early_mfe_bucket"].astype(str)
    grouped["runner_pct"] = 100.0 * pd.to_numeric(grouped["runner_pct"], errors="coerce")
    for col in ["avg_final_mfe", "avg_pnl", "runner_pct"]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").round(4)

    order_map = {label: i for i, label in enumerate(labels)}
    grouped["sort_key"] = grouped["early_mfe_bucket"].map(order_map)
    grouped = grouped.sort_values("sort_key").drop(columns=["sort_key"])
    return grouped.to_dict("records")


def _make_early_mfe_vs_capture_runner_figure(df: pd.DataFrame) -> go.Figure:
    work = df.copy()
    needed = ["runner_flag", "early_mfe_pnl_10000", "capture_ratio"]
    if not all(c in work.columns for c in needed):
        return _empty_figure("Early MFE vs Capture Ratio for Runners")

    for col in ["early_mfe_pnl_10000", "capture_ratio", "time_to_mfe_ticks", "mfe_pnl", "pnl"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["runner_flag"] = work["runner_flag"].fillna(False).astype(bool)
    work = work[work["runner_flag"]].dropna(subset=["early_mfe_pnl_10000", "capture_ratio"]).copy()
    if work.empty:
        return _empty_figure("No runner trades with early MFE / capture data.")

    custom = np.column_stack([
        work["product"].astype(str).to_numpy() if "product" in work.columns else np.array([""] * len(work)),
        work["direction"].astype(str).to_numpy() if "direction" in work.columns else np.array([""] * len(work)),
        work["time_to_mfe_ticks"].to_numpy() if "time_to_mfe_ticks" in work.columns else np.array([np.nan] * len(work)),
        work["mfe_pnl"].to_numpy() if "mfe_pnl" in work.columns else np.array([np.nan] * len(work)),
        work["pnl"].to_numpy() if "pnl" in work.columns else np.array([np.nan] * len(work)),
    ])

    fig = go.Figure(
        data=[go.Scatter(
            x=work["early_mfe_pnl_10000"],
            y=work["capture_ratio"],
            mode="markers",
            marker={"size": 8},
            customdata=custom,
            hovertemplate=(
                "Early MFE (10k)=%{x:.4f}<br>"
                "Capture Ratio=%{y:.4f}<br>"
                "Product=%{customdata[0]}<br>"
                "Direction=%{customdata[1]}<br>"
                "Time to MFE=%{customdata[2]:.2f}<br>"
                "Final MFE=%{customdata[3]:.4f}<br>"
                "Realized PnL=%{customdata[4]:.4f}<extra></extra>"
            ),
            name="Runners only",
        )]
    )
    fig.update_layout(
        title="Early MFE vs Capture Ratio for Runners Only",
        template="plotly_white",
        height=430,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Early MFE in First 10k Ticks")
    fig.update_yaxes(title_text="Capture Ratio")
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
        "entry_liquidity_bucket",
        "exit_liquidity_bucket",
        "round_trip_liquidity_bucket",
        "inventory_reference",
        "inventory_abs",
        "inventory_bucket",
        "imbalance_top3",
        "imbalance_z",
        "spread_z",
        "mid_z",
        "market_timestamp_at_entry",
        "mfe_pnl",
        "mae_pnl",
        "capture_ratio",
        "efficiency_gap",
        "mfe_price_move",
        "mae_price_move",
        "mfe_timestamp",
        "mae_timestamp",
        "time_to_mfe_ticks",
        "time_to_mae_ticks",
        "market_rows_in_trade",
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
        "entry_liquidity_bucket",
        "exit_liquidity_bucket",
        "round_trip_liquidity_bucket",
        "inventory_reference",
        "inventory_abs",
        "inventory_bucket",
        "imbalance_top3",
        "imbalance_z",
        "spread_z",
        "mid_z",
        "market_timestamp_at_entry",
        "mfe_pnl",
        "mae_pnl",
        "capture_ratio",
        "efficiency_gap",
        "mfe_price_move",
        "mae_price_move",
        "market_rows_in_trade",
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