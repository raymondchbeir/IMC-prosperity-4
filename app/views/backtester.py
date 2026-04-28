from __future__ import annotations

import base64
import math
import tempfile
from pathlib import Path

import dash_ag_grid as dag
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from app.backtesting.presets import PRESET_OPTIONS
from app.backtesting.runner import build_custom_data_root, parse_limit_overrides, run_backtests
from app.config import DEFAULT_BACKTEST_PRESET, DEFAULT_LIMIT_OVERRIDES_TEXT, DEFAULT_MATCH_TRADES
from app.models.schemas import BacktestRequest

MAX_ACTIVITY_ROWS_FOR_UI = 40000
MAX_SUBMISSION_ROWS_FOR_UI = 15000
MAX_SANDBOX_ROWS_FOR_UI = 5000
MAX_REALIZED_ROWS_FOR_UI = 15000


def _build_custom_data_root_from_session_store(session_store: dict, tmpdir_path: Path) -> Path | None:
    if not session_store:
        return None

    prices_df = pd.DataFrame(session_store.get("prices", []))
    trades_df = pd.DataFrame(session_store.get("trades", []))

    if prices_df.empty and trades_df.empty:
        return None

    data_root = tmpdir_path / "custom_backtest_data"
    data_root.mkdir(parents=True, exist_ok=True)

    price_cols = [
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
        "profit_and_loss",
    ]

    if not prices_df.empty:
        for (round_num, day_num), part in prices_df.groupby(["round", "day"], dropna=True):
            round_num = int(round_num)
            day_num = int(day_num)
            round_dir = data_root / f"round{round_num}"
            round_dir.mkdir(parents=True, exist_ok=True)

            out = part.copy()
            for col in price_cols:
                if col not in out.columns:
                    out[col] = pd.NA
            out = out[price_cols]
            out.to_csv(
                round_dir / f"prices_round_{round_num}_day_{day_num}.csv",
                sep=";",
                index=False,
            )

    if not trades_df.empty:
        for (round_num, day_num), part in trades_df.groupby(["round", "day"], dropna=True):
            round_num = int(round_num)
            day_num = int(day_num)
            round_dir = data_root / f"round{round_num}"
            round_dir.mkdir(parents=True, exist_ok=True)

            out = part.copy()
            if "product" in out.columns and "symbol" not in out.columns:
                out = out.rename(columns={"product": "symbol"})

            trade_cols = ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"]
            for col in trade_cols:
                if col not in out.columns:
                    out[col] = pd.NA

            out = out[trade_cols]
            out.to_csv(
                round_dir / f"trades_round_{round_num}_day_{day_num}.csv",
                sep=";",
                index=False,
            )

    return data_root


def build_backtester_layout():
    return html.Div(
        [
            dcc.Store(id="backtest-payload-store", data={}),
            html.H3("Strategy Backtester"),
            html.P(
                "Drop a Trader file here, choose a preset or manual round/day targets, and the dashboard will run the Prosperity 4 backtester and show fills, PnL, realized trades, logs, and positions."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Strategy file (.py)"),
                            dcc.Upload(
                                id="strategy-upload",
                                children=html.Div(
                                    [
                                        "Drop strategy file here or ",
                                        html.Span("click to select", style={"fontWeight": "bold"}),
                                    ]
                                ),
                                multiple=False,
                                style=_upload_box_style(),
                            ),
                            html.Div(id="strategy-upload-status", style={"marginTop": "8px"}),
                        ],
                        style={"flex": "2"},
                    ),
                    html.Div(
                        [
                            html.Label("Optional: custom backtest CSV data"),
                            dcc.Upload(
                                id="backtest-data-upload",
                                children=html.Div(
                                    [
                                        "Drop prices/trades/observations files here or ",
                                        html.Span("click to select", style={"fontWeight": "bold"}),
                                    ]
                                ),
                                multiple=True,
                                style=_upload_box_style(),
                            ),
                        ],
                        style={"flex": "3"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Preset"),
                            dcc.Dropdown(
                                id="backtest-preset-dropdown",
                                options=PRESET_OPTIONS,
                                value=DEFAULT_BACKTEST_PRESET,
                                clearable=False,
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                    html.Div(
                        [
                            html.Label("Targets"),
                            dcc.Input(
                                id="backtest-targets-input",
                                type="text",
                                value="0",
                                placeholder="Examples: 0, 1, 1-0, 1--1 1-0",
                                style={"width": "100%", "height": "38px", "padding": "0 10px"},
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                    html.Div(
                        [
                            html.Label("Match trades"),
                            dcc.Dropdown(
                                id="backtest-match-trades-dropdown",
                                options=[
                                    {"label": "all", "value": "all"},
                                    {"label": "worse", "value": "worse"},
                                    {"label": "none", "value": "none"},
                                ],
                                value=DEFAULT_MATCH_TRADES,
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Label("Extra market access"),
                            dcc.Dropdown(
                                id="backtest-extra-volume-dropdown",
                                options=[
                                    {"label": "Off", "value": 0.0},
                                    {"label": "+25% extra volume", "value": 0.25},
                                ],
                                value=0.0,
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Limit overrides"),
                            dcc.Textarea(
                                id="backtest-limit-overrides",
                                value=DEFAULT_LIMIT_OVERRIDES_TEXT,
                                style={"width": "100%", "height": "90px", "padding": "10px"},
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                    html.Div(
                        [
                            html.Label("Run"),
                            html.Button(
                                "Run backtest",
                                id="run-backtest-btn",
                                n_clicks=0,
                                style={
                                    "height": "44px",
                                    "padding": "0 18px",
                                    "borderRadius": "8px",
                                    "border": "1px solid #ccc",
                                    "backgroundColor": "#f3f4f6",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "justifyContent": "end"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
            ),
            html.Div(id="backtest-status", style={"marginBottom": "20px"}),
            html.Div(
                [
                    html.Button(
                        "Export Realized Trades CSV",
                        id="export-realized-trades-btn",
                        n_clicks=0,
                        disabled=True,
                        style={
                            "height": "42px",
                            "padding": "0 18px",
                            "borderRadius": "8px",
                            "border": "1px solid #ccc",
                            "backgroundColor": "#f3f4f6",
                            "cursor": "pointer",
                            "marginBottom": "16px",
                        },
                    ),
                    dcc.Download(id="download-realized-trades"),
                ],
                style={"marginBottom": "8px"},
            ),
            html.Div(id="backtest-summary-container", style={"marginBottom": "20px"}),
            html.H4("Combined charts"),
            dcc.Graph(id="backtest-pnl-graph", figure=_empty_figure("No backtest has been run yet.")),
            dcc.Graph(id="backtest-execution-graph", figure=_empty_figure("No backtest has been run yet.")),
            dcc.Graph(id="backtest-position-graph", figure=_empty_figure("No backtest has been run yet.")),
            html.H4("Per-product summary"),
            html.Div(id="backtest-product-summary-cards-container", style={"marginBottom": "20px"}),
            html.H4("PnL by product"),
            html.Div(id="backtest-pnl-by-product-container", style={"marginBottom": "20px"}),
            html.H4("Executions by product"),
            html.Div(id="backtest-execution-by-product-container", style={"marginBottom": "20px"}),
            html.H4("Positions by product"),
            html.Div(id="backtest-position-by-product-container", style={"marginBottom": "20px"}),
            html.Div(id="backtest-per-run-container", style={"marginBottom": "20px"}),
            html.Div(id="backtest-per-product-container", style={"marginBottom": "20px"}),
            html.Div(
                [
                    html.H4("Winning trades"),
                    html.P("Click one row to show that trade on the price graph below."),
                    dag.AgGrid(
                        id="backtest-winning-trades-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={
                            "pagination": True,
                            "paginationPageSize": 20,
                            "rowSelection": "single",
                            "suppressRowClickSelection": False,
                        },
                        style={"height": "360px", "width": "100%"},
                        className="ag-theme-alpine",
                    ),
                    dcc.Graph(
                        id="backtest-winning-trade-graph",
                        figure=_empty_figure("Run a backtest, then click a winning trade row."),
                    ),
                ],
                style={"marginBottom": "24px"},
            ),
            html.Div(
                [
                    html.H4("Losing trades"),
                    html.P("Click one row to show that trade on the price graph below."),
                    dag.AgGrid(
                        id="backtest-losing-trades-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={
                            "pagination": True,
                            "paginationPageSize": 20,
                            "rowSelection": "single",
                            "suppressRowClickSelection": False,
                        },
                        style={"height": "360px", "width": "100%"},
                        className="ag-theme-alpine",
                    ),
                    dcc.Graph(
                        id="backtest-losing-trade-graph",
                        figure=_empty_figure("Run a backtest, then click a losing trade row."),
                    ),
                ],
                style={"marginBottom": "24px"},
            ),
            html.Div(id="backtest-trades-container", style={"marginBottom": "20px"}),
            html.Div(id="backtest-activity-container", style={"marginBottom": "20px"}),
            html.Div(id="backtest-sandbox-container", style={"marginBottom": "20px"}),
        ]
    )


def register_backtester_callbacks(app):
    @app.callback(
        Output("strategy-upload-status", "children"),
        Input("strategy-upload", "filename"),
    )
    def show_strategy_upload_status(strategy_filename):
        if not strategy_filename:
            return html.Div(
                "No strategy uploaded yet.",
                style={
                    "padding": "8px 10px",
                    "borderRadius": "8px",
                    "backgroundColor": "#fafafa",
                    "border": "1px solid #ddd",
                    "color": "#555",
                },
            )

        return html.Div(
            [
                html.Strong(f"{strategy_filename} uploaded. "),
                html.Span("Ready to use in the backtester."),
            ],
            style={
                "padding": "8px 10px",
                "borderRadius": "8px",
                "backgroundColor": "#eef6ff",
                "border": "1px solid #cfe3ff",
                "color": "#1d4f91",
            },
        )

    @app.callback(
        Output("backtest-targets-input", "value"),
        Output("backtest-targets-input", "disabled"),
        Input("backtest-preset-dropdown", "value"),
        Input("round-dropdown", "value"),
        Input("day-dropdown", "value"),
    )
    def sync_backtest_targets(preset, selected_round, selected_day):
        if preset == "manual":
            return "", False
        if preset == "tutorial_round_0":
            return "0", True
        if preset == "selected_dashboard_round":
            return (str(selected_round) if selected_round is not None else ""), True
        if preset == "selected_dashboard_round_day":
            if selected_round is not None and selected_day is not None:
                return f"{selected_round}-{selected_day}", True
            return "", True
        return "", False

    @app.callback(
        Output("backtest-status", "children"),
        Output("backtest-payload-store", "data"),
        Input("run-backtest-btn", "n_clicks"),
        State("strategy-upload", "contents"),
        State("strategy-upload", "filename"),
        State("backtest-data-upload", "contents"),
        State("backtest-data-upload", "filename"),
        State("backtest-preset-dropdown", "value"),
        State("backtest-targets-input", "value"),
        State("backtest-match-trades-dropdown", "value"),
        State("backtest-limit-overrides", "value"),
        State("backtest-extra-volume-dropdown", "value"),
        State("round-dropdown", "value"),
        State("day-dropdown", "value"),
        prevent_initial_call=True,
    )
    def run_dashboard_backtest(
        n_clicks,
        strategy_contents,
        strategy_filename,
        data_contents_list,
        data_filenames,
        preset,
        targets_text,
        match_trades,
        limit_text,
        extra_volume_pct,
        selected_round,
        selected_day,
    ):
        if not n_clicks:
            return html.Div(), {}

        if not strategy_contents or not strategy_filename:
            return _error_box("Upload a strategy file first."), {}

        try:
            print(
                "RUN BACKTEST CALLBACK:",
                {
                    "n_clicks": n_clicks,
                    "strategy_filename": strategy_filename,
                    "preset": preset,
                    "targets_text": targets_text,
                    "selected_round": selected_round,
                    "selected_day": selected_day,
                    "extra_volume_pct": extra_volume_pct,
                    "has_custom_uploads": bool(data_contents_list and data_filenames),
                },
            )

            limit_overrides = parse_limit_overrides(limit_text, preset)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                strategy_path = _save_uploaded_file(strategy_contents, strategy_filename, tmpdir_path)

                custom_data_root = None
                if data_contents_list and data_filenames:
                    saved_data_filenames = []
                    for contents, filename in zip(data_contents_list, data_filenames):
                        _save_uploaded_file(contents, filename, tmpdir_path)
                        saved_data_filenames.append(filename)
                    custom_data_root = build_custom_data_root(tmpdir_path, saved_data_filenames)

                request = BacktestRequest(
                    preset=preset,
                    match_trades=match_trades,
                    targets_text=targets_text or "",
                    selected_round=selected_round,
                    selected_day=selected_day,
                    limit_overrides=limit_overrides,
                    use_custom_data=custom_data_root is not None,
                    extra_volume_pct=float(extra_volume_pct or 0.0),
                )
                payload = run_backtests(
                    strategy_path=strategy_path,
                    request=request,
                    custom_data_root=custom_data_root,
                )

            status = html.Div(
                [
                    html.Strong("Backtest complete."),
                    html.Span(f" Targets: {', '.join(payload.summary.get('targets', []))}. "),
                    html.Span(
                        "Using custom uploaded data."
                        if request.use_custom_data
                        else "Using local backtester round data."
                    ),
                    html.Span(
                        f" Extra market access: {int(round(100 * float(payload.summary.get('extra_volume_pct', 0.0))))}%.",
                    ),
                    html.Span(
                        _format_timing_line(payload.summary.get("timings", {})),
                        style={"display": "block", "marginTop": "6px", "fontSize": "13px", "color": "#4b5563"},
                    ),
                ],
                style={
                    "padding": "10px 14px",
                    "borderRadius": "10px",
                    "backgroundColor": "#eef6ff",
                    "border": "1px solid #cfe3ff",
                },
            )

            payload_data = {
                "summary": payload.summary,
                "targets": [target.label for target in payload.targets],
                    "activity_rows": _downsample_activity_rows(payload.activity_rows, MAX_ACTIVITY_ROWS_FOR_UI),
                    "submission_trade_rows": _limit_rows_for_ui(payload.submission_trade_rows, MAX_SUBMISSION_ROWS_FOR_UI),
                    "realized_trade_rows": _limit_rows_for_ui(payload.realized_trade_rows, MAX_REALIZED_ROWS_FOR_UI),
                    "sandbox_rows": _limit_rows_for_ui(payload.sandbox_rows, MAX_SANDBOX_ROWS_FOR_UI),
                "per_run_rows": payload.per_run_rows,
                "per_product_rows": payload.per_product_rows,
            }

            return status, payload_data

        except Exception as exc:
            print("RUN BACKTEST ERROR:", repr(exc))
            return _error_box(str(exc)), {}

    @app.callback(
        Output("backtest-summary-container", "children"),
        Output("backtest-pnl-graph", "figure"),
        Output("backtest-execution-graph", "figure"),
        Output("backtest-position-graph", "figure"),
        Output("backtest-product-summary-cards-container", "children"),
        Output("backtest-pnl-by-product-container", "children"),
        Output("backtest-execution-by-product-container", "children"),
        Output("backtest-position-by-product-container", "children"),
        Output("backtest-per-run-container", "children"),
        Output("backtest-per-product-container", "children"),
        Output("backtest-winning-trades-grid", "columnDefs"),
        Output("backtest-winning-trades-grid", "rowData"),
        Output("backtest-losing-trades-grid", "columnDefs"),
        Output("backtest-losing-trades-grid", "rowData"),
        Output("backtest-trades-container", "children"),
        Output("backtest-activity-container", "children"),
        Output("backtest-sandbox-container", "children"),
        Input("backtest-payload-store", "data"),
    )
    def render_backtest_payload(payload_data):
        empty_fig = _empty_figure("No backtest has been run yet.")
        empty_div = html.Div()
        simple_cols = _simple_trade_table_columns()

        if not payload_data:
            return (
                html.Div(),
                empty_fig,
                empty_fig,
                empty_fig,
                empty_div,
                empty_div,
                empty_div,
                empty_div,
                empty_div,
                empty_div,
                simple_cols,
                [],
                simple_cols,
                [],
                empty_div,
                empty_div,
                empty_div,
            )

        activity_df = pd.DataFrame(payload_data.get("activity_rows", []))
        trades_df = pd.DataFrame(payload_data.get("submission_trade_rows", []))
        realized_df = pd.DataFrame(payload_data.get("realized_trade_rows", []))
        sandbox_df = pd.DataFrame(payload_data.get("sandbox_rows", []))
        per_run_df = pd.DataFrame(payload_data.get("per_run_rows", []))
        per_product_df = pd.DataFrame(payload_data.get("per_product_rows", []))
        summary = payload_data.get("summary", {})

        winning_df = realized_df[realized_df["pnl"] > 0].copy() if not realized_df.empty else pd.DataFrame()
        losing_df = realized_df[realized_df["pnl"] < 0].copy() if not realized_df.empty else pd.DataFrame()

        return (
            _build_summary_cards(summary),
            _make_pnl_figure(activity_df),
            _make_execution_figure(activity_df, trades_df),
            _make_position_figure(trades_df),
            _build_product_metric_cards(per_product_df),
            _build_product_graphs_container(activity_df, trades_df, mode="pnl"),
            _build_product_graphs_container(activity_df, trades_df, mode="execution"),
            _build_product_graphs_container(activity_df, trades_df, mode="position"),
            _build_grid_section("Per-run summary", per_run_df),
            _build_grid_section("Per-product summary", per_product_df),
            simple_cols,
            _prepare_simple_trade_rows(winning_df, descending=True),
            simple_cols,
            _prepare_simple_trade_rows(losing_df, descending=False),
            _build_grid_section("Submission fills", trades_df),
            _build_grid_section("Activity log", activity_df),
            _build_grid_section("Sandbox / lambda logs", sandbox_df),
        )

    @app.callback(
        Output("backtest-winning-trade-graph", "figure"),
        Input("backtest-payload-store", "data"),
        Input("backtest-winning-trades-grid", "selectedRows"),
    )
    def update_winning_trade_graph(payload_data, selected_rows):
        if not payload_data:
            return _empty_figure("Run a backtest, then click a winning trade row.")

        activity_df = pd.DataFrame(payload_data.get("activity_rows", []))
        realized_df = pd.DataFrame(payload_data.get("realized_trade_rows", []))

        winning_df = realized_df[realized_df["pnl"] > 0].copy() if not realized_df.empty else pd.DataFrame()
        if winning_df.empty:
            return _empty_figure("No winning trades were realized in this backtest.")

        if not selected_rows:
            return _empty_figure("Click a winning trade row to show its entry and exit on the price chart.")

        return _make_trade_focus_figure(activity_df, selected_rows[0], title_prefix="Winning trade")

    @app.callback(
        Output("backtest-losing-trade-graph", "figure"),
        Input("backtest-payload-store", "data"),
        Input("backtest-losing-trades-grid", "selectedRows"),
    )
    def update_losing_trade_graph(payload_data, selected_rows):
        if not payload_data:
            return _empty_figure("Run a backtest, then click a losing trade row.")

        activity_df = pd.DataFrame(payload_data.get("activity_rows", []))
        realized_df = pd.DataFrame(payload_data.get("realized_trade_rows", []))

        losing_df = realized_df[realized_df["pnl"] < 0].copy() if not realized_df.empty else pd.DataFrame()
        if losing_df.empty:
            return _empty_figure("No losing trades were realized in this backtest.")

        if not selected_rows:
            return _empty_figure("Click a losing trade row to show its entry and exit on the price chart.")

        return _make_trade_focus_figure(activity_df, selected_rows[0], title_prefix="Losing trade")

    @app.callback(
        Output("export-realized-trades-btn", "disabled"),
        Input("backtest-payload-store", "data"),
    )
    def toggle_export_button(payload_data):
        if not payload_data:
            return True

        realized_rows = payload_data.get("realized_trade_rows", [])
        return len(realized_rows) == 0

    @app.callback(
        Output("download-realized-trades", "data"),
        Input("export-realized-trades-btn", "n_clicks"),
        State("backtest-payload-store", "data"),
        prevent_initial_call=True,
    )
    def export_realized_trades_csv(n_clicks, payload_data):
        print("EXPORT CLICKED:", n_clicks)

        if not n_clicks:
            raise PreventUpdate

        if not payload_data:
            print("No payload data available for export.")
            empty_df = pd.DataFrame({"message": ["No backtest run yet"]})
            return dcc.send_data_frame(
                empty_df.to_csv,
                "realized_trades.csv",
                index=False,
            )

        realized_rows = payload_data.get("realized_trade_rows", [])
        print("REALIZED TRADE ROW COUNT:", len(realized_rows))

        realized_df = pd.DataFrame(realized_rows)

        if realized_df.empty:
            print("No realized trades found. Exporting placeholder CSV.")
            realized_df = pd.DataFrame({"message": ["No realized trades"]})

        return dcc.send_data_frame(
            realized_df.to_csv,
            "realized_trades.csv",
            index=False,
        )


def _build_summary_cards(summary: dict):
    cards = [
        ("Total PnL", _fmt_number(summary.get("total_pnl"), decimals=2)),
        ("Runs", str(summary.get("num_runs", 0))),
        ("Submission fills", str(summary.get("submission_trade_count", 0))),
        ("Realized trades", str(summary.get("realized_trade_count", 0))),
        ("Winning trades", str(summary.get("winning_trade_count", 0))),
        ("Losing trades", str(summary.get("losing_trade_count", 0))),
        ("Average win / trade", _fmt_number(summary.get("average_win_per_trade"), decimals=2)),
        ("Win rate", _fmt_pct(summary.get("win_rate"))),
        ("Products traded", ", ".join(summary.get("products_traded", [])) or "None"),
        ("Extra market access", f"{int(round(100 * float(summary.get('extra_volume_pct', 0.0))))}%"),
    ]
    return html.Div(
        [
            html.Div(
                [html.H4(label), html.P(value)],
                style={
                    "border": "1px solid #ddd",
                    "borderRadius": "10px",
                    "padding": "14px",
                    "backgroundColor": "#fafafa",
                    "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                },
            )
            for label, value in cards
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
            "gap": "16px",
        },
    )


def _build_product_metric_cards(per_product_df: pd.DataFrame):
    if per_product_df.empty:
        return html.Div("No per-product metrics available.")

    cards = []
    for _, row in per_product_df.iterrows():
        product = str(row.get("product", "UNKNOWN"))
        pnl = row.get("total_final_pnl")
        qty = int(row.get("filled_quantity", 0))
        fills = int(row.get("submission_trades", 0))
        realized = int(row.get("realized_trade_count", 0))
        wins = int(row.get("winning_trade_count", 0))
        losses = int(row.get("losing_trade_count", 0))
        avg_win = row.get("average_win_per_trade")

        cards.append(
            html.Div(
                [
                    html.H4(product),
                    html.P(f"Final PnL: {_fmt_number(pnl, decimals=2)}"),
                    html.P(f"Filled Quantity: {qty}"),
                    html.P(f"Submission Fills: {fills}"),
                    html.P(f"Realized Trades: {realized}"),
                    html.P(f"Wins / Losses: {wins} / {losses}"),
                    html.P(f"Average Win / Trade: {_fmt_number(avg_win, decimals=2)}"),
                ],
                style={
                    "border": "1px solid #ddd",
                    "borderRadius": "10px",
                    "padding": "14px",
                    "backgroundColor": "#fafafa",
                    "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                },
            )
        )

    return html.Div(
        cards,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
            "gap": "16px",
        },
    )


def _simple_trade_table_columns() -> list[dict]:
    return [
        {"field": "product", "headerName": "Product", "flex": 1},
        {"field": "quantity", "headerName": "Qty", "flex": 1},
        {"field": "entry_timestamp", "headerName": "Entry Ts", "flex": 1},
        {"field": "exit_timestamp", "headerName": "Exit Ts", "flex": 1},
        {"field": "pnl", "headerName": "PnL", "flex": 1},
    ]


def _prepare_simple_trade_rows(df: pd.DataFrame, descending: bool) -> list[dict]:
    if df.empty:
        return []

    out = df.copy()
    keep_sort_cols = ["pnl", "product", "entry_timestamp"]
    for col in keep_sort_cols:
        if col not in out.columns:
            out[col] = None

    out = out.sort_values(
        ["pnl", "product", "entry_timestamp"],
        ascending=[not descending, True, True],
    ).reset_index(drop=True)

    return out.to_dict("records")


def _build_grid_section(title: str, df: pd.DataFrame):
    if df.empty:
        return html.Div([html.H4(title), html.P("No rows to display.")])

    column_defs = [
        {
            "field": col,
            "headerName": col.replace("_", " ").title(),
            "resizable": True,
            "sortable": True,
            "filter": True,
        }
        for col in df.columns
    ]
    return html.Div(
        [
            html.H4(title),
            dag.AgGrid(
                columnDefs=column_defs,
                rowData=df.to_dict("records"),
                defaultColDef={"resizable": True, "sortable": True, "filter": True},
                dashGridOptions={"pagination": True, "paginationPageSize": 25},
                style={"height": "460px", "width": "100%"},
                className="ag-theme-alpine",
            ),
        ]
    )


def _get_all_products(activity_df: pd.DataFrame, trades_df: pd.DataFrame) -> list[str]:
    products = set()

    if not activity_df.empty and "product" in activity_df.columns:
        products.update(activity_df["product"].dropna().astype(str).tolist())

    if not trades_df.empty and "product" in trades_df.columns:
        products.update(trades_df["product"].dropna().astype(str).tolist())

    return sorted(products)


def _build_product_graphs_container(activity_df: pd.DataFrame, trades_df: pd.DataFrame, mode: str):
    products = _get_all_products(activity_df, trades_df)
    if not products:
        return html.Div("No product-level backtest data available.")

    blocks = []
    for product in products:
        if mode == "pnl":
            fig = _make_product_pnl_figure(activity_df, product)
            title = f"{product} PnL"
        elif mode == "execution":
            fig = _make_product_execution_figure(activity_df, trades_df, product)
            title = f"{product} executions"
        elif mode == "position":
            fig = _make_product_position_figure(trades_df, product)
            title = f"{product} positions"
        else:
            continue

        blocks.append(
            html.Div(
                [
                    html.H5(title),
                    dcc.Graph(figure=fig),
                ],
                style={
                    "border": "1px solid #e5e7eb",
                    "borderRadius": "12px",
                    "padding": "12px",
                    "marginBottom": "16px",
                    "backgroundColor": "#ffffff",
                },
            )
        )

    return html.Div(blocks)


def _make_pnl_figure(activity_df: pd.DataFrame) -> go.Figure:
    if activity_df.empty:
        return _empty_figure("No activity logs returned.")

    curve = activity_df.groupby(["global_step", "run_label"], as_index=False)["profit_loss"].sum()
    curve = curve.sort_values("global_step")

    fig = go.Figure()
    for run_label, df_run in curve.groupby("run_label"):
        fig.add_trace(
            go.Scatter(
                x=df_run["global_step"],
                y=df_run["profit_loss"],
                mode="lines",
                name=run_label,
            )
        )

    fig.update_layout(
        title="Marked-to-market PnL (combined across all products)",
        template="plotly_white",
        height=380,
        hovermode="x unified",
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Global step")
    fig.update_yaxes(title_text="PnL")
    return fig


def _make_product_pnl_figure(activity_df: pd.DataFrame, product: str) -> go.Figure:
    df = activity_df[activity_df["product"] == product].copy()
    if df.empty:
        return _empty_figure(f"No PnL data returned for {product}.")

    curve = df.groupby(["global_step", "run_label"], as_index=False)["profit_loss"].sum()
    curve = curve.sort_values("global_step")

    fig = go.Figure()
    for run_label, df_run in curve.groupby("run_label"):
        fig.add_trace(
            go.Scatter(
                x=df_run["global_step"],
                y=df_run["profit_loss"],
                mode="lines",
                name=run_label,
            )
        )

    fig.update_layout(
        title=f"{product} marked-to-market PnL",
        template="plotly_white",
        height=350,
        hovermode="x unified",
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Global step")
    fig.update_yaxes(title_text="PnL")
    return fig


def _make_execution_figure(activity_df: pd.DataFrame, trades_df: pd.DataFrame) -> go.Figure:
    if activity_df.empty:
        return _empty_figure("No prices available from backtest activity log.")

    fig = go.Figure()

    for product, df_product in activity_df.groupby("product"):
        df_plot = df_product.copy()
        df_plot["mid_price"] = pd.to_numeric(df_plot["mid_price"], errors="coerce")
        df_plot.loc[df_plot["mid_price"] <= 0, "mid_price"] = np.nan
        df_plot = df_plot[df_plot["mid_price"].notna()]
        if df_plot.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=df_plot["global_step"],
                y=df_plot["mid_price"],
                mode="lines",
                name=f"{product} mid",
            )
        )

    if not trades_df.empty:
        for side in ["Buy", "Sell"]:
            df_side = trades_df[trades_df["side"] == side]
            if df_side.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df_side["global_step"],
                    y=df_side["price"],
                    mode="markers",
                    name=f"{side} fills",
                    marker={"size": 10, "symbol": "circle" if side == "Buy" else "x"},
                    customdata=df_side[["product", "quantity", "run_label", "timestamp"]].to_numpy(),
                    hovertemplate="Product=%{customdata[0]}<br>Qty=%{customdata[1]}<br>Run=%{customdata[2]}<br>Timestamp=%{customdata[3]}<br>Price=%{y}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Where your fills happened (all products)",
        template="plotly_white",
        height=430,
        hovermode="closest",
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Global step")
    fig.update_yaxes(title_text="Price")
    return fig


def _make_product_execution_figure(activity_df: pd.DataFrame, trades_df: pd.DataFrame, product: str) -> go.Figure:
    product_activity = activity_df[activity_df["product"] == product].copy()
    product_trades = trades_df[trades_df["product"] == product].copy()

    if product_activity.empty:
        return _empty_figure(f"No execution price data returned for {product}.")

    product_activity["mid_price"] = pd.to_numeric(product_activity["mid_price"], errors="coerce")
    product_activity.loc[product_activity["mid_price"] <= 0, "mid_price"] = np.nan
    product_activity = product_activity[product_activity["mid_price"].notna()]
    if product_activity.empty:
        return _empty_figure(f"No valid execution price data returned for {product}.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=product_activity["global_step"],
            y=product_activity["mid_price"],
            mode="lines",
            name=f"{product} mid",
        )
    )

    if not product_trades.empty:
        for side in ["Buy", "Sell"]:
            df_side = product_trades[product_trades["side"] == side]
            if df_side.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df_side["global_step"],
                    y=df_side["price"],
                    mode="markers",
                    name=f"{side} fills",
                    marker={"size": 10, "symbol": "circle" if side == "Buy" else "x"},
                    customdata=df_side[["quantity", "run_label", "timestamp"]].to_numpy(),
                    hovertemplate="Qty=%{customdata[0]}<br>Run=%{customdata[1]}<br>Timestamp=%{customdata[2]}<br>Price=%{y}<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"{product} executions",
        template="plotly_white",
        height=350,
        hovermode="closest",
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Global step")
    fig.update_yaxes(title_text="Price")
    return fig


def _make_position_figure(trades_df: pd.DataFrame) -> go.Figure:
    if trades_df.empty:
        return _empty_figure("No submission trades were filled.")

    fig = go.Figure()
    for product, df_product in trades_df.groupby("product"):
        df_product = df_product.sort_values(["round", "day", "timestamp"]).copy()
        df_product["global_trade_index"] = range(len(df_product))
        fig.add_trace(
            go.Scatter(
                x=df_product["global_trade_index"],
                y=df_product["position_after_trade"],
                mode="lines+markers",
                name=product,
                customdata=df_product[["run_label", "timestamp", "side", "price", "quantity"]].to_numpy(),
                hovertemplate="Run=%{customdata[0]}<br>Timestamp=%{customdata[1]}<br>Side=%{customdata[2]}<br>Price=%{customdata[3]}<br>Qty=%{customdata[4]}<br>Position=%{y}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Position after each fill (all products)",
        template="plotly_white",
        height=380,
        hovermode="closest",
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Fill index")
    fig.update_yaxes(title_text="Position")
    return fig


def _make_product_position_figure(trades_df: pd.DataFrame, product: str) -> go.Figure:
    df_product = trades_df[trades_df["product"] == product].copy()
    if df_product.empty:
        return _empty_figure(f"No filled trades returned for {product}.")

    df_product = df_product.sort_values(["round", "day", "timestamp"]).copy()
    df_product["product_fill_index"] = range(len(df_product))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_product["product_fill_index"],
            y=df_product["position_after_trade"],
            mode="lines+markers",
            name=product,
            customdata=df_product[["run_label", "timestamp", "side", "price", "quantity"]].to_numpy(),
            hovertemplate="Run=%{customdata[0]}<br>Timestamp=%{customdata[1]}<br>Side=%{customdata[2]}<br>Price=%{customdata[3]}<br>Qty=%{customdata[4]}<br>Position=%{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{product} position after each fill",
        template="plotly_white",
        height=350,
        hovermode="closest",
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(title_text="Fill index")
    fig.update_yaxes(title_text="Position")
    return fig


def _make_trade_focus_figure(activity_df: pd.DataFrame, trade_row: dict, title_prefix: str) -> go.Figure:
    if activity_df.empty:
        return _empty_figure("No activity log data is available for this trade.")

    product = trade_row.get("product")
    round_num = trade_row.get("round")
    day_num = trade_row.get("day")
    run_label = trade_row.get("run_label")
    entry_ts = trade_row.get("entry_timestamp")
    exit_ts = trade_row.get("exit_timestamp")
    entry_price = trade_row.get("entry_price")
    exit_price = trade_row.get("exit_price")
    qty = trade_row.get("quantity")
    pnl = trade_row.get("pnl")
    direction = trade_row.get("direction")

    df = activity_df.copy()
    if "product" in df.columns:
        df = df[df["product"] == product]
    if "round" in df.columns and round_num is not None:
        df = df[df["round"] == round_num]
    if "run_day" in df.columns and day_num is not None:
        df = df[df["run_day"] == day_num]

    if df.empty:
        return _empty_figure("Could not find matching price data for the selected trade.")

    df = df.sort_values("timestamp").copy()

    for col in ["bid_price_1", "ask_price_1", "mid_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] <= 0, col] = np.nan

    fig = go.Figure()

    if "bid_price_1" in df.columns and df["bid_price_1"].notna().any():
        df_bid = df[df["bid_price_1"].notna()]
        fig.add_trace(
            go.Scatter(
                x=df_bid["timestamp"],
                y=df_bid["bid_price_1"],
                mode="lines",
                name="Bid 1",
            )
        )

    if "ask_price_1" in df.columns and df["ask_price_1"].notna().any():
        df_ask = df[df["ask_price_1"].notna()]
        fig.add_trace(
            go.Scatter(
                x=df_ask["timestamp"],
                y=df_ask["ask_price_1"],
                mode="lines",
                name="Ask 1",
            )
        )

    if "mid_price" in df.columns and df["mid_price"].notna().any():
        df_mid = df[df["mid_price"].notna()]
        fig.add_trace(
            go.Scatter(
                x=df_mid["timestamp"],
                y=df_mid["mid_price"],
                mode="lines",
                name="Mid Price",
                line={"width": 3},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[entry_ts],
            y=[entry_price],
            mode="markers+text",
            name="Entry",
            text=["Entry"],
            textposition="top center",
            marker={"size": 14, "symbol": "triangle-up"},
            hovertemplate="Entry<br>Timestamp=%{x}<br>Price=%{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[exit_ts],
            y=[exit_price],
            mode="markers+text",
            name="Exit",
            text=["Exit"],
            textposition="top center",
            marker={"size": 14, "symbol": "triangle-down"},
            hovertemplate="Exit<br>Timestamp=%{x}<br>Price=%{y}<extra></extra>",
        )
    )

    fig.add_vline(x=entry_ts, line_dash="dash")
    fig.add_vline(x=exit_ts, line_dash="dash")

    title = (
        f"{title_prefix}: {product} | {run_label} | {direction} | "
        f"Qty={qty} | PnL={_fmt_number(pnl, 2)}"
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=430,
        hovermode="x unified",
        margin={"l": 50, "r": 20, "t": 70, "b": 50},
        annotations=[
            dict(
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                text=(
                    f"Product: {product}<br>"
                    f"Run: {run_label}<br>"
                    f"Direction: {direction}<br>"
                    f"Quantity: {qty}<br>"
                    f"Entry ts: {entry_ts} @ {_fmt_number(entry_price, 2)}<br>"
                    f"Exit ts: {exit_ts} @ {_fmt_number(exit_price, 2)}<br>"
                    f"PnL: {_fmt_number(pnl, 2)}"
                ),
                bordercolor="#d1d5db",
                borderwidth=1,
                bgcolor="#ffffff",
            )
        ],
    )
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Price")
    return fig


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=320,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
    )
    return fig


def _upload_box_style():
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


def _error_box(message: str):
    return html.Div(
        f"Backtest failed: {message}",
        style={
            "padding": "10px 14px",
            "borderRadius": "10px",
            "backgroundColor": "#fff1f0",
            "border": "1px solid #ffccc7",
            "color": "#a8071a",
        },
    )


def _save_uploaded_file(contents: str, filename: str, target_dir: Path) -> Path:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)

    file_path = target_dir / Path(filename).name
    file_path.write_bytes(decoded)
    return file_path


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


def _fmt_pct(value) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.2f}%"


def _format_timing_line(timings: dict) -> str:
    if not isinstance(timings, dict) or not timings:
        return ""

    total = timings.get("total_seconds")
    rust = timings.get("rust_seconds")
    parse = timings.get("parse_seconds")
    payload = timings.get("payload_seconds")

    parts = []
    if total is not None:
        parts.append(f"Total: {float(total):.2f}s")
    if rust is not None:
        parts.append(f"Rust: {float(rust):.2f}s")
    if parse is not None:
        parts.append(f"Parse: {float(parse):.2f}s")
    if payload is not None:
        parts.append(f"Payload: {float(payload):.2f}s")

    if not parts:
        return ""
    return "Timing - " + ", ".join(parts)


def _limit_rows_for_ui(rows: list[dict], max_rows: int) -> list[dict]:
    if not rows or max_rows <= 0 or len(rows) <= max_rows:
        return rows
    return rows[:max_rows]


def _downsample_activity_rows(rows: list[dict], max_rows: int) -> list[dict]:
    if not rows or max_rows <= 0 or len(rows) <= max_rows:
        return rows

    df = pd.DataFrame(rows)
    if df.empty:
        return rows[:max_rows]

    if "run_label" not in df.columns:
        df["run_label"] = "Run"
    if "product" not in df.columns:
        df["product"] = "UNKNOWN"
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    elif "global_step" in df.columns:
        df["timestamp"] = pd.to_numeric(df["global_step"], errors="coerce")
    else:
        df["timestamp"] = np.arange(len(df))

    out_parts: list[pd.DataFrame] = []
    grouped = list(df.groupby(["run_label", "product"], sort=False))
    budget_per_group = max(100, max_rows // max(1, len(grouped)))

    for _, group in grouped:
        group = group.sort_values("timestamp", kind="mergesort")
        if len(group) <= budget_per_group:
            out_parts.append(group)
            continue

        take = np.linspace(0, len(group) - 1, num=budget_per_group, dtype=int)
        out_parts.append(group.iloc[take])

    if not out_parts:
        return rows[:max_rows]

    sampled = pd.concat(out_parts, ignore_index=True)
    sampled = sampled.sort_values(["run_label", "product", "timestamp"], kind="mergesort")
    if len(sampled) > max_rows:
        sampled = sampled.iloc[:max_rows]
    return sampled.to_dict("records")