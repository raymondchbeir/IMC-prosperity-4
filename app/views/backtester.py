from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import dash_ag_grid as dag
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from app.backtesting.presets import PRESET_OPTIONS
from app.backtesting.runner import build_custom_data_root, parse_limit_overrides, run_backtests
from app.config import DEFAULT_BACKTEST_PRESET, DEFAULT_LIMIT_OVERRIDES_TEXT, DEFAULT_MATCH_TRADES
from app.models.schemas import BacktestRequest


def build_backtester_layout():
    return html.Div(
        [
            html.H3("Strategy Backtester"),
            html.P(
                "Drop a Trader file here, choose a preset or manual round/day targets, and the dashboard will run the Prosperity 4 backtester and show fills, PnL, logs, and positions."
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
            html.Div(id="backtest-summary-container", style={"marginBottom": "20px"}),
            html.H4("Combined charts"),
            dcc.Graph(id="backtest-pnl-graph"),
            dcc.Graph(id="backtest-execution-graph"),
            dcc.Graph(id="backtest-position-graph"),
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
        Output("backtest-status", "children"),
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
        Output("backtest-trades-container", "children"),
        Output("backtest-activity-container", "children"),
        Output("backtest-sandbox-container", "children"),
        Input("run-backtest-btn", "n_clicks"),
        State("strategy-upload", "contents"),
        State("strategy-upload", "filename"),
        State("backtest-data-upload", "contents"),
        State("backtest-data-upload", "filename"),
        State("backtest-preset-dropdown", "value"),
        State("backtest-targets-input", "value"),
        State("backtest-match-trades-dropdown", "value"),
        State("backtest-limit-overrides", "value"),
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
        selected_round,
        selected_day,
    ):
        empty_fig = _empty_figure("No backtest has been run yet.")
        empty_div = html.Div()

        if not n_clicks:
            return (
                html.Div(),
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
                empty_div,
                empty_div,
                empty_div,
            )

        if not strategy_contents or not strategy_filename:
            return (
                _error_box("Upload a strategy file first."),
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
                empty_div,
                empty_div,
                empty_div,
            )

        try:
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
                )
                payload = run_backtests(
                    strategy_path=strategy_path,
                    request=request,
                    custom_data_root=custom_data_root,
                )

            activity_df = pd.DataFrame(payload.activity_rows)
            trades_df = pd.DataFrame(payload.submission_trade_rows)
            sandbox_df = pd.DataFrame(payload.sandbox_rows)
            per_run_df = pd.DataFrame(payload.per_run_rows)
            per_product_df = pd.DataFrame(payload.per_product_rows)

            status = html.Div(
                [
                    html.Strong("Backtest complete."),
                    html.Span(f" Targets: {', '.join(payload.summary.get('targets', []))}. "),
                    html.Span(
                        "Using custom uploaded data."
                        if request.use_custom_data
                        else "Using bundled backtester round data."
                    ),
                ],
                style={
                    "padding": "10px 14px",
                    "borderRadius": "10px",
                    "backgroundColor": "#eef6ff",
                    "border": "1px solid #cfe3ff",
                },
            )

            return (
                status,
                _build_summary_cards(payload.summary),
                _make_pnl_figure(activity_df),
                _make_execution_figure(activity_df, trades_df),
                _make_position_figure(trades_df),
                _build_product_metric_cards(per_product_df),
                _build_product_graphs_container(activity_df, trades_df, mode="pnl"),
                _build_product_graphs_container(activity_df, trades_df, mode="execution"),
                _build_product_graphs_container(activity_df, trades_df, mode="position"),
                _build_grid_section("Per-run summary", per_run_df),
                _build_grid_section("Per-product summary", per_product_df),
                _build_grid_section("Submission fills", trades_df),
                _build_grid_section("Activity log", activity_df),
                _build_grid_section("Sandbox / lambda logs", sandbox_df),
            )

        except Exception as exc:
            return (
                _error_box(str(exc)),
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
                empty_div,
                empty_div,
                empty_div,
            )


def _build_summary_cards(summary: dict):
    cards = [
        ("Total PnL", f"{summary.get('total_pnl', 0):,.2f}"),
        ("Runs", str(summary.get("num_runs", 0))),
        ("Submission fills", str(summary.get("submission_trade_count", 0))),
        ("Products traded", ", ".join(summary.get("products_traded", [])) or "None"),
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
            "gridTemplateColumns": "repeat(4, minmax(220px, 1fr))",
            "gap": "16px",
        },
    )


def _build_product_metric_cards(per_product_df: pd.DataFrame):
    if per_product_df.empty:
        return html.Div("No per-product metrics available.")

    cards = []
    for _, row in per_product_df.iterrows():
        product = str(row.get("product", "UNKNOWN"))
        pnl = float(row.get("total_final_pnl", 0))
        qty = int(row.get("filled_quantity", 0))
        fills = int(row.get("submission_trades", 0))

        cards.append(
            html.Div(
                [
                    html.H4(product),
                    html.P(f"Final PnL: {pnl:,.2f}"),
                    html.P(f"Filled Quantity: {qty}"),
                    html.P(f"Submission Fills: {fills}"),
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
            "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
            "gap": "16px",
        },
    )


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
        fig.add_trace(
            go.Scatter(
                x=df_product["global_step"],
                y=df_product["mid_price"],
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