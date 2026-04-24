from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import dash_ag_grid as dag
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

from app.portal_logs.parser import parse_portal_files
from app.views.backtester import (
    _build_grid_section,
    _build_product_graphs_container,
    _build_product_metric_cards,
    _build_summary_cards,
    _empty_figure,
    _error_box,
    _make_execution_figure,
    _make_pnl_figure,
    _make_position_figure,
    _make_trade_focus_figure,
    _prepare_simple_trade_rows,
    _simple_trade_table_columns,
    _upload_box_style,
)


def build_portal_logs_layout():
    return html.Div(
        [
            dcc.Store(id="portal-log-payload-store", data={}),
            html.H3("Portal Log Analyzer"),
            html.P(
                "Upload IMC portal .log/.json outputs. This tab parses portal activity, fills, sandbox/lambda logs, and bot/market trades, then renders the same style of diagnostics as the Backtester tab."
            ),
            dcc.Upload(
                id="portal-log-upload",
                children=html.Div(
                    [
                        "Drop portal .log/.json files here or ",
                        html.Span("click to select", style={"fontWeight": "bold"}),
                    ]
                ),
                multiple=True,
                style=_upload_box_style(),
            ),
            html.Div(id="portal-log-status", style={"marginTop": "10px", "marginBottom": "20px"}),
            html.Div(id="portal-log-summary-container", style={"marginBottom": "20px"}),
            html.H4("Combined charts"),
            dcc.Graph(id="portal-log-pnl-graph", figure=_empty_figure("Upload portal logs to begin.")),
            dcc.Graph(id="portal-log-execution-graph", figure=_empty_figure("Upload portal logs to begin.")),
            dcc.Graph(id="portal-log-position-graph", figure=_empty_figure("Upload portal logs to begin.")),
            html.H4("Per-product summary"),
            html.Div(id="portal-log-product-summary-cards-container", style={"marginBottom": "20px"}),
            html.H4("PnL by product"),
            html.Div(id="portal-log-pnl-by-product-container", style={"marginBottom": "20px"}),
            html.H4("Executions by product"),
            html.Div(id="portal-log-execution-by-product-container", style={"marginBottom": "20px"}),
            html.H4("Positions by product"),
            html.Div(id="portal-log-position-by-product-container", style={"marginBottom": "20px"}),
            html.Div(
                [
                    html.H4("Winning trades"),
                    html.P("Click one row to show the realized trade on the price graph below."),
                    dag.AgGrid(
                        id="portal-log-winning-trades-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={"pagination": True, "paginationPageSize": 20, "rowSelection": "single"},
                        style={"height": "360px", "width": "100%"},
                        className="ag-theme-alpine",
                    ),
                    dcc.Graph(id="portal-log-winning-trade-graph", figure=_empty_figure("Click a winning trade row.")),
                ],
                style={"marginBottom": "24px"},
            ),
            html.Div(
                [
                    html.H4("Losing trades"),
                    html.P("Click one row to show the realized trade on the price graph below."),
                    dag.AgGrid(
                        id="portal-log-losing-trades-grid",
                        columnDefs=[],
                        rowData=[],
                        defaultColDef={"resizable": True, "sortable": True, "filter": True},
                        dashGridOptions={"pagination": True, "paginationPageSize": 20, "rowSelection": "single"},
                        style={"height": "360px", "width": "100%"},
                        className="ag-theme-alpine",
                    ),
                    dcc.Graph(id="portal-log-losing-trade-graph", figure=_empty_figure("Click a losing trade row.")),
                ],
                style={"marginBottom": "24px"},
            ),
            html.H3("Bot / Market Trade Distribution Heatmaps"),
            html.P(
                "Bot trades are inferred as portal tradeHistory entries where neither buyer nor seller is SUBMISSION. Each symbol gets its own heatmap panel. Color is total quantity traded, so bright cells show where bots traded the most."
            ),
            dcc.Graph(id="portal-log-bot-scatter-graph", figure=_empty_figure("Upload portal logs to begin.")),
            dcc.Graph(id="portal-log-bot-price-distribution-graph", figure=_empty_figure("Upload portal logs to begin.")),
            dcc.Graph(id="portal-log-bot-size-histogram-graph", figure=_empty_figure("Upload portal logs to begin.")),
            html.H4("Bot trade distribution by price"),
            html.Div(id="portal-log-bot-distribution-container", style={"marginBottom": "20px"}),
            html.H3("Normalized Bot Distribution Heatmaps"),
            html.P("For each symbol: x-axis is time, y-axis is distance from mid price in price units/ticks, and color is total quantity. Example: all bot trades at mid go into row 0; trades one tick above mid go into row +1."),
            dcc.Graph(id="portal-log-bot-normalized-graph", figure=_empty_figure("Upload portal logs to begin.")),
            dcc.Graph(id="portal-log-bot-spread-units-graph", figure=_empty_figure("Upload portal logs to begin.")),
            html.Div(id="portal-log-bot-normalized-container", style={"marginBottom": "20px"}),
            html.Div(id="portal-log-biggest-bot-trades-container", style={"marginBottom": "20px"}),
            html.Div(id="portal-log-submission-trades-container", style={"marginBottom": "20px"}),
            html.Div(id="portal-log-activity-container", style={"marginBottom": "20px"}),
            html.Div(id="portal-log-sandbox-container", style={"marginBottom": "20px"}),
        ]
    )


def register_portal_logs_callbacks(app):
    @app.callback(
        Output("portal-log-status", "children"),
        Output("portal-log-payload-store", "data"),
        Input("portal-log-upload", "contents"),
        State("portal-log-upload", "filename"),
        prevent_initial_call=True,
    )
    def parse_uploaded_portal_logs(contents_list, filenames):
        if not contents_list or not filenames:
            return html.Div(), {}
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                saved_paths = []
                for contents, filename in zip(contents_list, filenames):
                    saved_paths.append(_save_uploaded_file(contents, filename, tmpdir_path))

                payload = parse_portal_files(saved_paths)

            data = {
                "metadata": payload.metadata,
                "summary": payload.summary,
                "activity_rows": payload.activity_rows,
                "submission_trade_rows": payload.submission_trade_rows,
                "realized_trade_rows": payload.realized_trade_rows,
                "bot_trade_rows": payload.bot_trade_rows,
                "sandbox_rows": payload.sandbox_rows,
                "per_product_rows": payload.per_product_rows,
                "bot_distribution_rows": payload.bot_distribution_rows,
                "bot_normalized_distribution_rows": payload.bot_normalized_distribution_rows,
            }
            status = html.Div(
                [
                    html.Strong("Portal logs parsed."),
                    html.Span(f" Files: {', '.join(filenames)}. "),
                    html.Span(f"Submission fills: {len(payload.submission_trade_rows)}. "),
                    html.Span(f"Bot/market trades: {len(payload.bot_trade_rows)}."),
                ],
                style={"padding": "10px 14px", "borderRadius": "10px", "backgroundColor": "#eef6ff", "border": "1px solid #cfe3ff"},
            )
            return status, data
        except Exception as exc:
            return _error_box(str(exc)), {}

    @app.callback(
        Output("portal-log-summary-container", "children"),
        Output("portal-log-pnl-graph", "figure"),
        Output("portal-log-execution-graph", "figure"),
        Output("portal-log-position-graph", "figure"),
        Output("portal-log-product-summary-cards-container", "children"),
        Output("portal-log-pnl-by-product-container", "children"),
        Output("portal-log-execution-by-product-container", "children"),
        Output("portal-log-position-by-product-container", "children"),
        Output("portal-log-winning-trades-grid", "columnDefs"),
        Output("portal-log-winning-trades-grid", "rowData"),
        Output("portal-log-losing-trades-grid", "columnDefs"),
        Output("portal-log-losing-trades-grid", "rowData"),
        Output("portal-log-bot-scatter-graph", "figure"),
        Output("portal-log-bot-price-distribution-graph", "figure"),
        Output("portal-log-bot-size-histogram-graph", "figure"),
        Output("portal-log-bot-distribution-container", "children"),
        Output("portal-log-bot-normalized-graph", "figure"),
        Output("portal-log-bot-spread-units-graph", "figure"),
        Output("portal-log-bot-normalized-container", "children"),
        Output("portal-log-biggest-bot-trades-container", "children"),
        Output("portal-log-submission-trades-container", "children"),
        Output("portal-log-activity-container", "children"),
        Output("portal-log-sandbox-container", "children"),
        Input("portal-log-payload-store", "data"),
    )
    def render_portal_log_payload(payload_data):
        empty_fig = _empty_figure("Upload portal logs to begin.")
        empty_div = html.Div()
        simple_cols = _simple_trade_table_columns()
        if not payload_data:
            return (
                empty_div, empty_fig, empty_fig, empty_fig, empty_div, empty_div, empty_div, empty_div,
                simple_cols, [], simple_cols, [], empty_fig, empty_fig, empty_fig, empty_div,
                empty_fig, empty_fig, empty_div, empty_div, empty_div, empty_div, empty_div,
            )

        activity_df = pd.DataFrame(payload_data.get("activity_rows", []))
        submission_df = pd.DataFrame(payload_data.get("submission_trade_rows", []))
        realized_df = pd.DataFrame(payload_data.get("realized_trade_rows", []))
        bot_df = pd.DataFrame(payload_data.get("bot_trade_rows", []))
        sandbox_df = pd.DataFrame(payload_data.get("sandbox_rows", []))
        per_product_df = pd.DataFrame(payload_data.get("per_product_rows", []))
        bot_dist_df = pd.DataFrame(payload_data.get("bot_distribution_rows", []))
        bot_norm_df = pd.DataFrame(payload_data.get("bot_normalized_distribution_rows", []))
        summary = payload_data.get("summary", {})

        winning_df = realized_df[realized_df["pnl"] > 0].copy() if not realized_df.empty and "pnl" in realized_df.columns else pd.DataFrame()
        losing_df = realized_df[realized_df["pnl"] < 0].copy() if not realized_df.empty and "pnl" in realized_df.columns else pd.DataFrame()
        biggest_bot = _prepare_biggest_bot_trades(bot_df)

        return (
            _build_summary_cards(summary),
            _make_pnl_figure(activity_df),
            _make_execution_figure(activity_df, submission_df),
            _make_position_figure(submission_df),
            _build_product_metric_cards(per_product_df),
            _build_product_graphs_container(activity_df, submission_df, mode="pnl"),
            _build_product_graphs_container(activity_df, submission_df, mode="execution"),
            _build_product_graphs_container(activity_df, submission_df, mode="position"),
            simple_cols,
            _prepare_simple_trade_rows(winning_df, descending=True),
            simple_cols,
            _prepare_simple_trade_rows(losing_df, descending=False),
            _make_bot_scatter_figure(bot_df),
            _make_bot_price_level_heatmap(bot_df),
            _make_bot_size_histogram(bot_df),
            _build_grid_section("Bot distribution by product / price / inferred side", bot_dist_df),
            _make_bot_normalized_heatmap(bot_df),
            _make_bot_spread_level_heatmap(bot_df),
            _build_grid_section("Normalized bot distribution", bot_norm_df),
            _build_grid_section("Biggest bot / market trades", biggest_bot),
            _build_grid_section("Submission fills", submission_df),
            _build_grid_section("Activity log", activity_df),
            _build_grid_section("Sandbox / lambda logs", sandbox_df),
        )

    @app.callback(
        Output("portal-log-winning-trade-graph", "figure"),
        Input("portal-log-payload-store", "data"),
        Input("portal-log-winning-trades-grid", "selectedRows"),
    )
    def update_portal_winning_trade_graph(payload_data, selected_rows):
        if not payload_data or not selected_rows:
            return _empty_figure("Click a winning trade row to show it on the price chart.")
        return _make_trade_focus_figure(pd.DataFrame(payload_data.get("activity_rows", [])), selected_rows[0], title_prefix="Portal winning trade")

    @app.callback(
        Output("portal-log-losing-trade-graph", "figure"),
        Input("portal-log-payload-store", "data"),
        Input("portal-log-losing-trades-grid", "selectedRows"),
    )
    def update_portal_losing_trade_graph(payload_data, selected_rows):
        if not payload_data or not selected_rows:
            return _empty_figure("Click a losing trade row to show it on the price chart.")
        return _make_trade_focus_figure(pd.DataFrame(payload_data.get("activity_rows", [])), selected_rows[0], title_prefix="Portal losing trade")


def _save_uploaded_file(contents: str, filename: str, target_dir: Path) -> Path:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    file_path = target_dir / Path(filename).name
    file_path.write_bytes(decoded)
    return file_path


def _prepare_biggest_bot_trades(bot_df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    if bot_df.empty:
        return pd.DataFrame()
    out = bot_df.copy()
    if "gross_notional" not in out.columns and {"price", "quantity"}.issubset(out.columns):
        out["gross_notional"] = out["price"] * out["quantity"]
    sort_col = "quantity" if "quantity" in out.columns else out.columns[0]
    keep = [
        c for c in [
            "timestamp", "product", "price", "quantity", "gross_notional", "inferred_side",
            "mid_price", "spread", "price_minus_mid", "price_minus_mid_bps", "spread_units_from_mid",
            "buyer", "seller",
        ] if c in out.columns
    ]
    return out.sort_values(sort_col, ascending=False)[keep].head(n).reset_index(drop=True)


def _make_bot_scatter_figure(bot_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not bot_df.empty and {"timestamp", "price", "quantity", "product"}.issubset(bot_df.columns):
        for product, part in bot_df.groupby("product"):
            sizes = np.clip(np.sqrt(pd.to_numeric(part["quantity"], errors="coerce").fillna(0)) * 3.0, 5, 26)
            fig.add_trace(go.Scatter(
                x=part["timestamp"], y=part["price"], mode="markers", name=str(product),
                marker={"size": sizes, "opacity": 0.72},
                customdata=np.stack([part.get("quantity", pd.Series([None] * len(part))), part.get("inferred_side", pd.Series([None] * len(part)))], axis=-1),
                hovertemplate="Timestamp=%{x}<br>Price=%{y}<br>Qty=%{customdata[0]}<br>Side=%{customdata[1]}<extra></extra>",
            ))
    fig.update_layout(title="Bot / Market Trades Over Time", template="plotly_white", height=460, hovermode="closest", legend={"orientation": "h"})
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text="Price")
    return fig


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
    )
    return fig


def _make_bot_price_level_heatmap(bot_df: pd.DataFrame) -> go.Figure:
    if bot_df.empty or not {"timestamp", "price", "quantity", "product"}.issubset(bot_df.columns):
        return _empty_figure("No bot trades available for price-level heatmap.")
    df = bot_df.copy()
    df["time_bucket"] = _time_bucket_series(df["timestamp"], max_bins=80)
    df["price_bucket"] = _price_level_series(df["price"], max_levels=120)
    return _faceted_heatmap(
        df,
        y_col="price_bucket",
        title="Where Bots Traded Most: Price Level × Time by Symbol",
        x_title="Timestamp bucket",
        y_title="Price level",
        color_title="Total Qty",
    )


def _make_bot_size_histogram(bot_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not bot_df.empty and "quantity" in bot_df.columns:
        fig.add_trace(go.Histogram(x=pd.to_numeric(bot_df["quantity"], errors="coerce").dropna(), nbinsx=40, name="Bot trade size"))
    fig.update_layout(title="Bot / Market Trade Size Distribution", template="plotly_white", height=340)
    fig.update_xaxes(title_text="Quantity")
    fig.update_yaxes(title_text="Count")
    return fig


def _make_bot_normalized_heatmap(bot_df: pd.DataFrame) -> go.Figure:
    if bot_df.empty or not {"timestamp", "quantity", "product", "price_minus_mid"}.issubset(bot_df.columns):
        return _empty_figure("No normalized bot trades available.")
    df = bot_df.dropna(subset=["price_minus_mid"]).copy()
    if df.empty:
        return _empty_figure("No normalized bot trades available.")
    df["time_bucket"] = _time_bucket_series(df["timestamp"], max_bins=80)
    df["normalized_level"] = _normalized_level_series(df["price_minus_mid"], step=1.0)
    return _faceted_heatmap(
        df,
        y_col="normalized_level",
        title="Normalized Bot Distance: Price-Minus-Mid Level × Time by Symbol",
        x_title="Timestamp bucket",
        y_title="Distance from mid",
        color_title="Total Qty",
    )


def _make_bot_spread_level_heatmap(bot_df: pd.DataFrame) -> go.Figure:
    needed = {"spread", "quantity", "product", "spread_units_from_mid"}
    if bot_df.empty or not needed.issubset(bot_df.columns):
        return _empty_figure("No spread-level bot trades available.")
    df = bot_df.dropna(subset=["spread", "spread_units_from_mid"]).copy()
    if df.empty:
        return _empty_figure("No spread-level bot trades available.")

    spread_num = pd.to_numeric(df["spread"], errors="coerce")
    if spread_num.dropna().nunique() <= 30:
        df["spread_bucket"] = spread_num.astype("Int64").astype(str)
    else:
        df["spread_bucket"] = pd.cut(spread_num, bins=30, duplicates="drop").astype(str)

    bins = [-5, -2, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 2, 5]
    df["execution_level"] = pd.cut(df["spread_units_from_mid"], bins=bins, include_lowest=True, duplicates="drop").astype(str)

    products = sorted(df["product"].dropna().astype(str).unique().tolist())
    if not products:
        return _empty_figure("No spread-level bot trades available.")

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
        title="Where Bots Trade by Spread: Execution Level × Spread by Symbol",
        template="plotly_white",
        height=max(460, 310 * len(products)),
        coloraxis={"colorbar": {"title": "Total Qty"}},
        showlegend=False,
        hovermode="closest",
    )
    return fig
