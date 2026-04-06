from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import pandas as pd
from dash import Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from app.ingestion.session_builder import build_session
from app.rounds.registry import get_round_plugin
from app.views.market_overview import (
    build_market_overview_graphs_layout,
    build_shared_market_controls_layout,
    enrich_market_data,
    make_book_heatmap_figure,
    make_cross_product_figure,
    make_depth_figure,
    make_imbalance_figure,
    make_imbalance_forward_return_figure,
    make_imbalance_histogram_figure,
    make_price_book_figure,
    make_returns_volatility_figure,
    make_spread_figure,
    make_trade_size_histogram_figure,
    make_trade_volume_figure,
)
from app.views.session_summary import build_session_summary_components


def get_upload_layout():
    return html.Div(
        [
            dcc.Store(id="session-data-store"),
            dcc.Upload(
                id="imc-upload",
                children=html.Div(
                    [
                        "Drag and drop IMC CSV files here, or ",
                        html.Span("click to select files", style={"fontWeight": "bold"}),
                    ]
                ),
                multiple=True,
                style={
                    "width": "100%",
                    "height": "90px",
                    "lineHeight": "90px",
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "12px",
                    "textAlign": "center",
                    "marginBottom": "20px",
                    "backgroundColor": "#f8f9fb",
                },
            ),
            html.Div(id="upload-status", style={"marginBottom": "16px"}),
            html.Div(id="session-summary-container"),
            html.Div(id="shared-market-controls-container", style={"marginTop": "20px"}),
            dcc.Tabs(
                id="analysis-tabs",
                value="overview",
                children=[
                    dcc.Tab(
                        label="Overview",
                        value="overview",
                        children=[html.Div(id="market-overview-container")],
                    ),
                    dcc.Tab(
                        label="Round Analysis",
                        value="round-analysis",
                        children=[html.Div(id="round-analysis-container")],
                    ),
                ],
            ),
        ]
    )


def register_upload_callbacks(app):
    @app.callback(
        Output("upload-status", "children"),
        Output("session-summary-container", "children"),
        Output("shared-market-controls-container", "children"),
        Output("market-overview-container", "children"),
        Output("round-analysis-container", "children"),
        Output("session-data-store", "data"),
        Input("imc-upload", "contents"),
        State("imc-upload", "filename"),
        prevent_initial_call=True,
    )
    def handle_upload(contents_list, filenames):
        if not contents_list or not filenames:
            return no_update, no_update, no_update, no_update, no_update, no_update

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                saved_paths = []

                for contents, filename in zip(contents_list, filenames):
                    saved_path = _save_uploaded_file(contents, filename, tmpdir_path)
                    saved_paths.append(saved_path)

                session = build_session(saved_paths)

                status = html.Div(
                    [
                        html.Strong("Upload complete."),
                        html.Span(f" Valid files: {session.valid_file_count}."),
                        html.Span(f" Invalid files: {session.invalid_file_count}."),
                    ],
                    style={
                        "padding": "10px 14px",
                        "borderRadius": "10px",
                        "backgroundColor": "#eef6ff",
                        "border": "1px solid #cfe3ff",
                    },
                )

                summary = build_session_summary_components(session)
                controls_layout = build_shared_market_controls_layout()
                overview_layout = build_market_overview_graphs_layout()

                default_plugin = get_round_plugin(1)
                round_layout = (
                    default_plugin.build_round_analysis_layout()
                    if default_plugin is not None
                    else html.Div("No round plugin registered yet.")
                )

                store_data = {
                    "prices": session.prices_df.to_dict("records") if not session.prices_df.empty else [],
                    "trades": session.trades_df.to_dict("records") if not session.trades_df.empty else [],
                    "available_rounds": session.available_rounds,
                    "available_days": session.available_days,
                    "available_products": session.available_products,
                }

                return status, summary, controls_layout, overview_layout, round_layout, store_data

        except Exception as e:
            error_box = html.Div(
                f"Failed to process uploaded files: {e}",
                style={
                    "padding": "10px 14px",
                    "borderRadius": "10px",
                    "backgroundColor": "#fff1f0",
                    "border": "1px solid #ffccc7",
                    "color": "#a8071a",
                },
            )
            return error_box, html.Div(), html.Div(), html.Div(), html.Div(), None

    @app.callback(
        Output("product-dropdown", "options"),
        Output("product-dropdown", "value"),
        Output("compare-product-dropdown", "options"),
        Input("session-data-store", "data"),
        prevent_initial_call=True,
    )
    def populate_product_dropdown(store_data):
        if not store_data:
            return [], None, []

        products = sorted(store_data.get("available_products", []))
        options = [{"label": p, "value": p} for p in products]
        return options, (products[0] if products else None), options

    @app.callback(
        Output("round-dropdown", "options"),
        Output("round-dropdown", "value"),
        Input("session-data-store", "data"),
        Input("product-dropdown", "value"),
        prevent_initial_call=True,
    )
    def populate_round_dropdown(store_data, selected_product):
        if not store_data or not selected_product:
            return [], None

        prices_df = pd.DataFrame(store_data.get("prices", []))
        trades_df = pd.DataFrame(store_data.get("trades", []))

        rounds = sorted(
            set(
                (
                    prices_df.loc[prices_df["product"] == selected_product, "round"].dropna().astype(int).tolist()
                    if "product" in prices_df.columns and "round" in prices_df.columns
                    else []
                )
                + (
                    trades_df.loc[trades_df["product"] == selected_product, "round"].dropna().astype(int).tolist()
                    if "product" in trades_df.columns and "round" in trades_df.columns
                    else []
                )
            )
        )

        options = [{"label": str(r), "value": r} for r in rounds]
        return options, (rounds[0] if rounds else None)

    @app.callback(
        Output("day-dropdown", "options"),
        Output("day-dropdown", "value"),
        Output("timestamp-range-slider", "min"),
        Output("timestamp-range-slider", "max"),
        Output("timestamp-range-slider", "value"),
        Input("session-data-store", "data"),
        Input("product-dropdown", "value"),
        Input("round-dropdown", "value"),
        State("compare-days-toggle", "value"),
        prevent_initial_call=True,
    )
    def populate_day_dropdown_and_slider(store_data, selected_product, selected_round, compare_days_value):
        if not store_data or not selected_product or selected_round is None:
            return [], None, 0, 1, [0, 1]

        compare_days = "compare_days" in (compare_days_value or [])

        prices_df = pd.DataFrame(store_data.get("prices", []))
        trades_df = pd.DataFrame(store_data.get("trades", []))

        price_days = []
        trade_days = []

        if not prices_df.empty and {"product", "round", "day"}.issubset(prices_df.columns):
            price_days = prices_df[
                (prices_df["product"] == selected_product)
                & (prices_df["round"] == selected_round)
            ]["day"].dropna().astype(int).tolist()

        if not trades_df.empty and {"product", "round", "day"}.issubset(trades_df.columns):
            trade_days = trades_df[
                (trades_df["product"] == selected_product)
                & (trades_df["round"] == selected_round)
            ]["day"].dropna().astype(int).tolist()

        days = sorted(set(price_days + trade_days))
        day_options = [{"label": str(d), "value": d} for d in days]
        day_value = None if compare_days else (days[0] if days else None)

        combined = pd.concat([prices_df, trades_df], ignore_index=True) if (not prices_df.empty or not trades_df.empty) else pd.DataFrame()
        if not combined.empty and {"product", "round", "timestamp"}.issubset(combined.columns):
            combined = combined[
                (combined["product"] == selected_product)
                & (combined["round"] == selected_round)
            ]
            if not compare_days and day_value is not None and "day" in combined.columns:
                combined = combined[combined["day"] == day_value]

        if not combined.empty and "timestamp" in combined.columns:
            tmin = int(combined["timestamp"].min())
            tmax = int(combined["timestamp"].max())
            tvalue = [tmin, tmax]
        else:
            tmin, tmax, tvalue = 0, 1, [0, 1]

        return day_options, day_value, tmin, tmax, tvalue

    @app.callback(
        Output("price-book-graph", "figure"),
        Output("spread-graph", "figure"),
        Output("depth-graph", "figure"),
        Output("trade-volume-graph", "figure"),
        Output("imbalance-graph", "figure"),
        Output("imbalance-histogram-graph", "figure"),
        Output("imbalance-forward-return-graph", "figure"),
        Output("returns-volatility-graph", "figure"),
        Output("book-heatmap-graph", "figure"),
        Output("trade-size-histogram-graph", "figure"),
        Output("cross-product-graph", "figure"),
        Input("session-data-store", "data"),
        Input("product-dropdown", "value"),
        Input("round-dropdown", "value"),
        Input("day-dropdown", "value"),
        Input("compare-days-toggle", "value"),
        Input("compare-product-dropdown", "value"),
        Input("timestamp-range-slider", "value"),
        prevent_initial_call=True,
    )
    def update_market_graphs(store_data, selected_product, selected_round, selected_day, compare_days_value, compare_products, timestamp_range):
        if not store_data or not selected_product or selected_round is None:
            return {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        prices_df, trades_df, cross_df, compare_days = filter_selected_data(
            store_data,
            selected_product,
            selected_round,
            selected_day,
            compare_days_value,
            compare_products,
            timestamp_range,
        )

        prices_df, trades_df = enrich_market_data(prices_df, trades_df)
        cross_df, _ = enrich_market_data(cross_df, pd.DataFrame())

        return (
            make_price_book_figure(prices_df, trades_df, compare_days=compare_days),
            make_spread_figure(prices_df, compare_days=compare_days),
            make_depth_figure(prices_df),
            make_trade_volume_figure(trades_df),
            make_imbalance_figure(prices_df),
            make_imbalance_histogram_figure(prices_df),
            make_imbalance_forward_return_figure(prices_df),
            make_returns_volatility_figure(prices_df),
            make_book_heatmap_figure(prices_df),
            make_trade_size_histogram_figure(trades_df),
            make_cross_product_figure(cross_df, selected_product, compare_products or []),
        )

    @app.callback(
        Output("round-summary-cards", "children"),
        Output("round-edge-histogram-graph", "figure"),
        Output("round-extrema-signal-graph", "figure"),
        Input("session-data-store", "data"),
        Input("product-dropdown", "value"),
        Input("round-dropdown", "value"),
        Input("day-dropdown", "value"),
        Input("compare-days-toggle", "value"),
        Input("compare-product-dropdown", "value"),
        Input("timestamp-range-slider", "value"),
        prevent_initial_call=True,
    )
    def update_round_analysis(
        store_data,
        selected_product,
        selected_round,
        selected_day,
        compare_days_value,
        compare_products,
        timestamp_range,
    ):
        if not store_data or selected_round is None or not selected_product:
            return html.Div(), {}, {}

        plugin = get_round_plugin(selected_round)
        if plugin is None:
            return html.Div("No round-specific plugin registered for this round."), {}, {}

        prices_df, trades_df, _, _ = filter_selected_data(
            store_data,
            selected_product,
            selected_round,
            selected_day,
            compare_days_value,
            compare_products,
            timestamp_range,
        )
        prices_df, trades_df = enrich_market_data(prices_df, trades_df)

        return (
            plugin.build_round_summary_cards(prices_df, trades_df),
            plugin.make_edge_histogram_figure(prices_df),
            plugin.make_extrema_signal_figure(prices_df, trades_df),
        )

    @app.callback(
        Output("download-prices", "data"),
        Input("download-prices-btn", "n_clicks"),
        State("session-data-store", "data"),
        State("product-dropdown", "value"),
        State("round-dropdown", "value"),
        State("day-dropdown", "value"),
        State("compare-days-toggle", "value"),
        State("compare-product-dropdown", "value"),
        State("timestamp-range-slider", "value"),
        prevent_initial_call=True,
    )
    def download_filtered_prices(n_clicks, store_data, selected_product, selected_round, selected_day, compare_days_value, compare_products, timestamp_range):
        if not n_clicks or not store_data:
            raise PreventUpdate

        prices_df, _, _, _ = filter_selected_data(
            store_data,
            selected_product,
            selected_round,
            selected_day,
            compare_days_value,
            compare_products,
            timestamp_range,
        )
        if prices_df.empty:
            raise PreventUpdate

        return dcc.send_data_frame(prices_df.to_csv, "filtered_prices.csv", index=False)

    @app.callback(
        Output("download-trades", "data"),
        Input("download-trades-btn", "n_clicks"),
        State("session-data-store", "data"),
        State("product-dropdown", "value"),
        State("round-dropdown", "value"),
        State("day-dropdown", "value"),
        State("compare-days-toggle", "value"),
        State("compare-product-dropdown", "value"),
        State("timestamp-range-slider", "value"),
        prevent_initial_call=True,
    )
    def download_filtered_trades(n_clicks, store_data, selected_product, selected_round, selected_day, compare_days_value, compare_products, timestamp_range):
        if not n_clicks or not store_data:
            raise PreventUpdate

        _, trades_df, _, _ = filter_selected_data(
            store_data,
            selected_product,
            selected_round,
            selected_day,
            compare_days_value,
            compare_products,
            timestamp_range,
        )
        if trades_df.empty:
            raise PreventUpdate

        return dcc.send_data_frame(trades_df.to_csv, "filtered_trades.csv", index=False)


def filter_selected_data(
    store_data,
    selected_product,
    selected_round,
    selected_day,
    compare_days_value,
    compare_products,
    timestamp_range,
):
    compare_days = "compare_days" in (compare_days_value or [])
    compare_products = compare_products or []

    prices_df = pd.DataFrame(store_data.get("prices", []))
    trades_df = pd.DataFrame(store_data.get("trades", []))
    all_prices_df = prices_df.copy()

    if not prices_df.empty:
        prices_df = prices_df[
            (prices_df["product"] == selected_product)
            & (prices_df["round"] == selected_round)
        ]
        if not compare_days and selected_day is not None:
            prices_df = prices_df[prices_df["day"] == selected_day]

    if not trades_df.empty:
        trades_df = trades_df[
            (trades_df["product"] == selected_product)
            & (trades_df["round"] == selected_round)
        ]
        if not compare_days and selected_day is not None:
            trades_df = trades_df[trades_df["day"] == selected_day]
        elif compare_days:
            trades_df = pd.DataFrame()

    if timestamp_range and len(timestamp_range) == 2:
        t0, t1 = timestamp_range
        if not prices_df.empty:
            prices_df = prices_df[(prices_df["timestamp"] >= t0) & (prices_df["timestamp"] <= t1)]
        if not trades_df.empty:
            trades_df = trades_df[(trades_df["timestamp"] >= t0) & (trades_df["timestamp"] <= t1)]

    cross_df = pd.DataFrame()
    if not all_prices_df.empty:
        compare_set = [selected_product] + [p for p in compare_products if p != selected_product]
        cross_df = all_prices_df[
            (all_prices_df["product"].isin(compare_set))
            & (all_prices_df["round"] == selected_round)
        ]
        if not compare_days and selected_day is not None:
            cross_df = cross_df[cross_df["day"] == selected_day]
        if timestamp_range and len(timestamp_range) == 2:
            t0, t1 = timestamp_range
            cross_df = cross_df[(cross_df["timestamp"] >= t0) & (cross_df["timestamp"] <= t1)]

    return prices_df, trades_df, cross_df, compare_days


def _save_uploaded_file(contents: str, filename: str, target_dir: Path) -> Path:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)

    file_path = target_dir / Path(filename).name
    file_path.write_bytes(decoded)
    return file_path