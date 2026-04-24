from dash import Dash, dcc, html

import app.backtesting.runner as legacy_runner
from app.backtesting.rust_runner import run_rust_backtests
from app.views import backtester as backtester_view
from app.views.portal_logs import build_portal_logs_layout, register_portal_logs_callbacks
from app.views.upload import get_upload_layout, register_upload_callbacks

# Keep the existing dashboard/backtester UI intact, but route runs through the Rust CLI adapter.
legacy_runner.run_backtests = run_rust_backtests
backtester_view.run_backtests = run_rust_backtests


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "IMC Prosperity Dashboard"


def _append_portal_logs_tab(component):
    """Append the Portal Logs tab without rewriting the existing upload layout module."""
    if getattr(component, "id", None) == "analysis-tabs" and hasattr(component, "children"):
        existing = list(component.children or [])
        if not any(getattr(tab, "value", None) == "portal-logs" for tab in existing):
            existing.append(
                dcc.Tab(
                    label="Portal Logs",
                    value="portal-logs",
                    children=[
                        html.Div(
                            build_portal_logs_layout(),
                            id="portal-logs-container",
                        )
                    ],
                )
            )
        component.children = existing
        return component

    children = getattr(component, "children", None)
    if isinstance(children, list):
        component.children = [_append_portal_logs_tab(child) for child in children]
    elif children is not None and hasattr(children, "children"):
        component.children = _append_portal_logs_tab(children)
    return component


base_upload_layout = _append_portal_logs_tab(get_upload_layout())

app.layout = html.Div(
    [
        html.H1("IMC Prosperity Dashboard"),
        html.P(
            "Upload IMC CSV files to explore the market, compare products, run round-specific analysis, backtest your trader files with the Rust backtester, and analyze IMC portal logs."
        ),
        base_upload_layout,
    ],
    style={
        "maxWidth": "1500px",
        "margin": "0 auto",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
    },
)

register_upload_callbacks(app)
register_portal_logs_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
