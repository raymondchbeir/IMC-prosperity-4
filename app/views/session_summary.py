from __future__ import annotations

import dash_ag_grid as dag
from dash import html

from app.ingestion.session_builder import SessionBuildResult


def _build_file_rows(session: SessionBuildResult) -> list[dict]:
    rows = []
    for file_result in session.files:
        rows.append(
            {
                "source_file": file_result.source_file,
                "dataset_type": file_result.dataset_type,
                "round": file_result.round,
                "day": file_result.day,
                "row_count": file_result.row_count,
                "products": ", ".join(file_result.products),
                "status": file_result.status,
                "warnings": " | ".join(file_result.warnings),
                "errors": " | ".join(file_result.errors),
            }
        )
    return rows


def build_session_summary_components(session: SessionBuildResult):
    column_defs = [
        {"field": "source_file", "headerName": "Filename", "flex": 2},
        {"field": "dataset_type", "headerName": "Type", "flex": 1},
        {"field": "round", "headerName": "Round", "flex": 1},
        {"field": "day", "headerName": "Day", "flex": 1},
        {"field": "row_count", "headerName": "Rows", "flex": 1},
        {"field": "products", "headerName": "Products", "flex": 2},
        {"field": "status", "headerName": "Status", "flex": 1},
        {"field": "warnings", "headerName": "Warnings", "flex": 2},
        {"field": "errors", "headerName": "Errors", "flex": 2},
    ]

    file_rows = _build_file_rows(session)

    summary_cards = html.Div(
        [
            html.Div(
                [
                    html.H4("Available Rounds"),
                    html.P(", ".join(map(str, session.available_rounds)) or "None"),
                ],
                style=_card_style(),
            ),
            html.Div(
                [
                    html.H4("Available Days"),
                    html.P(", ".join(map(str, session.available_days)) or "None"),
                ],
                style=_card_style(),
            ),
            html.Div(
                [
                    html.H4("Available Products"),
                    html.P(", ".join(session.available_products) or "None"),
                ],
                style=_card_style(),
            ),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, minmax(220px, 1fr))",
            "gap": "16px",
            "marginBottom": "20px",
        },
    )

    grid = dag.AgGrid(
        id="session-summary-grid",
        columnDefs=column_defs,
        rowData=file_rows,
        defaultColDef={"resizable": True, "sortable": True, "filter": True},
        dashGridOptions={"pagination": True, "paginationPageSize": 10},
        style={"height": "420px", "width": "100%"},
        className="ag-theme-alpine",
    )

    return html.Div(
        [
            html.H3("Session Summary"),
            summary_cards,
            grid,
        ]
    )


def _card_style():
    return {
        "border": "1px solid #ddd",
        "borderRadius": "10px",
        "padding": "14px",
        "backgroundColor": "#fafafa",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
    }