from dash import Dash, html

from app.views.upload import get_upload_layout, register_upload_callbacks


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "IMC Prosperity Dashboard"

app.layout = html.Div(
    [
        html.H1("IMC Prosperity Dashboard"),
        html.P(
            "Upload IMC CSV files to explore the market, compare products, and run round-specific analysis."
        ),
        get_upload_layout(),
    ],
    style={
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
    },
)

register_upload_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)