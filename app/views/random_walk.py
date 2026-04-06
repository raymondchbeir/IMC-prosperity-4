from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
except Exception:
    acorr_ljungbox = None
    adfuller = None


RANDOM_WALK_MIN_POINTS = 40
RANDOM_WALK_ACF_LAGS = 10
RANDOM_WALK_VR_LAGS = (2, 4, 8)


def build_random_walk_summary_cards(prices_df: pd.DataFrame) -> html.Div:
    diagnostics = compute_random_walk_diagnostics(prices_df)
    interpretation = diagnostics.get("interpretation", "")

    cards = [
        ("Verdict", diagnostics.get("verdict", "No data")),
        ("Interpretation", interpretation or "No interpretation available."),
        ("Observations", diagnostics.get("n_obs", 0)),
        ("Days Used", diagnostics.get("n_days", 0)),
        ("ADF p-value", _fmt_metric(diagnostics.get("adf_pvalue"))),
        ("Ljung-Box p-value", _fmt_metric(diagnostics.get("ljungbox_pvalue"))),
        ("Lag-1 Return ACF", _fmt_metric(diagnostics.get("return_autocorr_lag1"))),
        ("Hurst", _fmt_metric(diagnostics.get("hurst_exponent"))),
        ("VR(2)", _fmt_metric(diagnostics.get("variance_ratio_2"))),
        ("VR(4)", _fmt_metric(diagnostics.get("variance_ratio_4"))),
        ("VR(8)", _fmt_metric(diagnostics.get("variance_ratio_8"))),
        (
            "Supports RW Score",
            f"{diagnostics.get('supporting_checks', 0)}/{diagnostics.get('total_checks', 0)}",
        ),
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.H4(label, style={"marginBottom": "8px"}),
                    html.P(str(value), style={"margin": 0}),
                ],
                style=_metric_card_style(emphasize=(label == "Verdict")),
            )
            for label, value in cards
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
            "gap": "16px",
        },
    )


def make_random_walk_diagnostics_figure(prices_df: pd.DataFrame) -> go.Figure:
    diagnostics = compute_random_walk_diagnostics(prices_df)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Return Autocorrelation", "Variance Ratios"),
        horizontal_spacing=0.12,
    )

    acf_vals = diagnostics.get("return_acf_values", [])
    acf_lags = diagnostics.get("return_acf_lags", [])
    vr_lags = diagnostics.get("variance_ratio_lags", [])
    vr_vals = diagnostics.get("variance_ratio_values", [])

    if acf_lags and acf_vals:
        fig.add_trace(
            go.Bar(
                x=acf_lags,
                y=acf_vals,
                name="Return ACF",
                hovertemplate="Lag=%{x}<br>ACF=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if vr_lags and vr_vals:
        fig.add_trace(
            go.Bar(
                x=[f"VR({lag})" for lag in vr_lags],
                y=vr_vals,
                name="Variance Ratio",
                hovertemplate="%{x}<br>Value=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    conf = diagnostics.get("acf_confidence_band")
    if conf is not None and np.isfinite(conf):
        fig.add_hline(y=conf, line_dash="dash", row=1, col=1)
        fig.add_hline(y=-conf, line_dash="dash", row=1, col=1)

    fig.add_hline(y=0, line_dash="dot", row=1, col=1)
    fig.add_hline(y=1, line_dash="dot", row=1, col=2)

    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_yaxes(title_text="Autocorrelation", row=1, col=1)
    fig.update_xaxes(title_text="Horizon", row=1, col=2)
    fig.update_yaxes(title_text="Variance Ratio", row=1, col=2)

    subtitle = diagnostics.get("figure_subtitle")
    title = "Random Walk Diagnostics"
    if subtitle:
        title = f"{title} — {subtitle}"

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=430,
        showlegend=False,
        margin={"l": 50, "r": 20, "t": 70, "b": 50},
    )
    return fig


def compute_random_walk_diagnostics(prices_df: pd.DataFrame) -> dict:
    base = {
        "verdict": "Insufficient data",
        "interpretation": "Need more filtered price observations to evaluate random-walk-like behavior.",
        "n_obs": 0,
        "n_days": int(prices_df["day"].nunique()) if (not prices_df.empty and "day" in prices_df.columns) else 0,
        "adf_pvalue": np.nan,
        "ljungbox_pvalue": np.nan,
        "return_autocorr_lag1": np.nan,
        "hurst_exponent": np.nan,
        "variance_ratio_2": np.nan,
        "variance_ratio_4": np.nan,
        "variance_ratio_8": np.nan,
        "variance_ratio_lags": list(RANDOM_WALK_VR_LAGS),
        "variance_ratio_values": [],
        "return_acf_lags": list(range(1, RANDOM_WALK_ACF_LAGS + 1)),
        "return_acf_values": [],
        "acf_confidence_band": np.nan,
        "supporting_checks": 0,
        "total_checks": 0,
        "figure_subtitle": "No usable mid-price data",
    }

    if prices_df.empty or "mid_price" not in prices_df.columns:
        return base

    series_df = prices_df.copy()
    sort_cols = [c for c in ["day", "timestamp"] if c in series_df.columns]
    if sort_cols:
        series_df = series_df.sort_values(sort_cols).copy()

    mids = pd.to_numeric(series_df["mid_price"], errors="coerce")
    mids = mids.replace([np.inf, -np.inf], np.nan)
    mids = mids[mids > 0].dropna()

    if len(mids) < RANDOM_WALK_MIN_POINTS:
        base["n_obs"] = int(len(mids))
        base["figure_subtitle"] = f"Only {len(mids)} usable mid-price points"
        return base

    log_price = np.log(mids.astype(float))
    log_return = log_price.diff().dropna()

    if len(log_return) < max(RANDOM_WALK_MIN_POINTS - 1, 20):
        base["n_obs"] = int(len(mids))
        base["figure_subtitle"] = f"Only {len(log_return)} usable return points"
        return base

    acf_lags = list(range(1, min(RANDOM_WALK_ACF_LAGS, len(log_return) - 1) + 1))
    acf_vals = [_safe_autocorr(log_return, lag) for lag in acf_lags]
    vr_vals = [_variance_ratio(log_price, lag) for lag in RANDOM_WALK_VR_LAGS]

    adf_pvalue = _adf_pvalue(log_price)
    ljungbox_pvalue = _ljungbox_pvalue(log_return, max_lag=min(10, len(log_return) - 1))
    hurst_exponent = _hurst_exponent(log_price)
    acf_conf_band = 1.96 / math.sqrt(len(log_return)) if len(log_return) > 0 else np.nan
    acf1 = acf_vals[0] if acf_vals else np.nan

    supporting_checks = 0
    total_checks = 0

    if np.isfinite(adf_pvalue):
        total_checks += 1
        if adf_pvalue > 0.05:
            supporting_checks += 1

    if np.isfinite(ljungbox_pvalue):
        total_checks += 1
        if ljungbox_pvalue > 0.05:
            supporting_checks += 1

    if np.isfinite(acf1):
        total_checks += 1
        if np.isfinite(acf_conf_band) and abs(acf1) <= acf_conf_band:
            supporting_checks += 1

    if np.isfinite(hurst_exponent):
        total_checks += 1
        if 0.45 <= hurst_exponent <= 0.55:
            supporting_checks += 1

    for vr in vr_vals:
        if np.isfinite(vr):
            total_checks += 1
            if abs(vr - 1.0) <= 0.15:
                supporting_checks += 1

    ratio = (supporting_checks / total_checks) if total_checks > 0 else 0.0

    if total_checks == 0:
        verdict = "Insufficient data"
        interpretation = "Not enough valid diagnostics could be computed on this filtered series."
    elif ratio >= 0.75:
        verdict = "Consistent with random walk"
        interpretation = "Price changes look fairly unpredictable in the mean over this selected window."
    elif ratio <= 0.40:
        verdict = "Evidence against random walk"
        interpretation = "The selected series shows structure inconsistent with a simple random walk."
    else:
        verdict = "Mixed evidence"
        interpretation = "Some diagnostics look random-walk-like, but others point to structure or mean reversion/trend."

    if np.isfinite(hurst_exponent):
        if hurst_exponent < 0.45:
            interpretation += " Hurst is below 0.5, which leans mean-reverting."
        elif hurst_exponent > 0.55:
            interpretation += " Hurst is above 0.5, which leans trending."
        else:
            interpretation += " Hurst is near 0.5, which is closer to random-walk-like behavior."

    out = base.copy()
    out.update(
        {
            "verdict": verdict,
            "interpretation": interpretation,
            "n_obs": int(len(mids)),
            "adf_pvalue": adf_pvalue,
            "ljungbox_pvalue": ljungbox_pvalue,
            "return_autocorr_lag1": acf1,
            "hurst_exponent": hurst_exponent,
            "variance_ratio_2": vr_vals[0] if len(vr_vals) > 0 else np.nan,
            "variance_ratio_4": vr_vals[1] if len(vr_vals) > 1 else np.nan,
            "variance_ratio_8": vr_vals[2] if len(vr_vals) > 2 else np.nan,
            "variance_ratio_values": [v for v in vr_vals if np.isfinite(v)],
            "variance_ratio_lags": [
                lag for lag, val in zip(RANDOM_WALK_VR_LAGS, vr_vals) if np.isfinite(val)
            ],
            "return_acf_lags": [lag for lag, val in zip(acf_lags, acf_vals) if np.isfinite(val)],
            "return_acf_values": [val for val in acf_vals if np.isfinite(val)],
            "acf_confidence_band": acf_conf_band,
            "supporting_checks": supporting_checks,
            "total_checks": total_checks,
            "figure_subtitle": verdict,
        }
    )
    return out


def _adf_pvalue(series: pd.Series) -> float:
    if adfuller is None:
        return np.nan
    try:
        return float(adfuller(series.dropna(), autolag="AIC")[1])
    except Exception:
        return np.nan


def _ljungbox_pvalue(series: pd.Series, max_lag: int) -> float:
    if acorr_ljungbox is None or max_lag < 1:
        return np.nan
    try:
        result = acorr_ljungbox(series.dropna(), lags=[max_lag], return_df=True)
        return float(result["lb_pvalue"].iloc[0])
    except Exception:
        return np.nan


def _safe_autocorr(series: pd.Series, lag: int) -> float:
    try:
        if lag <= 0 or len(series) <= lag:
            return np.nan
        return float(series.autocorr(lag=lag))
    except Exception:
        return np.nan


def _variance_ratio(log_price: pd.Series, lag: int) -> float:
    try:
        if lag <= 1 or len(log_price) <= lag:
            return np.nan

        diff_1 = log_price.diff().dropna()
        diff_k = log_price.diff(lag).dropna()

        var_1 = np.var(diff_1, ddof=1)
        var_k = np.var(diff_k, ddof=1)

        if not np.isfinite(var_1) or var_1 <= 0 or not np.isfinite(var_k):
            return np.nan

        return float(var_k / (lag * var_1))
    except Exception:
        return np.nan


def _hurst_exponent(series: pd.Series) -> float:
    try:
        values = pd.Series(series).dropna().astype(float).to_numpy()
        if len(values) < 30:
            return np.nan

        max_lag = min(20, len(values) // 2)
        lags = range(2, max_lag + 1)

        tau = []
        valid_lags = []
        for lag in lags:
            diff = values[lag:] - values[:-lag]
            std = np.std(diff)
            if np.isfinite(std) and std > 0:
                tau.append(std)
                valid_lags.append(lag)

        if len(tau) < 2:
            return np.nan

        poly = np.polyfit(np.log(valid_lags), np.log(tau), 1)
        return float(poly[0])
    except Exception:
        return np.nan


def _fmt_metric(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{float(value):.4f}"


def _metric_card_style(emphasize: bool = False) -> dict:
    return {
        "border": "1px solid #ddd",
        "borderRadius": "10px",
        "padding": "14px",
        "backgroundColor": "#f8fafc" if emphasize else "#fafafa",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
        "borderLeft": "5px solid #2563eb" if emphasize else "1px solid #ddd",
        "minHeight": "92px",
    }
