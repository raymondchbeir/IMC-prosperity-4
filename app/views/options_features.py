"""
Options Stats tab for a Dash/Plotly trading dashboard.

Drop this file into app/views/options_stats.py or wherever your dashboard keeps tab modules.
It is intentionally defensive: it accepts common column names from Prosperity-style price,
activity, and trades DataFrames, then builds all of the option diagnostics discussed:

1. Volatility smile + fitted IV curve
2. IV residuals over time
3. Black-Scholes fair price vs market price
4. Mispricing z-score / signal panel
5. Autocorrelation / mean reversion stats
6. Gamma scalping EV panel
7. Greeks exposure dashboard
8. Underlying mean reversion dashboard
9. Strategy attribution panel
10. Risk dashboard

Expected usage pattern:
    from app.views.options_stats import build_options_stats_tab, register_options_stats_callbacks

    # In layout/tab router:
    build_options_stats_tab()

    # During app setup:
    register_options_stats_callbacks(app, get_backtest_data)

Where get_backtest_data is a function that returns either:
    (activity_df, trades_df)
or:
    {"activity": activity_df, "trades": trades_df, "positions": positions_df}

If your app already keeps the latest backtest payload in dcc.Store, you can also adapt the
callback inputs at the bottom to read from that store.
"""

from __future__ import annotations

import base64
import io
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from dash import Dash, Input, Output, State, callback, dcc, html, dash_table
    from dash.exceptions import PreventUpdate
except Exception:  # pragma: no cover
    Dash = Any  # type: ignore
    Input = Output = State = callback = dcc = html = dash_table = PreventUpdate = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

UNDERLYING_ALIASES = (
    "VOLCANIC_ROCK",
    "VOLCANIC ROCK",
    "VR",
    "UNDERLYING",
)

OPTION_NAME_HINTS = (
    "VOUCHER",
    "CALL",
    "OPTION",
)

DEFAULT_DAYS_TO_EXPIRY = 7.0
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_ROLLING_WINDOW = 40
DEFAULT_Z_WINDOW = 80
EPS = 1e-12


@dataclass(frozen=True)
class OptionsStatsConfig:
    underlying_aliases: tuple[str, ...] = UNDERLYING_ALIASES
    option_name_hints: tuple[str, ...] = OPTION_NAME_HINTS
    default_days_to_expiry: float = DEFAULT_DAYS_TO_EXPIRY
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    z_window: int = DEFAULT_Z_WINDOW
    signal_entry_z: float = 1.5
    signal_exit_z: float = 0.35


# =============================================================================
# General helpers
# =============================================================================

def _empty_fig(title: str, message: str = "No usable data found") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_layout(title=title, template="plotly_white", height=420)
    return fig


def _normalize_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    lower_map = {c.lower(): c for c in out.columns}

    rename = {}
    candidates = {
        "timestamp": ["timestamp", "time", "t", "ts"],
        "product": ["product", "symbol", "instrument", "asset"],
        "bid_price_1": ["bid_price_1", "best_bid", "bid", "best_bid_price"],
        "ask_price_1": ["ask_price_1", "best_ask", "ask", "best_ask_price"],
        "bid_volume_1": ["bid_volume_1", "best_bid_volume", "bid_volume", "bid_qty"],
        "ask_volume_1": ["ask_volume_1", "best_ask_volume", "ask_volume", "ask_qty"],
        "mid_price": ["mid_price", "mid", "midprice", "fair_mid"],
        "price": ["price", "trade_price", "fill_price"],
        "quantity": ["quantity", "qty", "volume", "size"],
        "side": ["side", "direction"],
        "position": ["position", "pos", "inventory"],
        "realized_pnl": ["realized_pnl", "pnl", "profit", "profit_loss"],
    }
    for canonical, names in candidates.items():
        if canonical in out.columns:
            continue
        for name in names:
            if name in lower_map:
                rename[lower_map[name]] = canonical
                break
    out = out.rename(columns=rename)

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    if "product" in out.columns:
        out["product"] = out["product"].astype(str)

    for col in [
        "bid_price_1",
        "ask_price_1",
        "bid_volume_1",
        "ask_volume_1",
        "mid_price",
        "price",
        "quantity",
        "position",
        "realized_pnl",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "mid_price" not in out.columns and {"bid_price_1", "ask_price_1"}.issubset(out.columns):
        out["mid_price"] = (out["bid_price_1"] + out["ask_price_1"]) / 2.0

    return out


def _read_semicolon_safe_csv(path_or_buffer: Any, filename: str | None = None) -> pd.DataFrame:
    """Read Prosperity-style CSVs that may be semicolon-delimited or accidentally parsed as one column."""
    if path_or_buffer is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path_or_buffer, sep=None, engine="python")
    except Exception:
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            df = pd.read_csv(path_or_buffer, sep=";")
        except Exception:
            return pd.DataFrame()
    if df.shape[1] == 1:
        first_col = df.columns[0]
        header_has_semicolon = ";" in str(first_col)
        values_have_semicolon = df[first_col].astype(str).head(5).str.contains(";").any()
        if header_has_semicolon or values_have_semicolon:
            try:
                if hasattr(path_or_buffer, "seek"):
                    path_or_buffer.seek(0)
                    return pd.read_csv(path_or_buffer, sep=";")
                text = "\n".join([str(first_col)] + df[first_col].astype(str).tolist())
                return pd.read_csv(io.StringIO(text), sep=";")
            except Exception:
                return df
    return df


def _parse_upload_contents(contents: str | None, filename: str | None = None) -> pd.DataFrame:
    """Parse a Dash dcc.Upload contents string into a DataFrame."""
    if not contents:
        return pd.DataFrame()
    try:
        _prefix, b64data = contents.split(",", 1)
        decoded = base64.b64decode(b64data)
        text = decoded.decode("utf-8-sig", errors="replace")
        return _read_semicolon_safe_csv(io.StringIO(text), filename)
    except Exception:
        return pd.DataFrame()


def load_options_csv_files(price_files: Iterable[Any], trade_files: Optional[Iterable[Any]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate raw Round 3-style price/trade CSV files from paths or file-like objects."""
    price_frames: list[pd.DataFrame] = []
    for f in price_files or []:
        df = _read_semicolon_safe_csv(f, str(f))
        if not df.empty:
            price_frames.append(df)
    trade_frames: list[pd.DataFrame] = []
    for f in trade_files or []:
        df = _read_semicolon_safe_csv(f, str(f))
        if not df.empty:
            trade_frames.append(df)
    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return _normalize_columns(prices), _normalize_columns(trades)


def parse_uploaded_options_csvs(
    price_contents: Optional[list[str] | str],
    price_filenames: Optional[list[str] | str] = None,
    trade_contents: Optional[list[str] | str] = None,
    trade_filenames: Optional[list[str] | str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse files uploaded through dcc.Upload. Supports one or many files."""
    if isinstance(price_contents, str):
        price_contents = [price_contents]
    if isinstance(price_filenames, str) or price_filenames is None:
        price_filenames = [price_filenames] * len(price_contents or [])
    if isinstance(trade_contents, str):
        trade_contents = [trade_contents]
    if isinstance(trade_filenames, str) or trade_filenames is None:
        trade_filenames = [trade_filenames] * len(trade_contents or [])

    price_frames = []
    for contents, name in zip(price_contents or [], price_filenames or []):
        df = _parse_upload_contents(contents, name)
        if not df.empty:
            price_frames.append(df)

    trade_frames = []
    for contents, name in zip(trade_contents or [], trade_filenames or []):
        df = _parse_upload_contents(contents, name)
        if not df.empty:
            trade_frames.append(df)

    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return _normalize_columns(prices), _normalize_columns(trades)


def _uploaded_file_list(names: Any, kind: str) -> Any:
    if html is None:
        return None
    if not names:
        return html.Div(f"No {kind} CSVs uploaded yet.", style={"fontSize": "12px", "color": "#777"})
    if isinstance(names, str):
        names = [names]
    return html.Ul([html.Li(str(n)) for n in names], style={"margin": "4px 0 0 18px", "fontSize": "12px"})


def _infer_strike(product: str) -> Optional[float]:
    # Works for names like VOLCANIC_ROCK_VOUCHER_10000 or CALL_9750.
    nums = re.findall(r"(?<!\d)(\d{3,6})(?!\d)", str(product))
    if not nums:
        return None
    # For competition names, the strike is usually the last big integer.
    return float(nums[-1])


def _is_option_product(product: str, cfg: OptionsStatsConfig) -> bool:
    upper = str(product).upper()
    if any(h in upper for h in cfg.option_name_hints):
        return _infer_strike(upper) is not None
    return _infer_strike(upper) is not None and not _is_underlying_product(upper, cfg)


def _is_underlying_product(product: str, cfg: OptionsStatsConfig) -> bool:
    upper = str(product).upper()
    return any(alias in upper for alias in cfg.underlying_aliases) and not any(
        hint in upper for hint in cfg.option_name_hints
    )


def _safe_div(a: Any, b: Any) -> Any:
    return np.where(np.abs(b) > EPS, a / b, np.nan)


def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def _norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# =============================================================================
# Black-Scholes, implied volatility, and Greeks
# =============================================================================

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K * math.exp(-r * max(T, 0.0)), 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * float(_norm_cdf(d1)) - K * math.exp(-r * T) * float(_norm_cdf(d2))


def bs_call_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict[str, float]:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        delta = 1.0 if S > K else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = float(_norm_pdf(d1))
    delta = float(_norm_cdf(d1))
    gamma = pdf_d1 / (S * sigma * sqrtT)
    vega = S * pdf_d1 * sqrtT / 100.0
    theta = (-(S * pdf_d1 * sigma) / (2.0 * sqrtT) - r * K * math.exp(-r * T) * float(_norm_cdf(d2))) / 365.0
    rho = K * T * math.exp(-r * T) * float(_norm_cdf(d2)) / 100.0
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol_call(price: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    if any(pd.isna(x) for x in [price, S, K, T]) or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    upper = S
    if price < intrinsic - 1e-7 or price > upper + 1e-7:
        return np.nan

    lo, hi = 1e-5, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = bs_call_price(S, K, T, r, mid)
        if val > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _estimate_time_to_expiry(df: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.Series:
    # If a T/days_to_expiry column exists, use it. Otherwise linearly decay from default days.
    for col in ["time_to_expiry", "tte", "T", "days_to_expiry", "expiry_days"]:
        if col in df.columns:
            raw = pd.to_numeric(df[col], errors="coerce")
            if col in {"days_to_expiry", "expiry_days"}:
                return (raw / 365.0).clip(lower=1e-6)
            return raw.clip(lower=1e-6)
    if "timestamp" not in df.columns or df["timestamp"].nunique() <= 1:
        return pd.Series(cfg.default_days_to_expiry / 365.0, index=df.index)
    t = df["timestamp"].astype(float)
    progress = (t - t.min()) / max(t.max() - t.min(), EPS)
    days_left = cfg.default_days_to_expiry - progress * max(cfg.default_days_to_expiry - 1.0, 0.0)
    return (days_left / 365.0).clip(lower=1e-6)


# =============================================================================
# Data enrichment
# =============================================================================

def build_options_dataset(
    activity_df: Optional[pd.DataFrame],
    trades_df: Optional[pd.DataFrame] = None,
    cfg: OptionsStatsConfig = OptionsStatsConfig(),
) -> dict[str, pd.DataFrame]:
    activity = _normalize_columns(activity_df)
    trades = _normalize_columns(trades_df)
    if activity.empty or "product" not in activity.columns:
        return {"options": pd.DataFrame(), "underlying": pd.DataFrame(), "trades": trades}

    activity = activity.sort_values([c for c in ["timestamp", "product"] if c in activity.columns]).copy()

    products = activity["product"].dropna().unique().tolist()
    underlying_products = [p for p in products if _is_underlying_product(str(p), cfg)]
    if not underlying_products:
        # Fallback: product with no strike and the most observations.
        no_strike = [p for p in products if _infer_strike(str(p)) is None]
        if no_strike:
            counts = activity[activity["product"].isin(no_strike)]["product"].value_counts()
            underlying_products = [counts.index[0]]

    option_products = [p for p in products if _is_option_product(str(p), cfg)]

    if not underlying_products or not option_products:
        return {"options": pd.DataFrame(), "underlying": pd.DataFrame(), "trades": trades}

    underlying_product = underlying_products[0]
    underlying = activity[activity["product"] == underlying_product].copy()
    underlying = underlying[[c for c in ["timestamp", "mid_price", "bid_price_1", "ask_price_1"] if c in underlying.columns]]
    underlying = underlying.rename(columns={"mid_price": "underlying_mid"})

    options = activity[activity["product"].isin(option_products)].copy()
    options["strike"] = options["product"].map(_infer_strike)
    options = options.dropna(subset=["strike"])
    if "timestamp" in options.columns and "timestamp" in underlying.columns:
        options = options.merge(underlying[["timestamp", "underlying_mid"]], on="timestamp", how="left")
        options["underlying_mid"] = options["underlying_mid"].ffill().bfill()
    else:
        options["underlying_mid"] = np.nan

    options["T"] = _estimate_time_to_expiry(options, cfg)
    options["moneyness"] = _safe_div(options["underlying_mid"], options["strike"])
    options["log_moneyness"] = np.log(_safe_div(options["underlying_mid"], options["strike"]))

    if "mid_price" not in options.columns:
        options["mid_price"] = np.nan

    options["intrinsic"] = np.maximum(options["underlying_mid"] - options["strike"], 0.0)
    options["extrinsic"] = options["mid_price"] - options["intrinsic"]

    options["observed_iv"] = [
        implied_vol_call(p, s, k, t, cfg.risk_free_rate)
        for p, s, k, t in zip(options["mid_price"], options["underlying_mid"], options["strike"], options["T"])
    ]

    options = _fit_smile_by_timestamp(options)
    options["fair_price"] = [
        bs_call_price(s, k, t, cfg.risk_free_rate, iv)
        if not pd.isna(iv)
        else np.nan
        for s, k, t, iv in zip(options["underlying_mid"], options["strike"], options["T"], options["smile_iv"])
    ]
    options["mispricing"] = options["mid_price"] - options["fair_price"]
    options["normalized_mispricing"] = _safe_div(options["mispricing"], options["fair_price"])
    options["iv_residual"] = options["observed_iv"] - options["smile_iv"]

    greeks = [
        bs_call_greeks(s, k, t, cfg.risk_free_rate, iv if not pd.isna(iv) else obs_iv)
        for s, k, t, iv, obs_iv in zip(
            options["underlying_mid"], options["strike"], options["T"], options["smile_iv"], options["observed_iv"]
        )
    ]
    greeks_df = pd.DataFrame(greeks, index=options.index)
    options = pd.concat([options, greeks_df], axis=1)

    options = _add_rolling_stats(options, cfg)
    options = _add_gamma_scalping_ev(options)
    options = _add_signals(options, cfg)

    underlying = _add_underlying_stats(activity[activity["product"] == underlying_product].copy(), cfg)
    options = _add_cross_sectional_mispricing_features(options, underlying, cfg)

    return {"options": options, "underlying": underlying, "trades": trades}


def _fit_smile_by_timestamp(options: pd.DataFrame) -> pd.DataFrame:
    out = options.copy()
    out["smile_iv"] = np.nan
    out["smile_fit_ok"] = False
    out["smile_a"] = np.nan
    out["smile_b"] = np.nan
    out["smile_c"] = np.nan

    if "timestamp" not in out.columns:
        return out

    for ts, idx in out.groupby("timestamp").groups.items():
        sub = out.loc[idx]
        good = sub[["log_moneyness", "observed_iv", "extrinsic"]].replace([np.inf, -np.inf], np.nan).dropna()
        good = good[good["observed_iv"].between(1e-4, 5.0)]
        # Ignore points where option is basically intrinsic only because IV inversion gets unstable.
        good = good[good["extrinsic"] > 0.25]

        if len(good) >= 3:
            x = good["log_moneyness"].to_numpy(dtype=float)
            y = good["observed_iv"].to_numpy(dtype=float)
            try:
                coef = np.polyfit(x, y, deg=2)
                pred = np.polyval(coef, sub["log_moneyness"].to_numpy(dtype=float))
                out.loc[idx, "smile_iv"] = pred
                out.loc[idx, "smile_fit_ok"] = True
                out.loc[idx, ["smile_a", "smile_b", "smile_c"]] = coef
            except Exception:
                out.loc[idx, "smile_iv"] = sub["observed_iv"].median()
        else:
            # If there are too few strikes, use cross-sectional median as a conservative fallback.
            fallback = good["observed_iv"].median() if len(good) else sub["observed_iv"].median()
            out.loc[idx, "smile_iv"] = fallback

    out["smile_iv"] = out["smile_iv"].clip(lower=1e-5, upper=5.0)
    return out


def _add_rolling_stats(options: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.DataFrame:
    out = options.sort_values(["product", "timestamp"]).copy()
    for col in ["mispricing", "iv_residual", "normalized_mispricing"]:
        mean_col = f"{col}_roll_mean"
        std_col = f"{col}_roll_std"
        z_col = f"{col}_z"
        out[mean_col] = out.groupby("product")[col].transform(
            lambda s: s.rolling(cfg.z_window, min_periods=max(5, cfg.z_window // 5)).mean()
        )
        out[std_col] = out.groupby("product")[col].transform(
            lambda s: s.rolling(cfg.z_window, min_periods=max(5, cfg.z_window // 5)).std()
        )
        out[z_col] = _safe_div(out[col] - out[mean_col], out[std_col])

    out["option_return"] = out.groupby("product")["mid_price"].pct_change()
    out["mispricing_change"] = out.groupby("product")["mispricing"].diff()
    out["iv_residual_change"] = out.groupby("product")["iv_residual"].diff()
    return out


def _add_gamma_scalping_ev(options: pd.DataFrame) -> pd.DataFrame:
    out = options.sort_values(["product", "timestamp"]).copy()
    out["underlying_change"] = out.groupby("product")["underlying_mid"].diff()
    out["gamma_pnl_est"] = 0.5 * out["gamma"] * (out["underlying_change"] ** 2)
    out["theta_cost_est"] = out["theta"]  # already per-day Black-Scholes theta, usually negative
    out["net_gamma_scalp_ev"] = out["gamma_pnl_est"] + out["theta_cost_est"].fillna(0.0)
    out["gamma_theta_ratio"] = _safe_div(out["gamma"], np.abs(out["theta"]))
    return out


def _add_signals(options: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.DataFrame:
    out = options.copy()
    z = out["mispricing_z"] if "mispricing_z" in out.columns else out["mispricing_z"]
    out["signal"] = "HOLD"
    out.loc[z <= -cfg.signal_entry_z, "signal"] = "BUY_CHEAP_OPTION"
    out.loc[z >= cfg.signal_entry_z, "signal"] = "SELL_RICH_OPTION"
    out.loc[z.abs() <= cfg.signal_exit_z, "signal"] = "EXIT/FLAT"
    return out


def _add_underlying_stats(underlying: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.DataFrame:
    out = _normalize_columns(underlying)
    if out.empty or "mid_price" not in out.columns:
        return out
    out = out.sort_values("timestamp").copy() if "timestamp" in out.columns else out.copy()
    out["return"] = out["mid_price"].pct_change()
    out["fast_ema"] = out["mid_price"].ewm(span=cfg.rolling_window, adjust=False).mean()
    out["ema_deviation"] = out["mid_price"] - out["fast_ema"]
    out["ema_deviation_z"] = _safe_div(
        out["ema_deviation"] - out["ema_deviation"].rolling(cfg.z_window, min_periods=10).mean(),
        out["ema_deviation"].rolling(cfg.z_window, min_periods=10).std(),
    )
    out["rolling_autocorr_1"] = out["return"].rolling(cfg.z_window, min_periods=10).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    out["rolling_vol"] = out["return"].rolling(cfg.z_window, min_periods=10).std()
    out["jump_z"] = _safe_div(out["return"], out["rolling_vol"])
    return out


# =============================================================================
# Stats tables
# =============================================================================

def compute_autocorr_table(options: pd.DataFrame, underlying: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not options.empty:
        for product, sub in options.groupby("product"):
            sub = sub.sort_values("timestamp") if "timestamp" in sub.columns else sub
            for series_name, col in [
                ("option_return", "option_return"),
                ("mispricing_change", "mispricing_change"),
                ("iv_residual_change", "iv_residual_change"),
                ("mispricing_level", "mispricing"),
                ("iv_residual_level", "iv_residual"),
            ]:
                s = sub[col].replace([np.inf, -np.inf], np.nan).dropna() if col in sub.columns else pd.Series(dtype=float)
                rows.append(
                    {
                        "product": product,
                        "series": series_name,
                        "n": int(s.count()),
                        "lag_1_autocorr": float(s.autocorr(1)) if s.count() > 3 else np.nan,
                        "lag_2_autocorr": float(s.autocorr(2)) if s.count() > 4 else np.nan,
                        "mean": float(s.mean()) if s.count() else np.nan,
                        "std": float(s.std()) if s.count() else np.nan,
                    }
                )
    if not underlying.empty and "return" in underlying.columns:
        s = underlying["return"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "product": "UNDERLYING",
                "series": "underlying_return",
                "n": int(s.count()),
                "lag_1_autocorr": float(s.autocorr(1)) if s.count() > 3 else np.nan,
                "lag_2_autocorr": float(s.autocorr(2)) if s.count() > 4 else np.nan,
                "mean": float(s.mean()) if s.count() else np.nan,
                "std": float(s.std()) if s.count() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_greeks_summary(options: pd.DataFrame) -> pd.DataFrame:
    if options.empty:
        return pd.DataFrame()
    cols = ["delta", "gamma", "vega", "theta", "rho", "gamma_theta_ratio"]
    keep = [c for c in cols if c in options.columns]
    return (
        options.groupby("product")[keep]
        .agg(["mean", "median", "min", "max"])
        .round(6)
        .reset_index()
    )


def compute_latest_signal_table(options: pd.DataFrame) -> pd.DataFrame:
    if options.empty or "timestamp" not in options.columns:
        return pd.DataFrame()
    idx = options.groupby("product")["timestamp"].idxmax()
    cols = [
        "timestamp",
        "product",
        "strike",
        "underlying_mid",
        "mid_price",
        "observed_iv",
        "smile_iv",
        "iv_residual",
        "fair_price",
        "mispricing",
        "mispricing_z",
        "normalized_mispricing",
        "delta",
        "gamma",
        "theta",
        "gamma_theta_ratio",
        "signal",
    ]
    return options.loc[idx, [c for c in cols if c in options.columns]].sort_values("strike").round(6)


def compute_risk_table(options: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not options.empty:
        for product, sub in options.groupby("product"):
            s = sub.sort_values("timestamp")["mispricing"].dropna()
            rows.append(
                {
                    "product": product,
                    "risk_source": "mispricing",
                    "n": int(s.count()),
                    "mean": float(s.mean()) if s.count() else np.nan,
                    "std": float(s.std()) if s.count() else np.nan,
                    "var_95": float(s.quantile(0.05)) if s.count() else np.nan,
                    "max_drawdown_proxy": float((s.cumsum() - s.cumsum().cummax()).min()) if s.count() else np.nan,
                }
            )
    if not trades.empty and {"product", "realized_pnl"}.issubset(trades.columns):
        for product, sub in trades.groupby("product"):
            s = sub["realized_pnl"].dropna()
            rows.append(
                {
                    "product": product,
                    "risk_source": "realized_trade_pnl",
                    "n": int(s.count()),
                    "mean": float(s.mean()) if s.count() else np.nan,
                    "std": float(s.std()) if s.count() else np.nan,
                    "var_95": float(s.quantile(0.05)) if s.count() else np.nan,
                    "max_drawdown_proxy": float((s.cumsum() - s.cumsum().cummax()).min()) if s.count() else np.nan,
                }
            )
    return pd.DataFrame(rows).round(4)


def compute_strategy_attribution(options: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not trades.empty and {"product", "realized_pnl"}.issubset(trades.columns):
        pnl = trades.groupby("product")["realized_pnl"].sum().reset_index()
        for _, r in pnl.iterrows():
            rows.append({"sleeve": "realized_trading", "product": r["product"], "pnl_or_proxy": r["realized_pnl"]})
    if not options.empty:
        proxy = options.groupby("product").agg(
            iv_scalp_proxy=("mispricing", lambda x: -float(np.nansum(np.diff(x.dropna()))) if x.dropna().shape[0] > 1 else np.nan),
            gamma_scalp_ev=("net_gamma_scalp_ev", "sum"),
            avg_abs_mispricing=("mispricing", lambda x: float(np.nanmean(np.abs(x)))),
        ).reset_index()
        for _, r in proxy.iterrows():
            rows.append({"sleeve": "iv_scalp_proxy", "product": r["product"], "pnl_or_proxy": r["iv_scalp_proxy"]})
            rows.append({"sleeve": "gamma_scalp_ev", "product": r["product"], "pnl_or_proxy": r["gamma_scalp_ev"]})
            rows.append({"sleeve": "avg_abs_mispricing", "product": r["product"], "pnl_or_proxy": r["avg_abs_mispricing"]})
    return pd.DataFrame(rows).round(4)


# =============================================================================
# Figures
# =============================================================================

def fig_vol_smile(options: pd.DataFrame, timestamp: Optional[float] = None) -> go.Figure:
    if options.empty:
        return _empty_fig("Volatility Smile")
    df = options.copy()
    if timestamp is None and "timestamp" in df.columns:
        timestamp = df["timestamp"].max()
    if timestamp is not None and "timestamp" in df.columns:
        df = df[df["timestamp"] == timestamp]
    df = df.dropna(subset=["log_moneyness", "observed_iv", "smile_iv"])
    if df.empty:
        return _empty_fig("Volatility Smile", "No IV data available")

    df = df.sort_values("log_moneyness")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["log_moneyness"], y=df["observed_iv"], mode="markers+text", text=df["strike"],
        textposition="top center", name="Observed IV"
    ))
    fig.add_trace(go.Scatter(
        x=df["log_moneyness"], y=df["smile_iv"], mode="lines+markers", name="Fitted smile IV"
    ))
    fig.update_layout(
        title=f"Volatility Smile at timestamp {timestamp}",
        xaxis_title="log(S/K)", yaxis_title="Implied volatility", template="plotly_white", height=470,
    )
    return fig


def fig_iv_residuals(options: pd.DataFrame) -> go.Figure:
    if options.empty or "iv_residual" not in options.columns:
        return _empty_fig("IV Residuals Over Time")
    fig = px.line(options, x="timestamp", y="iv_residual", color="product", title="IV Residuals: observed IV minus smile IV")
    fig.update_layout(template="plotly_white", height=430)
    return fig


def fig_fair_vs_market(options: pd.DataFrame) -> go.Figure:
    if options.empty:
        return _empty_fig("Fair Price vs Market Price")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Market mid vs theoretical fair", "Mispricing"))
    for product, sub in options.groupby("product"):
        sub = sub.sort_values("timestamp")
        fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["mid_price"], mode="lines", name=f"{product} market"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["fair_price"], mode="lines", name=f"{product} fair"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["mispricing"], mode="lines", name=f"{product} mispricing"), row=2, col=1)
    fig.update_layout(title="Black-Scholes Smile Fair Price vs Market", template="plotly_white", height=650)
    return fig


def fig_mispricing_z(options: pd.DataFrame, entry_z: float = 1.5, exit_z: float = 0.35) -> go.Figure:
    if options.empty or "mispricing_z" not in options.columns:
        return _empty_fig("Mispricing Z-Score")
    fig = px.line(options, x="timestamp", y="mispricing_z", color="product", title="Mispricing Z-Score by Strike")
    for y, name in [(entry_z, "rich entry"), (-entry_z, "cheap entry"), (exit_z, "exit"), (-exit_z, "exit")]:
        fig.add_hline(y=y, line_dash="dash", annotation_text=name)
    fig.update_layout(template="plotly_white", height=450)
    return fig


def fig_autocorr(autocorr_df: pd.DataFrame) -> go.Figure:
    if autocorr_df.empty:
        return _empty_fig("Autocorrelation Stats")
    fig = px.bar(
        autocorr_df,
        x="product",
        y="lag_1_autocorr",
        color="series",
        barmode="group",
        title="Lag-1 Autocorrelation by Product and Series",
    )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(template="plotly_white", height=460)
    return fig


def fig_gamma_scalping(options: pd.DataFrame) -> go.Figure:
    if options.empty:
        return _empty_fig("Gamma Scalping EV")
    agg = options.groupby("product", as_index=False).agg(
        gamma_pnl_est=("gamma_pnl_est", "sum"),
        theta_cost_est=("theta_cost_est", "sum"),
        net_gamma_scalp_ev=("net_gamma_scalp_ev", "sum"),
        avg_gamma_theta_ratio=("gamma_theta_ratio", "mean"),
    )
    fig = go.Figure()
    for col in ["gamma_pnl_est", "theta_cost_est", "net_gamma_scalp_ev"]:
        fig.add_trace(go.Bar(x=agg["product"], y=agg[col], name=col))
    fig.update_layout(title="Gamma Scalping EV Proxy: Gamma PnL vs Theta Cost", barmode="group", template="plotly_white", height=450)
    return fig


def fig_greeks_heatmap(options: pd.DataFrame) -> go.Figure:
    if options.empty:
        return _empty_fig("Greeks Exposure Heatmap")
    latest = compute_latest_signal_table(options)
    greek_cols = [c for c in ["delta", "gamma", "vega", "theta", "rho", "gamma_theta_ratio"] if c in latest.columns]
    if latest.empty or not greek_cols:
        return _empty_fig("Greeks Exposure Heatmap")
    z = latest[greek_cols].to_numpy(dtype=float)
    fig = go.Figure(data=go.Heatmap(z=z, x=greek_cols, y=latest["product"], colorbar=dict(title="value")))
    fig.update_layout(title="Latest Greeks by Product", template="plotly_white", height=430)
    return fig


def fig_underlying_mean_reversion(underlying: pd.DataFrame) -> go.Figure:
    if underlying.empty or "mid_price" not in underlying.columns:
        return _empty_fig("Underlying Mean Reversion")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Price vs Fast EMA", "EMA Deviation Z", "Rolling Return Autocorrelation"))
    fig.add_trace(go.Scatter(x=underlying["timestamp"], y=underlying["mid_price"], mode="lines", name="underlying mid"), row=1, col=1)
    if "fast_ema" in underlying.columns:
        fig.add_trace(go.Scatter(x=underlying["timestamp"], y=underlying["fast_ema"], mode="lines", name="fast EMA"), row=1, col=1)
    if "ema_deviation_z" in underlying.columns:
        fig.add_trace(go.Scatter(x=underlying["timestamp"], y=underlying["ema_deviation_z"], mode="lines", name="EMA deviation z"), row=2, col=1)
    if "rolling_autocorr_1" in underlying.columns:
        fig.add_trace(go.Scatter(x=underlying["timestamp"], y=underlying["rolling_autocorr_1"], mode="lines", name="rolling autocorr 1"), row=3, col=1)
    fig.update_layout(title="Underlying Mean Reversion Diagnostics", template="plotly_white", height=760)
    return fig


def fig_strategy_attribution(attr: pd.DataFrame) -> go.Figure:
    if attr.empty:
        return _empty_fig("Strategy Attribution")
    fig = px.bar(attr, x="product", y="pnl_or_proxy", color="sleeve", barmode="group", title="PnL / Edge Proxy Attribution")
    fig.update_layout(template="plotly_white", height=470)
    return fig


def fig_risk_dashboard(risk: pd.DataFrame) -> go.Figure:
    if risk.empty:
        return _empty_fig("Risk Dashboard")
    fig = px.bar(risk, x="product", y="var_95", color="risk_source", barmode="group", title="95% VaR Proxy by Product / Sleeve")
    fig.update_layout(template="plotly_white", height=450)
    return fig


def fig_normalized_mispricing(options: pd.DataFrame) -> go.Figure:
    if options.empty or "normalized_mispricing" not in options.columns:
        return _empty_fig("Normalized Mispricing")
    fig = px.line(options, x="timestamp", y="normalized_mispricing", color="product", title="Normalized Mispricing: (market - fair) / fair")
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(template="plotly_white", height=430)
    return fig



# =============================================================================
# Goyal-Saretto and Xu-Wang inspired cross-sectional option diagnostics
# =============================================================================

def _add_cross_sectional_mispricing_features(options: pd.DataFrame, underlying: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.DataFrame:
    """Adds RV-IV spreads, Xu-Wang style residual IV mispricing, deciles, and delta-hedged return proxies."""
    out = options.copy()
    if out.empty:
        return out
    u = underlying.copy()
    if not u.empty and {"timestamp", "mid_price"}.issubset(u.columns):
        u = u.sort_values("timestamp")
        u["underlying_log_return"] = np.log(u["mid_price"]).diff()
        u["rv_short"] = u["underlying_log_return"].rolling(max(5, cfg.rolling_window // 2), min_periods=5).std()
        u["rv_long"] = u["underlying_log_return"].rolling(max(10, cfg.rolling_window * 2), min_periods=10).std()
        u["rv_trend"] = u["rv_short"] - u["rv_long"]
        u["rv_short_ann_proxy"] = u["rv_short"] * np.sqrt(365.0)
        u["rv_long_ann_proxy"] = u["rv_long"] * np.sqrt(365.0)
        out = out.merge(u[["timestamp", "rv_short", "rv_long", "rv_trend", "rv_short_ann_proxy", "rv_long_ann_proxy"]], on="timestamp", how="left")
    else:
        for c in ["rv_short", "rv_long", "rv_trend", "rv_short_ann_proxy", "rv_long_ann_proxy"]:
            out[c] = np.nan
    out["rv_iv_spread"] = out["rv_long_ann_proxy"] - out["observed_iv"]
    out["rv_iv_spread_short"] = out["rv_short_ann_proxy"] - out["observed_iv"]
    if {"ask_price_1", "bid_price_1"}.issubset(out.columns):
        out["quoted_spread"] = out["ask_price_1"] - out["bid_price_1"]
        out["quoted_spread_pct"] = _safe_div(out["quoted_spread"], out["mid_price"])
    else:
        out["quoted_spread"] = np.nan
        out["quoted_spread_pct"] = np.nan
    if {"bid_volume_1", "ask_volume_1"}.issubset(out.columns):
        out["book_imbalance"] = _safe_div(out["bid_volume_1"] - out["ask_volume_1"], out["bid_volume_1"] + out["ask_volume_1"])
        out["top_level_depth"] = out["bid_volume_1"].abs() + out["ask_volume_1"].abs()
    else:
        out["book_imbalance"] = np.nan
        out["top_level_depth"] = np.nan
    out = _add_xu_wang_residual(out)
    out = _add_cross_sectional_ranks(out)
    out = _add_delta_hedged_return_proxy(out)
    return out


def _add_xu_wang_residual(options: pd.DataFrame) -> pd.DataFrame:
    out = options.copy()
    out["xw_fitted_iv"] = np.nan
    out["xw_iv_mispricing"] = np.nan
    out["xw_price_mispricing"] = np.nan
    out["xw_model_ok"] = False
    features = ["rv_short_ann_proxy", "rv_long_ann_proxy", "log_moneyness", "T", "quoted_spread_pct", "book_imbalance", "top_level_depth"]
    available = [c for c in features if c in out.columns]
    if not available or "observed_iv" not in out.columns:
        return out
    if "timestamp" in out.columns:
        for _ts, idx in out.groupby("timestamp").groups.items():
            sub = out.loc[idx, available + ["observed_iv"]].replace([np.inf, -np.inf], np.nan)
            good = sub.dropna(subset=["observed_iv"])
            useful = [c for c in available if good[c].notna().sum() >= 3 and good[c].nunique(dropna=True) > 1]
            if len(good) >= max(3, len(useful) + 1) and useful:
                try:
                    med = good[useful].median()
                    X = good[useful].fillna(med).to_numpy(dtype=float)
                    X = np.column_stack([np.ones(len(X)), X])
                    y = good["observed_iv"].to_numpy(dtype=float)
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                    X_all = out.loc[idx, useful].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
                    X_all = np.column_stack([np.ones(len(X_all)), X_all])
                    out.loc[idx, "xw_fitted_iv"] = X_all @ beta
                    out.loc[idx, "xw_model_ok"] = True
                except Exception:
                    pass
    missing = out["xw_fitted_iv"].isna()
    pooled = out[available + ["observed_iv"]].replace([np.inf, -np.inf], np.nan).dropna(subset=["observed_iv"])
    useful = [c for c in available if pooled[c].notna().sum() >= 10 and pooled[c].nunique(dropna=True) > 1]
    if missing.any() and len(pooled) >= max(20, len(useful) + 5) and useful:
        try:
            med = pooled[useful].median()
            X = pooled[useful].fillna(med).to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(X)), X])
            y = pooled["observed_iv"].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            X_all = out.loc[missing, useful].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
            X_all = np.column_stack([np.ones(len(X_all)), X_all])
            out.loc[missing, "xw_fitted_iv"] = X_all @ beta
            out.loc[missing, "xw_model_ok"] = True
        except Exception:
            pass
    out["xw_iv_mispricing"] = out["observed_iv"] - out["xw_fitted_iv"]
    out["xw_price_mispricing"] = [
        bs_call_price(s, k, t, 0.0, iv) - bs_call_price(s, k, t, 0.0, fit_iv)
        if not any(pd.isna(x) for x in [s, k, t, iv, fit_iv]) else np.nan
        for s, k, t, iv, fit_iv in zip(out["underlying_mid"], out["strike"], out["T"], out["observed_iv"], out["xw_fitted_iv"])
    ]
    return out


def _safe_qcut(s: pd.Series, q: int = 10) -> pd.Series:
    vals = s.replace([np.inf, -np.inf], np.nan)
    try:
        return pd.qcut(vals.rank(method="first"), q=q, labels=False, duplicates="drop") + 1
    except Exception:
        return pd.Series(np.nan, index=s.index)


def _add_cross_sectional_ranks(options: pd.DataFrame) -> pd.DataFrame:
    out = options.copy()
    rank_cols = ["rv_iv_spread", "iv_residual", "mispricing", "xw_iv_mispricing", "xw_price_mispricing"]
    if "timestamp" not in out.columns:
        for c in rank_cols:
            if c in out.columns:
                out[f"{c}_rank"] = out[c].rank(pct=True)
                out[f"{c}_decile"] = _safe_qcut(out[c], 10)
        return out
    for c in rank_cols:
        if c in out.columns:
            out[f"{c}_rank"] = out.groupby("timestamp")[c].rank(pct=True)
            out[f"{c}_decile"] = out.groupby("timestamp", group_keys=False)[c].apply(_safe_qcut)
    return out


def _add_delta_hedged_return_proxy(options: pd.DataFrame) -> pd.DataFrame:
    out = options.sort_values(["product", "timestamp"]).copy() if "timestamp" in options.columns else options.copy()
    out["option_price_change"] = out.groupby("product")["mid_price"].diff()
    out["underlying_price_change_for_hedge"] = out.groupby("product")["underlying_mid"].diff()
    out["lag_delta"] = out.groupby("product")["delta"].shift(1)
    out["delta_hedged_pnl_proxy"] = out["option_price_change"] - out["lag_delta"] * out["underlying_price_change_for_hedge"]
    out["delta_hedged_return_proxy"] = _safe_div(out["delta_hedged_pnl_proxy"], out.groupby("product")["mid_price"].shift(1).abs())
    return out


def compute_cross_sectional_signal_table(options: pd.DataFrame) -> pd.DataFrame:
    if options.empty or "timestamp" not in options.columns:
        return pd.DataFrame()
    idx = options.groupby("product")["timestamp"].idxmax()
    cols = ["timestamp", "product", "strike", "observed_iv", "rv_long_ann_proxy", "rv_iv_spread", "rv_iv_spread_decile", "xw_fitted_iv", "xw_iv_mispricing", "xw_iv_mispricing_decile", "xw_price_mispricing", "rv_trend", "quoted_spread", "book_imbalance", "delta_hedged_return_proxy", "signal"]
    return options.loc[idx, [c for c in cols if c in options.columns]].sort_values("strike").round(6)


def compute_double_sort_table(options: pd.DataFrame) -> pd.DataFrame:
    if options.empty:
        return pd.DataFrame()
    needed = {"rv_iv_spread_decile", "xw_iv_mispricing_decile", "delta_hedged_return_proxy"}
    if not needed.issubset(options.columns):
        return pd.DataFrame()
    rows = []
    df = options.replace([np.inf, -np.inf], np.nan).dropna(subset=list(needed))
    for (rv_decile, xw_decile), sub in df.groupby(["rv_iv_spread_decile", "xw_iv_mispricing_decile"]):
        rows.append({"rv_iv_decile": int(rv_decile), "xw_mispricing_decile": int(xw_decile), "n": int(len(sub)), "avg_delta_hedged_return_proxy": float(sub["delta_hedged_return_proxy"].mean()), "avg_mispricing": float(sub.get("mispricing", pd.Series(dtype=float)).mean()), "avg_xw_iv_mispricing": float(sub["xw_iv_mispricing"].mean())})
    return pd.DataFrame(rows).round(6)


def fig_rv_iv_spread(options: pd.DataFrame) -> go.Figure:
    if options.empty or "rv_iv_spread" not in options.columns:
        return _empty_fig("Goyal-Saretto RV-IV Spread")
    fig = px.line(options, x="timestamp", y="rv_iv_spread", color="product", title="Goyal-Saretto Signal: Realized Volatility Proxy minus Implied Volatility")
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(template="plotly_white", height=450)
    return fig


def fig_xw_iv_mispricing(options: pd.DataFrame) -> go.Figure:
    if options.empty or "xw_iv_mispricing" not in options.columns:
        return _empty_fig("Xu-Wang IV Mispricing Residual")
    fig = px.line(options, x="timestamp", y="xw_iv_mispricing", color="product", title="Xu-Wang Style Two-Step IV Mispricing Residual")
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(template="plotly_white", height=450)
    return fig


def fig_cross_sectional_heatmap(options: pd.DataFrame) -> go.Figure:
    table = compute_double_sort_table(options)
    if table.empty:
        return _empty_fig("Double-Sort Cross-Sectional Diagnostics", "Need enough cross-sectional observations for decile sorting")
    pivot = table.pivot(index="rv_iv_decile", columns="xw_mispricing_decile", values="avg_delta_hedged_return_proxy")
    fig = go.Figure(data=go.Heatmap(z=pivot.to_numpy(dtype=float), x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index], colorbar=dict(title="avg DH return")))
    fig.update_layout(title="Double Sort: RV-IV Decile x Xu-Wang Mispricing Decile", xaxis_title="Xu-Wang IV mispricing decile", yaxis_title="RV-IV spread decile", template="plotly_white", height=480)
    return fig


def fig_vol_trend_condition(options: pd.DataFrame) -> go.Figure:
    if options.empty or "rv_trend" not in options.columns:
        return _empty_fig("Volatility Trend Condition")
    tmp = options.copy()
    tmp["vol_trend_regime"] = np.where(tmp["rv_trend"] >= 0, "short vol rising", "short vol falling")
    y = "delta_hedged_return_proxy" if "delta_hedged_return_proxy" in tmp.columns else "mispricing"
    fig = px.box(tmp, x="vol_trend_regime", y=y, color="vol_trend_regime", points="all", title="Conditional Option Return Proxy by Short-vs-Long Realized Vol Trend")
    fig.update_layout(template="plotly_white", height=430, showlegend=False)
    return fig

# =============================================================================
# Dash layout and callback registration
# =============================================================================

def _table_component(df: pd.DataFrame, table_id: str, page_size: int = 12):
    if dash_table is None:
        return html.Div("dash_table unavailable") if html else None
    if df is None or df.empty:
        return html.Div("No rows to display", className="text-muted")
    view = df.copy()
    view.columns = ["_".join(map(str, c)).strip("_") if isinstance(c, tuple) else str(c) for c in view.columns]
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": c, "id": c} for c in view.columns],
        data=view.replace([np.inf, -np.inf], np.nan).round(6).fillna("").to_dict("records"),
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Arial", "fontSize": 12, "padding": "6px", "textAlign": "left"},
        style_header={"fontWeight": "bold"},
    )


def _empty_table_message(message: str = "Click the load button above to build this table.") -> Any:
    return html.Div(message, style={"color": "#777", "padding": "10px 0"})


def _section(title: str, button_id: str, output_component: Any, description: str = "") -> Any:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(title, style={"margin": "0"}),
                            html.Div(description, style={"fontSize": "12px", "color": "#666", "marginTop": "4px"}) if description else html.Div(),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Button("Load", id=button_id, n_clicks=0, style={"height": "34px", "padding": "0 14px"}),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "12px", "borderBottom": "1px solid #eee", "paddingBottom": "8px", "marginBottom": "8px"},
            ),
            dcc.Loading(output_component, type="default"),
        ],
        style={"border": "1px solid #e5e7eb", "borderRadius": "12px", "padding": "12px", "margin": "12px 0", "background": "#fff"},
    )


def build_options_stats_tab() -> Any:
    """Returns the Dash layout for the Options Stats tab.

    This version is intentionally lazy:
    1. Upload CSVs.
    2. Click Build / Parse Options Data.
    3. Click Load on only the charts/tables you want.
    """
    if html is None:
        raise RuntimeError("Dash is not installed or importable in this environment.")

    return html.Div(
        [
            dcc.Store(id="options-stats-options-store", data=[]),
            dcc.Store(id="options-stats-underlying-store", data=[]),
            dcc.Store(id="options-stats-trades-store", data=[]),
            html.Div(
                [
                    html.H2("Options Stats"),
                    html.P(
                        "Upload raw option/underlying CSVs, build the normalized options dataset once, then load only the charts or tables you need.",
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Upload prices CSVs", style={"fontWeight": "700", "marginBottom": "4px"}),
                                    dcc.Upload(
                                        id="options-stats-prices-upload",
                                        children=html.Div(["Drag/drop or click to select price CSVs"]),
                                        multiple=True,
                                        style={
                                            "border": "1px dashed #aaa",
                                            "borderRadius": "10px",
                                            "padding": "14px",
                                            "textAlign": "center",
                                            "background": "#fafafa",
                                        },
                                    ),
                                    html.Div(id="options-stats-prices-upload-list"),
                                ],
                                style={"minWidth": "280px", "flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Div("Upload trades CSVs", style={"fontWeight": "700", "marginBottom": "4px"}),
                                    dcc.Upload(
                                        id="options-stats-trades-upload",
                                        children=html.Div(["Drag/drop or click to select trade CSVs"]),
                                        multiple=True,
                                        style={
                                            "border": "1px dashed #aaa",
                                            "borderRadius": "10px",
                                            "padding": "14px",
                                            "textAlign": "center",
                                            "background": "#fafafa",
                                        },
                                    ),
                                    html.Div(id="options-stats-trades-upload-list"),
                                ],
                                style={"minWidth": "280px", "flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "margin": "10px 0"},
                    ),
                    html.Div(
                        [
                            html.Label("Risk-free rate"),
                            dcc.Input(id="options-stats-r", type="number", value=0.0, step=0.001, style={"width": "120px"}),
                            html.Label("Default days to expiry", style={"marginLeft": "16px"}),
                            dcc.Input(id="options-stats-dte", type="number", value=7, step=1, style={"width": "120px"}),
                            html.Label("Rolling/Z window", style={"marginLeft": "16px"}),
                            dcc.Input(id="options-stats-window", type="number", value=80, step=5, style={"width": "120px"}),
                            html.Button("Build / Parse Options Data", id="options-stats-build-data", n_clicks=0, style={"marginLeft": "16px"}),
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"},
                    ),
                ],
                style={"padding": "14px", "borderBottom": "1px solid #ddd"},
            ),
            dcc.Loading(
                html.Div(id="options-stats-summary-cards", style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(160px, 1fr))", "gap": "10px", "padding": "12px"}),
                type="default",
            ),
            html.Div(
                [
                    _section("Volatility Smile", "options-stats-load-vol-smile", dcc.Graph(id="options-stats-vol-smile", figure=_empty_fig("Volatility Smile", "Click Load to render this chart.")), "Observed IV vs log-moneyness with fitted smile."),
                    _section("IV Residuals", "options-stats-load-iv-residuals", dcc.Graph(id="options-stats-iv-residuals", figure=_empty_fig("IV Residuals", "Click Load to render this chart.")), "Observed IV minus smile-implied IV over time."),
                    _section("Fair Price vs Market", "options-stats-load-fair-vs-market", dcc.Graph(id="options-stats-fair-vs-market", figure=_empty_fig("Fair Price vs Market", "Click Load to render this chart.")), "Market mid compared with Black-Scholes smile fair value."),
                    _section("Normalized Mispricing", "options-stats-load-normalized-mispricing", dcc.Graph(id="options-stats-normalized-mispricing", figure=_empty_fig("Normalized Mispricing", "Click Load to render this chart.")), "Market minus fair divided by fair."),
                    _section("Goyal-Saretto RV-IV Spread", "options-stats-load-rv-iv-spread", dcc.Graph(id="options-stats-rv-iv-spread", figure=_empty_fig("Goyal-Saretto RV-IV Spread", "Click Load to render this chart.")), "Realized volatility proxy minus implied volatility."),
                    _section("Xu-Wang IV Mispricing", "options-stats-load-xw-iv-mispricing", dcc.Graph(id="options-stats-xw-iv-mispricing", figure=_empty_fig("Xu-Wang IV Mispricing", "Click Load to render this chart.")), "Cross-sectional residual IV mispricing proxy."),
                    _section("Double-Sort Heatmap", "options-stats-load-cross-sectional-heatmap", dcc.Graph(id="options-stats-cross-sectional-heatmap", figure=_empty_fig("Double-Sort Heatmap", "Click Load to render this chart.")), "RV-IV decile by Xu-Wang mispricing decile."),
                    _section("Volatility Trend Condition", "options-stats-load-vol-trend-condition", dcc.Graph(id="options-stats-vol-trend-condition", figure=_empty_fig("Volatility Trend Condition", "Click Load to render this chart.")), "Option return proxy conditioned on short-vs-long realized volatility trend."),
                    _section("Mispricing Z-Score", "options-stats-load-mispricing-z", dcc.Graph(id="options-stats-mispricing-z", figure=_empty_fig("Mispricing Z-Score", "Click Load to render this chart.")), "Rolling z-score of option mispricing."),
                    _section("Autocorrelation", "options-stats-load-autocorr", dcc.Graph(id="options-stats-autocorr", figure=_empty_fig("Autocorrelation", "Click Load to render this chart.")), "Lag-1 autocorrelation by product and series."),
                    _section("Gamma Scalping EV", "options-stats-load-gamma-scalping", dcc.Graph(id="options-stats-gamma-scalping", figure=_empty_fig("Gamma Scalping EV", "Click Load to render this chart.")), "Gamma PnL proxy versus theta cost."),
                    _section("Greeks Heatmap", "options-stats-load-greeks-heatmap", dcc.Graph(id="options-stats-greeks-heatmap", figure=_empty_fig("Greeks Heatmap", "Click Load to render this chart.")), "Latest delta, gamma, vega, theta, rho, and gamma/theta ratio."),
                    _section("Underlying Mean Reversion", "options-stats-load-underlying-mr", dcc.Graph(id="options-stats-underlying-mr", figure=_empty_fig("Underlying Mean Reversion", "Click Load to render this chart.")), "Underlying price vs EMA, deviation z-score, and rolling autocorrelation."),
                    _section("Strategy Attribution", "options-stats-load-attribution", dcc.Graph(id="options-stats-attribution", figure=_empty_fig("Strategy Attribution", "Click Load to render this chart.")), "PnL and edge proxy by product and sleeve."),
                    _section("Risk Dashboard", "options-stats-load-risk", dcc.Graph(id="options-stats-risk", figure=_empty_fig("Risk Dashboard", "Click Load to render this chart.")), "VaR proxy by product and risk source."),
                    _section("Latest Signals Table", "options-stats-load-signal-table", html.Div(id="options-stats-signal-table", children=_empty_table_message()), "Latest per-strike signal values."),
                    _section("Cross-Sectional Research Signals Table", "options-stats-load-cross-sectional-table", html.Div(id="options-stats-cross-sectional-table", children=_empty_table_message()), "Goyal-Saretto and Xu-Wang inspired latest signal table."),
                    _section("Double-Sort Diagnostics Table", "options-stats-load-double-sort-table", html.Div(id="options-stats-double-sort-table", children=_empty_table_message()), "Average delta-hedged proxy by double-sort bucket."),
                    _section("Autocorrelation Stats Table", "options-stats-load-autocorr-table", html.Div(id="options-stats-autocorr-table", children=_empty_table_message()), "Autocorrelation values by product and series."),
                    _section("Risk Table", "options-stats-load-risk-table", html.Div(id="options-stats-risk-table", children=_empty_table_message()), "Numerical risk proxy table."),
                ],
                style={"padding": "0 8px 20px"},
            ),
        ],
        style={"padding": "0 8px 20px"},
    )


def _summary_card(title: str, value: Any, subtitle: str = "") -> Any:
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "color": "#666"}),
            html.Div(str(value), style={"fontSize": "22px", "fontWeight": "700"}),
            html.Div(subtitle, style={"fontSize": "11px", "color": "#888"}),
        ],
        style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "background": "white"},
    )


def _build_summary_cards(options: pd.DataFrame, underlying: pd.DataFrame, trades: pd.DataFrame) -> list[Any]:
    if html is None:
        return []
    if options.empty:
        return [_summary_card("Options detected", 0, "Check product naming and price columns")]
    latest = compute_latest_signal_table(options)
    rich = int((latest.get("signal", pd.Series(dtype=str)) == "SELL_RICH_OPTION").sum()) if not latest.empty else 0
    cheap = int((latest.get("signal", pd.Series(dtype=str)) == "BUY_CHEAP_OPTION").sum()) if not latest.empty else 0
    avg_abs_mis = float(np.nanmean(np.abs(options["mispricing"]))) if "mispricing" in options else np.nan
    avg_gamma_theta = float(np.nanmean(options["gamma_theta_ratio"].replace([np.inf, -np.inf], np.nan))) if "gamma_theta_ratio" in options else np.nan
    return [
        _summary_card("Options detected", options["product"].nunique(), "unique option products"),
        _summary_card("Cheap signals", cheap, "latest z-score <= entry band"),
        _summary_card("Rich signals", rich, "latest z-score >= entry band"),
        _summary_card("Avg abs mispricing", round(avg_abs_mis, 4), "market minus smile fair"),
        _summary_card("Avg gamma/theta", round(avg_gamma_theta, 6), "convexity per theta cost"),
        _summary_card("Underlying rows", len(underlying), "mean reversion sample"),
        _summary_card("Trade rows", len(trades), "if available"),
        _summary_card("Smile fit rows", int(options.get("smile_fit_ok", pd.Series(False)).sum()), "timestamps with parabola fit"),
    ]


def _store_df(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return df.replace([np.inf, -np.inf], np.nan).where(pd.notna(df), None).to_dict("records")


def _df_from_store(data: Any) -> pd.DataFrame:
    return pd.DataFrame(data or [])


def _options_from_store(data: Any) -> pd.DataFrame:
    return _df_from_store(data)


def _underlying_from_store(data: Any) -> pd.DataFrame:
    return _df_from_store(data)


def _trades_from_store(data: Any) -> pd.DataFrame:
    return _df_from_store(data)


def register_options_stats_callbacks(
    app: Dash,
    get_backtest_data: Optional[Callable[[], Any]] = None,
) -> None:
    """
    Register lazy callbacks for the tab.

    Build / Parse Options Data creates cached normalized DataFrames in dcc.Store.
    Each chart/table has its own Load button and renders independently.
    """

    @app.callback(
        Output("options-stats-prices-upload-list", "children"),
        Output("options-stats-trades-upload-list", "children"),
        Input("options-stats-prices-upload", "filename"),
        Input("options-stats-trades-upload", "filename"),
        prevent_initial_call=False,
    )
    def _show_uploaded_option_file_names(price_names, trade_names):
        return (
            _uploaded_file_list(price_names, "price"),
            _uploaded_file_list(trade_names, "trade"),
        )

    @app.callback(
        Output("options-stats-options-store", "data"),
        Output("options-stats-underlying-store", "data"),
        Output("options-stats-trades-store", "data"),
        Output("options-stats-summary-cards", "children"),
        Input("options-stats-build-data", "n_clicks"),
        State("options-stats-prices-upload", "contents"),
        State("options-stats-trades-upload", "contents"),
        State("options-stats-prices-upload", "filename"),
        State("options-stats-trades-upload", "filename"),
        State("options-stats-r", "value"),
        State("options-stats-dte", "value"),
        State("options-stats-window", "value"),
        prevent_initial_call=True,
    )
    def _build_options_data(_n_clicks: int, price_contents, trade_contents, price_names, trade_names, r: float, dte: float, window: int):
        if not _n_clicks:
            raise PreventUpdate

        activity_df, trades_df = parse_uploaded_options_csvs(price_contents, price_names, trade_contents, trade_names)

        if activity_df.empty and get_backtest_data is not None:
            raw = get_backtest_data()
            if isinstance(raw, dict):
                activity_df = raw.get("activity", None)
                if activity_df is None:
                    activity_df = raw.get("activity_df", None)
                if activity_df is None:
                    activity_df = raw.get("prices", None)
                if activity_df is None:
                    activity_df = raw.get("price_df", None)
                trades_df = raw.get("trades", None)
                if trades_df is None:
                    trades_df = raw.get("trades_df", None)
                if trades_df is None:
                    trades_df = raw.get("realized_trades", None)
            elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
                activity_df, trades_df = raw[0], raw[1]
            else:
                activity_df, trades_df = pd.DataFrame(), pd.DataFrame()

        cfg = OptionsStatsConfig(
            default_days_to_expiry=float(dte or DEFAULT_DAYS_TO_EXPIRY),
            risk_free_rate=float(r or 0.0),
            rolling_window=max(5, int(window or DEFAULT_ROLLING_WINDOW)),
            z_window=max(10, int(window or DEFAULT_Z_WINDOW)),
        )
        data = build_options_dataset(activity_df, trades_df, cfg)
        options = data["options"]
        underlying = data["underlying"]
        trades = data["trades"]

        return (
            _store_df(options),
            _store_df(underlying),
            _store_df(trades),
            _build_summary_cards(options, underlying, trades),
        )

    @app.callback(Output("options-stats-vol-smile", "figure"), Input("options-stats-load-vol-smile", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_vol_smile(n, options_data):
        if not n: raise PreventUpdate
        return fig_vol_smile(_options_from_store(options_data))

    @app.callback(Output("options-stats-iv-residuals", "figure"), Input("options-stats-load-iv-residuals", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_iv_residuals(n, options_data):
        if not n: raise PreventUpdate
        return fig_iv_residuals(_options_from_store(options_data))

    @app.callback(Output("options-stats-fair-vs-market", "figure"), Input("options-stats-load-fair-vs-market", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_fair_vs_market(n, options_data):
        if not n: raise PreventUpdate
        return fig_fair_vs_market(_options_from_store(options_data))

    @app.callback(Output("options-stats-normalized-mispricing", "figure"), Input("options-stats-load-normalized-mispricing", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_normalized_mispricing(n, options_data):
        if not n: raise PreventUpdate
        return fig_normalized_mispricing(_options_from_store(options_data))

    @app.callback(Output("options-stats-rv-iv-spread", "figure"), Input("options-stats-load-rv-iv-spread", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_rv_iv_spread(n, options_data):
        if not n: raise PreventUpdate
        return fig_rv_iv_spread(_options_from_store(options_data))

    @app.callback(Output("options-stats-xw-iv-mispricing", "figure"), Input("options-stats-load-xw-iv-mispricing", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_xw_iv_mispricing(n, options_data):
        if not n: raise PreventUpdate
        return fig_xw_iv_mispricing(_options_from_store(options_data))

    @app.callback(Output("options-stats-cross-sectional-heatmap", "figure"), Input("options-stats-load-cross-sectional-heatmap", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_cross_sectional_heatmap(n, options_data):
        if not n: raise PreventUpdate
        return fig_cross_sectional_heatmap(_options_from_store(options_data))

    @app.callback(Output("options-stats-vol-trend-condition", "figure"), Input("options-stats-load-vol-trend-condition", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_vol_trend_condition(n, options_data):
        if not n: raise PreventUpdate
        return fig_vol_trend_condition(_options_from_store(options_data))

    @app.callback(Output("options-stats-mispricing-z", "figure"), Input("options-stats-load-mispricing-z", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_mispricing_z(n, options_data):
        if not n: raise PreventUpdate
        return fig_mispricing_z(_options_from_store(options_data))

    @app.callback(Output("options-stats-autocorr", "figure"), Input("options-stats-load-autocorr", "n_clicks"), State("options-stats-options-store", "data"), State("options-stats-underlying-store", "data"), prevent_initial_call=True)
    def _load_autocorr(n, options_data, underlying_data):
        if not n: raise PreventUpdate
        return fig_autocorr(compute_autocorr_table(_options_from_store(options_data), _underlying_from_store(underlying_data)))

    @app.callback(Output("options-stats-gamma-scalping", "figure"), Input("options-stats-load-gamma-scalping", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_gamma_scalping(n, options_data):
        if not n: raise PreventUpdate
        return fig_gamma_scalping(_options_from_store(options_data))

    @app.callback(Output("options-stats-greeks-heatmap", "figure"), Input("options-stats-load-greeks-heatmap", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_greeks_heatmap(n, options_data):
        if not n: raise PreventUpdate
        return fig_greeks_heatmap(_options_from_store(options_data))

    @app.callback(Output("options-stats-underlying-mr", "figure"), Input("options-stats-load-underlying-mr", "n_clicks"), State("options-stats-underlying-store", "data"), prevent_initial_call=True)
    def _load_underlying_mr(n, underlying_data):
        if not n: raise PreventUpdate
        return fig_underlying_mean_reversion(_underlying_from_store(underlying_data))

    @app.callback(Output("options-stats-attribution", "figure"), Input("options-stats-load-attribution", "n_clicks"), State("options-stats-options-store", "data"), State("options-stats-trades-store", "data"), prevent_initial_call=True)
    def _load_attribution(n, options_data, trades_data):
        if not n: raise PreventUpdate
        return fig_strategy_attribution(compute_strategy_attribution(_options_from_store(options_data), _trades_from_store(trades_data)))

    @app.callback(Output("options-stats-risk", "figure"), Input("options-stats-load-risk", "n_clicks"), State("options-stats-options-store", "data"), State("options-stats-trades-store", "data"), prevent_initial_call=True)
    def _load_risk(n, options_data, trades_data):
        if not n: raise PreventUpdate
        return fig_risk_dashboard(compute_risk_table(_options_from_store(options_data), _trades_from_store(trades_data)))

    @app.callback(Output("options-stats-signal-table", "children"), Input("options-stats-load-signal-table", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_signal_table(n, options_data):
        if not n: raise PreventUpdate
        return _table_component(compute_latest_signal_table(_options_from_store(options_data)), "options-stats-latest-signals-table")

    @app.callback(Output("options-stats-cross-sectional-table", "children"), Input("options-stats-load-cross-sectional-table", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_cross_sectional_table(n, options_data):
        if not n: raise PreventUpdate
        return _table_component(compute_cross_sectional_signal_table(_options_from_store(options_data)), "options-stats-cross-sectional-data-table")

    @app.callback(Output("options-stats-double-sort-table", "children"), Input("options-stats-load-double-sort-table", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_double_sort_table(n, options_data):
        if not n: raise PreventUpdate
        return _table_component(compute_double_sort_table(_options_from_store(options_data)), "options-stats-double-sort-data-table")

    @app.callback(Output("options-stats-autocorr-table", "children"), Input("options-stats-load-autocorr-table", "n_clicks"), State("options-stats-options-store", "data"), State("options-stats-underlying-store", "data"), prevent_initial_call=True)
    def _load_autocorr_table(n, options_data, underlying_data):
        if not n: raise PreventUpdate
        autocorr_df = compute_autocorr_table(_options_from_store(options_data), _underlying_from_store(underlying_data))
        return _table_component(autocorr_df.round(6), "options-stats-autocorr-data-table")

    @app.callback(Output("options-stats-risk-table", "children"), Input("options-stats-load-risk-table", "n_clicks"), State("options-stats-options-store", "data"), State("options-stats-trades-store", "data"), prevent_initial_call=True)
    def _load_risk_table(n, options_data, trades_data):
        if not n: raise PreventUpdate
        risk = compute_risk_table(_options_from_store(options_data), _trades_from_store(trades_data))
        return _table_component(risk.round(6), "options-stats-risk-data-table")


# =============================================================================
# Optional direct rendering helper for apps that do not use callbacks
# =============================================================================

def render_options_stats_payload(
    activity_df: pd.DataFrame,
    trades_df: Optional[pd.DataFrame] = None,
    cfg: OptionsStatsConfig = OptionsStatsConfig(),
) -> dict[str, Any]:
    """Useful for testing in notebooks or existing dashboard render functions."""
    data = build_options_dataset(activity_df, trades_df, cfg)
    options = data["options"]
    underlying = data["underlying"]
    trades = data["trades"]
    autocorr_df = compute_autocorr_table(options, underlying)
    attr = compute_strategy_attribution(options, trades)
    risk = compute_risk_table(options, trades)
    latest = compute_latest_signal_table(options)
    cross_sectional = compute_cross_sectional_signal_table(options)
    double_sort = compute_double_sort_table(options)
    return {
        "options_df": options,
        "underlying_df": underlying,
        "latest_signals": latest,
        "autocorr_table": autocorr_df,
        "risk_table": risk,
        "attribution_table": attr,
        "cross_sectional_table": cross_sectional,
        "double_sort_table": double_sort,
        "figures": {
            "vol_smile": fig_vol_smile(options),
            "iv_residuals": fig_iv_residuals(options),
            "fair_vs_market": fig_fair_vs_market(options),
            "normalized_mispricing": fig_normalized_mispricing(options),
            "rv_iv_spread": fig_rv_iv_spread(options),
            "xu_wang_iv_mispricing": fig_xw_iv_mispricing(options),
            "cross_sectional_heatmap": fig_cross_sectional_heatmap(options),
            "vol_trend_condition": fig_vol_trend_condition(options),
            "mispricing_z": fig_mispricing_z(options, cfg.signal_entry_z, cfg.signal_exit_z),
            "autocorr": fig_autocorr(autocorr_df),
            "gamma_scalping": fig_gamma_scalping(options),
            "greeks_heatmap": fig_greeks_heatmap(options),
            "underlying_mean_reversion": fig_underlying_mean_reversion(underlying),
            "strategy_attribution": fig_strategy_attribution(attr),
            "risk": fig_risk_dashboard(risk),
        },
    }
