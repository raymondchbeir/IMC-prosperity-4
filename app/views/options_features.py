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
import uuid
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
    "VELVETFRUIT_EXTRACT",
    "VELVETFRUIT",
    "VR",
    "UNDERLYING",
)

OPTION_NAME_HINTS = (
    "VOUCHER",
    "CALL",
    "OPTION",
    "VEV_",
    "VEV",
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

    if "day" in out.columns:
        out["day"] = pd.to_numeric(out["day"], errors="coerce")
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
        df = _add_day_from_filename_if_missing(df, str(f))
        if not df.empty:
            price_frames.append(df)
    trade_frames: list[pd.DataFrame] = []
    for f in trade_files or []:
        df = _read_semicolon_safe_csv(f, str(f))
        df = _add_day_from_filename_if_missing(df, str(f))
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
        df = _add_day_from_filename_if_missing(df, name)
        if not df.empty:
            price_frames.append(df)

    trade_frames = []
    for contents, name in zip(trade_contents or [], trade_filenames or []):
        df = _parse_upload_contents(contents, name)
        df = _add_day_from_filename_if_missing(df, name)
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



def _extract_day_from_filename(filename: str | None) -> Optional[int]:
    """Extract day number from filenames like prices_round_3_day_0.csv."""
    if not filename:
        return None
    m = re.search(r"day[_\-\s]?(-?\d+)", str(filename), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _add_day_from_filename_if_missing(df: pd.DataFrame, filename: str | None) -> pd.DataFrame:
    if df is None or df.empty or "day" in df.columns:
        return df
    day = _extract_day_from_filename(filename)
    if day is not None:
        out = df.copy()
        out["day"] = day
        return out
    return df


def _time_key_cols(df: pd.DataFrame) -> list[str]:
    """Use day + timestamp when available so Round 3 day 0/1/2 rows do not collide."""
    if df is None or df.empty:
        return []
    keys = []
    if "day" in df.columns:
        keys.append("day")
    if "timestamp" in df.columns:
        keys.append("timestamp")
    return keys


def _latest_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest day/timestamp slice without mixing identical timestamps across days."""
    if df is None or df.empty:
        return df
    out = df
    if "day" in out.columns and out["day"].notna().any():
        latest_day = out["day"].max()
        out = out[out["day"] == latest_day]
    if "timestamp" in out.columns and out["timestamp"].notna().any():
        latest_ts = out["timestamp"].max()
        out = out[out["timestamp"] == latest_ts]
    return out


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
# Volatility Risk Premium helpers
# =============================================================================

def _rolling_zscore_by_product(df: pd.DataFrame, value_col: str, window: int) -> pd.Series:
    """Rolling z-score helper used by VRP and option diagnostics."""
    if df.empty or value_col not in df.columns or "product" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    min_periods = max(5, window // 5)
    grouped = df.groupby("product")[value_col]
    mean = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    std = grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std())
    return pd.Series(_safe_div(df[value_col] - mean, std), index=df.index)


def _add_vrp_features(options: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.DataFrame:
    """
    Adds Volatility Risk Premium features.

    VRP = implied volatility minus realized volatility.

    Positive VRP means options are rich versus recent realized volatility.
    Negative VRP means options are cheap versus recent realized volatility.
    The dashboard uses smile IV as the primary IV when available because it is cleaner
    than raw observed IV across strikes.
    """
    out = options.copy()
    if out.empty:
        return out

    out["realized_vol_short"] = pd.to_numeric(out.get("rv_short_ann_proxy", np.nan), errors="coerce")
    out["realized_vol_long"] = pd.to_numeric(out.get("rv_long_ann_proxy", np.nan), errors="coerce")

    for iv_col, prefix in [("observed_iv", "observed"), ("smile_iv", "smile")]:
        if iv_col not in out.columns:
            continue
        iv = pd.to_numeric(out[iv_col], errors="coerce")
        out[f"{prefix}_vrp_short"] = iv - out["realized_vol_short"]
        out[f"{prefix}_vrp_long"] = iv - out["realized_vol_long"]
        out[f"{prefix}_vrp_ratio_short"] = _safe_div(iv, out["realized_vol_short"])
        out[f"{prefix}_vrp_ratio_long"] = _safe_div(iv, out["realized_vol_long"])

    if "smile_vrp_long" in out.columns:
        out["vrp"] = out["smile_vrp_long"]
        out["vrp_short"] = out.get("smile_vrp_short", np.nan)
        out["vrp_ratio"] = out.get("smile_vrp_ratio_long", np.nan)
    elif "observed_vrp_long" in out.columns:
        out["vrp"] = out["observed_vrp_long"]
        out["vrp_short"] = out.get("observed_vrp_short", np.nan)
        out["vrp_ratio"] = out.get("observed_vrp_ratio_long", np.nan)
    else:
        out["vrp"] = np.nan
        out["vrp_short"] = np.nan
        out["vrp_ratio"] = np.nan

    out["vrp_pct"] = 100.0 * out["vrp"]
    out["vrp_short_pct"] = 100.0 * out["vrp_short"]
    out["vrp_z"] = _rolling_zscore_by_product(out, "vrp", cfg.z_window)
    out["vrp_short_z"] = _rolling_zscore_by_product(out, "vrp_short", cfg.z_window)

    if "iv_residual" in out.columns:
        out["vrp_residual_combo"] = out["iv_residual"] + out["vrp"]
        out["vrp_residual_combo_z"] = _rolling_zscore_by_product(out, "vrp_residual_combo", cfg.z_window)
    else:
        out["vrp_residual_combo"] = np.nan
        out["vrp_residual_combo_z"] = np.nan

    return out


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



def bs_call_price_vectorized(S: Any, K: Any, T: Any, r: float, sigma: Any) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    out = np.maximum(S - K * np.exp(-r * np.maximum(T, 0.0)), 0.0)
    good = (S > 0) & (K > 0) & (T > 0) & (sigma > 0) & np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(sigma)
    if good.any():
        sqrtT = np.sqrt(T[good])
        d1 = (np.log(S[good] / K[good]) + (r + 0.5 * sigma[good] * sigma[good]) * T[good]) / (sigma[good] * sqrtT)
        d2 = d1 - sigma[good] * sqrtT
        out[good] = S[good] * _norm_cdf(d1) - K[good] * np.exp(-r * T[good]) * _norm_cdf(d2)
    out[~np.isfinite(out)] = np.nan
    return out


def implied_vol_call_vectorized(price: Any, S: Any, K: Any, T: Any, r: float = 0.0, iterations: int = 32) -> np.ndarray:
    price = np.asarray(price, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    iv = np.full_like(price, np.nan, dtype=float)
    intrinsic = np.maximum(S - K * np.exp(-r * np.maximum(T, 0.0)), 0.0)
    good = (
        np.isfinite(price) & np.isfinite(S) & np.isfinite(K) & np.isfinite(T)
        & (price > 0) & (S > 0) & (K > 0) & (T > 0)
        & (price >= intrinsic - 1e-7) & (price <= S + 1e-7)
    )
    if not good.any():
        return iv
    lo = np.full(good.sum(), 1e-5)
    hi = np.full(good.sum(), 5.0)
    Sg, Kg, Tg, pg = S[good], K[good], T[good], price[good]
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        val = bs_call_price_vectorized(Sg, Kg, Tg, r, mid)
        hi = np.where(val > pg, mid, hi)
        lo = np.where(val <= pg, mid, lo)
    iv[good] = 0.5 * (lo + hi)
    return iv


def bs_call_greeks_vectorized(S: Any, K: Any, T: Any, r: float, sigma: Any) -> pd.DataFrame:
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(S)
    delta = np.where(S > K, 1.0, 0.0).astype(float)
    gamma = np.zeros(n, dtype=float)
    vega = np.zeros(n, dtype=float)
    theta = np.zeros(n, dtype=float)
    rho = np.zeros(n, dtype=float)
    good = (S > 0) & (K > 0) & (T > 0) & (sigma > 0) & np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(sigma)
    if good.any():
        sqrtT = np.sqrt(T[good])
        d1 = (np.log(S[good] / K[good]) + (r + 0.5 * sigma[good] * sigma[good]) * T[good]) / (sigma[good] * sqrtT)
        d2 = d1 - sigma[good] * sqrtT
        pdf_d1 = _norm_pdf(d1)
        delta[good] = _norm_cdf(d1)
        gamma[good] = pdf_d1 / (S[good] * sigma[good] * sqrtT)
        vega[good] = S[good] * pdf_d1 * sqrtT / 100.0
        theta[good] = (-(S[good] * pdf_d1 * sigma[good]) / (2.0 * sqrtT) - r * K[good] * np.exp(-r * T[good]) * _norm_cdf(d2)) / 365.0
        rho[good] = K[good] * T[good] * np.exp(-r * T[good]) * _norm_cdf(d2) / 100.0
    return pd.DataFrame({"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho})


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

    # Round 3 has day 0/1/2 with timestamps restarted each day. Build a monotonic clock.
    t = pd.to_numeric(df["timestamp"], errors="coerce").astype(float)
    if "day" in df.columns:
        day = pd.to_numeric(df["day"], errors="coerce").fillna(0).astype(float)
        span = max(float(t.max() - t.min()), 1.0)
        clock = day * (span + 1.0) + t
    else:
        clock = t

    progress = (clock - clock.min()) / max(clock.max() - clock.min(), EPS)
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

    sort_cols = [c for c in ["day", "timestamp", "product"] if c in activity.columns]
    activity = activity.sort_values(sort_cols).copy()

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
    time_keys = _time_key_cols(activity)

    underlying = activity[activity["product"] == underlying_product].copy()
    keep_underlying_cols = [c for c in [*time_keys, "mid_price", "bid_price_1", "ask_price_1"] if c in underlying.columns]
    underlying = underlying[keep_underlying_cols].rename(columns={"mid_price": "underlying_mid"})
    # Critical for Round 3: timestamps repeat across days, so merge on day+timestamp when day exists.
    underlying = underlying.drop_duplicates(subset=time_keys, keep="last") if time_keys else underlying

    options = activity[activity["product"].isin(option_products)].copy()
    options["strike"] = options["product"].map(_infer_strike)
    options = options.dropna(subset=["strike"]).reset_index(drop=True)
    if time_keys and all(c in underlying.columns for c in time_keys):
        options = options.merge(underlying[[*time_keys, "underlying_mid"]], on=time_keys, how="left", validate="many_to_one")
        options["underlying_mid"] = options.groupby("product")["underlying_mid"].ffill().bfill()
    else:
        options["underlying_mid"] = np.nan

    options["T"] = _estimate_time_to_expiry(options, cfg)
    options["moneyness"] = _safe_div(options["underlying_mid"], options["strike"])
    options["log_moneyness"] = np.log(_safe_div(options["underlying_mid"], options["strike"]))

    if "mid_price" not in options.columns:
        options["mid_price"] = np.nan

    options["intrinsic"] = np.maximum(options["underlying_mid"] - options["strike"], 0.0)
    options["extrinsic"] = options["mid_price"] - options["intrinsic"]

    # Vectorized IV is much faster than looping through every row and prevents the dashboard from hanging.
    options["observed_iv"] = implied_vol_call_vectorized(
        options["mid_price"].to_numpy(),
        options["underlying_mid"].to_numpy(),
        options["strike"].to_numpy(),
        options["T"].to_numpy(),
        cfg.risk_free_rate,
    )

    options = _fit_smile_by_timestamp(options)

    options["fair_price"] = bs_call_price_vectorized(
        options["underlying_mid"].to_numpy(),
        options["strike"].to_numpy(),
        options["T"].to_numpy(),
        cfg.risk_free_rate,
        options["smile_iv"].to_numpy(),
    )
    options["mispricing"] = options["mid_price"] - options["fair_price"]
    options["normalized_mispricing"] = _safe_div(options["mispricing"], options["fair_price"])
    options["iv_residual"] = options["observed_iv"] - options["smile_iv"]

    greek_sigma = options["smile_iv"].where(options["smile_iv"].notna(), options["observed_iv"])
    greeks_df = bs_call_greeks_vectorized(
        options["underlying_mid"].to_numpy(),
        options["strike"].to_numpy(),
        options["T"].to_numpy(),
        cfg.risk_free_rate,
        greek_sigma.to_numpy(),
    )
    greeks_df.index = options.index
    options = pd.concat([options, greeks_df], axis=1)

    options = _add_rolling_stats(options, cfg)
    options = _add_gamma_scalping_ev(options)
    options = _add_signals(options, cfg)

    underlying_stats_input = activity[activity["product"] == underlying_product].copy()
    underlying = _add_underlying_stats(underlying_stats_input, cfg)
    options = _add_cross_sectional_mispricing_features(options, underlying, cfg)
    options = _add_vrp_features(options, cfg)
    options = _add_cross_sectional_ranks(options)

    # Keep stores lighter. Raw book fields have already been converted into derived diagnostics.
    useful_cols = [
        "day", "timestamp", "product", "strike", "underlying_mid", "mid_price", "bid_price_1", "ask_price_1",
        "T", "moneyness", "log_moneyness",
        "observed_iv", "smile_iv", "smile_fit_ok", "fair_price", "mispricing", "normalized_mispricing",
        "iv_residual", "mispricing_z", "iv_residual_z", "normalized_mispricing_z", "delta", "gamma",
        "vega", "theta", "rho", "gamma_pnl_est", "theta_cost_est", "net_gamma_scalp_ev",
        "gamma_theta_ratio", "signal", "rv_short_ann_proxy", "rv_long_ann_proxy", "rv_iv_spread",
        "rv_iv_spread_short", "realized_vol_short", "realized_vol_long", "vrp", "vrp_pct",
        "vrp_short", "vrp_short_pct", "vrp_ratio", "vrp_z", "vrp_short_z",
        "observed_vrp_long", "observed_vrp_short", "smile_vrp_long", "smile_vrp_short",
        "observed_vrp_ratio_long", "observed_vrp_ratio_short", "smile_vrp_ratio_long", "smile_vrp_ratio_short",
        "vrp_residual_combo", "vrp_residual_combo_z", "rv_trend", "quoted_spread", "quoted_spread_pct", "book_imbalance",
        "top_level_depth", "xw_fitted_iv", "xw_iv_mispricing", "xw_price_mispricing",
        "rv_iv_spread_rank", "rv_iv_spread_decile", "vrp_rank", "vrp_decile",
        "vrp_residual_combo_rank", "vrp_residual_combo_decile", "iv_residual_rank", "iv_residual_decile",
        "mispricing_rank", "mispricing_decile", "xw_iv_mispricing_rank", "xw_iv_mispricing_decile",
        "xw_price_mispricing_rank", "xw_price_mispricing_decile", "delta_hedged_return_proxy",
    ]
    options = options[[c for c in useful_cols if c in options.columns]]

    return {"options": options, "underlying": underlying, "trades": trades}


def _fit_smile_by_timestamp(options: pd.DataFrame) -> pd.DataFrame:
    """Fast per-day/timestamp smile fit.

    The original version used many pandas .loc writes inside 30k tiny groups. Round 3 has
    roughly 10 option rows per timestamp across 3 days, so array writes are much faster.
    """
    out = options.reset_index(drop=True).copy()
    n = len(out)
    out["smile_iv"] = np.nan
    out["smile_fit_ok"] = False
    out["smile_a"] = np.nan
    out["smile_b"] = np.nan
    out["smile_c"] = np.nan

    time_keys = _time_key_cols(out)
    if not time_keys or n == 0:
        return out

    x_all = out["log_moneyness"].to_numpy(dtype=float)
    y_all = out["observed_iv"].to_numpy(dtype=float)
    extr_all = out["extrinsic"].to_numpy(dtype=float)

    smile_iv = np.full(n, np.nan, dtype=float)
    smile_ok = np.zeros(n, dtype=bool)
    smile_a = np.full(n, np.nan, dtype=float)
    smile_b = np.full(n, np.nan, dtype=float)
    smile_c = np.full(n, np.nan, dtype=float)

    # observed_iv must be finite and in a sane range; extrinsic > 0.25 avoids unstable near-intrinsic IV points.
    for _, idx in out.groupby(time_keys, sort=False, dropna=False).indices.items():
        idx = np.asarray(idx, dtype=int)
        valid = (
            np.isfinite(x_all[idx])
            & np.isfinite(y_all[idx])
            & (y_all[idx] >= 1e-4)
            & (y_all[idx] <= 5.0)
            & np.isfinite(extr_all[idx])
            & (extr_all[idx] > 0.25)
        )
        good_idx = idx[valid]
        if len(good_idx) >= 3:
            try:
                coef = np.polyfit(x_all[good_idx], y_all[good_idx], deg=2)
                smile_iv[idx] = np.polyval(coef, x_all[idx])
                smile_ok[idx] = True
                smile_a[idx], smile_b[idx], smile_c[idx] = coef
                continue
            except Exception:
                pass

        fallback_vals = y_all[good_idx] if len(good_idx) else y_all[idx][np.isfinite(y_all[idx])]
        fallback = float(np.nanmedian(fallback_vals)) if len(fallback_vals) else np.nan
        smile_iv[idx] = fallback

    out["smile_iv"] = pd.Series(smile_iv).clip(lower=1e-5, upper=5.0)
    out["smile_fit_ok"] = smile_ok
    out["smile_a"] = smile_a
    out["smile_b"] = smile_b
    out["smile_c"] = smile_c
    return out


def _add_rolling_stats(options: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.DataFrame:
    out = options.sort_values([c for c in ["product", "day", "timestamp"] if c in options.columns]).copy()
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
    out = options.sort_values([c for c in ["product", "day", "timestamp"] if c in options.columns]).copy()
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
    out = out.sort_values([c for c in ["day", "timestamp"] if c in out.columns]).copy() if "timestamp" in out.columns else out.copy()
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
    latest = _latest_slice(options)
    idx = latest.groupby("product")["timestamp"].idxmax()
    cols = [
        "timestamp",
        "product",
        "strike",
        "underlying_mid",
        "mid_price",
        "observed_iv",
        "smile_iv",
        "iv_residual",
        "realized_vol_long",
        "vrp",
        "vrp_pct",
        "vrp_z",
        "vrp_ratio",
        "vrp_residual_combo",
        "vrp_residual_combo_z",
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
    return latest.loc[idx, [c for c in cols if c in latest.columns]].sort_values("strike").round(6)


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

def _repair_options_for_plotting(options: pd.DataFrame) -> pd.DataFrame:
    """Make stored/cached options data robust after reloads or older builds.

    Dash can keep an old dcc.Store payload in the browser after code changes. This
    helper recreates lightweight derived columns that plots need instead of throwing
    KeyError.
    """
    if options is None or options.empty:
        return pd.DataFrame()
    out = options.copy()
    for col in ["timestamp", "strike", "underlying_mid", "mid_price", "observed_iv", "smile_iv"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "strike" not in out.columns and "product" in out.columns:
        out["strike"] = out["product"].map(_infer_strike)
    if "log_moneyness" not in out.columns and {"underlying_mid", "strike"}.issubset(out.columns):
        out["log_moneyness"] = np.log(_safe_div(out["underlying_mid"], out["strike"]))
    if "moneyness" not in out.columns and {"underlying_mid", "strike"}.issubset(out.columns):
        out["moneyness"] = _safe_div(out["underlying_mid"], out["strike"])
    if "observed_iv" not in out.columns:
        out["observed_iv"] = np.nan
    if "smile_iv" not in out.columns:
        out["smile_iv"] = out["observed_iv"]
    if "iv_residual" not in out.columns and {"observed_iv", "smile_iv"}.issubset(out.columns):
        out["iv_residual"] = pd.to_numeric(out["observed_iv"], errors="coerce") - pd.to_numeric(out["smile_iv"], errors="coerce")
    return out


def fig_vol_smile(options: pd.DataFrame, timestamp: Optional[float] = None) -> go.Figure:
    options = _repair_options_for_plotting(options)
    if options.empty:
        return _empty_fig("Volatility Smile", "No options data available. Click Build / Parse Options Data first.")
    df = options.copy()

    required_base = {"log_moneyness", "observed_iv", "smile_iv"}
    missing = sorted(required_base - set(df.columns))
    if missing:
        return _empty_fig("Volatility Smile", f"Missing required columns after repair: {missing}")

    if timestamp is None:
        if "timestamp" in df.columns:
            usable = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_moneyness", "observed_iv"])
            if not usable.empty:
                latest_ts = usable["timestamp"].max()
                df = df[df["timestamp"] == latest_ts]
                timestamp = latest_ts
            else:
                df = _latest_slice(df)
                timestamp = df["timestamp"].max() if "timestamp" in df.columns and not df.empty else None
    elif "timestamp" in df.columns:
        df = df[df["timestamp"] == timestamp]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_moneyness", "observed_iv", "smile_iv"])
    if df.empty:
        return _empty_fig("Volatility Smile", "No IV data available at the selected/latest timestamp")

    df = df.sort_values("log_moneyness")
    label = df["strike"] if "strike" in df.columns else df.get("product", None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["log_moneyness"], y=df["observed_iv"], mode="markers+text", text=label,
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
        u = u.sort_values([c for c in ["day", "timestamp"] if c in u.columns])
        u["underlying_log_return"] = np.log(u["mid_price"]).diff()
        u["rv_short"] = u["underlying_log_return"].rolling(max(5, cfg.rolling_window // 2), min_periods=5).std()
        u["rv_long"] = u["underlying_log_return"].rolling(max(10, cfg.rolling_window * 2), min_periods=10).std()
        u["rv_trend"] = u["rv_short"] - u["rv_long"]
        u["rv_short_ann_proxy"] = u["rv_short"] * np.sqrt(365.0)
        u["rv_long_ann_proxy"] = u["rv_long"] * np.sqrt(365.0)
        time_keys = _time_key_cols(out)
        merge_cols = [*time_keys, "rv_short", "rv_long", "rv_trend", "rv_short_ann_proxy", "rv_long_ann_proxy"]
        out = out.merge(u[[c for c in merge_cols if c in u.columns]].drop_duplicates(subset=time_keys), on=time_keys, how="left")
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
    """Pooled Xu-Wang style IV residual.

    A per-timestamp least-squares regression is too slow for the full Round 3 upload.
    This keeps the same idea, but uses one pooled cross-sectional/time-series fit.
    """
    out = options.copy()
    out["xw_fitted_iv"] = np.nan
    out["xw_iv_mispricing"] = np.nan
    out["xw_price_mispricing"] = np.nan
    out["xw_model_ok"] = False

    features = ["rv_short_ann_proxy", "rv_long_ann_proxy", "log_moneyness", "T", "quoted_spread_pct", "book_imbalance", "top_level_depth"]
    available = [c for c in features if c in out.columns]
    if not available or "observed_iv" not in out.columns:
        return out

    pooled = out[available + ["observed_iv"]].replace([np.inf, -np.inf], np.nan).dropna(subset=["observed_iv"])
    useful = [c for c in available if pooled[c].notna().sum() >= 20 and pooled[c].nunique(dropna=True) > 1]
    if len(pooled) >= max(50, len(useful) + 5) and useful:
        try:
            med = pooled[useful].median()
            X = pooled[useful].fillna(med).to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(X)), X])
            y = pooled["observed_iv"].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)

            X_all = out[useful].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(dtype=float)
            X_all = np.column_stack([np.ones(len(X_all)), X_all])
            out["xw_fitted_iv"] = X_all @ beta
            out["xw_model_ok"] = True
        except Exception:
            pass

    out["xw_iv_mispricing"] = out["observed_iv"] - out["xw_fitted_iv"]
    price_obs = bs_call_price_vectorized(out["underlying_mid"], out["strike"], out["T"], 0.0, out["observed_iv"])
    price_fit = bs_call_price_vectorized(out["underlying_mid"], out["strike"], out["T"], 0.0, out["xw_fitted_iv"])
    out["xw_price_mispricing"] = price_obs - price_fit
    return out


def _safe_qcut(s: pd.Series, q: int = 10) -> pd.Series:
    vals = s.replace([np.inf, -np.inf], np.nan)
    try:
        return pd.qcut(vals.rank(method="first"), q=q, labels=False, duplicates="drop") + 1
    except Exception:
        return pd.Series(np.nan, index=s.index)


def _add_cross_sectional_ranks(options: pd.DataFrame) -> pd.DataFrame:
    out = options.copy()
    rank_cols = [
        "rv_iv_spread",
        "vrp",
        "vrp_residual_combo",
        "iv_residual",
        "mispricing",
        "xw_iv_mispricing",
        "xw_price_mispricing",
    ]
    time_keys = _time_key_cols(out)

    for c in rank_cols:
        if c not in out.columns:
            continue
        if time_keys:
            pct = out.groupby(time_keys)[c].rank(pct=True)
        else:
            pct = out[c].rank(pct=True)
        out[f"{c}_rank"] = pct
        out[f"{c}_decile"] = np.ceil(pct * 10.0).clip(lower=1, upper=10)
    return out


def _add_delta_hedged_return_proxy(options: pd.DataFrame) -> pd.DataFrame:
    out = options.sort_values([c for c in ["product", "day", "timestamp"] if c in options.columns]).copy() if "timestamp" in options.columns else options.copy()
    out["option_price_change"] = out.groupby("product")["mid_price"].diff()
    out["underlying_price_change_for_hedge"] = out.groupby("product")["underlying_mid"].diff()
    out["lag_delta"] = out.groupby("product")["delta"].shift(1)
    out["delta_hedged_pnl_proxy"] = out["option_price_change"] - out["lag_delta"] * out["underlying_price_change_for_hedge"]
    out["delta_hedged_return_proxy"] = _safe_div(out["delta_hedged_pnl_proxy"], out.groupby("product")["mid_price"].shift(1).abs())
    return out


def compute_cross_sectional_signal_table(options: pd.DataFrame) -> pd.DataFrame:
    if options.empty or "timestamp" not in options.columns:
        return pd.DataFrame()
    latest = _latest_slice(options)
    idx = latest.groupby("product")["timestamp"].idxmax()
    cols = ["timestamp", "product", "strike", "observed_iv", "rv_long_ann_proxy", "realized_vol_long", "rv_iv_spread", "rv_iv_spread_decile", "vrp", "vrp_z", "vrp_decile", "vrp_residual_combo", "vrp_residual_combo_z", "vrp_residual_combo_decile", "xw_fitted_iv", "xw_iv_mispricing", "xw_iv_mispricing_decile", "xw_price_mispricing", "rv_trend", "quoted_spread", "book_imbalance", "delta_hedged_return_proxy", "signal"]
    return latest.loc[idx, [c for c in cols if c in latest.columns]].sort_values("strike").round(6)


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


def fig_vrp(options: pd.DataFrame) -> go.Figure:
    if options.empty or "vrp" not in options.columns:
        return _empty_fig("Volatility Risk Premium")
    fig = px.line(
        options,
        x="timestamp",
        y="vrp",
        color="product",
        title="Volatility Risk Premium: Fitted IV minus Realized Volatility",
    )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(template="plotly_white", height=450, yaxis_title="VRP")
    return fig


def fig_vrp_zscore(options: pd.DataFrame) -> go.Figure:
    if options.empty or "vrp_z" not in options.columns:
        return _empty_fig("VRP Z-Score")
    fig = px.line(
        options,
        x="timestamp",
        y="vrp_z",
        color="product",
        title="Volatility Risk Premium Z-Score",
    )
    for y, name in [(1.5, "rich vol"), (-1.5, "cheap vol"), (0, "neutral")]:
        fig.add_hline(y=y, line_dash="dash", annotation_text=name)
    fig.update_layout(template="plotly_white", height=450, yaxis_title="VRP z-score")
    return fig

def _add_display_day_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw day labels to dashboard Day 0, Day 1, Day 2 labels.

    Prosperity uploads often use raw day labels like -1, 0, 1, while the dashboard
    sections are labeled Day 0, Day 1, Day 2. This helper keeps the raw day in
    raw_day and adds display_day = 0, 1, 2 based on sorted trading-day order.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "day" not in out.columns:
        out["day"] = 0

    out["raw_day"] = pd.to_numeric(out["day"], errors="coerce").fillna(0).astype(int)
    ordered_days = sorted(out["raw_day"].dropna().unique().tolist())
    day_map = {raw_day: i for i, raw_day in enumerate(ordered_days)}
    out["display_day"] = out["raw_day"].map(day_map).astype("Int64")
    return out


def _midday_option_slice_by_day(options: pd.DataFrame) -> pd.DataFrame:
    """Return one representative mid-day options slice per displayed trading day.

    Uses display_day labels so raw Prosperity days -1/0/1 still render as
    dashboard Day 0/1/2.
    """
    options = _repair_options_for_plotting(options)
    if options.empty or "timestamp" not in options.columns:
        return pd.DataFrame()

    out = options.copy()
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if out.empty:
        return pd.DataFrame()

    out = _add_display_day_labels(out)

    frames: list[pd.DataFrame] = []
    for display_day, sub in out.groupby("display_day", sort=True):
        if sub.empty:
            continue
        target_ts = float(sub["timestamp"].median())
        available_ts = sub["timestamp"].dropna().unique()
        if len(available_ts) == 0:
            continue
        selected_ts = min(available_ts, key=lambda x: abs(float(x) - target_ts))
        frames.append(sub[sub["timestamp"] == selected_ts].copy())

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _option_smile_axis(df: pd.DataFrame) -> pd.Series:
    """Use the normalized moneyness axis from the screenshot when possible."""
    if {"strike", "underlying_mid", "T"}.issubset(df.columns):
        sqrt_t = np.sqrt(pd.to_numeric(df["T"], errors="coerce").clip(lower=EPS))
        return pd.Series(
            np.log(_safe_div(pd.to_numeric(df["strike"], errors="coerce"), pd.to_numeric(df["underlying_mid"], errors="coerce"))) / sqrt_t,
            index=df.index,
        )
    if "log_moneyness" in df.columns:
        return -pd.to_numeric(df["log_moneyness"], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def fig_daily_smile_outliers(options: pd.DataFrame) -> go.Figure:
    """Show one mid-day IV smile panel per day, with strike labels and fitted smile."""
    df = _midday_option_slice_by_day(options)
    if df.empty:
        return _empty_fig("The Smile and Its Outliers", "No options data available for daily smile panels")

    required = {"observed_iv", "smile_iv", "strike"}
    missing = sorted(required - set(df.columns))
    if missing:
        return _empty_fig("The Smile and Its Outliers", f"Missing required columns: {missing}")

    df = df.copy()
    df["smile_axis"] = _option_smile_axis(df)
    df["observed_iv_pct"] = 100.0 * pd.to_numeric(df["observed_iv"], errors="coerce")
    df["smile_iv_pct"] = 100.0 * pd.to_numeric(df["smile_iv"], errors="coerce")
    df["strike_label"] = pd.to_numeric(df["strike"], errors="coerce").round(0).astype("Int64").astype(str)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["smile_axis", "observed_iv_pct", "strike"])
    if df.empty:
        return _empty_fig("The Smile and Its Outliers", "No usable IV smile points found")

    if "display_day" not in df.columns:
        df = _add_display_day_labels(df)
    days = sorted(df["display_day"].dropna().astype(int).unique().tolist()) if "display_day" in df.columns else [0]
    ncols = min(max(len(days), 1), 4)
    fig = make_subplots(
        rows=1,
        cols=ncols,
        subplot_titles=[f"Day {int(day)} IV Smile at mid-day" for day in days[:ncols]],
        shared_yaxes=True,
    )

    palette = px.colors.qualitative.Plotly
    for col_idx, day in enumerate(days[:ncols], start=1):
        sub = df[df["display_day"] == day].sort_values("smile_axis").copy() if "display_day" in df.columns else df.sort_values("smile_axis").copy()
        if sub.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=sub["smile_axis"],
                y=sub["observed_iv_pct"],
                mode="markers+text",
                text=sub["strike_label"],
                textposition="top center",
                marker=dict(size=9, color=[palette[i % len(palette)] for i in range(len(sub))]),
                name=f"Day {int(day)} observed IV",
                showlegend=False,
                hovertemplate="Strike %{text}<br>m=%{x:.3f}<br>IV=%{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

        fit_sub = sub.dropna(subset=["smile_iv_pct"]).copy()
        if len(fit_sub) >= 3:
            # If the data contains deep wing strikes like 4000/4500, exclude them from the dashed fit display.
            non_wing = fit_sub[fit_sub["strike"] > 4500]
            if len(non_wing) >= 3:
                fit_sub = non_wing
        fit_sub = fit_sub.sort_values("smile_axis")
        fig.add_trace(
            go.Scatter(
                x=fit_sub["smile_axis"],
                y=fit_sub["smile_iv_pct"],
                mode="lines",
                line=dict(dash="dash", width=2, color="gray"),
                name="Fitted smile",
                showlegend=(col_idx == 1),
                hovertemplate="m=%{x:.3f}<br>Fitted IV=%{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(title_text="Moneyness m = log(K/S) / sqrt(T)", row=1, col=col_idx)

    fig.update_yaxes(title_text="Implied Vol (%)", row=1, col=1)
    fig.update_layout(
        title="The Smile and Its Outliers",
        template="plotly_white",
        height=470,
        margin=dict(l=60, r=30, t=80, b=70),
    )
    return fig




def fig_single_day_iv_smile(options: pd.DataFrame, day: int) -> go.Figure:
    """Large standalone IV smile plot for one displayed trading day.

    The day argument is the displayed dashboard day, not necessarily the raw CSV day.
    For example, raw days -1/0/1 become displayed Day 0/1/2.
    """
    df = _midday_option_slice_by_day(options)
    if df.empty:
        return _empty_fig(f"Day {day} IV Smile", "No options data available for this day")

    if "display_day" not in df.columns:
        df = _add_display_day_labels(df)

    available_days = sorted(df["display_day"].dropna().astype(int).unique().tolist()) if "display_day" in df.columns else []
    df = df[df["display_day"] == int(day)].copy()
    if df.empty:
        return _empty_fig(
            f"Day {day} IV Smile",
            f"No mid-day option slice found for displayed day {day}. Available displayed days: {available_days}",
        )

    required = {"observed_iv", "smile_iv", "strike"}
    missing = sorted(required - set(df.columns))
    if missing:
        return _empty_fig(f"Day {day} IV Smile", f"Missing required columns: {missing}")

    df["smile_axis"] = _option_smile_axis(df)
    df["observed_iv_pct"] = 100.0 * pd.to_numeric(df["observed_iv"], errors="coerce")
    df["smile_iv_pct"] = 100.0 * pd.to_numeric(df["smile_iv"], errors="coerce")
    df["iv_residual_pct"] = 100.0 * pd.to_numeric(df.get("iv_residual", np.nan), errors="coerce")
    df["strike_label"] = pd.to_numeric(df["strike"], errors="coerce").round(0).astype("Int64").astype(str)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["smile_axis", "observed_iv_pct", "strike"])
    if df.empty:
        return _empty_fig(f"Day {day} IV Smile", "No usable IV smile points found")

    df = df.sort_values("smile_axis")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["smile_axis"],
            y=df["observed_iv_pct"],
            mode="markers+text",
            text=df["strike_label"],
            textposition="top center",
            marker=dict(
                size=13,
                color=df["iv_residual_pct"],
                colorscale=[
                    [0.00, "#08306b"],
                    [0.25, "#2171b5"],
                    [0.50, "#f7f7f7"],
                    [0.75, "#de2d26"],
                    [1.00, "#67000d"],
                ],
                cmid=0,
                colorbar=dict(title="IV residual (%)"),
                line=dict(width=1.2, color="black"),
            ),
            name="Observed IV",
            hovertemplate="Strike %{text}<br>m=%{x:.3f}<br>IV=%{y:.2f}%<br>Residual=%{marker.color:+.2f}%<extra></extra>",
        )
    )

    fit_sub = df.dropna(subset=["smile_iv_pct"]).copy()
    if len(fit_sub) >= 3:
        non_wing = fit_sub[fit_sub["strike"] > 4500]
        if len(non_wing) >= 3:
            fit_sub = non_wing
    fit_sub = fit_sub.sort_values("smile_axis")
    fig.add_trace(
        go.Scatter(
            x=fit_sub["smile_axis"],
            y=fit_sub["smile_iv_pct"],
            mode="lines",
            line=dict(dash="dash", width=2.5, color="gray"),
            name="Fitted smile",
            hovertemplate="m=%{x:.3f}<br>Fitted IV=%{y:.2f}%<extra></extra>",
        )
    )

    raw_day_text = ""
    if "raw_day" in df.columns and df["raw_day"].notna().any():
        raw_day_text = f" (raw day {int(df['raw_day'].iloc[0])})"

    fig.add_hline(y=0, line_dash="dot", opacity=0.25)
    fig.update_layout(
        title=f"Day {day} IV Smile at Mid-Day{raw_day_text}",
        xaxis_title="Moneyness m = log(K/S) / sqrt(T)",
        yaxis_title="Implied Vol (%)",
        template="plotly_white",
        height=560,
        margin=dict(l=70, r=40, t=80, b=70),
    )
    return fig

def fig_mispricing_heatmap(options: pd.DataFrame) -> go.Figure:
    """Mean IV residual heatmap by displayed day and strike.

    Shows exactly one row per displayed trading day, so raw Prosperity days like
    -1/0/1 are rendered as Day 0/1/2. Negative residuals are cheap and positive
    residuals are rich.
    """
    options = _repair_options_for_plotting(options)
    if options.empty or "iv_residual" not in options.columns:
        return _empty_fig("Mispricing Heatmap", "Need IV residuals from observed IV minus fitted smile")

    df = options.copy()
    df = _add_display_day_labels(df)

    if "strike" not in df.columns and "product" in df.columns:
        df["strike"] = df["product"].map(_infer_strike)

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["iv_residual_pct"] = 100.0 * pd.to_numeric(df["iv_residual"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["display_day", "strike", "iv_residual_pct"])
    if df.empty:
        return _empty_fig("Mispricing Heatmap", "No usable IV residual data found")

    pivot = (
        df.groupby(["display_day", "strike"], as_index=False)["iv_residual_pct"]
        .mean()
        .pivot(index="display_day", columns="strike", values="iv_residual_pct")
        .sort_index()
        .sort_index(axis=1)
    )

    # Force the heatmap to keep one row for each displayed trading day present in the upload.
    all_days = sorted(df["display_day"].dropna().astype(int).unique().tolist())
    pivot = pivot.reindex(all_days)

    if pivot.empty:
        return _empty_fig("Mispricing Heatmap", "No day by strike residual grid found")

    try:
        text = pivot.map(lambda x: "" if pd.isna(x) else f"{x:+.2f}").to_numpy()
    except AttributeError:
        text = pivot.applymap(lambda x: "" if pd.isna(x) else f"{x:+.2f}").to_numpy()

    z = pivot.to_numpy(dtype=float)
    finite_abs = np.abs(z[np.isfinite(z)])
    if finite_abs.size:
        # Use a robust cap so small but meaningful residuals still show strong color.
        robust_cap = float(np.nanpercentile(finite_abs, 85))
        max_abs = float(np.nanmax(finite_abs))
        color_cap = max(0.05, min(max_abs, robust_cap if robust_cap > EPS else max_abs))
    else:
        color_cap = 1.0

    strike_labels = [str(int(c)) if float(c).is_integer() else str(c) for c in pivot.columns]
    day_labels = [f"Day {int(i)}" for i in pivot.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=strike_labels,
            y=day_labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "black"},
            colorscale=[
                [0.00, "#08306b"],
                [0.18, "#2171b5"],
                [0.36, "#6baed6"],
                [0.50, "#f7f7f7"],
                [0.64, "#fcae91"],
                [0.82, "#de2d26"],
                [1.00, "#67000d"],
            ],
            zmid=0,
            zmin=-color_cap,
            zmax=color_cap,
            xgap=2,
            ygap=2,
            colorbar=dict(title="IV residual (%)"),
            hovertemplate="%{y}<br>Strike %{x}<br>Mean IV residual %{z:+.3f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Mispricing Heatmap: Mean IV Residual vs Smile Fit",
        xaxis_title="Strike",
        yaxis_title="Trading day",
        template="plotly_white",
        height=540,
        margin=dict(l=80, r=40, t=90, b=70),
        annotations=[
            dict(
                text="Blue = CHEAP, IV below fitted smile. Red = RICH, IV above fitted smile.",
                x=0.5,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
            )
        ],
    )
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
                    _section("Volatility Risk Premium", "options-stats-load-vrp", dcc.Graph(id="options-stats-vrp", figure=_empty_fig("Volatility Risk Premium", "Click Load to render this chart.")), "Fitted IV minus realized volatility. Positive means rich vol, negative means cheap vol."),
                    _section("VRP Z-Score", "options-stats-load-vrp-zscore", dcc.Graph(id="options-stats-vrp-zscore", figure=_empty_fig("VRP Z-Score", "Click Load to render this chart.")), "Rolling z-score of volatility risk premium by contract."),
                    _section("Xu-Wang IV Mispricing", "options-stats-load-xw-iv-mispricing", dcc.Graph(id="options-stats-xw-iv-mispricing", figure=_empty_fig("Xu-Wang IV Mispricing", "Click Load to render this chart.")), "Cross-sectional residual IV mispricing proxy."),
                    _section("Smile Outliers by Day", "options-stats-load-daily-smiles", dcc.Graph(id="options-stats-daily-smiles", figure=_empty_fig("Smile Outliers by Day", "Click Load to render this chart.")), "Mid-day IV smile panels for each trading day with strike labels and fitted smile."),
                    _section("Day 0 IV Smile", "options-stats-load-day0-smile", dcc.Graph(id="options-stats-day0-smile", figure=_empty_fig("Day 0 IV Smile", "Click Load to render this chart.")), "Large standalone mid-day IV smile for day 0."),
                    _section("Day 1 IV Smile", "options-stats-load-day1-smile", dcc.Graph(id="options-stats-day1-smile", figure=_empty_fig("Day 1 IV Smile", "Click Load to render this chart.")), "Large standalone mid-day IV smile for day 1."),
                    _section("Day 2 IV Smile", "options-stats-load-day2-smile", dcc.Graph(id="options-stats-day2-smile", figure=_empty_fig("Day 2 IV Smile", "Click Load to render this chart.")), "Large standalone mid-day IV smile for day 2."),
                    _section("Mispricing Heatmap", "options-stats-load-mispricing-heatmap", dcc.Graph(id="options-stats-mispricing-heatmap", figure=_empty_fig("Mispricing Heatmap", "Click Load to render this chart.")), "Mean IV residual by day and strike. Blue means cheap, red means rich."),
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
    avg_vrp = float(np.nanmean(options["vrp"].replace([np.inf, -np.inf], np.nan))) if "vrp" in options else np.nan
    return [
        _summary_card("Options detected", options["product"].nunique(), "unique option products"),
        _summary_card("Cheap signals", cheap, "latest z-score <= entry band"),
        _summary_card("Rich signals", rich, "latest z-score >= entry band"),
        _summary_card("Avg abs mispricing", round(avg_abs_mis, 4), "market minus smile fair"),
        _summary_card("Avg gamma/theta", round(avg_gamma_theta, 6), "convexity per theta cost"),
        _summary_card("Avg VRP", round(avg_vrp, 6), "fitted IV minus realized vol"),
        _summary_card("Underlying rows", len(underlying), "mean reversion sample"),
        _summary_card("Trade rows", len(trades), "if available"),
        _summary_card("Smile fit rows", int(options.get("smile_fit_ok", pd.Series(False)).sum()), "timestamps with parabola fit"),
    ]


OPTIONS_STATS_CACHE: dict[str, pd.DataFrame] = {}


def _store_df(df: pd.DataFrame) -> Any:
    if df is None or df.empty:
        return []
    # Avoid shipping hundreds of thousands of option rows through the browser as JSON.
    # Large DataFrames stay server-side and the dcc.Store only holds a small cache key.
    if len(df) > 5000:
        key = f"options_stats_df_{uuid.uuid4().hex}"
        OPTIONS_STATS_CACHE[key] = df.copy()
        # Keep the cache from growing forever during repeated reloads.
        if len(OPTIONS_STATS_CACHE) > 12:
            for old_key in list(OPTIONS_STATS_CACHE.keys())[:-12]:
                OPTIONS_STATS_CACHE.pop(old_key, None)
        return {"__options_stats_cache_key__": key, "rows": int(len(df)), "columns": list(map(str, df.columns))}
    return df.replace([np.inf, -np.inf], np.nan).where(pd.notna(df), None).to_dict("records")


def _df_from_store(data: Any) -> pd.DataFrame:
    if isinstance(data, dict) and "__options_stats_cache_key__" in data:
        cached = OPTIONS_STATS_CACHE.get(data["__options_stats_cache_key__"])
        return cached.copy() if cached is not None else pd.DataFrame()
    return pd.DataFrame(data or [])


def _options_from_store(data: Any) -> pd.DataFrame:
    return _repair_options_for_plotting(_df_from_store(data))


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

    @app.callback(Output("options-stats-vrp", "figure"), Input("options-stats-load-vrp", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_vrp(n, options_data):
        if not n: raise PreventUpdate
        return fig_vrp(_options_from_store(options_data))

    @app.callback(Output("options-stats-vrp-zscore", "figure"), Input("options-stats-load-vrp-zscore", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_vrp_zscore(n, options_data):
        if not n: raise PreventUpdate
        return fig_vrp_zscore(_options_from_store(options_data))

    @app.callback(Output("options-stats-xw-iv-mispricing", "figure"), Input("options-stats-load-xw-iv-mispricing", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_xw_iv_mispricing(n, options_data):
        if not n: raise PreventUpdate
        return fig_xw_iv_mispricing(_options_from_store(options_data))

    @app.callback(Output("options-stats-daily-smiles", "figure"), Input("options-stats-load-daily-smiles", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_daily_smiles(n, options_data):
        if not n: raise PreventUpdate
        return fig_daily_smile_outliers(_options_from_store(options_data))

    @app.callback(Output("options-stats-day0-smile", "figure"), Input("options-stats-load-day0-smile", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_day0_smile(n, options_data):
        if not n: raise PreventUpdate
        return fig_single_day_iv_smile(_options_from_store(options_data), 0)

    @app.callback(Output("options-stats-day1-smile", "figure"), Input("options-stats-load-day1-smile", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_day1_smile(n, options_data):
        if not n: raise PreventUpdate
        return fig_single_day_iv_smile(_options_from_store(options_data), 1)

    @app.callback(Output("options-stats-day2-smile", "figure"), Input("options-stats-load-day2-smile", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_day2_smile(n, options_data):
        if not n: raise PreventUpdate
        return fig_single_day_iv_smile(_options_from_store(options_data), 2)

    @app.callback(Output("options-stats-mispricing-heatmap", "figure"), Input("options-stats-load-mispricing-heatmap", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_mispricing_heatmap(n, options_data):
        if not n: raise PreventUpdate
        return fig_mispricing_heatmap(_options_from_store(options_data))

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
            "vrp": fig_vrp(options),
            "vrp_zscore": fig_vrp_zscore(options),
            "daily_smile_outliers": fig_daily_smile_outliers(options),
            "day0_iv_smile": fig_single_day_iv_smile(options, 0),
            "day1_iv_smile": fig_single_day_iv_smile(options, 1),
            "day2_iv_smile": fig_single_day_iv_smile(options, 2),
            "mispricing_heatmap": fig_mispricing_heatmap(options),
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
