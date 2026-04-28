"""
Options Stats tab for a Dash/Plotly trading dashboard.

Drop this file into app/views/options_stats.py or wherever your dashboard keeps tab modules.
It is intentionally defensive: it accepts common column names from Prosperity-style price,
activity, and trades DataFrames, then builds all of the option diagnostics discussed:

1. Volatility smile + fitted IV curve
2. IV residuals over time
3. Heston fair price vs market price
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

# --- Time-to-expiry conventions -------------------------------------------
# Round 3 convention: data stamped at 0, 100, 200, ... 999_900 within each day.
# Vouchers expire N days from the start of day 0. User specified 5 DTE at
# the start of day 0 (so day 2 end = ~3 DTE).
START_DAYS_TO_EXPIRY   = 5.0             # DTE at day 0 timestamp 0
TRADING_YEAR           = 250.0           # trading days per year for annualization
TICKS_PER_DAY          = 1_000_000       # Prosperity timestamp range per day
DEFAULT_SAMPLE_INTERVAL = 100             # typical timestamp stride between rows
DEFAULT_BINOMIAL_STEPS = 40                # CRR tree steps for European voucher pricing
DEFAULT_DAYS_TO_EXPIRY = START_DAYS_TO_EXPIRY

DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_ROLLING_WINDOW = 40
DEFAULT_Z_WINDOW = 80
EPS = 1e-12


def _np_trapz(y, x=None, axis=-1):
    """NumPy 2.x removed np.trapz; use trapezoid when available."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


@dataclass(frozen=True)
class OptionsStatsConfig:
    underlying_aliases: tuple[str, ...] = UNDERLYING_ALIASES
    option_name_hints: tuple[str, ...] = OPTION_NAME_HINTS
    # v2: time-to-expiry in trading-year units, starts at 5 DTE on day 0.
    start_days_to_expiry: float = START_DAYS_TO_EXPIRY
    trading_year:         float = TRADING_YEAR
    ticks_per_day:        float = TICKS_PER_DAY
    # legacy field preserved for any code that still reads it; no longer used
    # in _estimate_time_to_expiry.
    default_days_to_expiry: float = START_DAYS_TO_EXPIRY
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    z_window: int = DEFAULT_Z_WINDOW
    signal_entry_z: float = 1.5
    signal_exit_z: float = 0.35
    binomial_steps: int = DEFAULT_BINOMIAL_STEPS
    max_tradeable_spread: float = 20.0
    min_tradeable_delta: float = 0.15
    max_tradeable_delta: float = 0.85
    min_executable_edge: float = 2.0
    edge_spread_mult: float = 0.5


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
# Heston option pricing, implied volatility, and Greeks
# =============================================================================

# The dashboard previously used Cox-Ross-Rubinstein binomial trees as the theoretical
# fair-value layer. The compatibility function names below are preserved so the rest
# of the dashboard does not break, but the model-facing fair value and Greeks now use
# Heston stochastic volatility.

HESTON_DEFAULT_KAPPA = 3.0          # mean-reversion speed of variance
HESTON_DEFAULT_THETA = 0.04         # long-run variance, 20% vol squared
HESTON_DEFAULT_VOL_OF_VOL = 0.65    # volatility of variance
HESTON_DEFAULT_RHO = -0.45          # asset-vol correlation; negative creates downside skew
HESTON_INTEGRATION_LIMIT = 80.0
HESTON_INTEGRATION_STEPS = 64
HESTON_CHUNK_SIZE = 3500


def _bs_call_price_vectorized(S: Any, K: Any, T: Any, r: float, sigma: Any) -> np.ndarray:
    """Black-Scholes call price used only for IV inversion/fallbacks, not fair value."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    out = np.maximum(S - K * np.exp(-r * np.maximum(T, 0.0)), 0.0)
    good = (
        (S > 0) & (K > 0) & (T > 0) & (sigma > 0)
        & np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(sigma)
    )
    if not good.any():
        out[~np.isfinite(out)] = np.nan
        return out
    sqrtT = np.sqrt(T[good])
    d1 = (np.log(S[good] / K[good]) + (r + 0.5 * sigma[good] ** 2) * T[good]) / (sigma[good] * sqrtT)
    d2 = d1 - sigma[good] * sqrtT
    out[good] = S[good] * _norm_cdf(d1) - K[good] * np.exp(-r * T[good]) * _norm_cdf(d2)
    out[~np.isfinite(out)] = np.nan
    return out

def _bs_call_greeks_fast_vectorized(S: Any, K: Any, T: Any, r: float, sigma: Any) -> pd.DataFrame:
    """Fast Black-Scholes Greeks for full Round 3 datasets.

    The observed/fitted IV layer is still model-independent. This keeps the dashboard
    responsive on 300k option rows while the SABR/SVI smile comparison accounts for
    all uploaded days and strikes.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    S, K, T, sigma = np.broadcast_arrays(S, K, T, sigma)
    n = S.size
    delta = np.zeros(n, dtype=float)
    gamma = np.zeros(n, dtype=float)
    vega = np.zeros(n, dtype=float)
    theta = np.zeros(n, dtype=float)
    rho = np.zeros(n, dtype=float)
    good = (
        (S > 0) & (K > 0) & (T > 0) & (sigma > 0)
        & np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(sigma)
    )
    if good.any():
        sqrtT = np.sqrt(T[good])
        d1 = (np.log(S[good] / K[good]) + (r + 0.5 * sigma[good] ** 2) * T[good]) / (sigma[good] * sqrtT)
        d2 = d1 - sigma[good] * sqrtT
        pdf = _norm_pdf(d1)
        delta[good] = _norm_cdf(d1)
        gamma[good] = pdf / np.maximum(S[good] * sigma[good] * sqrtT, EPS)
        vega[good] = S[good] * pdf * sqrtT / 100.0
        theta[good] = (
            -(S[good] * pdf * sigma[good]) / (2.0 * sqrtT)
            - r * K[good] * np.exp(-r * T[good]) * _norm_cdf(d2)
        ) / 365.0
        rho[good] = K[good] * T[good] * np.exp(-r * T[good]) * _norm_cdf(d2) / 100.0
    return pd.DataFrame({"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho})


def implied_vol_bs_call_vectorized(price: Any, S: Any, K: Any, T: Any, r: float = 0.0, iterations: int = 28) -> np.ndarray:
    """Model-independent observed IV: invert Black-Scholes from market mid prices."""
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
        val = _bs_call_price_vectorized(Sg, Kg, Tg, r, mid)
        hi = np.where(val > pg, mid, hi)
        lo = np.where(val <= pg, mid, lo)
    iv[good] = 0.5 * (lo + hi)
    return iv


def _heston_cf(u: np.ndarray, S: np.ndarray, T: np.ndarray, r: float, kappa: np.ndarray, theta: np.ndarray, vol_of_vol: np.ndarray, rho: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """Heston characteristic function, vectorized over rows x integration grid."""
    i = 1j
    u = np.asarray(u, dtype=complex)[None, :]
    S = np.asarray(S, dtype=float)[:, None]
    T = np.asarray(T, dtype=float)[:, None]
    kappa = np.asarray(kappa, dtype=float)[:, None]
    theta = np.asarray(theta, dtype=float)[:, None]
    vol_of_vol = np.asarray(vol_of_vol, dtype=float)[:, None]
    rho = np.asarray(rho, dtype=float)[:, None]
    v0 = np.asarray(v0, dtype=float)[:, None]

    vol2 = np.maximum(vol_of_vol ** 2, 1e-10)
    a = kappa * theta
    b = kappa - rho * vol_of_vol * i * u
    d = np.sqrt(b * b + vol2 * (u * u + i * u))
    g = (b - d) / np.where(np.abs(b + d) > EPS, b + d, EPS)
    exp_neg_dT = np.exp(-d * T)
    one_minus_gexp = 1.0 - g * exp_neg_dT
    one_minus_g = 1.0 - g
    log_term = np.log(one_minus_gexp / np.where(np.abs(one_minus_g) > EPS, one_minus_g, EPS))
    C = r * i * u * T + (a / vol2) * ((b - d) * T - 2.0 * log_term)
    D = ((b - d) / vol2) * ((1.0 - exp_neg_dT) / np.where(np.abs(one_minus_gexp) > EPS, one_minus_gexp, EPS))
    return np.exp(C + D * v0 + i * u * np.log(np.maximum(S, EPS)))


def heston_call_price_vectorized(
    S: Any,
    K: Any,
    T: Any,
    r: float,
    kappa: Any = HESTON_DEFAULT_KAPPA,
    theta: Any = HESTON_DEFAULT_THETA,
    vol_of_vol: Any = HESTON_DEFAULT_VOL_OF_VOL,
    rho: Any = HESTON_DEFAULT_RHO,
    v0: Any = HESTON_DEFAULT_THETA,
    integration_limit: float = HESTON_INTEGRATION_LIMIT,
    integration_steps: int = HESTON_INTEGRATION_STEPS,
) -> np.ndarray:
    """European call price under Heston using semi-closed-form Fourier integration."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    n = len(np.atleast_1d(S))
    S, K, T = np.broadcast_arrays(S, K, T)
    kappa = np.broadcast_to(np.asarray(kappa, dtype=float), S.shape).copy()
    theta = np.broadcast_to(np.asarray(theta, dtype=float), S.shape).copy()
    vol_of_vol = np.broadcast_to(np.asarray(vol_of_vol, dtype=float), S.shape).copy()
    rho = np.broadcast_to(np.asarray(rho, dtype=float), S.shape).copy()
    v0 = np.broadcast_to(np.asarray(v0, dtype=float), S.shape).copy()

    out = np.maximum(S - K * np.exp(-r * np.maximum(T, 0.0)), 0.0).astype(float)
    good = (
        (S > 0) & (K > 0) & (T > 0)
        & np.isfinite(S) & np.isfinite(K) & np.isfinite(T)
        & np.isfinite(kappa) & np.isfinite(theta) & np.isfinite(vol_of_vol) & np.isfinite(rho) & np.isfinite(v0)
    )
    if not good.any():
        out[~np.isfinite(out)] = np.nan
        return out

    # If vol-of-vol is essentially zero, Heston collapses toward BS with variance v0.
    near_bs = good & (np.abs(vol_of_vol) < 1e-5)
    if near_bs.any():
        out[near_bs] = _bs_call_price_vectorized(S[near_bs], K[near_bs], T[near_bs], r, np.sqrt(np.maximum(v0[near_bs], 1e-8)))
        good = good & ~near_bs
    if not good.any():
        return np.clip(out, 0.0, S)

    u_grid = np.linspace(1e-5, float(integration_limit), int(max(16, integration_steps)))
    idx_all = np.flatnonzero(good)
    for start in range(0, len(idx_all), HESTON_CHUNK_SIZE):
        idx = idx_all[start:start + HESTON_CHUNK_SIZE]
        Sg, Kg, Tg = S[idx], K[idx], T[idx]
        kappag = np.maximum(kappa[idx], 1e-5)
        thetag = np.maximum(theta[idx], 1e-8)
        volvolg = np.maximum(vol_of_vol[idx], 1e-5)
        rhog = np.clip(rho[idx], -0.995, 0.995)
        v0g = np.maximum(v0[idx], 1e-8)

        phi_u = _heston_cf(u_grid, Sg, Tg, r, kappag, thetag, volvolg, rhog, v0g)
        phi_um_i = _heston_cf(u_grid - 1j, Sg, Tg, r, kappag, thetag, volvolg, rhog, v0g)
        phi_minus_i = _heston_cf(np.array([-1j]), Sg, Tg, r, kappag, thetag, volvolg, rhog, v0g)[:, 0]
        denominator = 1j * u_grid[None, :]
        discount_strike = np.exp(-1j * u_grid[None, :] * np.log(np.maximum(Kg, EPS))[:, None])
        integrand_p1 = np.real(discount_strike * phi_um_i / np.where(np.abs(phi_minus_i[:, None]) > EPS, phi_minus_i[:, None], EPS) / denominator)
        integrand_p2 = np.real(discount_strike * phi_u / denominator)
        p1 = 0.5 + (1.0 / math.pi) * _np_trapz(integrand_p1, u_grid, axis=1)
        p2 = 0.5 + (1.0 / math.pi) * _np_trapz(integrand_p2, u_grid, axis=1)
        vals = Sg * p1 - Kg * np.exp(-r * Tg) * p2
        out[idx] = np.clip(vals, 0.0, Sg)

    out[~np.isfinite(out)] = np.nan
    return out


def heston_call_greeks_vectorized(S: Any, K: Any, T: Any, r: float, kappa: Any, theta: Any, vol_of_vol: Any, rho: Any, v0: Any) -> pd.DataFrame:
    """Heston Greeks via stable finite differences around the Heston fair value."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    theta = np.asarray(theta, dtype=float)
    vol_of_vol = np.asarray(vol_of_vol, dtype=float)
    rho = np.asarray(rho, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    p_mid = heston_call_price_vectorized(S, K, T, r, kappa, theta, vol_of_vol, rho, v0)
    bump_S = np.maximum(1.0, 0.001 * np.maximum(np.abs(S), 1.0))
    p_up = heston_call_price_vectorized(S + bump_S, K, T, r, kappa, theta, vol_of_vol, rho, v0)
    p_dn = heston_call_price_vectorized(np.maximum(S - bump_S, EPS), K, T, r, kappa, theta, vol_of_vol, rho, v0)
    delta = _safe_div(p_up - p_dn, 2.0 * bump_S)
    gamma = _safe_div(p_up - 2.0 * p_mid + p_dn, bump_S ** 2)

    # Heston vega is sensitivity to initial volatility, not Black-Scholes flat-vol vega.
    bump_v = 0.0001
    p_v_up = heston_call_price_vectorized(S, K, T, r, kappa, theta, vol_of_vol, rho, np.maximum(v0 + bump_v, 1e-8))
    p_v_dn = heston_call_price_vectorized(S, K, T, r, kappa, theta, vol_of_vol, rho, np.maximum(v0 - bump_v, 1e-8))
    vega = (p_v_up - p_v_dn) / (2.0 * bump_v * 100.0)

    dt_year = 1.0 / 365.0
    p_later = heston_call_price_vectorized(S, K, np.maximum(T - dt_year, 1e-12), r, kappa, theta, vol_of_vol, rho, v0)
    theta_greek = p_later - p_mid

    bump_r = 0.0001
    p_r_up = heston_call_price_vectorized(S, K, T, r + bump_r, kappa, theta, vol_of_vol, rho, v0)
    p_r_dn = heston_call_price_vectorized(S, K, T, r - bump_r, kappa, theta, vol_of_vol, rho, v0)
    rho_greek = (p_r_up - p_r_dn) / (2.0 * bump_r * 100.0)

    for arr in [delta, gamma, vega, theta_greek, rho_greek]:
        arr[~np.isfinite(arr)] = 0.0
    return pd.DataFrame({"delta": delta, "gamma": gamma, "vega": vega, "theta": theta_greek, "rho": rho_greek})


def _add_heston_parameters(options: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.DataFrame:
    """Infer fast Heston parameters from the observed smile at each day/timestamp.

    This is intentionally a fast estimator rather than a heavy nonlinear calibration at
    every tick. It preserves the main Heston advantages for this dashboard: stochastic
    variance, skew/smile curvature, and asset-vol correlation through rho.
    """
    out = options.copy()
    if out.empty:
        return out
    for col in ["heston_kappa", "heston_theta", "heston_vol_of_vol", "heston_rho", "heston_v0"]:
        out[col] = np.nan

    time_keys = _time_key_cols(out)
    if not time_keys:
        time_keys = ["timestamp"] if "timestamp" in out.columns else []
    if not time_keys:
        atm_iv = pd.to_numeric(out.get("smile_iv", out.get("observed_iv", np.nan)), errors="coerce").median()
        var = float(np.clip((atm_iv if np.isfinite(atm_iv) else 0.2) ** 2, 1e-6, 4.0))
        out["heston_kappa"] = HESTON_DEFAULT_KAPPA
        out["heston_theta"] = var
        out["heston_vol_of_vol"] = HESTON_DEFAULT_VOL_OF_VOL
        out["heston_rho"] = HESTON_DEFAULT_RHO
        out["heston_v0"] = var
        return out

    for _, idx in out.groupby(time_keys, sort=False, dropna=False).indices.items():
        idx = np.asarray(idx, dtype=int)
        sub = out.iloc[idx]
        x = pd.to_numeric(sub.get("log_moneyness", np.nan), errors="coerce").to_numpy(dtype=float)
        iv_source = sub["smile_iv"] if "smile_iv" in sub.columns else sub.get("observed_iv", np.nan)
        y = pd.to_numeric(iv_source, errors="coerce").to_numpy(dtype=float)
        obs = pd.to_numeric(sub.get("observed_iv", np.nan), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y) & (y > 0.01) & (y < 5.0)
        if valid.sum() < 2:
            y_valid = obs[np.isfinite(obs) & (obs > 0.01) & (obs < 5.0)]
            atm_iv = float(np.nanmedian(y_valid)) if len(y_valid) else 0.20
            slope = 0.0
            curvature = 0.0
        else:
            xv, yv = x[valid], y[valid]
            atm_idx = int(np.nanargmin(np.abs(xv)))
            atm_iv = float(np.clip(yv[atm_idx], 0.01, 2.0))
            if len(xv) >= 3:
                try:
                    a, b, _c = np.polyfit(xv, yv, 2)
                    slope = float(b)
                    curvature = float(a)
                except Exception:
                    slope = float(np.polyfit(xv, yv, 1)[0]) if len(xv) >= 2 else 0.0
                    curvature = 0.0
            else:
                slope = float(np.polyfit(xv, yv, 1)[0]) if len(xv) >= 2 else 0.0
                curvature = 0.0

        if "rv_long_ann_proxy" in sub.columns:
            rv = pd.to_numeric(sub["rv_long_ann_proxy"], errors="coerce").to_numpy(dtype=float)
            rv_finite = rv[np.isfinite(rv)]
            rv_med = float(np.nanmedian(rv_finite)) if len(rv_finite) else atm_iv
        else:
            rv_med = atm_iv
        v0 = float(np.clip(atm_iv ** 2, 1e-6, 4.0))
        theta = float(np.clip((0.65 * atm_iv + 0.35 * rv_med) ** 2, 1e-6, 4.0))
        # Slope maps to asset-vol correlation: negative slope implies negative rho/skew.
        rho = float(np.clip(3.0 * slope, -0.90, 0.90))
        # Curvature and slope increase vol-of-vol. Keep it bounded for numerical stability.
        vol_of_vol = float(np.clip(0.25 + 1.75 * abs(curvature) + 0.75 * abs(slope), 0.05, 2.50))
        kappa = float(np.clip(2.0 + 4.0 * abs(curvature), 0.25, 12.0))

        out.loc[out.index[idx], "heston_kappa"] = kappa
        out.loc[out.index[idx], "heston_theta"] = theta
        out.loc[out.index[idx], "heston_vol_of_vol"] = vol_of_vol
        out.loc[out.index[idx], "heston_rho"] = rho
        out.loc[out.index[idx], "heston_v0"] = v0

    return out



# =============================================================================
# SABR daily refit and SVI/quadratic comparison layer
# =============================================================================

SABR_BETA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
SABR_RHO_GRID = tuple(np.linspace(-0.90, 0.90, 13))
SABR_NU_GRID = tuple(np.geomspace(0.05, 3.00, 16))
SABR_ALPHA_SCALE_GRID = (0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50)
SABR_WING_ABS_MONEYNESS_CUTOFF = 0.015
SABR_MIN_IMPROVEMENT = 0.0005


def hagan_sabr_iv_vectorized(F: Any, K: Any, T: Any, alpha: Any, beta: Any, rho: Any, nu: Any) -> np.ndarray:
    """Hagan lognormal SABR implied-vol approximation, dependency-free."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    F, K, T = np.broadcast_arrays(F, K, T)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), F.shape).copy()
    beta = np.broadcast_to(np.asarray(beta, dtype=float), F.shape).copy()
    rho = np.broadcast_to(np.asarray(rho, dtype=float), F.shape).copy()
    nu = np.broadcast_to(np.asarray(nu, dtype=float), F.shape).copy()
    out = np.full(F.shape, np.nan, dtype=float)
    good = (F > 0) & (K > 0) & (T >= 0) & np.isfinite(F) & np.isfinite(K) & np.isfinite(T) & np.isfinite(alpha) & np.isfinite(beta) & np.isfinite(rho) & np.isfinite(nu) & (alpha > 0) & (nu >= 0)
    if not good.any():
        return out
    Fg, Kg, Tg = F[good], K[good], np.maximum(T[good], 1e-8)
    ag = np.maximum(alpha[good], 1e-8)
    bg = np.clip(beta[good], 0.0, 1.0)
    rg = np.clip(rho[good], -0.999, 0.999)
    ng = np.maximum(nu[good], 1e-8)
    omb = 1.0 - bg
    log_fk = np.log(Fg / Kg)
    fk_beta = (Fg * Kg) ** (0.5 * omb)
    log2, log4 = log_fk ** 2, log_fk ** 4
    denom = fk_beta * (1.0 + (omb ** 2 / 24.0) * log2 + (omb ** 4 / 1920.0) * log4)
    base = ag / np.maximum(denom, EPS)
    z = (ng / ag) * fk_beta * log_fk
    sqrt_term = np.sqrt(np.maximum(1.0 - 2.0 * rg * z + z * z, EPS))
    x_z = np.log(np.maximum((sqrt_term + z - rg) / np.maximum(1.0 - rg, EPS), EPS))
    z_over_xz = np.where(np.abs(z) < 1e-7, 1.0 - 0.5 * rg * z, z / np.where(np.abs(x_z) > EPS, x_z, EPS))
    term1 = (omb ** 2 / 24.0) * ag ** 2 / np.maximum((Fg * Kg) ** omb, EPS)
    term2 = 0.25 * rg * bg * ng * ag / np.maximum((Fg * Kg) ** (0.5 * omb), EPS)
    term3 = ((2.0 - 3.0 * rg ** 2) / 24.0) * ng ** 2
    vol = base * z_over_xz * (1.0 + (term1 + term2 + term3) * Tg)
    atm = np.abs(log_fk) < 1e-7
    if atm.any():
        F_atm, b_atm, a_atm = Fg[atm], bg[atm], ag[atm]
        r_atm, n_atm, T_atm = rg[atm], ng[atm], Tg[atm]
        omb_atm = 1.0 - b_atm
        F_pow = np.maximum(F_atm ** omb_atm, EPS)
        atm_corr = 1.0 + ((omb_atm ** 2 / 24.0) * a_atm ** 2 / np.maximum(F_atm ** (2.0 * omb_atm), EPS) + 0.25 * r_atm * b_atm * n_atm * a_atm / F_pow + ((2.0 - 3.0 * r_atm ** 2) / 24.0) * n_atm ** 2) * T_atm
        vol[atm] = (a_atm / F_pow) * atm_corr
    out[good] = np.clip(vol, 1e-5, 5.0)
    return out


def _sabr_objective_for_slice(F: float, K: np.ndarray, T: np.ndarray, iv: np.ndarray, alpha: float, beta: float, rho: float, nu: float) -> float:
    pred = hagan_sabr_iv_vectorized(np.full_like(K, F, dtype=float), K, T, alpha, beta, rho, nu)
    valid = np.isfinite(pred) & np.isfinite(iv)
    if valid.sum() < 3:
        return float("inf")
    m = np.abs(np.log(np.maximum(K[valid], EPS) / max(F, EPS)))
    weights = 1.0 + 2.0 * (m / max(np.nanmax(m), EPS))
    err = pred[valid] - iv[valid]
    return float(np.average(err * err, weights=weights))


def _fit_sabr_params_for_slice(sub: pd.DataFrame) -> dict[str, float]:
    req = ["underlying_mid", "strike", "T", "observed_iv"]
    if sub.empty or any(c not in sub.columns for c in req):
        return {"alpha": np.nan, "beta": np.nan, "rho": np.nan, "nu": np.nan, "rmse": np.nan, "n": 0}
    data = sub[req + (["quoted_spread"] if "quoted_spread" in sub.columns else [])].replace([np.inf, -np.inf], np.nan).dropna(subset=req).copy()
    data = data[(data["underlying_mid"] > 0) & (data["strike"] > 0) & (data["observed_iv"] > 0.01) & (data["observed_iv"] < 3.0)]
    if "quoted_spread" in data.columns and data["quoted_spread"].notna().sum() >= 3:
        q = data["quoted_spread"].quantile(0.80)
        data = data[(data["quoted_spread"].isna()) | (data["quoted_spread"] <= q)]
    if len(data) < 3:
        return {"alpha": np.nan, "beta": np.nan, "rho": np.nan, "nu": np.nan, "rmse": np.nan, "n": int(len(data))}
    F = float(np.nanmedian(data["underlying_mid"]))
    K = data["strike"].to_numpy(dtype=float)
    T = data["T"].to_numpy(dtype=float)
    iv = data["observed_iv"].to_numpy(dtype=float)
    atm_idx = int(np.nanargmin(np.abs(np.log(K / F))))
    atm_iv = float(np.clip(iv[atm_idx], 0.01, 2.0))
    best = {"alpha": np.nan, "beta": np.nan, "rho": np.nan, "nu": np.nan, "rmse": np.inf, "n": int(len(data))}
    for beta in SABR_BETA_GRID:
        base_alpha = float(np.clip(atm_iv * (F ** (1.0 - beta)), 1e-5, 1e6))
        for alpha_scale in SABR_ALPHA_SCALE_GRID:
            alpha = base_alpha * alpha_scale
            for rho in SABR_RHO_GRID:
                for nu in SABR_NU_GRID:
                    mse = _sabr_objective_for_slice(F, K, T, iv, alpha, beta, float(rho), float(nu))
                    if mse < best["rmse"]:
                        best = {"alpha": alpha, "beta": float(beta), "rho": float(rho), "nu": float(nu), "rmse": float(math.sqrt(mse)), "n": int(len(data))}
    return best


def _representative_sabr_slice_for_day(day_df: pd.DataFrame) -> pd.DataFrame:
    if day_df.empty or "timestamp" not in day_df.columns:
        return day_df
    tmp = day_df.copy()
    tmp["timestamp"] = pd.to_numeric(tmp["timestamp"], errors="coerce")
    tmp = tmp.dropna(subset=["timestamp"])
    if tmp.empty:
        return tmp
    unique_ts = np.asarray(sorted(tmp["timestamp"].dropna().unique()), dtype=float)
    median_ts = float(np.nanmedian(unique_ts))
    candidates = sorted(unique_ts, key=lambda x: abs(float(x) - median_ts))[:25]
    best_slice, best_score = pd.DataFrame(), -1.0
    for ts in candidates:
        sub = tmp[tmp["timestamp"] == ts].copy()
        usable = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["strike", "observed_iv", "underlying_mid", "T"])
        usable = usable[(usable["observed_iv"] > 0.01) & (usable["observed_iv"] < 3.0)]
        if usable.empty:
            continue
        spread_penalty = 0.0
        if "quoted_spread" in usable.columns:
            spread_penalty = float(np.nanmedian(pd.to_numeric(usable["quoted_spread"], errors="coerce")))
            if not np.isfinite(spread_penalty):
                spread_penalty = 0.0
        score = float(usable["strike"].nunique()) - 0.01 * spread_penalty
        if score > best_score:
            best_score, best_slice = score, usable
    return best_slice if not best_slice.empty else tmp[tmp["timestamp"] == min(unique_ts, key=lambda x: abs(float(x) - median_ts))]


def _add_sabr_features(options: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.DataFrame:
    out = options.copy()
    if out.empty:
        return out
    for c in ["sabr_alpha", "sabr_beta", "sabr_rho", "sabr_nu", "sabr_fit_rmse", "sabr_fit_n", "sabr_iv", "sabr_iv_residual", "sabr_abs_residual", "svi_abs_residual", "sabr_residual_minus_svi_residual", "sabr_beats_svi", "sabr_wing_beats_svi", "preferred_smile_model", "best_structural_iv", "best_structural_iv_residual"]:
        out[c] = np.nan
    try:
        labeled = _add_display_day_labels(out)
        out["display_day"] = labeled.get("display_day", 0)
    except Exception:
        out["display_day"] = 0
    if "iv_residual" not in out.columns and {"observed_iv", "smile_iv"}.issubset(out.columns):
        out["iv_residual"] = out["observed_iv"] - out["smile_iv"]
    param_by_day = {}
    for display_day, day_df in out.groupby("display_day", sort=True, dropna=False):
        try:
            day_key = int(display_day)
        except Exception:
            day_key = 0
        param_by_day[day_key] = _fit_sabr_params_for_slice(_representative_sabr_slice_for_day(day_df))
    for display_day, params in param_by_day.items():
        mask = out["display_day"].astype("Int64") == int(display_day)
        if not mask.any():
            continue
        for col, key in [("sabr_alpha", "alpha"), ("sabr_beta", "beta"), ("sabr_rho", "rho"), ("sabr_nu", "nu"), ("sabr_fit_rmse", "rmse"), ("sabr_fit_n", "n")]:
            out.loc[mask, col] = params.get(key, np.nan)
        if np.isfinite(params.get("alpha", np.nan)):
            out.loc[mask, "sabr_iv"] = hagan_sabr_iv_vectorized(out.loc[mask, "underlying_mid"].to_numpy(dtype=float), out.loc[mask, "strike"].to_numpy(dtype=float), out.loc[mask, "T"].to_numpy(dtype=float), params["alpha"], params["beta"], params["rho"], params["nu"])
    out["sabr_iv_residual"] = pd.to_numeric(out.get("observed_iv", np.nan), errors="coerce") - pd.to_numeric(out.get("sabr_iv", np.nan), errors="coerce")
    out["sabr_abs_residual"] = np.abs(out["sabr_iv_residual"])
    out["svi_abs_residual"] = np.abs(pd.to_numeric(out.get("iv_residual", np.nan), errors="coerce"))
    out["sabr_residual_minus_svi_residual"] = out["sabr_abs_residual"] - out["svi_abs_residual"]
    out["sabr_beats_svi"] = out["sabr_abs_residual"] + SABR_MIN_IMPROVEMENT < out["svi_abs_residual"]
    m_abs = np.abs(pd.to_numeric(out.get("log_moneyness", np.nan), errors="coerce"))
    wing = m_abs >= SABR_WING_ABS_MONEYNESS_CUTOFF
    out["sabr_wing_beats_svi"] = wing & out["sabr_beats_svi"]
    out["preferred_smile_model"] = "SVI_QUADRATIC"
    out.loc[out["sabr_wing_beats_svi"], "preferred_smile_model"] = "SABR_WING"
    out["best_structural_iv"] = np.where(out["sabr_wing_beats_svi"], out["sabr_iv"], out.get("smile_iv", np.nan))
    out["best_structural_iv_residual"] = pd.to_numeric(out.get("observed_iv", np.nan), errors="coerce") - pd.to_numeric(out["best_structural_iv"], errors="coerce")
    return out


# Compatibility wrappers: callers keep working, but model fair value is Heston now.
def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return float(_bs_call_price_vectorized([S], [K], [T], r, [sigma])[0])


def bs_call_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict[str, float]:
    var = max(float(sigma) ** 2, 1e-8)
    df = heston_call_greeks_vectorized([S], [K], [T], r, [HESTON_DEFAULT_KAPPA], [var], [1e-5], [0.0], [var])
    return {k: float(df.iloc[0][k]) for k in ["delta", "gamma", "vega", "theta", "rho"]}


def implied_vol_call(price: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    return float(implied_vol_bs_call_vectorized([price], [S], [K], [T], r=r)[0])


def bs_call_price_vectorized(S: Any, K: Any, T: Any, r: float, sigma: Any) -> np.ndarray:
    return _bs_call_price_vectorized(S, K, T, r, sigma)


def implied_vol_call_vectorized(price: Any, S: Any, K: Any, T: Any, r: float = 0.0, iterations: int = 28) -> np.ndarray:
    return implied_vol_bs_call_vectorized(price, S, K, T, r=r, iterations=iterations)


def bs_call_greeks_vectorized(S: Any, K: Any, T: Any, r: float, sigma: Any) -> pd.DataFrame:
    sigma_arr = np.asarray(sigma, dtype=float)
    var = np.maximum(sigma_arr ** 2, 1e-8)
    return heston_call_greeks_vectorized(S, K, T, r, HESTON_DEFAULT_KAPPA, var, 1e-5, 0.0, var)

def _add_executable_edge_features(options: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.DataFrame:
    """Add bid/ask-aware edge columns and a contract quality filter."""
    out = options.copy()
    if out.empty:
        return out

    if {"ask_price_1", "bid_price_1"}.issubset(out.columns):
        out["quoted_spread"] = out["ask_price_1"] - out["bid_price_1"]
        out["quoted_spread_pct"] = _safe_div(out["quoted_spread"], out["mid_price"])
        out["buy_edge"] = out["fair_price"] - out["ask_price_1"]
        out["sell_edge"] = out["bid_price_1"] - out["fair_price"]
    else:
        out["quoted_spread"] = np.nan
        out["quoted_spread_pct"] = np.nan
        out["buy_edge"] = np.nan
        out["sell_edge"] = np.nan

    buy = pd.to_numeric(out["buy_edge"], errors="coerce").to_numpy(dtype=float)
    sell = pd.to_numeric(out["sell_edge"], errors="coerce").to_numpy(dtype=float)
    out["executable_edge"] = np.nanmax(np.vstack([buy, sell]), axis=0)
    out["required_edge"] = np.maximum(
        cfg.min_executable_edge,
        cfg.edge_spread_mult * pd.to_numeric(out["quoted_spread"], errors="coerce"),
    )

    delta = pd.to_numeric(out.get("delta", np.nan), errors="coerce")
    spread = pd.to_numeric(out.get("quoted_spread", np.nan), errors="coerce")
    edge = pd.to_numeric(out.get("executable_edge", np.nan), errors="coerce")
    req = pd.to_numeric(out.get("required_edge", np.nan), errors="coerce")
    out["contract_matters"] = (
        delta.between(cfg.min_tradeable_delta, cfg.max_tradeable_delta)
        & spread.between(0, cfg.max_tradeable_spread)
        & (edge > req)
    )
    out["edge_side"] = np.where(out["buy_edge"] > out["sell_edge"], "BUY_CHEAP_OPTION", "SELL_RICH_OPTION")
    out.loc[~out["contract_matters"], "edge_side"] = "IGNORE"
    out["edge_score"] = _safe_div(out["executable_edge"], np.maximum(1.0, out["quoted_spread"]))
    out["pricing_model"] = "SVI_SABR_FAST_FULL_DATA"
    return out


def _estimate_time_to_expiry(df: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.Series:
    """Time to expiry in TRADING YEARS.

    v2 fix (2025-04-24): previous implementation normalized progress across the
    *loaded sample* regardless of the real schedule, and annualized with /365
    (calendar days). Both were wrong. The Prosperity Round 3 convention is:
      - Data is day 0, 1, 2, ... with timestamps 0 .. ticks_per_day within each day.
      - Vouchers expire cfg.start_days_to_expiry days after day 0 timestamp 0.
      - Annualization uses TRADING days (cfg.trading_year, default 250).

    Override order:
      1. Explicit `T` / `time_to_expiry` / `tte` column (already in years) — used as-is.
      2. Explicit `days_to_expiry` / `expiry_days` column — divided by trading_year.
      3. Compute from `day` + `timestamp` vs cfg.start_days_to_expiry.
    """
    for col in ["time_to_expiry", "tte", "T"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").clip(lower=1e-6)
    for col in ["days_to_expiry", "expiry_days"]:
        if col in df.columns:
            raw = pd.to_numeric(df[col], errors="coerce")
            return (raw / cfg.trading_year).clip(lower=1e-6)

    if "timestamp" not in df.columns:
        return pd.Series(cfg.start_days_to_expiry / cfg.trading_year, index=df.index)

    t = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0).astype(float)
    if "day" in df.columns:
        day = pd.to_numeric(df["day"], errors="coerce").fillna(0.0).astype(float)
    else:
        day = pd.Series(0.0, index=df.index)

    # Normalize day numbering so the earliest day in the upload is treated as day 0.
    # Handles raw Prosperity days -1/0/1 uploads the same way display_day does.
    if len(day):
        day = day - day.min()

    days_elapsed = day + t / float(cfg.ticks_per_day)
    days_left    = (cfg.start_days_to_expiry - days_elapsed).clip(lower=1e-6)
    return days_left / float(cfg.trading_year)


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

    # Observed IV is model-independent: invert Black-Scholes from market mids.
    # The theoretical fair-value layer below is Heston, not binomial.
    options["observed_iv"] = implied_vol_bs_call_vectorized(
        options["mid_price"].to_numpy(),
        options["underlying_mid"].to_numpy(),
        options["strike"].to_numpy(),
        options["T"].to_numpy(),
        cfg.risk_free_rate,
    )

    options = _fit_smile_by_timestamp(options)
    options = _add_sabr_features(options, cfg)

    # Keep Heston parameter columns available for research tables, but do not
    # estimate them per timestamp during the dashboard build. The uploaded Round 3
    # files produce roughly 30k timestamp slices, and per-slice Heston estimation
    # is unnecessary for the first structure plot and SABR/SVI comparison.
    options["heston_kappa"] = HESTON_DEFAULT_KAPPA
    options["heston_theta"] = np.square(pd.to_numeric(options.get("best_structural_iv", options.get("smile_iv", 0.20)), errors="coerce").fillna(0.20))
    options["heston_vol_of_vol"] = HESTON_DEFAULT_VOL_OF_VOL
    options["heston_rho"] = HESTON_DEFAULT_RHO
    options["heston_v0"] = options["heston_theta"]

    # Full Round 3 uploads contain ~300k option rows. A Heston FFT fair-price and
    # finite-difference Greek pass over every row is too slow for the dashboard build.
    # For the production dashboard, use the fast structural IV layer: SABR where it
    # improves the wing fit, otherwise the quadratic/SVI-style smile. The first
    # structure plot and residual diagnostics still account for every uploaded day,
    # timestamp, strike, and voucher row.
    structural_iv = options.get("best_structural_iv", options.get("smile_iv", options.get("observed_iv", np.nan)))
    structural_iv = pd.to_numeric(structural_iv, errors="coerce")
    structural_iv = structural_iv.where(structural_iv.notna(), pd.to_numeric(options.get("smile_iv", np.nan), errors="coerce"))
    structural_iv = structural_iv.where(structural_iv.notna(), pd.to_numeric(options.get("observed_iv", np.nan), errors="coerce"))
    options["fair_price"] = _bs_call_price_vectorized(
        options["underlying_mid"].to_numpy(),
        options["strike"].to_numpy(),
        options["T"].to_numpy(),
        cfg.risk_free_rate,
        structural_iv.to_numpy(),
    )
    options["mispricing"] = options["mid_price"] - options["fair_price"]
    options["normalized_mispricing"] = _safe_div(options["mispricing"], options["fair_price"])
    options["iv_residual"] = options["observed_iv"] - options["smile_iv"]

    greeks_df = _bs_call_greeks_fast_vectorized(
        options["underlying_mid"].to_numpy(),
        options["strike"].to_numpy(),
        options["T"].to_numpy(),
        cfg.risk_free_rate,
        structural_iv.to_numpy(),
    )
    greeks_df.index = options.index
    options = pd.concat([options, greeks_df], axis=1)
    options = _add_executable_edge_features(options, cfg)

    options = _add_rolling_stats(options, cfg)
    options = _add_gamma_scalping_ev(options, cfg)
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
        "observed_iv", "smile_iv", "smile_fit_ok", "sabr_iv", "sabr_iv_residual", "sabr_abs_residual",
        "svi_abs_residual", "sabr_residual_minus_svi_residual", "sabr_beats_svi", "sabr_wing_beats_svi",
        "preferred_smile_model", "best_structural_iv", "best_structural_iv_residual",
        "sabr_alpha", "sabr_beta", "sabr_rho", "sabr_nu", "sabr_fit_rmse", "sabr_fit_n",
        "display_day", "heston_kappa", "heston_theta",
        "heston_vol_of_vol", "heston_rho", "heston_v0", "fair_price", "mispricing",
        "normalized_mispricing", "pricing_model",
        "buy_edge", "sell_edge", "executable_edge", "required_edge", "edge_score", "edge_side", "contract_matters",
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

    x_all   = out["log_moneyness"].to_numpy(dtype=float)
    y_all   = out["observed_iv"].to_numpy(dtype=float)
    extr_all = out["extrinsic"].to_numpy(dtype=float)
    mid_all  = out["mid_price"].to_numpy(dtype=float) if "mid_price" in out.columns else np.full(n, np.nan)

    smile_iv = np.full(n, np.nan, dtype=float)
    smile_ok = np.zeros(n, dtype=bool)
    smile_a  = np.full(n, np.nan, dtype=float)
    smile_b  = np.full(n, np.nan, dtype=float)
    smile_c  = np.full(n, np.nan, dtype=float)

    # v2 fix (2025-04-24): tighter wing exclusion. The old `extrinsic > 0.25`
    # absolute threshold was too lenient for thin-priced wings (VEV_4000/4500
    # near intrinsic, VEV_6500 stuck at 0.5 floor) and let them distort the
    # parabolic fit — producing the −48% IV residual we saw on day 0.
    # New rules:
    #   - mid price >= 2.0    (avoid 0.5/1.0 floor-rounded prices)
    #   - extrinsic / mid >= 0.05  (>=5% time value; excludes deep-ITM dominated by intrinsic)
    #   - 0.05 <= IV <= 1.5   (sensible range for 5-day options at this vol level)
    MIN_MID            = 2.0
    MIN_EXTR_FRACTION  = 0.05
    IV_LOW             = 0.05
    IV_HIGH            = 1.5

    for _, idx in out.groupby(time_keys, sort=False, dropna=False).indices.items():
        idx = np.asarray(idx, dtype=int)
        ratio = np.where(mid_all[idx] > 0, extr_all[idx] / np.maximum(mid_all[idx], 1e-9), 0.0)
        valid = (
            np.isfinite(x_all[idx])
            & np.isfinite(y_all[idx])
            & (y_all[idx] >= IV_LOW)
            & (y_all[idx] <= IV_HIGH)
            & np.isfinite(extr_all[idx])
            & np.isfinite(mid_all[idx])
            & (mid_all[idx] >= MIN_MID)
            & (ratio >= MIN_EXTR_FRACTION)
        )
        good_idx = idx[valid]
        if len(good_idx) >= 3:
            try:
                coef = np.polyfit(x_all[good_idx], y_all[good_idx], deg=2)
                # Only assign smile_iv to strikes USED in the fit. Extrapolating
                # to excluded wings (deep ITM/OTM) produces nonsense (e.g. the
                # earlier dashboard's -48% IV residual on VEV_4000). Wings stay
                # at NaN, so their iv_residual is also NaN — they don't appear
                # on the heatmap or in cross-sectional signals.
                smile_iv[good_idx] = np.polyval(coef, x_all[good_idx])
                smile_ok[good_idx] = True
                # Per-timestamp coefficients still recorded for ALL rows in
                # this timestamp so callers can do their own extrapolation if
                # they really want to.
                smile_a[idx], smile_b[idx], smile_c[idx] = coef
                continue
            except Exception:
                pass

        fallback_vals = y_all[good_idx] if len(good_idx) else y_all[idx][np.isfinite(y_all[idx])]
        fallback = float(np.nanmedian(fallback_vals)) if len(fallback_vals) else np.nan
        smile_iv[good_idx if len(good_idx) else idx] = fallback

    out["smile_iv"]     = pd.Series(smile_iv).clip(lower=1e-5, upper=5.0)
    out["smile_fit_ok"] = smile_ok
    out["smile_a"]      = smile_a
    out["smile_b"]      = smile_b
    out["smile_c"]      = smile_c
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


def _add_gamma_scalping_ev(options: pd.DataFrame, cfg: OptionsStatsConfig = None) -> pd.DataFrame:
    """Per-tick gamma-scalping PnL proxy.

    v2 fix (2025-04-24): theta from bs_call_greeks is in PER-CALENDAR-DAY units
    (the formula divides by 365). v1 stored that raw value on every per-tick row
    and the chart aggregated it with sum(), counting every tick as a full day —
    inflating theta cost by ~10,000x. Now we scale by dt_days so theta_cost_est
    is the *per-tick* time decay, comparable to gamma_pnl_est which is also
    per-tick.
    """
    out = options.sort_values([c for c in ["product", "day", "timestamp"] if c in options.columns]).copy()
    out["underlying_change"] = out.groupby("product")["underlying_mid"].diff()
    out["gamma_pnl_est"] = 0.5 * out["gamma"] * (out["underlying_change"] ** 2)

    # Detect typical sample interval to scale daily theta -> per-tick theta.
    if "timestamp" in out.columns:
        ts_diff = pd.to_numeric(out["timestamp"], errors="coerce").diff().dropna()
        # within-day strides only (skip the giant gap at day boundaries)
        ticks_per_day = float(cfg.ticks_per_day) if cfg is not None else float(TICKS_PER_DAY)
        ts_diff = ts_diff[(ts_diff > 0) & (ts_diff < ticks_per_day / 2.0)]
        sample_interval = float(ts_diff.median()) if len(ts_diff) else float(DEFAULT_SAMPLE_INTERVAL)
    else:
        ticks_per_day   = float(TICKS_PER_DAY)
        sample_interval = float(DEFAULT_SAMPLE_INTERVAL)
    dt_days = sample_interval / max(ticks_per_day, 1.0)

    # bs_call_greeks already returned theta in /365 calendar-day units. Convert
    # /365 -> /trading_year first (the rest of the dashboard is on /trading_year),
    # then scale to per-tick.
    trading_year = float(cfg.trading_year) if cfg is not None else float(TRADING_YEAR)
    theta_per_tick = out["theta"] * (365.0 / trading_year) * dt_days

    out["theta_cost_est"] = theta_per_tick
    out["net_gamma_scalp_ev"] = out["gamma_pnl_est"] + out["theta_cost_est"].fillna(0.0)
    out["gamma_theta_ratio"]  = _safe_div(out["gamma"], np.abs(out["theta"]))
    return out


def _add_signals(options: pd.DataFrame, cfg: OptionsStatsConfig) -> pd.DataFrame:
    out = options.copy()
    # v2 fix: the old version had a tautology (both branches of the ternary
    # returned out["mispricing_z"]), which raised KeyError if the column was
    # absent. Fall back to zeros when the z-score hasn't been computed.
    if "mispricing_z" in out.columns:
        z = pd.to_numeric(out["mispricing_z"], errors="coerce")
    else:
        z = pd.Series(0.0, index=out.index)
    out["signal"] = "HOLD"
    out.loc[z <= -cfg.signal_entry_z, "signal"] = "BUY_CHEAP_OPTION"
    out.loc[z >=  cfg.signal_entry_z, "signal"] = "SELL_RICH_OPTION"
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
        "sabr_iv",
        "sabr_iv_residual",
        "sabr_abs_residual",
        "svi_abs_residual",
        "sabr_residual_minus_svi_residual",
        "sabr_beats_svi",
        "sabr_wing_beats_svi",
        "preferred_smile_model",
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
    if "sabr_iv" not in out.columns:
        out["sabr_iv"] = np.nan
    if "sabr_iv_residual" not in out.columns and {"observed_iv", "sabr_iv"}.issubset(out.columns):
        out["sabr_iv_residual"] = pd.to_numeric(out["observed_iv"], errors="coerce") - pd.to_numeric(out["sabr_iv"], errors="coerce")
    if "svi_abs_residual" not in out.columns:
        out["svi_abs_residual"] = np.abs(pd.to_numeric(out.get("iv_residual", np.nan), errors="coerce"))
    if "sabr_abs_residual" not in out.columns:
        out["sabr_abs_residual"] = np.abs(pd.to_numeric(out.get("sabr_iv_residual", np.nan), errors="coerce"))
    if "sabr_residual_minus_svi_residual" not in out.columns:
        out["sabr_residual_minus_svi_residual"] = out["sabr_abs_residual"] - out["svi_abs_residual"]
    if "preferred_smile_model" not in out.columns:
        out["preferred_smile_model"] = "SVI_QUADRATIC"
    return out


def fig_vol_smile(options: pd.DataFrame, timestamp: Optional[float] = None) -> go.Figure:
    """Simple first plot: implied volatility versus moneyness with a fitted parabola.

    This is intentionally broader than a single-timestamp smile. It pools usable
    option observations across the loaded sample so the user can see the overall
    smile structure, while filtering out historical points whose extrinsic value
    is too low to produce reliable implied vol estimates.
    """
    options = _repair_options_for_plotting(options)
    if options.empty:
        return _empty_fig("Implied Volatility vs Moneyness", "No options data available. Click Build / Parse Options Data first.")

    df = options.copy()
    if "display_day" not in df.columns:
        df = _add_display_day_labels(df)

    if "moneyness" not in df.columns and {"underlying_mid", "strike"}.issubset(df.columns):
        # User-facing moneyness axis: m_t = log(K / S).
        df["moneyness"] = np.log(_safe_div(pd.to_numeric(df["strike"], errors="coerce"), pd.to_numeric(df["underlying_mid"], errors="coerce")))
    elif "log_moneyness" in df.columns:
        # Existing column is log(S / K); flip sign so the axis reads log(K / S).
        df["moneyness"] = -pd.to_numeric(df["log_moneyness"], errors="coerce")

    if "extrinsic" not in df.columns and {"mid_price", "underlying_mid", "strike"}.issubset(df.columns):
        intrinsic = np.maximum(pd.to_numeric(df["underlying_mid"], errors="coerce") - pd.to_numeric(df["strike"], errors="coerce"), 0.0)
        df["extrinsic"] = pd.to_numeric(df["mid_price"], errors="coerce") - intrinsic

    required = {"moneyness", "observed_iv"}
    missing = sorted(required - set(df.columns))
    if missing:
        return _empty_fig("Implied Volatility vs Moneyness", f"Missing required columns after repair: {missing}")

    df["moneyness"] = pd.to_numeric(df["moneyness"], errors="coerce")
    df["observed_iv"] = pd.to_numeric(df["observed_iv"], errors="coerce")
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    if "mid_price" in df.columns:
        df["mid_price"] = pd.to_numeric(df["mid_price"], errors="coerce")
    if "extrinsic" in df.columns:
        df["extrinsic"] = pd.to_numeric(df["extrinsic"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["moneyness", "observed_iv"])
    if df.empty:
        return _empty_fig("Implied Volatility vs Moneyness", "No usable IV data found")

    # Exclude bottom-left historical points where extrinsic value is too low.
    mid = pd.to_numeric(df.get("mid_price", np.nan), errors="coerce")
    extr = pd.to_numeric(df.get("extrinsic", np.nan), errors="coerce")
    ratio = np.where(mid > 0, extr / np.maximum(mid, 1e-9), np.nan)
    low_extrinsic = (
        np.isfinite(mid)
        & np.isfinite(extr)
        & ((mid < 2.0) | (ratio < 0.05) | (extr <= 0.25))
    )
    df["fit_inlier"] = ~low_extrinsic & df["observed_iv"].between(0.05, 1.5)

    inliers = df[df["fit_inlier"]].copy()
    if len(inliers) < 3:
        inliers = df.copy()
        inliers["fit_inlier"] = True

    coef = None
    if len(inliers) >= 3:
        try:
            coef = np.polyfit(inliers["moneyness"].to_numpy(dtype=float), inliers["observed_iv"].to_numpy(dtype=float), deg=2)
            df["vhat"] = np.polyval(coef, df["moneyness"].to_numpy(dtype=float))
        except Exception:
            coef = None
            df["vhat"] = np.nan
    else:
        df["vhat"] = np.nan

    # The fit above uses ALL usable uploaded observations. For rendering, cap the
    # visible cloud deterministically so the browser does not choke on ~300k points.
    # We sample within displayed day x strike x inlier/outlier groups, preserving the
    # shape across every voucher and timeframe.
    fit_n_total = int(len(df))
    fit_n_inliers = int(df["fit_inlier"].sum())
    max_plot_points = 60000
    if len(df) > max_plot_points:
        sample_parts = []
        group_cols = [c for c in ["display_day", "strike", "fit_inlier"] if c in df.columns]
        if group_cols:
            groups = list(df.groupby(group_cols, sort=True, dropna=False))
            per_group = max(50, int(math.ceil(max_plot_points / max(1, len(groups)))))
            for _, g in groups:
                if len(g) > per_group:
                    sample_parts.append(g.sample(n=per_group, random_state=7))
                else:
                    sample_parts.append(g)
            plot_df = pd.concat(sample_parts, ignore_index=False) if sample_parts else df
            if len(plot_df) > max_plot_points:
                plot_df = plot_df.sample(n=max_plot_points, random_state=7)
        else:
            plot_df = df.sample(n=max_plot_points, random_state=7)
    else:
        plot_df = df

    fig = go.Figure()

    if (~plot_df["fit_inlier"]).any():
        excluded = plot_df[~plot_df["fit_inlier"]].copy()
        fig.add_trace(go.Scattergl(
            x=excluded["moneyness"],
            y=excluded["observed_iv"],
            mode="markers",
            name="Excluded low-extrinsic outliers",
            marker=dict(symbol="x", size=8, color="rgba(140,140,140,0.7)"),
            customdata=np.column_stack([
                excluded.get("product", pd.Series([""] * len(excluded))).astype(str),
                excluded.get("strike", pd.Series([np.nan] * len(excluded))),
                excluded.get("display_day", pd.Series([np.nan] * len(excluded))),
                excluded.get("timestamp", pd.Series([np.nan] * len(excluded))),
                excluded.get("extrinsic", pd.Series([np.nan] * len(excluded))),
            ]),
            hovertemplate="Excluded point<br>Product %{customdata[0]}<br>Strike %{customdata[1]}<br>Day %{customdata[2]}<br>Timestamp %{customdata[3]}<br>Extrinsic %{customdata[4]:.3f}<br>m_t %{x:.4f}<br>v_t %{y:.4f}<extra></extra>",
        ))

    color_series = plot_df["display_day"].astype(str) if "display_day" in plot_df.columns else plot_df.get("product", pd.Series(["all"] * len(plot_df))).astype(str)
    for color_value in sorted(color_series.dropna().unique().tolist()):
        sub = plot_df[color_series == color_value].copy()
        label = f"Day {color_value}" if "display_day" in df.columns else str(color_value)
        fig.add_trace(go.Scattergl(
            x=sub["moneyness"],
            y=sub["observed_iv"],
            mode="markers",
            name=label,
            opacity=0.65,
            marker=dict(size=8),
            customdata=np.column_stack([
                sub.get("product", pd.Series([""] * len(sub))).astype(str),
                sub.get("strike", pd.Series([np.nan] * len(sub))),
                sub.get("timestamp", pd.Series([np.nan] * len(sub))),
                sub.get("vhat", pd.Series([np.nan] * len(sub))),
                sub.get("extrinsic", pd.Series([np.nan] * len(sub))),
                sub["fit_inlier"].astype(str),
            ]),
            hovertemplate="Product %{customdata[0]}<br>Strike %{customdata[1]}<br>Timestamp %{customdata[2]}<br>m_t %{x:.4f}<br>v_t %{y:.4f}<br>v̂_t %{customdata[3]:.4f}<br>Extrinsic %{customdata[4]:.3f}<br>Used in fit %{customdata[5]}<extra></extra>",
        ))

    if coef is not None:
        x_grid = np.linspace(float(np.nanmin(inliers["moneyness"])), float(np.nanmax(inliers["moneyness"])), 250)
        y_grid = np.polyval(coef, x_grid)
        fig.add_trace(go.Scatter(
            x=x_grid,
            y=y_grid,
            mode="lines",
            name="Fitted parabola v̂_t",
            line=dict(width=3, dash="dash", color="black"),
            hovertemplate="m_t %{x:.4f}<br>v̂_t %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title="Implied Volatility v_t versus Moneyness m_t",
        xaxis_title="Moneyness m_t = log(K / S)",
        yaxis_title="Implied volatility v_t",
        template="plotly_white",
        height=520,
        legend_title_text="Timeframe",
        annotations=[
            dict(
                text=f"Dashed parabola = fitted fair IV v̂_t. Fit uses all {fit_n_total:,} usable observations ({fit_n_inliers:,} inliers); visible cloud capped at {len(plot_df):,} points for speed.",
                x=0.5,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=13),
            )
        ],
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
    fig.update_layout(title="SVI/SABR Structural-IV Fair Price vs Market", template="plotly_white", height=650)
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

        # v2 fix (2025-04-24): proper annualization in TRADING-YEAR units.
        # Per-sample std → per-year std requires sqrt(samples_per_year), where
        #   samples_per_year = trading_year × (ticks_per_day / sample_interval).
        # The old code used sqrt(365), under-annualizing by ~83× for Round 3 data.
        ts_diff = pd.to_numeric(u["timestamp"], errors="coerce").diff().dropna()
        # Prefer the *within-day* typical stride — diff at a day boundary can be huge.
        ts_diff = ts_diff[(ts_diff > 0) & (ts_diff < cfg.ticks_per_day / 2)]
        sample_interval = float(ts_diff.median()) if len(ts_diff) else float(DEFAULT_SAMPLE_INTERVAL)
        samples_per_year = cfg.trading_year * (cfg.ticks_per_day / max(sample_interval, 1.0))
        ann_factor = float(np.sqrt(samples_per_year))
        u["rv_short_ann_proxy"] = u["rv_short"] * ann_factor
        u["rv_long_ann_proxy"]  = u["rv_long"]  * ann_factor
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
    # XW residuals are expressed in IV space; convert IV residual into a price residual
    # with Black-Scholes purely as a display transform, not as the fair-value model.
    price_obs = _bs_call_price_vectorized(out["underlying_mid"], out["strike"], out["T"], 0.0, out["observed_iv"])
    price_fit = _bs_call_price_vectorized(out["underlying_mid"], out["strike"], out["T"], 0.0, out["xw_fitted_iv"])
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
    cols = ["timestamp", "product", "strike", "observed_iv", "smile_iv", "iv_residual", "sabr_iv", "sabr_iv_residual", "sabr_residual_minus_svi_residual", "sabr_beats_svi", "sabr_wing_beats_svi", "preferred_smile_model", "sabr_alpha", "sabr_beta", "sabr_rho", "sabr_nu", "sabr_fit_rmse", "rv_long_ann_proxy", "realized_vol_long", "rv_iv_spread", "rv_iv_spread_decile", "vrp", "vrp_z", "vrp_decile", "vrp_residual_combo", "vrp_residual_combo_z", "vrp_residual_combo_decile", "xw_fitted_iv", "xw_iv_mispricing", "xw_iv_mispricing_decile", "xw_price_mispricing", "rv_trend", "quoted_spread", "book_imbalance", "delta_hedged_return_proxy", "signal"]
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


def fig_sabr_svi_comparison(options: pd.DataFrame) -> go.Figure:
    """Compare observed IV, fast quadratic/SVI smile IV, and once-per-day SABR IV."""
    df = _midday_option_slice_by_day(options)
    if df.empty:
        return _empty_fig("SABR vs SVI Smile Comparison", "No options data available for daily SABR comparison")
    required = {"observed_iv", "smile_iv", "sabr_iv", "strike"}
    missing = sorted(required - set(df.columns))
    if missing:
        return _empty_fig("SABR vs SVI Smile Comparison", f"Missing required columns: {missing}")

    df = df.copy()
    df["smile_axis"] = _option_smile_axis(df)
    df["observed_iv_pct"] = 100.0 * pd.to_numeric(df["observed_iv"], errors="coerce")
    df["svi_iv_pct"] = 100.0 * pd.to_numeric(df["smile_iv"], errors="coerce")
    df["sabr_iv_pct"] = 100.0 * pd.to_numeric(df["sabr_iv"], errors="coerce")
    df["strike_label"] = pd.to_numeric(df["strike"], errors="coerce").round(0).astype("Int64").astype(str)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["smile_axis", "observed_iv_pct", "strike"])
    if df.empty:
        return _empty_fig("SABR vs SVI Smile Comparison", "No usable SABR comparison points found")
    if "display_day" not in df.columns:
        df = _add_display_day_labels(df)

    days = sorted(df["display_day"].dropna().astype(int).unique().tolist()) if "display_day" in df.columns else [0]
    ncols = min(max(len(days), 1), 4)
    fig = make_subplots(
        rows=1,
        cols=ncols,
        subplot_titles=[f"Day {int(day)}: Observed vs SVI vs SABR" for day in days[:ncols]],
        shared_yaxes=True,
    )

    for col_idx, day in enumerate(days[:ncols], start=1):
        sub = df[df["display_day"] == day].sort_values("smile_axis").copy()
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["smile_axis"], y=sub["observed_iv_pct"], mode="markers+text",
                text=sub["strike_label"], textposition="top center", name="Observed IV",
                showlegend=(col_idx == 1),
                hovertemplate="Strike %{text}<br>m=%{x:.3f}<br>Observed IV=%{y:.2f}%<extra></extra>",
            ),
            row=1, col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["smile_axis"], y=sub["svi_iv_pct"], mode="lines+markers",
                line=dict(dash="dash"), name="SVI/quadratic IV", showlegend=(col_idx == 1),
                hovertemplate="m=%{x:.3f}<br>SVI/quadratic IV=%{y:.2f}%<extra></extra>",
            ),
            row=1, col=col_idx,
        )
        sabr_sub = sub.dropna(subset=["sabr_iv_pct"]).sort_values("smile_axis")
        fig.add_trace(
            go.Scatter(
                x=sabr_sub["smile_axis"], y=sabr_sub["sabr_iv_pct"], mode="lines+markers",
                name="SABR IV", showlegend=(col_idx == 1),
                hovertemplate="m=%{x:.3f}<br>SABR IV=%{y:.2f}%<extra></extra>",
            ),
            row=1, col=col_idx,
        )
        fig.update_xaxes(title_text="m = log(K/S) / sqrt(T)", row=1, col=col_idx)

    fig.update_yaxes(title_text="Implied Vol (%)", row=1, col=1)
    fig.update_layout(
        title="SABR vs SVI/Quadratic Smile Comparison",
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=30, t=90, b=70),
    )
    return fig


def fig_sabr_residual_edge_heatmap(options: pd.DataFrame) -> go.Figure:
    """Heatmap of SABR absolute residual minus SVI absolute residual by day and strike."""
    options = _repair_options_for_plotting(options)
    if options.empty:
        return _empty_fig("SABR Residual Edge Heatmap", "No options data available")
    required = {"sabr_residual_minus_svi_residual", "strike"}
    missing = sorted(required - set(options.columns))
    if missing:
        return _empty_fig("SABR Residual Edge Heatmap", f"Missing required columns: {missing}")

    df = _add_display_day_labels(options.copy())
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["sabr_minus_svi_abs_residual_pct"] = 100.0 * pd.to_numeric(df["sabr_residual_minus_svi_residual"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["display_day", "strike", "sabr_minus_svi_abs_residual_pct"])
    if df.empty:
        return _empty_fig("SABR Residual Edge Heatmap", "No usable SABR residual comparison data found")

    pivot = (
        df.groupby(["display_day", "strike"], as_index=False)["sabr_minus_svi_abs_residual_pct"]
        .mean()
        .pivot(index="display_day", columns="strike", values="sabr_minus_svi_abs_residual_pct")
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.empty:
        return _empty_fig("SABR Residual Edge Heatmap", "No day by strike SABR residual grid found")

    try:
        text = pivot.map(lambda x: "" if pd.isna(x) else f"{x:+.2f}").to_numpy()
    except AttributeError:
        text = pivot.applymap(lambda x: "" if pd.isna(x) else f"{x:+.2f}").to_numpy()

    z = pivot.to_numpy(dtype=float)
    finite_abs = np.abs(z[np.isfinite(z)])
    color_cap = max(0.02, float(np.nanpercentile(finite_abs, 90))) if finite_abs.size else 1.0
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
                [0.20, "#2171b5"],
                [0.40, "#9ecae1"],
                [0.50, "#f7f7f7"],
                [0.60, "#fcbba1"],
                [0.80, "#de2d26"],
                [1.00, "#67000d"],
            ],
            zmid=0,
            zmin=-color_cap,
            zmax=color_cap,
            xgap=2,
            ygap=2,
            colorbar=dict(title="SABR abs residual - SVI abs residual (%)"),
            hovertemplate="%{y}<br>Strike %{x}<br>SABR - SVI abs residual %{z:+.3f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="SABR Residual Edge Heatmap",
        xaxis_title="Strike",
        yaxis_title="Trading day",
        template="plotly_white",
        height=540,
        margin=dict(l=80, r=40, t=90, b=70),
        annotations=[
            dict(
                text="Blue = SABR improves residuals. Red = SVI/quadratic is better.",
                x=0.5, y=1.08, xref="paper", yref="paper", showarrow=False, font=dict(size=14),
            )
        ],
    )
    return fig


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
                    _section("Implied Volatility vs Moneyness", "options-stats-load-vol-smile", dcc.Graph(id="options-stats-vol-smile", figure=_empty_fig("Implied Volatility vs Moneyness", "Click Load to render this chart.")), "Scatter of implied volatility v_t versus moneyness m_t across the sample, with a fitted parabola v̂_t and low-extrinsic outliers excluded."),
                    _section("IV Residuals", "options-stats-load-iv-residuals", dcc.Graph(id="options-stats-iv-residuals", figure=_empty_fig("IV Residuals", "Click Load to render this chart.")), "Observed IV minus smile-implied IV over time."),
                    _section("Fair Price vs Market", "options-stats-load-fair-vs-market", dcc.Graph(id="options-stats-fair-vs-market", figure=_empty_fig("Fair Price vs Market", "Click Load to render this chart.")), "Market mid compared with Heston smile/skew fair value."),
                    _section("Normalized Mispricing", "options-stats-load-normalized-mispricing", dcc.Graph(id="options-stats-normalized-mispricing", figure=_empty_fig("Normalized Mispricing", "Click Load to render this chart.")), "Market minus fair divided by fair."),
                    _section("Goyal-Saretto RV-IV Spread", "options-stats-load-rv-iv-spread", dcc.Graph(id="options-stats-rv-iv-spread", figure=_empty_fig("Goyal-Saretto RV-IV Spread", "Click Load to render this chart.")), "Realized volatility proxy minus implied volatility."),
                    _section("Volatility Risk Premium", "options-stats-load-vrp", dcc.Graph(id="options-stats-vrp", figure=_empty_fig("Volatility Risk Premium", "Click Load to render this chart.")), "Fitted IV minus realized volatility. Positive means rich vol, negative means cheap vol."),
                    _section("VRP Z-Score", "options-stats-load-vrp-zscore", dcc.Graph(id="options-stats-vrp-zscore", figure=_empty_fig("VRP Z-Score", "Click Load to render this chart.")), "Rolling z-score of volatility risk premium by contract."),
                    _section("Xu-Wang IV Mispricing", "options-stats-load-xw-iv-mispricing", dcc.Graph(id="options-stats-xw-iv-mispricing", figure=_empty_fig("Xu-Wang IV Mispricing", "Click Load to render this chart.")), "Cross-sectional residual IV mispricing proxy."),
                    _section("Smile Outliers by Day", "options-stats-load-daily-smiles", dcc.Graph(id="options-stats-daily-smiles", figure=_empty_fig("Smile Outliers by Day", "Click Load to render this chart.")), "Mid-day IV smile panels for each trading day with strike labels and fitted smile."),
                    _section("SABR vs SVI Smile Comparison", "options-stats-load-sabr-svi-comparison", dcc.Graph(id="options-stats-sabr-svi-comparison", figure=_empty_fig("SABR vs SVI Smile Comparison", "Click Load to render this chart.")), "Observed IV versus fast SVI/quadratic smile and once-per-day SABR fit."),
                    _section("SABR Residual Edge Heatmap", "options-stats-load-sabr-residual-edge", dcc.Graph(id="options-stats-sabr-residual-edge", figure=_empty_fig("SABR Residual Edge Heatmap", "Click Load to render this chart.")), "SABR absolute residual minus SVI absolute residual by day and strike. Blue means SABR improves the fit."),
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

    @app.callback(Output("options-stats-sabr-svi-comparison", "figure"), Input("options-stats-load-sabr-svi-comparison", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_sabr_svi_comparison(n, options_data):
        if not n: raise PreventUpdate
        return fig_sabr_svi_comparison(_options_from_store(options_data))

    @app.callback(Output("options-stats-sabr-residual-edge", "figure"), Input("options-stats-load-sabr-residual-edge", "n_clicks"), State("options-stats-options-store", "data"), prevent_initial_call=True)
    def _load_sabr_residual_edge(n, options_data):
        if not n: raise PreventUpdate
        return fig_sabr_residual_edge_heatmap(_options_from_store(options_data))

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
            "sabr_svi_comparison": fig_sabr_svi_comparison(options),
            "sabr_residual_edge_heatmap": fig_sabr_residual_edge_heatmap(options),
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
