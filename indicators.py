from __future__ import annotations

import numpy as np
import pandas as pd

from config import ScannerConfig


def add_indicators(
    df: pd.DataFrame,
    benchmark_close: pd.Series,
    cfg: ScannerConfig,
) -> pd.DataFrame:
    out = df.copy()

    for period in (cfg.ma_short, cfg.ma_fast, cfg.ma_mid, cfg.ma_long):
        out[f"SMA{period}"] = out["Close"].rolling(period).mean()

    out["RSI"] = _rsi(out["Close"], cfg.rsi_period)
    out["ATR"] = _atr(out, cfg.atr_period)

    # MA slopes expressed as percentage change over slope_lookback trading days.
    for period in (cfg.ma_fast, cfg.ma_mid, cfg.ma_long):
        ma = out[f"SMA{period}"]
        out[f"SMA{period}_SLOPE"] = ma / ma.shift(cfg.slope_lookback) - 1.0

    # Relative strength: stock return minus benchmark return.
    bench = benchmark_close.reindex(out.index).ffill()
    out["RS20"] = out["Close"].pct_change(20) - bench.pct_change(20)

    # Ratio-based relative-strength trend. Positive means stock/benchmark ratio
    # improved during the latest slope lookback window.
    rs_ratio = out["Close"] / bench
    out["RS_RATIO"] = rs_ratio
    out["RS_RATIO_SLOPE"] = rs_ratio / rs_ratio.shift(cfg.slope_lookback) - 1.0

    out["VOL20"] = out["Volume"].rolling(20).mean()
    out["PRIOR_HIGH_60"] = out["High"].shift(1).rolling(60).max()
    out["SPACE_TO_HIGH60"] = out["PRIOR_HIGH_60"] / out["Close"] - 1.0

    # Average volume on up/down days over the latest 20 sessions.
    up_mask = out["Close"].diff() > 0
    down_mask = out["Close"].diff() < 0
    out["UP_VOL20"] = out["Volume"].where(up_mask).rolling(20, min_periods=5).mean()
    out["DOWN_VOL20"] = out["Volume"].where(down_mask).rolling(20, min_periods=5).mean()

    return out


def confirmed_swing_lows(df: pd.DataFrame, span: int = 3) -> pd.DataFrame:
    """Return confirmed swing lows using only information available by df.index[-1].

    A candidate at t is confirmed only after `span` later trading days have
    occurred. Since callers truncate data at the evaluation date first, this
    does not use data after the historical as-of date.
    """
    lows = df["Low"]
    window = 2 * span + 1
    local_min = lows.rolling(window=window, center=True).min()
    pivots = df.loc[lows.eq(local_min), ["Low"]].copy()

    # The final `span` rows cannot yet be confirmed at the as-of date.
    if len(df) > span:
        cutoff = df.index[-span - 1]
        pivots = pivots.loc[pivots.index <= cutoff]
    else:
        pivots = pivots.iloc[0:0]

    return pivots


def higher_low(df: pd.DataFrame, span: int = 3, tolerance: float = 0.005) -> tuple[bool, float | None, float | None]:
    pivots = confirmed_swing_lows(df, span=span)
    if len(pivots) < 2:
        return False, None, None

    prev_low = float(pivots.iloc[-2]["Low"])
    last_low = float(pivots.iloc[-1]["Low"])

    # Small tolerance avoids treating nearly identical lows as a failure.
    is_higher = last_low >= prev_low * (1.0 - tolerance)
    return is_higher, prev_low, last_low


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100.0).where(avg_gain.notna())


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
