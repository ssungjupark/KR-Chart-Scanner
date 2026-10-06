from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from config import ScannerConfig
from indicators import higher_low


@dataclass
class ScanResult:
    ticker: str
    as_of: str
    score: int
    grade: str
    metrics: dict[str, Any]
    checks: dict[str, bool]


def score_reversal(
    ticker: str,
    df: pd.DataFrame,
    cfg: ScannerConfig,
) -> ScanResult:
    if len(df) < cfg.min_history:
        raise ValueError(
            f"Not enough history: {len(df)} rows, need at least {cfg.min_history}"
        )

    row = df.iloc[-1]
    as_of = df.index[-1].strftime("%Y-%m-%d")

    is_hl, prev_swing_low, last_swing_low = higher_low(df, span=cfg.swing_span)

    ma20 = float(row[f"SMA{cfg.ma_fast}"])
    ma60 = float(row[f"SMA{cfg.ma_mid}"])
    ma120 = float(row[f"SMA{cfg.ma_long}"])
    close = float(row["Close"])

    ma_gap = abs(ma20 / ma60 - 1.0)
    dist_ma20 = close / ma20 - 1.0
    dist_ma60 = close / ma60 - 1.0

    fast_slope = float(row[f"SMA{cfg.ma_fast}_SLOPE"])
    mid_slope = float(row[f"SMA{cfg.ma_mid}_SLOPE"])
    long_slope = float(row[f"SMA{cfg.ma_long}_SLOPE"])

    rsi = float(row["RSI"])
    rs20 = float(row["RS20"])
    rs_slope = float(row["RS_RATIO_SLOPE"])
    space_to_high60 = float(row["SPACE_TO_HIGH60"])

    up_vol = float(row["UP_VOL20"]) if pd.notna(row["UP_VOL20"]) else float("nan")
    down_vol = float(row["DOWN_VOL20"]) if pd.notna(row["DOWN_VOL20"]) else float("nan")

    checks = {
        "higher_low": is_hl,
        "sma20_rising": fast_slope >= cfg.fast_slope_min,
        "sma60_flat_or_rising": mid_slope >= cfg.mid_slope_flat_min,
        "ma20_ma60_converged": ma_gap <= cfg.ma_gap_max,
        "price_near_ma20": abs(dist_ma20) <= cfg.ma_price_band,
        "price_near_ma60": abs(dist_ma60) <= cfg.ma_price_band,
        "rsi_normal": cfg.rsi_low <= rsi <= cfg.rsi_high,
        "rs20_positive": rs20 >= cfg.rs20_min,
        "rs_trend_positive": rs_slope >= cfg.rs_slope_min,
        "correction_volume_ok": pd.notna(down_vol)
        and pd.notna(up_vol)
        and down_vol <= up_vol * 1.10,
        "resistance_space_ok": space_to_high60 >= cfg.resistance_space_min,
    }

    # Initial weights. These are hypotheses, not truths. Backtesting will decide
    # whether to keep, reduce or remove each factor.
    weights = {
        "higher_low": 15,
        "sma20_rising": 15,
        "sma60_flat_or_rising": 10,
        "ma20_ma60_converged": 10,
        "price_near_ma20": 5,
        "price_near_ma60": 5,
        "rsi_normal": 5,
        "rs20_positive": 10,
        "rs_trend_positive": 5,
        "correction_volume_ok": 10,
        "resistance_space_ok": 10,
    }

    score = sum(weights[name] for name, passed in checks.items() if passed)

    if score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    else:
        grade = "D"

    metrics = {
        "close": close,
        "sma20": ma20,
        "sma60": ma60,
        "sma120": ma120,
        "sma20_slope_5d": fast_slope,
        "sma60_slope_5d": mid_slope,
        "sma120_slope_5d": long_slope,
        "ma20_ma60_gap": ma_gap,
        "distance_to_sma20": dist_ma20,
        "distance_to_sma60": dist_ma60,
        "rsi14": rsi,
        "rs20_excess_return": rs20,
        "rs_ratio_slope_5d": rs_slope,
        "space_to_prior_60d_high": space_to_high60,
        "up_day_volume_20": up_vol,
        "down_day_volume_20": down_vol,
        "previous_swing_low": prev_swing_low,
        "latest_swing_low": last_swing_low,
    }

    return ScanResult(
        ticker=ticker,
        as_of=as_of,
        score=int(score),
        grade=grade,
        metrics=metrics,
        checks=checks,
    )
