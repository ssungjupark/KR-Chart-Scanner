from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import market_validation_v2 as core
from config import DEFAULT_CONFIG
from data_loader import align_benchmark_ohlc
from indicators import add_indicators
from scanner import REVERSAL_WEIGHTS


CHECKS = list(REVERSAL_WEIGHTS)
HORIZONS = (5, 10, 20)


def finite_feature_frame(stock: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    cfg = DEFAULT_CONFIG
    bench = align_benchmark_ohlc(stock, benchmark)
    x = add_indicators(stock, bench["Close"], cfg).copy()

    x["ADV20"] = (x["Close"] * x["Volume"]).rolling(20).mean()
    x["higher_low"] = core._confirmed_higher_low_state(x, cfg.swing_span)
    x["sma20_rising"] = x[f"SMA{cfg.ma_fast}_SLOPE"] >= cfg.fast_slope_min
    x["sma60_flat_or_rising"] = x[f"SMA{cfg.ma_mid}_SLOPE"] >= cfg.mid_slope_flat_min
    x["ma20_ma60_converged"] = (
        (x[f"SMA{cfg.ma_fast}"] / x[f"SMA{cfg.ma_mid}"] - 1).abs() <= cfg.ma_gap_max
    )
    x["price_near_ma20"] = ((x["Close"] / x[f"SMA{cfg.ma_fast}"] - 1).abs() <= cfg.ma_price_band)
    x["price_near_ma60"] = ((x["Close"] / x[f"SMA{cfg.ma_mid}"] - 1).abs() <= cfg.ma_price_band)
    x["rsi_normal"] = x["RSI"].between(cfg.rsi_low, cfg.rsi_high)
    x["rs20_positive"] = x["RS20"] >= cfg.rs20_min
    x["rs_trend_positive"] = x["RS_RATIO_SLOPE"] >= cfg.rs_slope_min
    x["correction_volume_ok"] = (
        x["DOWN_VOL20"].notna()
        & x["UP_VOL20"].notna()
        & (x["DOWN_VOL20"] <= x["UP_VOL20"] * 1.10)
    )
    x["resistance_space_ok"] = x["SPACE_TO_HIGH60"] >= cfg.resistance_space_min

    x["base_score"] = 0
    for factor, weight in REVERSAL_WEIGHTS.items():
        x["base_score"] += x[factor].fillna(False).astype(int) * weight

    bench_sma20 = bench["Close"].rolling(20).mean()
    bench_sma60 = bench["Close"].rolling(60).mean()
    bench_sma120 = bench["Close"].rolling(120).mean()
    x["market_bull"] = (bench["Close"] > bench_sma120) & (bench_sma20 > bench_sma60)

    # Scanner runs after the close. A realistic backtest enters at next session's open.
    entry = x["Open"].shift(-1).where(x["Open"].shift(-1) > 0)
    bench_entry = bench["Open"].shift(-1).where(bench["Open"].shift(-1) > 0)
    x["entry_next_open"] = entry

    for h in HORIZONS:
        stock_ret = x["Close"].shift(-h) / entry - 1
        bench_ret = bench["Close"].shift(-h) / bench_entry - 1
        x[f"ret_{h}d"] = stock_ret.replace([np.inf, -np.inf], np.nan)
        x[f"excess_{h}d"] = (stock_ret - bench_ret).replace([np.inf, -np.inf], np.nan)

    x["mfe_20d"] = (core._future_max(x["High"], 20) / entry - 1).replace([np.inf, -np.inf], np.nan)
    x["mae_20d"] = (core._future_min(x["Low"], 20) / entry - 1).replace([np.inf, -np.inf], np.nan)
    x["row_pos"] = np.arange(len(x))
    return x


def finite_summary(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {"sample": label, "signals": 0}
    clean = frame.copy()
    numeric = [
        "ret_5d", "ret_10d", "ret_20d", "excess_20d", "mfe_20d", "mae_20d"
    ]
    for col in numeric:
        if col in clean:
            clean[col] = pd.to_numeric(clean[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    r20 = clean["ret_20d"].dropna()
    ex20 = clean["excess_20d"].dropna()
    return {
        "sample": label,
        "signals": len(clean),
        "avg_5d": clean["ret_5d"].mean(),
        "avg_10d": clean["ret_10d"].mean(),
        "avg_20d": r20.mean(),
        "median_20d": r20.median(),
        "win_rate_20d": (r20 > 0).mean(),
        "avg_excess_20d": ex20.mean(),
        "excess_win_rate_20d": (ex20 > 0).mean(),
        "avg_mfe_20d": clean["mfe_20d"].mean(),
        "avg_mae_20d": clean["mae_20d"].mean(),
    }


def finite_factor_diagnostics(frame: pd.DataFrame, sample: str) -> pd.DataFrame:
    clean = frame.copy()
    clean["excess_20d"] = pd.to_numeric(clean["excess_20d"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    rows = []
    for factor in CHECKS:
        passed = clean[clean[factor].astype(bool)]["excess_20d"].dropna()
        failed = clean[~clean[factor].astype(bool)]["excess_20d"].dropna()
        pmean = passed.mean()
        fmean = failed.mean()
        rows.append({
            "sample": sample,
            "factor": factor,
            "pass_n": len(passed),
            "fail_n": len(failed),
            "pass_excess_20d": pmean,
            "fail_excess_20d": fmean,
            "edge_excess_20d": pmean - fmean if pd.notna(pmean) and pd.notna(fmean) else np.nan,
        })
    return pd.DataFrame(rows)


def finite_ridge_weights(train: pd.DataFrame, ridge: float = 25.0) -> np.ndarray:
    clean = train.copy()
    clean["excess_20d"] = pd.to_numeric(clean["excess_20d"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["excess_20d"])
    if len(clean) < 200:
        return np.array([REVERSAL_WEIGHTS[c] for c in CHECKS], dtype=float)

    X = clean[CHECKS].astype(float).to_numpy()
    y = clean["excess_20d"].to_numpy(float)
    finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[finite], y[finite]
    X1 = np.column_stack([np.ones(len(X)), X])
    penalty = np.eye(X1.shape[1]) * ridge
    penalty[0, 0] = 0
    beta = np.linalg.solve(X1.T @ X1 + penalty, X1.T @ y)[1:]
    positive = np.clip(np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
    if positive.sum() <= 0:
        return np.array([REVERSAL_WEIGHTS[c] for c in CHECKS], dtype=float)
    return positive / positive.sum() * 100


def finite_utility(frame: pd.DataFrame) -> float:
    if len(frame) < 100:
        return -999.0
    ex = pd.to_numeric(frame["excess_20d"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mae = pd.to_numeric(frame["mae_20d"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if ex.notna().sum() < 100:
        return -999.0
    return float(ex.mean() + 0.20 * mae.mean())


# Replace the research-core functions with guarded versions while retaining the
# tested orchestration, universe handling, cooldown logic and artifact outputs.
core._feature_frame = finite_feature_frame
core._summary = finite_summary
core._factor_diagnostics = finite_factor_diagnostics
core._ridge_learned_weights = finite_ridge_weights
core._utility = finite_utility


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KRX validation v3: next-open, finite-safe, market-specific")
    p.add_argument("--start", default="2022-01-03")
    p.add_argument("--end", default="2026-09-04")
    p.add_argument("--train-end", default="2023-12-28")
    p.add_argument("--validation-end", default="2024-12-30")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--min-score", type=int, default=60)
    p.add_argument("--cooldown", type=int, default=20)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--output-dir", default="output_market_v3")
    a = p.parse_args()
    core.run(
        start=a.start,
        end=a.end,
        train_end=a.train_end,
        validation_end=a.validation_end,
        min_adv=a.min_adv,
        min_price=a.min_price,
        min_score=a.min_score,
        cooldown=a.cooldown,
        sample_every=a.sample_every,
        workers=a.workers,
        max_symbols=None if a.max_symbols <= 0 else a.max_symbols,
        output_dir=a.output_dir,
    )
