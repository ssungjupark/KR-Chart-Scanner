from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from data_loader import align_benchmark, load_benchmark_range, load_price_range
from indicators import add_indicators
from scanner import REVERSAL_WEIGHTS
from universe import load_krx_universe


CHECKS = list(REVERSAL_WEIGHTS)
HORIZONS = (5, 10, 20)


def _confirmed_higher_low_state(df: pd.DataFrame, span: int, tolerance: float = 0.005) -> pd.Series:
    """Vectorized state of the latest two confirmed swing lows.

    A pivot is not allowed to affect the state until `span` later trading rows
    exist, so the feature is safe for historical signal generation.
    """
    lows = df["Low"].astype(float)
    window = 2 * span + 1
    local_min = lows.eq(lows.rolling(window, center=True).min())
    pivot_pos = np.flatnonzero(local_min.to_numpy())

    state = np.zeros(len(df), dtype=bool)
    if len(pivot_pos) < 2:
        return pd.Series(state, index=df.index)

    confirmation_points: list[tuple[int, bool]] = []
    for j in range(1, len(pivot_pos)):
        prev_pos = int(pivot_pos[j - 1])
        curr_pos = int(pivot_pos[j])
        confirm_pos = curr_pos + span
        if confirm_pos >= len(df):
            continue
        is_higher = lows.iloc[curr_pos] >= lows.iloc[prev_pos] * (1.0 - tolerance)
        confirmation_points.append((confirm_pos, bool(is_higher)))

    for i, (start, value) in enumerate(confirmation_points):
        end = confirmation_points[i + 1][0] if i + 1 < len(confirmation_points) else len(df)
        state[start:end] = value

    return pd.Series(state, index=df.index)


def _future_max(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]


def _future_min(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]


def _feature_frame(stock: pd.DataFrame, benchmark_close: pd.Series) -> pd.DataFrame:
    cfg = DEFAULT_CONFIG
    x = add_indicators(stock, benchmark_close, cfg).copy()

    x["ADV20"] = (x["Close"] * x["Volume"]).rolling(20).mean()
    x["higher_low"] = _confirmed_higher_low_state(x, cfg.swing_span)
    x["sma20_rising"] = x[f"SMA{cfg.ma_fast}_SLOPE"] >= cfg.fast_slope_min
    x["sma60_flat_or_rising"] = x[f"SMA{cfg.ma_mid}_SLOPE"] >= cfg.mid_slope_flat_min
    x["ma20_ma60_converged"] = (
        (x[f"SMA{cfg.ma_fast}"] / x[f"SMA{cfg.ma_mid}"] - 1.0).abs() <= cfg.ma_gap_max
    )
    x["price_near_ma20"] = (
        (x["Close"] / x[f"SMA{cfg.ma_fast}"] - 1.0).abs() <= cfg.ma_price_band
    )
    x["price_near_ma60"] = (
        (x["Close"] / x[f"SMA{cfg.ma_mid}"] - 1.0).abs() <= cfg.ma_price_band
    )
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
    for name, weight in REVERSAL_WEIGHTS.items():
        x["base_score"] += x[name].fillna(False).astype(int) * weight

    for horizon in HORIZONS:
        x[f"ret_{horizon}d"] = x["Close"].shift(-horizon) / x["Close"] - 1.0
        bench_ret = benchmark_close.shift(-horizon) / benchmark_close - 1.0
        x[f"excess_{horizon}d"] = x[f"ret_{horizon}d"] - bench_ret

    x["mfe_20d"] = _future_max(x["High"], 20) / x["Close"] - 1.0
    x["mae_20d"] = _future_min(x["Low"], 20) / x["Close"] - 1.0
    x["row_pos"] = np.arange(len(x))
    return x


def _cooldown_positions(frame: pd.DataFrame, mask: pd.Series, cooldown: int) -> pd.DataFrame:
    positions = frame.loc[mask, "row_pos"].astype(int).to_numpy()
    chosen: list[int] = []
    last = -10_000
    for pos in positions:
        if pos - last >= cooldown:
            chosen.append(pos)
            last = pos
    if not chosen:
        return frame.iloc[0:0]
    return frame[frame["row_pos"].isin(chosen)]


def _process_symbol(
    ticker: str,
    name: str,
    benchmark: pd.DataFrame,
    start: str,
    end: str,
    min_adv: float,
    min_price: float,
    min_score: int,
    cooldown: int,
    sample_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    try:
        stock = load_price_range(ticker, start, end, warmup_days=550, forward_days=75)
        benchmark_close = align_benchmark(stock, benchmark)
        x = _feature_frame(stock, benchmark_close)

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        cfg = DEFAULT_CONFIG

        eligible = (
            (x.index >= start_ts)
            & (x.index <= end_ts)
            & (x["row_pos"] >= cfg.min_history - 1)
            & (x["ADV20"] >= min_adv)
            & (x["Close"] >= min_price)
            & x["excess_20d"].notna()
        )

        # Sparse, score-agnostic research observations for factor calibration.
        eligible_pos = x.loc[eligible, "row_pos"].astype(int)
        sample_mask = eligible & x["row_pos"].isin(eligible_pos.iloc[::sample_every].tolist())
        obs_cols = [
            "row_pos",
            "Close",
            "ADV20",
            "base_score",
            *CHECKS,
            *[f"ret_{h}d" for h in HORIZONS],
            *[f"excess_{h}d" for h in HORIZONS],
            "mfe_20d",
            "mae_20d",
        ]
        observations = x.loc[sample_mask, obs_cols].copy()
        observations.insert(0, "date", observations.index.strftime("%Y-%m-%d"))
        observations.insert(0, "name", name)
        observations.insert(0, "ticker", ticker)

        signal_mask = eligible & (x["base_score"] >= min_score)
        signals = _cooldown_positions(x, signal_mask, cooldown)
        signal_cols = obs_cols
        signals = signals[signal_cols].copy()
        signals.insert(0, "date", signals.index.strftime("%Y-%m-%d"))
        signals.insert(0, "name", name)
        signals.insert(0, "ticker", ticker)
        return observations.reset_index(drop=True), signals.reset_index(drop=True), None
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"{ticker},{name},{type(exc).__name__}: {exc}"


def _summary(frame: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    if frame.empty:
        return {"sample": label, "signals": 0}
    r20 = frame["ret_20d"].dropna()
    ex20 = frame["excess_20d"].dropna()
    return {
        "sample": label,
        "signals": int(len(frame)),
        "avg_5d": float(frame["ret_5d"].mean()),
        "avg_10d": float(frame["ret_10d"].mean()),
        "avg_20d": float(r20.mean()),
        "median_20d": float(r20.median()),
        "win_rate_20d": float((r20 > 0).mean()),
        "avg_excess_20d": float(ex20.mean()),
        "excess_win_rate_20d": float((ex20 > 0).mean()),
        "avg_mfe_20d": float(frame["mfe_20d"].mean()),
        "avg_mae_20d": float(frame["mae_20d"].mean()),
    }


def _factor_diagnostics(obs: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    rows = []
    for factor in CHECKS:
        passed = obs[obs[factor].astype(bool)]
        failed = obs[~obs[factor].astype(bool)]
        rows.append(
            {
                "sample": sample_name,
                "factor": factor,
                "pass_n": len(passed),
                "fail_n": len(failed),
                "pass_excess_20d": passed["excess_20d"].mean(),
                "fail_excess_20d": failed["excess_20d"].mean(),
                "edge_excess_20d": passed["excess_20d"].mean() - failed["excess_20d"].mean(),
                "pass_win_rate": (passed["excess_20d"] > 0).mean(),
                "fail_win_rate": (failed["excess_20d"] > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def _fit_calibrated_weights(train: pd.DataFrame, ridge: float = 20.0) -> tuple[dict[str, int], pd.DataFrame]:
    clean = train.dropna(subset=["excess_20d"]).copy()
    if len(clean) < 100:
        return REVERSAL_WEIGHTS.copy(), pd.DataFrame()

    X = clean[CHECKS].astype(float).to_numpy()
    y = clean["excess_20d"].to_numpy(dtype=float)
    X1 = np.column_stack([np.ones(len(X)), X])

    penalty = np.eye(X1.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X1.T @ X1 + penalty, X1.T @ y)
    coefs = beta[1:]

    positive = np.clip(coefs, 0.0, None)
    if positive.sum() <= 0:
        learned = np.array([REVERSAL_WEIGHTS[c] for c in CHECKS], dtype=float)
    else:
        learned = positive / positive.sum() * 100.0

    prior = np.array([REVERSAL_WEIGHTS[c] for c in CHECKS], dtype=float)
    blended = 0.5 * prior + 0.5 * learned
    rounded = np.floor(blended + 0.5).astype(int)
    diff = 100 - int(rounded.sum())
    if diff != 0:
        order = np.argsort(-(blended - np.floor(blended))) if diff > 0 else np.argsort(blended - np.floor(blended))
        for idx in order[: abs(diff)]:
            rounded[idx] += 1 if diff > 0 else -1

    weights = {factor: int(weight) for factor, weight in zip(CHECKS, rounded)}
    detail = pd.DataFrame(
        {
            "factor": CHECKS,
            "prior_weight": [REVERSAL_WEIGHTS[c] for c in CHECKS],
            "ridge_coef": coefs,
            "learned_raw_weight": learned,
            "calibrated_weight": [weights[c] for c in CHECKS],
        }
    )
    return weights, detail


def _apply_score(frame: pd.DataFrame, weights: dict[str, int], column: str) -> pd.DataFrame:
    out = frame.copy()
    out[column] = 0
    for factor, weight in weights.items():
        out[column] += out[factor].astype(int) * weight
    return out


def _cooldown_table(frame: pd.DataFrame, score_col: str, threshold: float, cooldown: int) -> pd.DataFrame:
    selected = []
    for _, group in frame.sort_values(["ticker", "row_pos"]).groupby("ticker", sort=False):
        last = -10_000
        for idx, row in group.iterrows():
            if row[score_col] < threshold:
                continue
            pos = int(row["row_pos"])
            if pos - last >= cooldown:
                selected.append(idx)
                last = pos
    return frame.loc[selected].copy() if selected else frame.iloc[0:0].copy()


def run_validation(
    start: str,
    end: str,
    split: str,
    benchmark_symbol: str,
    min_adv: float,
    min_price: float,
    min_score: int,
    cooldown: int,
    sample_every: int,
    workers: int,
    max_symbols: int | None,
    output_dir: str,
) -> None:
    print("Loading current KRX common-stock universe...")
    universe = load_krx_universe(max_symbols=max_symbols)
    print(f"Universe: {len(universe):,} symbols")
    print(f"Liquidity filter: historical 20D average trading value >= {min_adv/1e8:,.0f}억원")

    benchmark = load_benchmark_range(benchmark_symbol, start, end, warmup_days=550, forward_days=75)

    observations_parts: list[pd.DataFrame] = []
    signals_parts: list[pd.DataFrame] = []
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_symbol,
                ticker,
                name,
                benchmark,
                start,
                end,
                min_adv,
                min_price,
                min_score,
                cooldown,
                sample_every,
            ): (ticker, name)
            for ticker, name in universe.items()
        }
        total = len(futures)
        for i, future in enumerate(as_completed(futures), start=1):
            ticker, name = futures[future]
            obs, sig, failure = future.result()
            if not obs.empty:
                observations_parts.append(obs)
            if not sig.empty:
                signals_parts.append(sig)
            if failure:
                failures.append(failure)
            if i % 100 == 0 or i == total:
                print(f"Processed {i:,}/{total:,} | failures={len(failures):,}")

    observations = pd.concat(observations_parts, ignore_index=True) if observations_parts else pd.DataFrame()
    signals = pd.concat(signals_parts, ignore_index=True) if signals_parts else pd.DataFrame()
    if observations.empty:
        raise RuntimeError("No research observations were produced")

    observations["date"] = pd.to_datetime(observations["date"])
    signals["date"] = pd.to_datetime(signals["date"])
    split_ts = pd.Timestamp(split)

    train = observations[observations["date"] < split_ts].copy()
    test = observations[observations["date"] >= split_ts].copy()
    base_train = signals[signals["date"] < split_ts].copy()
    base_test = signals[signals["date"] >= split_ts].copy()

    weights, weight_detail = _fit_calibrated_weights(train)
    train_scored = _apply_score(train, weights, "calibrated_score")
    test_scored = _apply_score(test, weights, "calibrated_score")

    # Preserve the training-period selection rate of the original 60-point rule.
    base_rate = float((train_scored["base_score"] >= min_score).mean())
    quantile = max(0.0, min(1.0, 1.0 - base_rate))
    calibrated_threshold = float(train_scored["calibrated_score"].quantile(quantile))
    calibrated_test = _cooldown_table(test_scored, "calibrated_score", calibrated_threshold, cooldown)

    summary_rows = [
        _summary(base_train, "base_train"),
        _summary(base_test, "base_oos"),
        _summary(calibrated_test, "calibrated_oos"),
    ]
    summary = pd.DataFrame(summary_rows)

    factor_diag = pd.concat(
        [
            _factor_diagnostics(train, "train"),
            _factor_diagnostics(test, "oos"),
        ],
        ignore_index=True,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    observations.to_csv(out / "market_observations.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(out / "market_base_signals.csv", index=False, encoding="utf-8-sig")
    calibrated_test.to_csv(out / "market_calibrated_oos_signals.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "market_validation_summary.csv", index=False, encoding="utf-8-sig")
    factor_diag.to_csv(out / "factor_diagnostics.csv", index=False, encoding="utf-8-sig")
    weight_detail.to_csv(out / "calibrated_weights.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"failure": failures}).to_csv(out / "data_failures.csv", index=False, encoding="utf-8-sig")

    print("=" * 96)
    print("KR-Chart-Scanner | Broad KRX validation")
    print(f"Period          : {start} ~ {end}")
    print(f"Train / OOS     : before {split} / from {split}")
    print(f"Universe        : {len(universe):,} current KOSPI/KOSDAQ common stocks")
    print(f"Observations    : {len(observations):,}")
    print(f"Base signals    : {len(signals):,}")
    print(f"Failures        : {len(failures):,}")
    print(f"Calib threshold : {calibrated_threshold:.1f} (same train selection rate {base_rate:.2%})")
    print("=" * 96)

    percent_cols = [
        "avg_5d", "avg_10d", "avg_20d", "median_20d", "win_rate_20d",
        "avg_excess_20d", "excess_win_rate_20d", "avg_mfe_20d", "avg_mae_20d",
    ]
    printable = summary.copy()
    for col in percent_cols:
        if col in printable:
            printable[col] = printable[col].map(lambda v: "" if pd.isna(v) else f"{v*100:+.2f}%")
    print(printable.to_string(index=False))

    print("\nCalibrated weights")
    print(weight_detail[["factor", "prior_weight", "ridge_coef", "calibrated_weight"]].to_string(index=False))

    print("\nOOS factor edge, pass minus fail (20D excess return)")
    oos_diag = factor_diag[factor_diag["sample"] == "oos"].sort_values("edge_excess_20d", ascending=False)
    show = oos_diag[["factor", "pass_n", "fail_n", "edge_excess_20d"]].copy()
    show["edge_excess_20d"] = show["edge_excess_20d"].map(lambda v: f"{v*100:+.2f}%")
    print(show.to_string(index=False))

    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Broad KRX validation with out-of-sample factor calibration.")
    parser.add_argument("--start", default="2022-01-03")
    parser.add_argument("--end", default="2026-09-04")
    parser.add_argument("--split", default="2025-01-02")
    parser.add_argument("--benchmark", default="KS11")
    parser.add_argument("--min-adv", type=float, default=2_000_000_000, help="Historical 20D average trading value in KRW")
    parser.add_argument("--min-price", type=float, default=1000)
    parser.add_argument("--min-score", type=int, default=60)
    parser.add_argument("--cooldown", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-symbols", type=int, default=0, help="0 means all current KOSPI/KOSDAQ common stocks")
    parser.add_argument("--output-dir", default="output_market")
    args = parser.parse_args()

    run_validation(
        start=args.start,
        end=args.end,
        split=args.split,
        benchmark_symbol=args.benchmark,
        min_adv=args.min_adv,
        min_price=args.min_price,
        min_score=args.min_score,
        cooldown=args.cooldown,
        sample_every=args.sample_every,
        workers=args.workers,
        max_symbols=None if args.max_symbols <= 0 else args.max_symbols,
        output_dir=args.output_dir,
    )
