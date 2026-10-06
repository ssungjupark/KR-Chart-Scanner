from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from data_loader import (
    align_benchmark_ohlc,
    load_benchmark_range,
    load_delisted_price_range,
    load_price_range,
)
from indicators import add_indicators
from scanner import REVERSAL_WEIGHTS
from universe import load_krx_research_universe


CHECKS = list(REVERSAL_WEIGHTS)
HORIZONS = (5, 10, 20)


def _confirmed_higher_low_state(df: pd.DataFrame, span: int, tolerance: float = 0.005) -> pd.Series:
    lows = df["Low"].astype(float)
    local_min = lows.eq(lows.rolling(2 * span + 1, center=True).min())
    pivots = np.flatnonzero(local_min.to_numpy())
    state = np.zeros(len(df), dtype=bool)
    if len(pivots) < 2:
        return pd.Series(state, index=df.index)

    confirms: list[tuple[int, bool]] = []
    for j in range(1, len(pivots)):
        prev_pos = int(pivots[j - 1])
        curr_pos = int(pivots[j])
        confirm_pos = curr_pos + span
        if confirm_pos < len(df):
            confirms.append((confirm_pos, bool(lows.iloc[curr_pos] >= lows.iloc[prev_pos] * (1 - tolerance))))

    for i, (start, value) in enumerate(confirms):
        end = confirms[i + 1][0] if i + 1 < len(confirms) else len(df)
        state[start:end] = value
    return pd.Series(state, index=df.index)


def _future_max(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]


def _future_min(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]


def _feature_frame(stock: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    cfg = DEFAULT_CONFIG
    bench = align_benchmark_ohlc(stock, benchmark)
    x = add_indicators(stock, bench["Close"], cfg).copy()

    x["ADV20"] = (x["Close"] * x["Volume"]).rolling(20).mean()
    x["higher_low"] = _confirmed_higher_low_state(x, cfg.swing_span)
    x["sma20_rising"] = x[f"SMA{cfg.ma_fast}_SLOPE"] >= cfg.fast_slope_min
    x["sma60_flat_or_rising"] = x[f"SMA{cfg.ma_mid}_SLOPE"] >= cfg.mid_slope_flat_min
    x["ma20_ma60_converged"] = (
        (x[f"SMA{cfg.ma_fast}"] / x[f"SMA{cfg.ma_mid}"] - 1).abs() <= cfg.ma_gap_max
    )
    x["price_near_ma20"] = (
        (x["Close"] / x[f"SMA{cfg.ma_fast}"] - 1).abs() <= cfg.ma_price_band
    )
    x["price_near_ma60"] = (
        (x["Close"] / x[f"SMA{cfg.ma_mid}"] - 1).abs() <= cfg.ma_price_band
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
    for factor, weight in REVERSAL_WEIGHTS.items():
        x["base_score"] += x[factor].fillna(False).astype(int) * weight

    # Market regime is deliberately kept separate from the stock score.
    bench_sma20 = bench["Close"].rolling(20).mean()
    bench_sma60 = bench["Close"].rolling(60).mean()
    bench_sma120 = bench["Close"].rolling(120).mean()
    x["market_bull"] = (bench["Close"] > bench_sma120) & (bench_sma20 > bench_sma60)

    # Executable convention: signal is known after today's close; entry is next session open.
    entry = x["Open"].shift(-1)
    bench_entry = bench["Open"].shift(-1)
    x["entry_next_open"] = entry
    for h in HORIZONS:
        x[f"ret_{h}d"] = x["Close"].shift(-h) / entry - 1
        bench_ret = bench["Close"].shift(-h) / bench_entry - 1
        x[f"excess_{h}d"] = x[f"ret_{h}d"] - bench_ret

    x["mfe_20d"] = _future_max(x["High"], 20) / entry - 1
    x["mae_20d"] = _future_min(x["Low"], 20) / entry - 1
    x["row_pos"] = np.arange(len(x))
    return x


def _cooldown(frame: pd.DataFrame, mask: pd.Series, sessions: int) -> pd.DataFrame:
    positions = frame.loc[mask, "row_pos"].astype(int).to_numpy()
    chosen: list[int] = []
    last = -10_000
    for pos in positions:
        if pos - last >= sessions:
            chosen.append(pos)
            last = pos
    return frame[frame["row_pos"].isin(chosen)].copy() if chosen else frame.iloc[0:0].copy()


def _process_symbol(
    meta: dict,
    benchmarks: dict[str, pd.DataFrame],
    start: str,
    end: str,
    min_adv: float,
    min_price: float,
    min_score: int,
    cooldown: int,
    sample_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    ticker = str(meta["ticker"])
    name = str(meta["name"])
    market = str(meta["market"])
    source = str(meta["source"])
    try:
        loader = load_delisted_price_range if source == "delisted" else load_price_range
        stock = loader(ticker, start, end, warmup_days=550, forward_days=75)
        benchmark = benchmarks[market]
        x = _feature_frame(stock, benchmark)

        cfg = DEFAULT_CONFIG
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        eligible = (
            (x.index >= start_ts)
            & (x.index <= end_ts)
            & (x["row_pos"] >= cfg.min_history - 1)
            & (x["ADV20"] >= min_adv)
            & (x["Close"] >= min_price)
            & x["ret_20d"].notna()
            & x["excess_20d"].notna()
        )

        common_cols = [
            "row_pos", "Close", "entry_next_open", "ADV20", "base_score", "market_bull",
            *CHECKS,
            *[f"ret_{h}d" for h in HORIZONS],
            *[f"excess_{h}d" for h in HORIZONS],
            "mfe_20d", "mae_20d",
        ]

        eligible_positions = x.loc[eligible, "row_pos"].astype(int).tolist()
        sampled_positions = set(eligible_positions[::sample_every])
        obs = x.loc[eligible & x["row_pos"].isin(sampled_positions), common_cols].copy()

        sig = _cooldown(x, eligible & (x["base_score"] >= min_score), cooldown)[common_cols].copy()

        for frame in (obs, sig):
            frame.insert(0, "date", frame.index.strftime("%Y-%m-%d"))
            frame.insert(0, "source", source)
            frame.insert(0, "market", market)
            frame.insert(0, "name", name)
            frame.insert(0, "ticker", ticker)

        return obs.reset_index(drop=True), sig.reset_index(drop=True), None
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"{ticker},{name},{market},{source},{type(exc).__name__}: {exc}"


def _summary(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {"sample": label, "signals": 0}
    r20 = frame["ret_20d"].dropna()
    ex20 = frame["excess_20d"].dropna()
    return {
        "sample": label,
        "signals": len(frame),
        "avg_5d": frame["ret_5d"].mean(),
        "avg_10d": frame["ret_10d"].mean(),
        "avg_20d": r20.mean(),
        "median_20d": r20.median(),
        "win_rate_20d": (r20 > 0).mean(),
        "avg_excess_20d": ex20.mean(),
        "excess_win_rate_20d": (ex20 > 0).mean(),
        "avg_mfe_20d": frame["mfe_20d"].mean(),
        "avg_mae_20d": frame["mae_20d"].mean(),
    }


def _factor_diagnostics(frame: pd.DataFrame, sample: str) -> pd.DataFrame:
    rows = []
    for factor in CHECKS:
        passed = frame[frame[factor].astype(bool)]
        failed = frame[~frame[factor].astype(bool)]
        rows.append(
            {
                "sample": sample,
                "factor": factor,
                "pass_n": len(passed),
                "fail_n": len(failed),
                "pass_excess_20d": passed["excess_20d"].mean(),
                "fail_excess_20d": failed["excess_20d"].mean(),
                "edge_excess_20d": passed["excess_20d"].mean() - failed["excess_20d"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _ridge_learned_weights(train: pd.DataFrame, ridge: float = 25.0) -> np.ndarray:
    clean = train.dropna(subset=["excess_20d"])
    X = clean[CHECKS].astype(float).to_numpy()
    y = clean["excess_20d"].to_numpy(float)
    X1 = np.column_stack([np.ones(len(X)), X])
    penalty = np.eye(X1.shape[1]) * ridge
    penalty[0, 0] = 0
    beta = np.linalg.solve(X1.T @ X1 + penalty, X1.T @ y)[1:]
    positive = np.clip(beta, 0, None)
    if positive.sum() == 0:
        return np.array([REVERSAL_WEIGHTS[c] for c in CHECKS], dtype=float)
    return positive / positive.sum() * 100


def _weights_for_alpha(learned: np.ndarray, alpha: float) -> dict[str, int]:
    prior = np.array([REVERSAL_WEIGHTS[c] for c in CHECKS], dtype=float)
    raw = (1 - alpha) * prior + alpha * learned
    rounded = np.floor(raw + 0.5).astype(int)
    diff = 100 - int(rounded.sum())
    if diff:
        residual = raw - np.floor(raw)
        order = np.argsort(-residual) if diff > 0 else np.argsort(residual)
        for idx in order[: abs(diff)]:
            rounded[idx] += 1 if diff > 0 else -1
    return {f: int(w) for f, w in zip(CHECKS, rounded)}


def _score(frame: pd.DataFrame, weights: dict[str, int], col: str) -> pd.DataFrame:
    out = frame.copy()
    out[col] = 0
    for factor, weight in weights.items():
        out[col] += out[factor].astype(int) * weight
    return out


def _select_from_observations(
    frame: pd.DataFrame,
    score_col: str,
    threshold: float,
    cooldown: int,
    bull_only: bool,
) -> pd.DataFrame:
    chosen: list[int] = []
    for _, group in frame.sort_values(["ticker", "row_pos"]).groupby("ticker", sort=False):
        last = -10_000
        for idx, row in group.iterrows():
            if float(row[score_col]) < threshold:
                continue
            if bull_only and not bool(row["market_bull"]):
                continue
            pos = int(row["row_pos"])
            if pos - last >= cooldown:
                chosen.append(idx)
                last = pos
    return frame.loc[chosen].copy() if chosen else frame.iloc[0:0].copy()


def _utility(frame: pd.DataFrame) -> float:
    if len(frame) < 100:
        return -999.0
    # Reward market-relative edge and penalize adverse excursion.
    return float(frame["excess_20d"].mean() + 0.20 * frame["mae_20d"].mean())


def run(
    start: str,
    end: str,
    train_end: str,
    validation_end: str,
    min_adv: float,
    min_price: float,
    min_score: int,
    cooldown: int,
    sample_every: int,
    workers: int,
    max_symbols: int | None,
    output_dir: str,
) -> None:
    universe = load_krx_research_universe(start, end, include_delisted=True, max_symbols=max_symbols)
    print(f"Research universe: {len(universe):,} current + period-delisted KOSPI/KOSDAQ common stocks")
    print(universe["source"].value_counts().to_string())

    benchmarks = {
        "KOSPI": load_benchmark_range("KS11", start, end, warmup_days=550, forward_days=75),
        "KOSDAQ": load_benchmark_range("KQ11", start, end, warmup_days=550, forward_days=75),
    }

    obs_parts: list[pd.DataFrame] = []
    sig_parts: list[pd.DataFrame] = []
    failures: list[str] = []
    records = universe.to_dict("records")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_symbol, meta, benchmarks, start, end, min_adv, min_price,
                min_score, cooldown, sample_every,
            ): meta
            for meta in records
        }
        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            obs, sig, failure = future.result()
            if not obs.empty:
                obs_parts.append(obs)
            if not sig.empty:
                sig_parts.append(sig)
            if failure:
                failures.append(failure)
            if i % 100 == 0 or i == total:
                print(f"Processed {i:,}/{total:,}; failures={len(failures):,}")

    observations = pd.concat(obs_parts, ignore_index=True)
    base_signals = pd.concat(sig_parts, ignore_index=True) if sig_parts else pd.DataFrame()
    observations["date"] = pd.to_datetime(observations["date"])
    base_signals["date"] = pd.to_datetime(base_signals["date"])

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(validation_end)
    train = observations[observations["date"] <= train_end_ts].copy()
    validation = observations[(observations["date"] > train_end_ts) & (observations["date"] <= val_end_ts)].copy()
    test = observations[observations["date"] > val_end_ts].copy()

    learned = _ridge_learned_weights(train)
    base_rate = float((train["base_score"] >= min_score).mean())

    model_rows: list[dict] = []
    candidates: list[tuple[float, bool, dict[str, int], float, float]] = []
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        weights = _weights_for_alpha(learned, alpha)
        train_s = _score(train, weights, "candidate_score")
        threshold = float(train_s["candidate_score"].quantile(1 - base_rate))
        val_s = _score(validation, weights, "candidate_score")
        for bull_only in (False, True):
            selected = _select_from_observations(val_s, "candidate_score", threshold, cooldown, bull_only)
            util = _utility(selected)
            row = _summary(selected, f"alpha={alpha:.2f},bull={bull_only}")
            row.update({"alpha": alpha, "bull_only": bull_only, "threshold": threshold, "utility": util})
            model_rows.append(row)
            candidates.append((util, bull_only, weights, threshold, alpha))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_utility, best_bull, best_weights, best_threshold, best_alpha = candidates[0]

    test_scored = _score(test, best_weights, "calibrated_score")
    calibrated_test = _select_from_observations(
        test_scored, "calibrated_score", best_threshold, cooldown, best_bull
    )

    base_train = base_signals[base_signals["date"] <= train_end_ts]
    base_val = base_signals[(base_signals["date"] > train_end_ts) & (base_signals["date"] <= val_end_ts)]
    base_test = base_signals[base_signals["date"] > val_end_ts]

    summaries = pd.DataFrame(
        [
            _summary(base_train, "base_train"),
            _summary(base_val, "base_validation"),
            _summary(base_test, "base_test"),
            _summary(base_test[base_test["market_bull"]], "base_test_bull_only"),
            _summary(calibrated_test, "selected_model_test"),
        ]
    )

    diagnostics = pd.concat(
        [
            _factor_diagnostics(train, "train"),
            _factor_diagnostics(validation, "validation"),
            _factor_diagnostics(test, "test"),
        ],
        ignore_index=True,
    )

    weight_table = pd.DataFrame(
        {
            "factor": CHECKS,
            "prior_weight": [REVERSAL_WEIGHTS[c] for c in CHECKS],
            "ridge_positive_weight": learned,
            "selected_weight": [best_weights[c] for c in CHECKS],
        }
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    observations.to_csv(out / "observations.csv", index=False, encoding="utf-8-sig")
    base_signals.to_csv(out / "base_signals.csv", index=False, encoding="utf-8-sig")
    calibrated_test.to_csv(out / "selected_model_test_signals.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(out / "factor_diagnostics.csv", index=False, encoding="utf-8-sig")
    weight_table.to_csv(out / "selected_weights.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(model_rows).sort_values("utility", ascending=False).to_csv(out / "model_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"failure": failures}).to_csv(out / "failures.csv", index=False, encoding="utf-8-sig")

    print("=" * 104)
    print("KR-Chart-Scanner | Market validation v2")
    print(f"Period            : {start} ~ {end}")
    print(f"Train             : through {train_end}")
    print(f"Validation        : {pd.Timestamp(train_end) + pd.Timedelta(days=1):%Y-%m-%d} ~ {validation_end}")
    print(f"Holdout test      : after {validation_end}")
    print(f"Historical ADV20  : >= {min_adv / 1e8:,.0f}억원")
    print(f"Universe rows     : {len(universe):,}")
    print(f"Observations      : {len(observations):,}")
    print(f"Base signals      : {len(base_signals):,}")
    print(f"Data failures     : {len(failures):,}")
    print(f"Selected alpha    : {best_alpha:.2f}")
    print(f"Selected bull gate: {best_bull}")
    print(f"Selected threshold: {best_threshold:.1f}")
    print(f"Validation utility: {best_utility:+.4f}")
    print("=" * 104)

    show = summaries.copy()
    pct_cols = [
        "avg_5d", "avg_10d", "avg_20d", "median_20d", "win_rate_20d",
        "avg_excess_20d", "excess_win_rate_20d", "avg_mfe_20d", "avg_mae_20d",
    ]
    for col in pct_cols:
        if col in show:
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v*100:+.2f}%")
    print(show.to_string(index=False))
    print("\nSelected weights")
    print(weight_table.to_string(index=False))
    print("\nSaved to", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
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
    p.add_argument("--output-dir", default="output_market_v2")
    a = p.parse_args()
    run(
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
