from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from data_loader import align_benchmark_ohlc, load_benchmark_range, load_price_range
from indicators import add_indicators
from market_validation_v2 import _confirmed_higher_low_state, _future_max, _future_min
from universe import load_krx_universe_frame


HORIZONS = (5, 10, 20)


def _features(stock: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    cfg = DEFAULT_CONFIG
    bench = align_benchmark_ohlc(stock, benchmark)
    x = add_indicators(stock, bench["Close"], cfg).copy()

    x["row_pos"] = np.arange(len(x))
    x["ADV20"] = (x["Close"] * x["Volume"]).rolling(20).mean()
    x["HL"] = _confirmed_higher_low_state(x, cfg.swing_span)

    x["DIST20"] = x["Close"] / x["SMA20"] - 1
    x["DIST60"] = x["Close"] / x["SMA60"] - 1
    x["MA_GAP_20_60"] = (x["SMA20"] / x["SMA60"] - 1).abs()

    x["PRIOR_HIGH20"] = x["High"].shift(1).rolling(20).max()
    x["PRIOR_HIGH60_X"] = x["High"].shift(1).rolling(60).max()
    x["PRIOR_LOW20"] = x["Low"].shift(1).rolling(20).min()
    x["DD20"] = x["Close"] / x["PRIOR_HIGH20"] - 1
    x["DD60"] = x["Close"] / x["PRIOR_HIGH60_X"] - 1

    prev_vol20 = x["Volume"].shift(1).rolling(20).mean()
    x["VOL_RATIO"] = x["Volume"] / prev_vol20.replace(0, np.nan)
    x["NATR"] = x["ATR"] / x["Close"]
    x["NATR_Q40_60"] = x["NATR"].rolling(60).quantile(0.40)

    trend_up = (
        (x["Close"] > x["SMA120"])
        & (x["SMA20"] > x["SMA60"])
        & (x["SMA60"] > x["SMA120"])
        & (x["SMA20_SLOPE"] > 0)
        & (x["SMA60_SLOPE"] > 0)
    )
    rs_good = (x["RS20"] > 0) & (x["RS_RATIO_SLOPE"] > 0)
    correction_vol = (
        x["DOWN_VOL20"].notna()
        & x["UP_VOL20"].notna()
        & (x["DOWN_VOL20"] <= x["UP_VOL20"] * 1.10)
    )

    # 1) Early reversal: close to the Iljin 2026-09-04 setup, but requiring RS confirmation.
    x["reversal_early"] = (
        x["HL"]
        & (x["SMA20_SLOPE"] > 0)
        & (x["MA_GAP_20_60"] <= 0.04)
        & x["DIST20"].between(-0.03, 0.03)
        & x["DIST60"].between(-0.03, 0.03)
        & rs_good
    )

    # 2) Trend pullback to the rising 20-day average.
    x["pullback20"] = (
        trend_up
        & x["DIST20"].between(-0.03, 0.03)
        & x["DD20"].between(-0.12, -0.02)
        & (x["RS20"] > 0)
        & correction_vol
        & x["RSI"].between(40, 65)
    )

    # 3) Deeper trend pullback to the rising 60-day average.
    x["pullback60"] = (
        (x["Close"] > x["SMA120"] * 0.97)
        & (x["SMA60"] > x["SMA120"])
        & (x["SMA60_SLOPE"] > 0)
        & x["DIST60"].between(-0.04, 0.04)
        & x["DD60"].between(-0.25, -0.07)
        & (x["RS20"] > 0)
        & x["RSI"].between(35, 60)
    )

    # 4) 20-day breakout with participation.
    x["breakout20"] = (
        (x["Close"] > x["PRIOR_HIGH20"])
        & (x["VOL_RATIO"] >= 1.5)
        & (x["Close"] > x["SMA60"])
        & (x["SMA20"] > x["SMA60"])
        & (x["SMA20_SLOPE"] > 0)
        & (x["RS20"] > 0)
    )

    # 5) Stronger 60-day breakout.
    x["breakout60"] = (
        (x["Close"] > x["PRIOR_HIGH60_X"])
        & (x["VOL_RATIO"] >= 1.5)
        & (x["Close"] > x["SMA120"])
        & (x["SMA20"] > x["SMA60"])
        & (x["SMA60_SLOPE"] > 0)
        & (x["RS20"] > 0)
    )

    # 6) Volatility/MA compression followed by a 20-day breakout.
    range20 = x["PRIOR_HIGH20"] / x["PRIOR_LOW20"] - 1
    x["squeeze_breakout"] = (
        (range20 <= 0.18)
        & (x["MA_GAP_20_60"] <= 0.04)
        & (x["NATR"] <= x["NATR_Q40_60"])
        & (x["Close"] > x["PRIOR_HIGH20"])
        & (x["VOL_RATIO"] >= 1.3)
        & (x["RS20"] > 0)
    )

    # 7) Mature trend setup near the 20-day line, looser than pullback20.
    x["trend_near20"] = (
        trend_up
        & x["DIST20"].between(-0.02, 0.05)
        & (x["RS20"] > 0)
        & x["RSI"].between(45, 70)
    )

    bench_sma20 = bench["Close"].rolling(20).mean()
    bench_sma60 = bench["Close"].rolling(60).mean()
    bench_sma120 = bench["Close"].rolling(120).mean()
    x["MARKET_BULL"] = (bench["Close"] > bench_sma120) & (bench_sma20 > bench_sma60)

    entry = x["Open"].shift(-1).where(x["Open"].shift(-1) > 0)
    bench_entry = bench["Open"].shift(-1).where(bench["Open"].shift(-1) > 0)
    x["ENTRY"] = entry
    for h in HORIZONS:
        stock_ret = x["Close"].shift(-h) / entry - 1
        bench_ret = bench["Close"].shift(-h) / bench_entry - 1
        x[f"RET{h}"] = stock_ret.replace([np.inf, -np.inf], np.nan)
        x[f"EXCESS{h}"] = (stock_ret - bench_ret).replace([np.inf, -np.inf], np.nan)
    x["MFE20"] = (_future_max(x["High"], 20) / entry - 1).replace([np.inf, -np.inf], np.nan)
    x["MAE20"] = (_future_min(x["Low"], 20) / entry - 1).replace([np.inf, -np.inf], np.nan)
    return x


STRATEGIES = [
    "reversal_early",
    "pullback20",
    "pullback60",
    "breakout20",
    "breakout60",
    "squeeze_breakout",
    "trend_near20",
]


def _cooldown_positions(x: pd.DataFrame, mask: pd.Series, cooldown: int) -> list[int]:
    positions = x.loc[mask, "row_pos"].astype(int).to_numpy()
    chosen: list[int] = []
    last = -10000
    for pos in positions:
        if pos - last >= cooldown:
            chosen.append(pos)
            last = pos
    return chosen


def _process(meta: dict, benchmarks: dict[str, pd.DataFrame], start: str, end: str,
             min_adv: float, min_price: float, cooldown: int) -> tuple[pd.DataFrame, str | None]:
    ticker, name, market = str(meta["ticker"]), str(meta["name"]), str(meta["market"])
    try:
        stock = load_price_range(ticker, start, end, warmup_days=550, forward_days=75)
        x = _features(stock, benchmarks[market])
        eligible = (
            (x.index >= pd.Timestamp(start))
            & (x.index <= pd.Timestamp(end))
            & (x["row_pos"] >= DEFAULT_CONFIG.min_history - 1)
            & (x["ADV20"] >= min_adv)
            & (x["Close"] >= min_price)
            & x["RET20"].notna()
            & x["EXCESS20"].notna()
        )
        rows: list[pd.DataFrame] = []
        keep_cols = [
            "row_pos", "Close", "ENTRY", "ADV20", "RSI", "RS20", "RS_RATIO_SLOPE",
            "DIST20", "DIST60", "DD20", "DD60", "VOL_RATIO", "MARKET_BULL",
            "RET5", "RET10", "RET20", "EXCESS5", "EXCESS10", "EXCESS20", "MFE20", "MAE20",
        ]
        for strategy in STRATEGIES:
            positions = _cooldown_positions(x, eligible & x[strategy].fillna(False), cooldown)
            if not positions:
                continue
            part = x[x["row_pos"].isin(positions)][keep_cols].copy()
            part.insert(0, "strategy", strategy)
            part.insert(0, "date", part.index.strftime("%Y-%m-%d"))
            part.insert(0, "market", market)
            part.insert(0, "name", name)
            part.insert(0, "ticker", ticker)
            rows.append(part.reset_index(drop=True))
        return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), None
    except Exception as exc:
        return pd.DataFrame(), f"{ticker},{name},{market},{type(exc).__name__}: {exc}"


def _stats(frame: pd.DataFrame, strategy: str, period: str, regime: str) -> dict:
    if frame.empty:
        return {"strategy": strategy, "period": period, "regime": regime, "signals": 0}
    return {
        "strategy": strategy,
        "period": period,
        "regime": regime,
        "signals": len(frame),
        "avg_5d": frame["RET5"].mean(),
        "avg_10d": frame["RET10"].mean(),
        "avg_20d": frame["RET20"].mean(),
        "median_20d": frame["RET20"].median(),
        "win_rate_20d": (frame["RET20"] > 0).mean(),
        "avg_excess_20d": frame["EXCESS20"].mean(),
        "excess_win_rate_20d": (frame["EXCESS20"] > 0).mean(),
        "avg_mfe_20d": frame["MFE20"].mean(),
        "avg_mae_20d": frame["MAE20"].mean(),
        "payoff_ratio": (
            frame.loc[frame["RET20"] > 0, "RET20"].mean()
            / abs(frame.loc[frame["RET20"] <= 0, "RET20"].mean())
            if (frame["RET20"] > 0).any() and (frame["RET20"] <= 0).any() else np.nan
        ),
    }


def run(start: str, end: str, train_end: str, validation_end: str,
        min_adv: float, min_price: float, cooldown: int, workers: int,
        max_symbols: int | None, output_dir: str) -> None:
    universe = load_krx_universe_frame(max_symbols=max_symbols)
    print(f"Universe: {len(universe):,} currently listed KOSPI/KOSDAQ common stocks")
    print("NOTE: current-listing universe has survivorship bias; FDR delisted listing endpoint is empty in this runtime.")

    benchmarks = {
        "KOSPI": load_benchmark_range("KS11", start, end, warmup_days=550, forward_days=75),
        "KOSDAQ": load_benchmark_range("KQ11", start, end, warmup_days=550, forward_days=75),
    }

    parts: list[pd.DataFrame] = []
    failures: list[str] = []
    records = universe.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process, meta, benchmarks, start, end, min_adv, min_price, cooldown): meta
            for meta in records
        }
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            part, failure = fut.result()
            if not part.empty:
                parts.append(part)
            if failure:
                failures.append(failure)
            if i % 100 == 0 or i == total:
                print(f"Processed {i:,}/{total:,}; failures={len(failures):,}")

    signals = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if signals.empty:
        raise RuntimeError("No signals found")
    signals["date"] = pd.to_datetime(signals["date"])

    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    periods = {
        "train": signals[signals["date"] <= train_end_ts],
        "validation": signals[(signals["date"] > train_end_ts) & (signals["date"] <= validation_end_ts)],
        "test": signals[signals["date"] > validation_end_ts],
    }

    rows: list[dict] = []
    for period_name, period_df in periods.items():
        for strategy in STRATEGIES:
            base = period_df[period_df["strategy"] == strategy]
            rows.append(_stats(base, strategy, period_name, "all"))
            rows.append(_stats(base[base["MARKET_BULL"]], strategy, period_name, "bull"))
    summary = pd.DataFrame(rows)

    # Stability score is intentionally simple and transparent. It rewards positive
    # validation/test excess return, positive test median, and enough observations.
    ranking_rows = []
    for strategy in STRATEGIES:
        for regime in ("all", "bull"):
            v = summary[(summary.strategy == strategy) & (summary.period == "validation") & (summary.regime == regime)]
            t = summary[(summary.strategy == strategy) & (summary.period == "test") & (summary.regime == regime)]
            if v.empty or t.empty or int(t.iloc[0].get("signals", 0)) == 0:
                continue
            vr, tr = v.iloc[0], t.iloc[0]
            score = 0
            score += 2 if vr.get("avg_excess_20d", np.nan) > 0 else 0
            score += 3 if tr.get("avg_excess_20d", np.nan) > 0 else 0
            score += 2 if tr.get("median_20d", np.nan) > 0 else 0
            score += 1 if tr.get("win_rate_20d", 0) >= 0.50 else 0
            score += 1 if tr.get("avg_mae_20d", -1) > -0.10 else 0
            score += 1 if int(tr.get("signals", 0)) >= 200 else 0
            ranking_rows.append({
                "strategy": strategy,
                "regime": regime,
                "stability_score": score,
                "validation_signals": int(vr.get("signals", 0)),
                "validation_excess20": vr.get("avg_excess_20d", np.nan),
                "test_signals": int(tr.get("signals", 0)),
                "test_avg20": tr.get("avg_20d", np.nan),
                "test_median20": tr.get("median_20d", np.nan),
                "test_win20": tr.get("win_rate_20d", np.nan),
                "test_excess20": tr.get("avg_excess_20d", np.nan),
                "test_mfe20": tr.get("avg_mfe_20d", np.nan),
                "test_mae20": tr.get("avg_mae_20d", np.nan),
            })
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["stability_score", "test_excess20", "test_avg20"], ascending=False
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out / "strategy_signals.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "strategy_summary.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(out / "strategy_ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"failure": failures}).to_csv(out / "failures.csv", index=False, encoding="utf-8-sig")

    print("=" * 112)
    print("KR-Chart-Scanner | Traditional pattern comparison")
    print(f"Period      : {start} ~ {end}")
    print(f"Train       : through {train_end}")
    print(f"Validation  : through {validation_end}")
    print(f"Holdout test: after {validation_end}")
    print(f"ADV20 filter: >= {min_adv/1e8:,.0f}억원")
    print(f"Signals     : {len(signals):,}")
    print(f"Failures    : {len(failures):,}")
    print("=" * 112)
    show = ranking.head(14).copy()
    for col in ["validation_excess20", "test_avg20", "test_median20", "test_win20", "test_excess20", "test_mfe20", "test_mae20"]:
        if col in show:
            show[col] = show[col].map(lambda z: "" if pd.isna(z) else f"{z*100:+.2f}%")
    print(show.to_string(index=False))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-01-03")
    p.add_argument("--end", default="2026-09-04")
    p.add_argument("--train-end", default="2023-12-28")
    p.add_argument("--validation-end", default="2024-12-30")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--output-dir", default="output_strategy")
    a = p.parse_args()
    run(a.start, a.end, a.train_end, a.validation_end, a.min_adv, a.min_price,
        a.cooldown, a.workers, None if a.max_symbols <= 0 else a.max_symbols, a.output_dir)
