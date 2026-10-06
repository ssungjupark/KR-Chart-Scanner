from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from data_loader import align_benchmark, load_benchmark_range, load_price_range
from indicators import add_indicators
from scanner import score_reversal
from universe import KOSPI_RESEARCH_PANEL


HORIZONS = (5, 10, 20)


def _forward_return(close: pd.Series, pos: int, horizon: int) -> float:
    target = pos + horizon
    if target >= len(close):
        return float("nan")
    return float(close.iloc[target] / close.iloc[pos] - 1.0)


def _future_excursion(df: pd.DataFrame, pos: int, horizon: int = 20) -> tuple[float, float]:
    end = min(pos + horizon + 1, len(df))
    future = df.iloc[pos + 1 : end]
    if future.empty:
        return float("nan"), float("nan")

    entry = float(df["Close"].iloc[pos])
    mfe = float(future["High"].max() / entry - 1.0)
    mae = float(future["Low"].min() / entry - 1.0)
    return mfe, mae


def _summarize(signals: pd.DataFrame, key: str) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    def win_rate(series: pd.Series) -> float:
        valid = series.dropna()
        return float((valid > 0).mean()) if len(valid) else float("nan")

    summary = (
        signals.groupby(key, observed=True)
        .agg(
            signals=("ticker", "size"),
            avg_5d=("ret_5d", "mean"),
            avg_10d=("ret_10d", "mean"),
            avg_20d=("ret_20d", "mean"),
            median_20d=("ret_20d", "median"),
            win_rate_20d=("ret_20d", win_rate),
            avg_excess_20d=("excess_20d", "mean"),
            avg_mfe_20d=("mfe_20d", "mean"),
            avg_mae_20d=("mae_20d", "mean"),
        )
        .reset_index()
    )
    return summary


def _print_summary(title: str, df: pd.DataFrame) -> None:
    print()
    print(title)
    if df.empty:
        print("No completed signals.")
        return

    show = df.copy()
    pct_cols = [
        "avg_5d",
        "avg_10d",
        "avg_20d",
        "median_20d",
        "win_rate_20d",
        "avg_excess_20d",
        "avg_mfe_20d",
        "avg_mae_20d",
    ]
    for col in pct_cols:
        if col in show:
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x * 100:+.2f}%")
    print(show.to_string(index=False))


def backtest(
    start: str,
    end: str,
    benchmark: str,
    min_score: int,
    cooldown: int,
    tickers: dict[str, str],
    output_dir: str,
) -> pd.DataFrame:
    cfg = DEFAULT_CONFIG
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    print(f"Loading benchmark {benchmark}...")
    bench = load_benchmark_range(benchmark, start, end)

    records: list[dict] = []
    failures: list[tuple[str, str]] = []

    for n, (ticker, name) in enumerate(tickers.items(), start=1):
        print(f"[{n:02d}/{len(tickers):02d}] {ticker} {name}")
        try:
            stock = load_price_range(ticker, start, end)
            benchmark_close = align_benchmark(stock, bench)
            enriched = add_indicators(stock, benchmark_close, cfg)

            evaluation_dates = enriched.index[
                (enriched.index >= start_ts) & (enriched.index <= end_ts)
            ]
            last_signal_pos = -10_000

            for date in evaluation_dates:
                pos = int(stock.index.get_loc(date))
                if pos - last_signal_pos < cooldown:
                    continue

                history = enriched.loc[:date]
                if len(history) < cfg.min_history:
                    continue

                result = score_reversal(ticker, history, cfg)
                if result.score < min_score:
                    continue

                mfe20, mae20 = _future_excursion(stock, pos, 20)
                row = {
                    "ticker": ticker,
                    "name": name,
                    "date": date.strftime("%Y-%m-%d"),
                    "score": result.score,
                    "grade": result.grade,
                    "close": result.metrics["close"],
                    "mfe_20d": mfe20,
                    "mae_20d": mae20,
                }

                for horizon in HORIZONS:
                    stock_ret = _forward_return(stock["Close"], pos, horizon)
                    bench_ret = _forward_return(benchmark_close, pos, horizon)
                    row[f"ret_{horizon}d"] = stock_ret
                    row[f"benchmark_{horizon}d"] = bench_ret
                    row[f"excess_{horizon}d"] = (
                        stock_ret - bench_ret
                        if np.isfinite(stock_ret) and np.isfinite(bench_ret)
                        else float("nan")
                    )

                for check_name, passed in result.checks.items():
                    row[f"check_{check_name}"] = bool(passed)

                records.append(row)
                last_signal_pos = pos

        except Exception as exc:  # Keep panel research running if one symbol fails.
            failures.append((ticker, str(exc)))
            print(f"  FAILED: {exc}")

    signals = pd.DataFrame(records)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    signals_path = out / "backtest_signals.csv"
    signals.to_csv(signals_path, index=False, encoding="utf-8-sig")

    if signals.empty:
        print("No signals met the threshold.")
        return signals

    signals["score_band"] = pd.cut(
        signals["score"],
        bins=[59, 69, 79, 89, 100],
        labels=["60-69", "70-79", "80-89", "90-100"],
        include_lowest=True,
    )

    by_grade = _summarize(signals, "grade")
    by_score = _summarize(signals, "score_band")
    by_grade.to_csv(out / "backtest_summary_by_grade.csv", index=False, encoding="utf-8-sig")
    by_score.to_csv(out / "backtest_summary_by_score.csv", index=False, encoding="utf-8-sig")

    print("=" * 88)
    print("KR-Chart-Scanner | Panel backtest")
    print(f"Period       : {start} ~ {end}")
    print(f"Panel        : {len(tickers)} KOSPI research stocks")
    print(f"Minimum score: {min_score}")
    print(f"Cooldown     : {cooldown} trading sessions")
    print(f"Signals      : {len(signals)}")
    print("=" * 88)
    _print_summary("Performance by grade", by_grade)
    _print_summary("Performance by score band", by_score)

    if failures:
        print()
        print("Data failures")
        for ticker, reason in failures:
            print(f"{ticker}: {reason}")

    print()
    print(f"Saved: {signals_path}")
    return signals


def parse_tickers(raw: str | None) -> dict[str, str]:
    if not raw:
        return KOSPI_RESEARCH_PANEL

    requested = [x.strip() for x in raw.split(",") if x.strip()]
    return {ticker: KOSPI_RESEARCH_PANEL.get(ticker, ticker) for ticker in requested}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest reversal scores without using future data in signal generation."
    )
    parser.add_argument("--start", default="2024-01-02", help="Evaluation start YYYY-MM-DD")
    parser.add_argument("--end", default="2026-09-04", help="Evaluation end YYYY-MM-DD")
    parser.add_argument("--benchmark", default="KS11", help="Benchmark symbol")
    parser.add_argument("--min-score", type=int, default=60, help="Minimum score to record")
    parser.add_argument(
        "--cooldown",
        type=int,
        default=20,
        help="Minimum trading sessions between signals for one stock",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Optional comma-separated ticker list. Default: research panel",
    )
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    backtest(
        start=args.start,
        end=args.end,
        benchmark=args.benchmark,
        min_score=args.min_score,
        cooldown=args.cooldown,
        tickers=parse_tickers(args.tickers),
        output_dir=args.output_dir,
    )
