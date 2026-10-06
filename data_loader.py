from __future__ import annotations

from datetime import timedelta

import FinanceDataReader as fdr
import pandas as pd


def load_price_data(ticker: str, as_of: str, lookback_days: int = 550) -> pd.DataFrame:
    """Load OHLCV data strictly up to as_of.

    The dataframe is truncated again after download so historical tests cannot
    accidentally see rows after the evaluation date.
    """
    end = pd.Timestamp(as_of).normalize()
    start = end - timedelta(days=lookback_days)

    df = fdr.DataReader(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        raise ValueError(f"No price data returned for ticker={ticker}")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.loc[df.index <= end]

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df.sort_index()


def load_benchmark_data(symbol: str, as_of: str, lookback_days: int = 550) -> pd.DataFrame:
    """Load benchmark index data, e.g. KS11 for KOSPI."""
    return load_price_data(symbol, as_of, lookback_days)


def align_benchmark(stock: pd.DataFrame, benchmark: pd.DataFrame) -> pd.Series:
    """Align benchmark close to stock trading dates without future filling."""
    bench_close = benchmark["Close"].reindex(stock.index).ffill()
    return bench_close
