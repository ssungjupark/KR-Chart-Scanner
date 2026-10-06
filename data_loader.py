from __future__ import annotations

from datetime import timedelta

import FinanceDataReader as fdr
import pandas as pd


def _validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"No price data returned for ticker={ticker}")

    out = df.copy()
    out.index = pd.to_datetime(out.index)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return out.sort_index()


def load_price_data(ticker: str, as_of: str, lookback_days: int = 550) -> pd.DataFrame:
    """Load OHLCV data strictly up to as_of."""
    end = pd.Timestamp(as_of).normalize()
    start = end - timedelta(days=lookback_days)

    df = fdr.DataReader(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    out = _validate_ohlcv(df, ticker)
    return out.loc[out.index <= end]


def _range_dates(start: str, end: str, warmup_days: int, forward_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    return start_ts - timedelta(days=warmup_days), end_ts + timedelta(days=forward_days)


def load_price_range(
    ticker: str,
    start: str,
    end: str,
    warmup_days: int = 550,
    forward_days: int = 60,
) -> pd.DataFrame:
    """Load a research range with warm-up history and forward outcome rows."""
    download_start, download_end = _range_dates(start, end, warmup_days, forward_days)
    df = fdr.DataReader(
        ticker,
        download_start.strftime("%Y-%m-%d"),
        download_end.strftime("%Y-%m-%d"),
    )
    return _validate_ohlcv(df, ticker)


def load_delisted_price_range(
    ticker: str,
    start: str,
    end: str,
    warmup_days: int = 550,
    forward_days: int = 60,
) -> pd.DataFrame:
    """Load OHLCV for a delisted KRX stock through FinanceDataReader."""
    download_start, download_end = _range_dates(start, end, warmup_days, forward_days)
    symbol = f"KRX-DELISTING:{ticker}"
    df = fdr.DataReader(
        symbol,
        download_start.strftime("%Y-%m-%d"),
        download_end.strftime("%Y-%m-%d"),
    )
    return _validate_ohlcv(df, ticker)


def load_benchmark_data(symbol: str, as_of: str, lookback_days: int = 550) -> pd.DataFrame:
    """Load benchmark index data, e.g. KS11 for KOSPI."""
    return load_price_data(symbol, as_of, lookback_days)


def load_benchmark_range(
    symbol: str,
    start: str,
    end: str,
    warmup_days: int = 550,
    forward_days: int = 60,
) -> pd.DataFrame:
    """Load benchmark history for a panel backtest."""
    return load_price_range(symbol, start, end, warmup_days, forward_days)


def align_benchmark(stock: pd.DataFrame, benchmark: pd.DataFrame) -> pd.Series:
    """Align benchmark close to stock trading dates without future filling."""
    return benchmark["Close"].reindex(stock.index).ffill()


def align_benchmark_ohlc(stock: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Align benchmark OHLC to stock dates for executable next-open outcome tests."""
    cols = [c for c in ("Open", "High", "Low", "Close") if c in benchmark.columns]
    return benchmark[cols].reindex(stock.index).ffill()
