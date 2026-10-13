from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from config import DEFAULT_CONFIG
from indicators import add_indicators
from universe import load_krx_universe_frame


MODEL_ID = "V15_FORWARD_OOS_10PCT"
OOS_START_SIGNAL_DATE = pd.Timestamp("2026-10-14")
MEGA_TICKERS = {"005930", "000660"}
MIN_ADV = 2_000_000_000.0
MIN_PRICE = 1_000.0
COOLDOWN_SESSIONS = 20
HOLD_SESSIONS = 20
SIGNAL_WEIGHT = 0.10
ROUND_TRIP_COST = 0.004
SIDE_COST = ROUND_TRIP_COST / 2.0
MAX_POSITIONS = 10
PERCENTILE_WINDOW = 756
PERCENTILE_MIN = 252
PRICE_HISTORY_YEARS = 5

SIGNAL_SPEC = {
    "model": MODEL_ID,
    "oos_start_signal_date": str(OOS_START_SIGNAL_DATE.date()),
    "stock": {
        "min_history": 150,
        "min_adv20_krw": MIN_ADV,
        "min_raw_price": MIN_PRICE,
        "setup": "Close>SMA120; SMA20>SMA60>SMA120; SMA20/60 slope>0; dist20 [-3%,+4%]; dd60 [-16%,-2%]; RSI [40,72]; down-vol20 <= 1.1*up-vol20",
        "trigger": "Close > prior 3-session high; Close>SMA5; SMA5 rising",
        "rs": "RS20>0; RS ratio slope>0; cross-sectional RS20>=50th, RS60>=60th, RS120>=70th percentile",
        "ticker_cooldown_sessions": COOLDOWN_SESSIONS,
        "excluded_direct_trades": sorted(MEGA_TICKERS),
    },
    "market": {
        "KOSPI_above_SMA120": True,
        "KOSPI_RET60_trailing756_percentile_min": 0.65,
        "LIQ_BREADTH60_trailing756_percentile_band": [0.30, 0.70],
        "rolling_percentile_min_periods": PERCENTILE_MIN,
    },
    "portfolio": {
        "core": "KOSPI",
        "signal_weight": SIGNAL_WEIGHT,
        "max_positions": MAX_POSITIONS,
        "strategic_cash": 0.0,
        "leverage": False,
        "entry": "next observed stock-session open after signal",
        "exit": "20th observed stock-session close after signal",
        "round_trip_cost": ROUND_TRIP_COST,
    },
}
SPEC_HASH = hashlib.sha256(json.dumps(SIGNAL_SPEC, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class DataBundle:
    panel: pd.DataFrame
    daily: pd.DataFrame
    prices: dict[str, pd.DataFrame]
    benchmark: pd.DataFrame
    universe: pd.DataFrame
    marcap_commit: str
    liquidity_fallback_rows: int


def _ensure_dirs(state_dir: Path, cache_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "prices").mkdir(parents=True, exist_ok=True)
    (cache_dir / "marcap").mkdir(parents=True, exist_ok=True)


def fetch_naver_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch observed daily bars from NAVER's date-range endpoint."""
    r = requests.get(
        "https://api.finance.naver.com/siseJson.naver",
        params={
            "symbol": symbol,
            "requestType": 1,
            "startTime": start.replace("-", ""),
            "endTime": end.replace("-", ""),
            "timeframe": "day",
        },
        timeout=35,
    )
    r.raise_for_status()
    records = ast.literal_eval(r.text.strip())
    if not isinstance(records, list) or len(records) < 2:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    header = [str(x).strip() for x in records[0]]
    x = pd.DataFrame(records[1:], columns=header)
    date_col = next((c for c in x.columns if c.lower() in {"날짜", "date"}), x.columns[0])
    rename = {}
    aliases = {
        "시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume",
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    }
    for c in x.columns:
        key = str(c).strip().lower()
        if key in aliases:
            rename[c] = aliases[key]
        elif str(c).strip() in aliases:
            rename[c] = aliases[str(c).strip()]
    x = x.rename(columns=rename)
    x[date_col] = pd.to_datetime(x[date_col].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    x = x.dropna(subset=[date_col]).set_index(date_col).sort_index()
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"NAVER missing columns for {symbol}: {missing}")
    x = x[required].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    return x


def sync_price(symbol: str, start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / "prices" / f"{symbol}.csv.gz"
    old = pd.DataFrame()
    fetch_start = start
    if path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        if not old.empty:
            fetch_start = max(start, old.index.max() - pd.Timedelta(days=10))
    fresh = fetch_naver_history(symbol, str(fetch_start.date()), str(end.date()))
    if old.empty:
        out = fresh
    else:
        out = pd.concat([old.loc[old.index < fetch_start], fresh]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
    out = out.loc[(out.index >= start) & (out.index <= end)]
    if out.empty:
        raise ValueError(f"No price history for {symbol}")
    out.to_csv(path, compression="gzip", index_label="Date")
    return out


def sync_all_prices(universe: pd.DataFrame, as_of: pd.Timestamp, cache_dir: Path, workers: int = 12) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    start = as_of - pd.DateOffset(years=PRICE_HISTORY_YEARS)
    symbols = ["KOSPI", *universe["ticker"].astype(str).tolist()]
    data: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(sync_price, s, start, as_of, cache_dir): s for s in symbols}
        for n, fut in enumerate(as_completed(futs), 1):
            s = futs[fut]
            try:
                data[s] = fut.result()
            except Exception as exc:
                errors.append(f"{s}: {type(exc).__name__}: {exc}")
            if n % 100 == 0 or n == len(futs):
                print(f"forward prices {n}/{len(futs)}, errors={len(errors)}", flush=True)
    if "KOSPI" not in data:
        raise RuntimeError("KOSPI benchmark price is unavailable")
    # A few unavailable stocks are allowed, but broad data failure is not.
    if len(errors) > max(10, int(len(symbols) * 0.03)):
        raise RuntimeError(f"Too many price failures: {len(errors)}/{len(symbols)}")
    if errors:
        print("price warnings:\n" + "\n".join(errors[:20]), flush=True)
    benchmark = data.pop("KOSPI")
    return data, benchmark


def _latest_marcap_commit() -> str:
    r = requests.get("https://api.github.com/repos/FinanceData/marcap/commits/master", timeout=30)
    r.raise_for_status()
    return str(r.json()["sha"])


def load_marcap(as_of: pd.Timestamp, cache_dir: Path, codes: set[str]) -> tuple[pd.DataFrame, str]:
    """Load raw KRX close/volume/amount/membership used by V15 eligibility.

    Past-year files are cached as a point-in-time snapshot. The current year is
    refreshed from the latest FinanceData/marcap commit each run. The commit is
    persisted in the signal ledger for source provenance.
    """
    commit = _latest_marcap_commit()
    start_year = (as_of - pd.DateOffset(years=PRICE_HISTORY_YEARS)).year
    years = range(start_year, as_of.year + 1)
    parts = []
    mdir = cache_dir / "marcap"
    for year in years:
        path = mdir / f"marcap-{year}.parquet"
        refresh = year == as_of.year or not path.exists()
        if refresh:
            url = f"https://raw.githubusercontent.com/FinanceData/marcap/{commit}/data/marcap-{year}.parquet"
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            x = pd.read_parquet(io.BytesIO(r.content))
            if "Date" not in x.columns:
                x = x.reset_index()
            keep = [c for c in ["Date", "Code", "Close", "Volume", "Amount", "Market"] if c in x.columns]
            x = x[keep].copy()
            x["Code"] = x["Code"].astype(str).str.zfill(6)
            x = x[x["Code"].isin(codes)]
            x.to_parquet(path, index=False)
        else:
            x = pd.read_parquet(path)
        parts.append(x)
    raw = pd.concat(parts, ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["Code"] = raw["Code"].astype(str).str.zfill(6)
    raw = raw[raw["Date"] <= as_of]
    return raw, commit


def add_v15_stock_features(stock: pd.DataFrame, benchmark: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    inputs = stock.copy()
    aligned = raw.reindex(stock.index)
    # Match the long-history research accounting: actual KRX amount and raw close
    # define liquidity/price eligibility; adjusted volume is rescaled so volume
    # comparisons remain economically consistent with the adjusted price series.
    inputs["Amount"] = aligned["Amount"]
    inputs["RawClose"] = aligned["Close"]
    ratio = aligned["Close"] / stock["Close"].replace(0, np.nan)
    inputs["Volume"] = aligned["Volume"] * ratio

    bc = benchmark["Close"].reindex(stock.index).ffill()
    x = add_indicators(inputs, bc, DEFAULT_CONFIG)
    x["row_pos"] = np.arange(len(x))
    x["ADV20"] = inputs["Amount"].rolling(20).mean()
    x["filter_price"] = inputs["RawClose"]
    for h in (60, 120):
        x[f"RS{h}"] = x["Close"].pct_change(h) - bc.pct_change(h)

    dist20 = x["Close"] / x["SMA20"] - 1.0
    dd60 = x["Close"] / x["High"].shift(1).rolling(60).max() - 1.0
    correction = x["DOWN_VOL20"].notna() & x["UP_VOL20"].notna() & (x["DOWN_VOL20"] <= x["UP_VOL20"] * 1.1)
    x["setup"] = (
        (x["Close"] > x["SMA120"])
        & (x["SMA20"] > x["SMA60"])
        & (x["SMA60"] > x["SMA120"])
        & (x["SMA20_SLOPE"] > 0)
        & (x["SMA60_SLOPE"] > 0)
        & dist20.between(-0.03, 0.04, inclusive="both")
        & dd60.between(-0.16, -0.02, inclusive="both")
        & x["RSI"].between(40, 72, inclusive="both")
        & correction
    )
    x["trigger"] = (
        (x["Close"] > x["High"].shift(1).rolling(3).max())
        & (x["Close"] > x["SMA5"])
        & (x["SMA5"] > x["SMA5"].shift(1))
    )
    bma120 = bc.rolling(120).mean()
    x["KOSPI_RET60"] = bc / bc.shift(60) - 1.0
    x["KOSPI_ABOVE120"] = bc > bma120
    x["ABOVE60"] = x["Close"] > x["SMA60"]
    return x


def rolling_last_percentile(s: pd.Series, window: int = PERCENTILE_WINDOW, min_periods: int = PERCENTILE_MIN) -> pd.Series:
    def pct(a: np.ndarray) -> float:
        if len(a) == 0 or not np.isfinite(a[-1]):
            return np.nan
        finite = a[np.isfinite(a)]
        if len(finite) == 0:
            return np.nan
        return float(np.mean(finite <= a[-1]))
    return s.rolling(window, min_periods=min_periods).apply(pct, raw=True)


def build_signal_bundle(as_of: pd.Timestamp, cache_dir: Path, workers: int = 12) -> DataBundle:
    universe = load_krx_universe_frame(markets=("KOSPI",)).copy()
    universe["ticker"] = universe["ticker"].astype(str).str.zfill(6)
    prices, benchmark = sync_all_prices(universe, as_of, cache_dir, workers=workers)
    raw_all, marcap_commit = load_marcap(as_of, cache_dir, set(universe["ticker"]))
    raw_all["Market"] = raw_all["Market"].astype(str)
    raw_by_code = {t: g.set_index("Date").sort_index() for t, g in raw_all.groupby("Code")}

    pieces: list[pd.DataFrame] = []
    fallback_rows = 0
    for n, meta in enumerate(universe.to_dict("records"), 1):
        ticker = meta["ticker"]
        stock = prices.get(ticker)
        if stock is None or stock.empty:
            continue
        raw = raw_by_code.get(ticker, pd.DataFrame(columns=["Close", "Volume", "Amount", "Market"])).reindex(stock.index)

        # If marcap has not published the latest observed session yet, use the
        # same day's unadjusted NAVER close*volume only for those trailing rows.
        # This is explicitly counted and written to the ledger/summary.
        trailing_missing = raw["Amount"].isna() & (stock.index >= (as_of - pd.Timedelta(days=7)))
        if trailing_missing.any():
            raw.loc[trailing_missing, "Close"] = stock.loc[trailing_missing, "Close"]
            raw.loc[trailing_missing, "Volume"] = stock.loc[trailing_missing, "Volume"]
            raw.loc[trailing_missing, "Amount"] = stock.loc[trailing_missing, "Close"] * stock.loc[trailing_missing, "Volume"]
            raw.loc[trailing_missing, "Market"] = "KOSPI"
            fallback_rows += int(trailing_missing.sum())

        x = add_v15_stock_features(stock, benchmark, raw)
        valid = stock[["Open", "High", "Low", "Close", "Volume"]].gt(0).all(axis=1)
        eligible = (
            (x["row_pos"] >= DEFAULT_CONFIG.min_history - 1)
            & (x["ADV20"] >= MIN_ADV)
            & (x["filter_price"] >= MIN_PRICE)
            & x[["RS20", "RS60", "RS120", "RS_RATIO_SLOPE"]].notna().all(axis=1)
            & raw["Market"].eq("KOSPI")
            & valid
            & valid.shift(1).rolling(3).sum().eq(3)
        )
        fields = [
            "row_pos", "Close", "ADV20", "RS20", "RS60", "RS120", "RS_RATIO_SLOPE",
            "setup", "trigger", "KOSPI_RET60", "KOSPI_ABOVE120", "ABOVE60",
        ]
        p = x.loc[eligible, fields].copy()
        if p.empty:
            continue
        p["date"] = p.index
        p["ticker"] = ticker
        p["name"] = meta["name"]
        p["is_mega"] = ticker in MEGA_TICKERS
        pieces.append(p.reset_index(drop=True))
        if n % 100 == 0:
            print(f"forward features {n}/{len(universe)}", flush=True)

    if not pieces:
        raise RuntimeError("No eligible KOSPI panel rows")
    panel = pd.concat(pieces, ignore_index=True)
    for h in (20, 60, 120):
        panel[f"RS{h}_PCTL"] = panel.groupby("date")[f"RS{h}"].rank(pct=True)

    daily = panel.groupby("date")["ABOVE60"].mean().to_frame("LIQ_BREADTH60")
    daily["KOSPI_RET60"] = panel.groupby("date")["KOSPI_RET60"].first()
    daily["KOSPI_ABOVE120"] = panel.groupby("date")["KOSPI_ABOVE120"].first()
    daily = daily.sort_index()
    daily["KOSPI_RET60_PCTL756"] = rolling_last_percentile(daily["KOSPI_RET60"])
    daily["LIQ_BREADTH60_PCTL756"] = rolling_last_percentile(daily["LIQ_BREADTH60"])
    panel = panel.merge(daily, left_on="date", right_index=True, how="left", suffixes=("", "_STATE"))
    # Prefer state fields when merge duplicates names.
    if "KOSPI_RET60_STATE" in panel:
        panel["KOSPI_RET60"] = panel["KOSPI_RET60_STATE"]
    if "KOSPI_ABOVE120_STATE" in panel:
        panel["KOSPI_ABOVE120"] = panel["KOSPI_ABOVE120_STATE"]
    return DataBundle(panel, daily, prices, benchmark, universe, marcap_commit, fallback_rows)


def v15_mask(panel: pd.DataFrame) -> pd.Series:
    base = (
        panel["setup"].astype(bool)
        & panel["trigger"].astype(bool)
        & (panel["RS20"] > 0)
        & (panel["RS_RATIO_SLOPE"] > 0)
        & (panel["RS20_PCTL"] >= 0.50)
        & (panel["RS60_PCTL"] >= 0.60)
        & (panel["RS120_PCTL"] >= 0.70)
    )
    regime = (
        panel["KOSPI_ABOVE120"].astype(bool)
        & (panel["KOSPI_RET60_PCTL756"] >= 0.65)
        & panel["LIQ_BREADTH60_PCTL756"].between(0.30, 0.70, inclusive="both")
    )
    return (base & regime & (~panel["is_mega"])).fillna(False)


def apply_cooldown(frame: pd.DataFrame, sessions: int = COOLDOWN_SESSIONS) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    keep: list[int] = []
    for _, g in frame.sort_values(["ticker", "row_pos"]).groupby("ticker", sort=False):
        last = -100_000
        for idx, pos in zip(g.index.to_numpy(), g["row_pos"].astype(int).to_numpy()):
            if pos - last >= sessions:
                keep.append(int(idx))
                last = pos
    return frame.loc[keep].sort_values(["date", "ticker"]).copy()


def selected_signals_for_date(bundle: DataBundle, signal_date: pd.Timestamp) -> pd.DataFrame:
    all_selected = apply_cooldown(bundle.panel.loc[v15_mask(bundle.panel)].copy())
    out = all_selected[all_selected["date"].eq(signal_date)].copy()
    cols = [
        "date", "ticker", "name", "Close", "ADV20", "RS20", "RS60", "RS120",
        "RS20_PCTL", "RS60_PCTL", "RS120_PCTL", "KOSPI_RET60",
        "KOSPI_RET60_PCTL756", "LIQ_BREADTH60", "LIQ_BREADTH60_PCTL756",
    ]
    return out[[c for c in cols if c in out.columns]].sort_values("ticker")


def _empty_state() -> dict:
    return {
        "model_id": MODEL_ID,
        "spec_hash": SPEC_HASH,
        "oos_start_signal_date": str(OOS_START_SIGNAL_DATE.date()),
        "initialized": False,
        "initial_session": None,
        "last_processed_market_date": None,
        "last_signal_scan_date": None,
        "index_units": None,
        "benchmark_units": None,
        "active_positions": [],
        "pending_signals": [],
        "next_signal_id": 1,
    }


def load_state(path: Path) -> dict:
    if not path.exists():
        return _empty_state()
    state = json.loads(path.read_text())
    if state.get("model_id") != MODEL_ID or state.get("spec_hash") != SPEC_HASH:
        raise RuntimeError("Forward paper state model/spec hash mismatch; refusing to mutate frozen OOS ledger")
    return state


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists() and path.stat().st_size:
        return pd.read_csv(path, dtype={"ticker": str, "signal_id": str})
    return pd.DataFrame(columns=columns)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def next_stock_session(stock: pd.DataFrame, after: pd.Timestamp) -> pd.Timestamp | None:
    dates = stock.index[stock.index > after]
    return pd.Timestamp(dates[0]) if len(dates) else None


def exit_stock_session(stock: pd.DataFrame, signal_date: pd.Timestamp, hold: int = HOLD_SESSIONS) -> pd.Timestamp | None:
    dates = stock.index[stock.index > signal_date]
    return pd.Timestamp(dates[hold - 1]) if len(dates) >= hold else None


def update_signal_status(signals: pd.DataFrame, signal_id: str, status: str, **values) -> pd.DataFrame:
    mask = signals["signal_id"].astype(str).eq(str(signal_id))
    signals.loc[mask, "status"] = status
    for k, v in values.items():
        if k not in signals.columns:
            signals[k] = np.nan
        signals.loc[mask, k] = v
    return signals


def _portfolio_value(state: dict, d: pd.Timestamp, bundle: DataBundle, field: str) -> tuple[float, float, float]:
    b = bundle.benchmark
    bp = float(b.loc[d, field])
    index_value = float(state["index_units"]) * bp
    stock_value = 0.0
    for p in state["active_positions"]:
        stock = bundle.prices.get(p["ticker"])
        if stock is None:
            raise RuntimeError(f"Missing active-position price {p['ticker']}")
        hist = stock.loc[:d, field]
        if hist.empty:
            raise RuntimeError(f"No mark for active position {p['ticker']} on {d.date()}")
        stock_value += float(p["shares"]) * float(hist.iloc[-1])
    return index_value + stock_value, index_value, stock_value


def process_market_day(
    d: pd.Timestamp,
    state: dict,
    bundle: DataBundle,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    b = bundle.benchmark
    bo = float(b.loc[d, "Open"])
    bc = float(b.loc[d, "Close"])

    # Fill pre-existing signals at the first observed stock-session open after signal.
    due = []
    still_pending = []
    for p in state["pending_signals"]:
        stock = bundle.prices.get(p["ticker"])
        entry_day = next_stock_session(stock, pd.Timestamp(p["signal_date"])) if stock is not None else None
        if entry_day is not None and entry_day == d:
            due.append(p)
        else:
            still_pending.append(p)
    state["pending_signals"] = still_pending

    equity_open, _, _ = _portfolio_value(state, d, bundle, "Open")
    for p in sorted(due, key=lambda x: x["ticker"]):
        target = equity_open * SIGNAL_WEIGHT
        available_net_index = float(state["index_units"]) * bo * (1.0 - SIDE_COST)
        signal_id = str(p["signal_id"])
        if len(state["active_positions"]) >= MAX_POSITIONS or available_net_index + 1e-12 < target:
            signals = update_signal_status(signals, signal_id, "skipped_capacity", decision_date=str(d.date()))
            continue
        stock = bundle.prices[p["ticker"]]
        ep = float(stock.loc[d, "Open"])
        if not np.isfinite(ep) or ep <= 0:
            signals = update_signal_status(signals, signal_id, "skipped_bad_open", decision_date=str(d.date()))
            continue
        state["index_units"] -= target / (bo * (1.0 - SIDE_COST))
        if state["index_units"] < -1e-10:
            raise AssertionError("KOSPI core went negative")
        state["index_units"] = max(float(state["index_units"]), 0.0)
        shares = target / (ep * (1.0 + SIDE_COST))
        exit_day = exit_stock_session(stock, pd.Timestamp(p["signal_date"]))
        pos = {
            "signal_id": signal_id,
            "ticker": p["ticker"],
            "name": p["name"],
            "signal_date": p["signal_date"],
            "entry_date": str(d.date()),
            "entry_price": ep,
            "shares": float(shares),
            "target_exit_date": str(exit_day.date()) if exit_day is not None else None,
            "benchmark_entry_open": bo,
        }
        state["active_positions"].append(pos)
        signals = update_signal_status(
            signals, signal_id, "active", decision_date=str(d.date()), entry_date=str(d.date()), entry_price=ep
        )

    # Exit on the 20th observed stock-session close after signal.
    active_next = []
    for p in state["active_positions"]:
        stock = bundle.prices[p["ticker"]]
        exit_day = exit_stock_session(stock, pd.Timestamp(p["signal_date"]))
        if exit_day is not None and exit_day == d:
            xp = float(stock.loc[d, "Close"])
            proceeds = float(p["shares"]) * xp * (1.0 - SIDE_COST)
            state["index_units"] += proceeds / (bc * (1.0 + SIDE_COST))
            net_ret = (xp * (1.0 - SIDE_COST)) / (float(p["entry_price"]) * (1.0 + SIDE_COST)) - 1.0
            bench_ret = bc / float(p["benchmark_entry_open"]) - 1.0
            row = {
                "signal_id": p["signal_id"], "ticker": p["ticker"], "name": p["name"],
                "signal_date": p["signal_date"], "entry_date": p["entry_date"], "entry_price": p["entry_price"],
                "exit_date": str(d.date()), "exit_price": xp, "hold_rule": "signal+20 observed stock sessions",
                "round_trip_cost": ROUND_TRIP_COST, "net_return": net_ret,
                "benchmark_return": bench_ret, "excess_return": net_ret - bench_ret,
            }
            trades = pd.concat([trades, pd.DataFrame([row])], ignore_index=True)
            signals = update_signal_status(
                signals, str(p["signal_id"]), "closed", exit_date=str(d.date()), exit_price=xp,
                net_return=net_ret, benchmark_return=bench_ret, excess_return=net_ret - bench_ret,
            )
        else:
            if exit_day is not None:
                p["target_exit_date"] = str(exit_day.date())
            active_next.append(p)
    state["active_positions"] = active_next

    equity_close, index_value, stock_value = _portfolio_value(state, d, bundle, "Close")
    kospi_nav = float(state["benchmark_units"]) * bc
    total = index_value + stock_value
    if total <= 0 or abs((index_value / total + stock_value / total) - 1.0) > 1e-9:
        raise AssertionError("Portfolio exposure accounting failed")
    daily_row = {
        "date": str(d.date()), "overlay_nav": equity_close, "kospi_nav": kospi_nav,
        "relative_nav": equity_close / kospi_nav, "excess_since_start": equity_close / kospi_nav - 1.0,
        "index_exposure": index_value / total, "signal_stock_exposure": stock_value / total,
        "total_equity_exposure": 1.0, "active_positions": len(state["active_positions"]),
    }
    return state, signals, trades, daily_row


def append_new_signals(
    selected: pd.DataFrame,
    signal_date: pd.Timestamp,
    state: dict,
    signals: pd.DataFrame,
    bundle: DataBundle,
) -> tuple[dict, pd.DataFrame]:
    existing = set(signals.get("signal_key", pd.Series(dtype=str)).astype(str))
    for _, r in selected.sort_values("ticker").iterrows():
        key = f"{signal_date.date()}|{r.ticker}"
        if key in existing:
            continue
        sid = f"V15-{int(state['next_signal_id']):06d}"
        state["next_signal_id"] = int(state["next_signal_id"]) + 1
        row = {
            "signal_id": sid, "signal_key": key, "signal_date": str(signal_date.date()),
            "ticker": str(r.ticker).zfill(6), "name": r["name"], "status": "pending_entry",
            "signal_close": float(r.Close), "adv20": float(r.ADV20),
            "rs20": float(r.RS20), "rs60": float(r.RS60), "rs120": float(r.RS120),
            "rs20_pctl": float(r.RS20_PCTL), "rs60_pctl": float(r.RS60_PCTL), "rs120_pctl": float(r.RS120_PCTL),
            "kospi_ret60": float(r.KOSPI_RET60), "kospi_ret60_pctl756": float(r.KOSPI_RET60_PCTL756),
            "liq_breadth60": float(r.LIQ_BREADTH60), "liq_breadth60_pctl756": float(r.LIQ_BREADTH60_PCTL756),
            "signal_weight": SIGNAL_WEIGHT, "round_trip_cost": ROUND_TRIP_COST,
            "spec_hash": SPEC_HASH, "marcap_commit": bundle.marcap_commit,
            "liquidity_fallback_rows_run": bundle.liquidity_fallback_rows,
        }
        signals = pd.concat([signals, pd.DataFrame([row])], ignore_index=True)
        state["pending_signals"].append({
            "signal_id": sid, "ticker": str(r.ticker).zfill(6), "name": r["name"],
            "signal_date": str(signal_date.date()),
        })
        existing.add(key)
    return state, signals


def write_latest(state_dir: Path, state: dict, signals: pd.DataFrame, trades: pd.DataFrame, portfolio: pd.DataFrame, market_date: pd.Timestamp | None, bundle: DataBundle | None) -> None:
    active = len(state.get("active_positions", []))
    pending = len(state.get("pending_signals", []))
    closed = int((signals.get("status", pd.Series(dtype=str)) == "closed").sum()) if len(signals) else 0
    nav_line = "아직 시작 전"
    if len(portfolio):
        last = portfolio.iloc[-1]
        nav_line = f"Overlay NAV {float(last.overlay_nav):.4f} / KOSPI NAV {float(last.kospi_nav):.4f} / 상대 {float(last.excess_since_start)*100:+.2f}%"
    fallback = bundle.liquidity_fallback_rows if bundle is not None else 0
    text = f"""# V15 Forward OOS Paper\n\n- 모델: `{MODEL_ID}`\n- frozen spec hash: `{SPEC_HASH}`\n- OOS 신호 시작일: **{OOS_START_SIGNAL_DATE.date()}**\n- 최근 시장일: **{market_date.date() if market_date is not None else 'N/A'}**\n- 마지막 신호 스캔일: **{state.get('last_signal_scan_date')}**\n- 활성 포지션: **{active}**\n- 진입 대기: **{pending}**\n- 완료 거래: **{closed}**\n- {nav_line}\n- 이번 실행 marcap trailing fallback rows: **{fallback}**\n\n규칙은 V15 10% fully-invested overlay로 동결되어 있습니다. 과거 신호 backfill 및 threshold 재튜닝은 금지합니다.\n"""
    (state_dir / "latest.md").write_text(text, encoding="utf-8")


def run(as_of: pd.Timestamp, state_dir: Path, cache_dir: Path, workers: int = 12) -> None:
    _ensure_dirs(state_dir, cache_dir)
    state_path = state_dir / "state.json"
    state = load_state(state_path)
    signals = _read_csv(state_dir / "signals.csv", ["signal_id", "signal_key", "signal_date", "ticker", "name", "status"])
    trades = _read_csv(state_dir / "trades.csv", ["signal_id", "ticker", "name", "signal_date", "entry_date", "exit_date"])
    portfolio = _read_csv(state_dir / "portfolio.csv", ["date", "overlay_nav", "kospi_nav"])

    bundle = build_signal_bundle(as_of, cache_dir, workers=workers)
    benchmark = bundle.benchmark.loc[:as_of]
    if benchmark.empty:
        raise RuntimeError("Benchmark has no completed market session")
    market_date = pd.Timestamp(benchmark.index[-1])

    start_sessions = benchmark.index[benchmark.index >= OOS_START_SIGNAL_DATE]
    if not state["initialized"] and len(start_sessions):
        first = pd.Timestamp(start_sessions[0])
        op = float(benchmark.loc[first, "Open"])
        state["initialized"] = True
        state["initial_session"] = str(first.date())
        state["index_units"] = 1.0 / (op * (1.0 + SIDE_COST))
        state["benchmark_units"] = 1.0 / (op * (1.0 + SIDE_COST))
        # Process from the first OOS session close; signals are generated only after it.
        state["last_processed_market_date"] = None

    if state["initialized"]:
        first = pd.Timestamp(state["initial_session"])
        last = pd.Timestamp(state["last_processed_market_date"]) if state.get("last_processed_market_date") else first - pd.Timedelta(days=1)
        dates = benchmark.index[(benchmark.index >= first) & (benchmark.index > last) & (benchmark.index <= market_date)]
        rows = []
        for d in dates:
            state, signals, trades, row = process_market_day(pd.Timestamp(d), state, bundle, signals, trades)
            rows.append(row)
            state["last_processed_market_date"] = str(pd.Timestamp(d).date())
        if rows:
            portfolio = pd.concat([portfolio, pd.DataFrame(rows)], ignore_index=True)
            portfolio = portfolio.drop_duplicates("date", keep="last").sort_values("date")

        # Do not manufacture missed historical signals. Only scan the latest
        # completed market date. Weekend/holiday runs may safely scan Friday
        # because no later market open has occurred.
        last_scan = pd.Timestamp(state["last_signal_scan_date"]) if state.get("last_signal_scan_date") else None
        if market_date >= OOS_START_SIGNAL_DATE and (last_scan is None or market_date > last_scan):
            selected = selected_signals_for_date(bundle, market_date)
            state, signals = append_new_signals(selected, market_date, state, signals, bundle)
            state["last_signal_scan_date"] = str(market_date.date())

    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(signals, state_dir / "signals.csv")
    _write_csv(trades, state_dir / "trades.csv")
    _write_csv(portfolio, state_dir / "portfolio.csv")
    write_latest(state_dir, state, signals, trades, portfolio, market_date, bundle)
    print((state_dir / "latest.md").read_text(encoding="utf-8"))


def validate_config() -> None:
    assert OOS_START_SIGNAL_DATE == pd.Timestamp("2026-10-14")
    assert SIGNAL_WEIGHT == 0.10
    assert MAX_POSITIONS == 10
    assert HOLD_SESSIONS == 20
    assert ROUND_TRIP_COST == 0.004
    assert MEGA_TICKERS == {"005930", "000660"}
    assert SIGNAL_SPEC["market"]["KOSPI_RET60_trailing756_percentile_min"] == 0.65
    assert SIGNAL_SPEC["market"]["LIQ_BREADTH60_trailing756_percentile_band"] == [0.30, 0.70]
    print(f"{MODEL_ID} spec_hash={SPEC_HASH}")


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen V15 forward out-of-sample paper tracker")
    p.add_argument("--as-of", default=pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d"))
    p.add_argument("--state-dir", default="paper/v15")
    p.add_argument("--cache-dir", default="paper_cache/v15")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--validate-config", action="store_true")
    args = p.parse_args()
    if args.validate_config:
        validate_config()
        return
    run(pd.Timestamp(args.as_of).normalize(), Path(args.state_dir), Path(args.cache_dir), workers=args.workers)


if __name__ == "__main__":
    main()
