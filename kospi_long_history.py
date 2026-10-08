"""Research-only KOSPI retrospective, frozen v6 rules, no model selection."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from config import DEFAULT_CONFIG
from indicators import add_indicators
import robustness_v7 as v7

HORIZONS = (5, 10, 20, 40, 60)
VARIANTS = ("v6", "rs_only", "rs_trigger", "rs_setup")
COLS = ["Date", "Open", "High", "Low", "Close", "Volume", "foreign"]


def fetch_history(symbol, start, end):
    """NAVER's date-range endpoint avoids the chart endpoint's 3000-bar cap."""
    r = requests.get("https://api.finance.naver.com/siseJson.naver", params={
        "symbol": symbol, "requestType": 1, "startTime": start.replace("-", ""),
        "endTime": end.replace("-", ""), "timeframe": "day"}, timeout=35)
    r.raise_for_status()
    records = ast.literal_eval(r.text.strip())
    x = pd.DataFrame(records[1:])
    x.columns = COLS[:len(x.columns)]
    if x.empty:
        raise ValueError("No NAVER date-range history")
    x["Date"] = pd.to_datetime(x["Date"], format="%Y%m%d")
    x = x.set_index("Date").sort_index()
    x = x[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric)
    if x.index.has_duplicates or (x["Close"] <= 0).any():
        raise ValueError("Invalid dates or closing prices")
    return x.loc[start:end]


def collect(args):
    cache = Path(args.cache_dir)
    cache.mkdir(exist_ok=True, parents=True)
    universe = pd.read_csv(args.universe, dtype={"ticker": str})
    universe = universe[universe.market.eq("KOSPI")].copy()
    universe.to_csv(cache / "universe.csv", index=False)
    symbols = ["KOSPI", *universe.ticker.tolist()]
    def worker(symbol):
        path = cache / f"{symbol}.csv.gz"
        if path.exists():
            x = pd.read_csv(path, index_col=0, parse_dates=True)
            if args.phase == "extend" and x.index.min() <= pd.Timestamp("1999-01-15"):
                for attempt in range(3):
                    try:
                        early = fetch_history(symbol, "1989-01-01", "1999-01-15")
                        break
                    except Exception as exc:
                        if attempt == 2:
                            return {"ticker": symbol, "error": f"Early history: {exc}"}
                x = pd.concat([early.loc[early.index < x.index.min()], x]).sort_index()
                x.to_csv(path, compression="gzip", index_label="Date")
            return {"ticker": symbol, "first": str(x.index.min().date()),
                    "last": str(x.index.max().date()), "rows": len(x), "error": ""}
        for attempt in range(3):
            try:
                x = fetch_history(symbol, args.download_start, args.download_end)
                x.to_csv(path, compression="gzip", index_label="Date")
                return {"ticker": symbol, "first": str(x.index.min().date()),
                        "last": str(x.index.max().date()), "rows": len(x), "error": ""}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < 2:
                    time.sleep(1 + attempt)
        return {"ticker": symbol, "error": error}
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, s) for s in symbols]
        for n, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if n % 50 == 0 or n == len(symbols):
                pd.DataFrame(rows).to_csv(cache / "coverage.csv", index=False)
                print(f"Download {n}/{len(symbols)}, failures={sum(bool(r['error']) for r in rows)}", flush=True)
    pd.DataFrame(rows).to_csv(cache / "coverage.csv", index=False)


def features(stock, benchmark):
    # Only trailing observations define eligibility and entry conditions.
    bc = benchmark.Close.reindex(stock.index).ffill()
    x = add_indicators(stock, bc, DEFAULT_CONFIG)
    x["row_pos"] = np.arange(len(x))
    x["ADV20"] = stock.Amount.rolling(20).mean() if "Amount" in stock else (x.Close * x.Volume).rolling(20).mean()
    x["filter_price"] = stock.RawClose if "RawClose" in stock else stock.Close
    for h in (60, 120):
        x[f"RS{h}"] = x.Close.pct_change(h) - bc.pct_change(h)
    dist = x.Close / x.SMA20 - 1
    dd = x.Close / x.High.shift(1).rolling(60).max() - 1
    correction = x.DOWN_VOL20.notna() & x.UP_VOL20.notna() & (x.DOWN_VOL20 <= x.UP_VOL20 * 1.1)
    x["setup"] = ((x.Close > x.SMA120) & (x.SMA20 > x.SMA60)
                  & (x.SMA60 > x.SMA120) & (x.SMA20_SLOPE > 0)
                  & (x.SMA60_SLOPE > 0) & dist.between(-.03, .04)
                  & dd.between(-.16, -.02) & x.RSI.between(40, 72) & correction)
    x["trigger"] = ((x.Close > x.High.shift(1).rolling(3).max())
                    & (x.Close > x.SMA5) & (x.SMA5 > x.SMA5.shift(1)))
    b20, b60, b120 = bc.rolling(20).mean(), bc.rolling(60).mean(), bc.rolling(120).mean()
    bull = (bc > b120) & (b20 > b60) & (b20 > b20.shift(5))
    bear = (bc < b120) & (b20 < b60) & (b20 < b20.shift(5))
    x["regime"] = np.select([bull, bear], ["bull", "bear"], default="mixed")
    x["dist20"], x["dd60"] = dist, dd
    return x


def repair_index_bars(benchmark, cache):
    bad = benchmark[["Open", "High", "Low", "Close"]].le(0).any(axis=1)
    rows = []
    for year in sorted(set(benchmark.index[bad].year)):
        url = f"https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/refs/heads/master/data/index/year_ks11/{year}.csv"
        r = requests.get(url, timeout=35)
        r.raise_for_status()
        import io
        raw = pd.read_csv(io.StringIO(r.text), index_col="Date", parse_dates=True)
        for d in benchmark.index[bad & (benchmark.index.year == year)]:
            if d not in raw.index or raw.loc[d, ["Open", "High", "Low", "Close"]].le(0).any():
                raise ValueError(f"No observed index candle to repair {d}")
            if abs(raw.loc[d, "Close"] / benchmark.loc[d, "Close"] - 1) > .001:
                raise ValueError("Index sources disagree on close")
            rows.append({"date": str(d.date()), "old_open": float(benchmark.loc[d, "Open"]),
                         "new_open": float(raw.loc[d, "Open"]), "source": url})
            benchmark.loc[d, ["Open", "High", "Low", "Close"]] = raw.loc[d, ["Open", "High", "Low", "Close"]]
    if rows:
        benchmark.to_csv(cache / "KOSPI.csv.gz", compression="gzip", index_label="Date")
        (cache / "index_repair.json").write_text(json.dumps(rows, indent=2))
    return benchmark


def stats(f):
    if f.empty:
        return {"signals": 0}
    ex = f.EXCESS20
    return {"signals": len(f), "tickers": f.ticker.nunique(), "mean_net": f.NET20.mean(),
            "mean_excess": ex.mean(), "median_excess": ex.median(),
            "excess_win_rate": (ex > 0).mean(), "win_rate": (f.NET20 > 0).mean(),
            "mean_mae": f.MAE20.mean(), "mean_mfe": f.MFE20.mean(),
            "mean_gap": f.entry_gap.mean(),
            "top3_removed_excess": ex.sort_values().iloc[:-3].mean() if len(ex) > 3 else np.nan}


def bootstrap(f, draws=2000):
    if f.empty:
        return {"cluster_ci_low": np.nan, "cluster_ci_high": np.nan}
    g = f.groupby("date").EXCESS20.agg(["sum", "count"])
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(g), (draws, len(g)))
    means = g["sum"].to_numpy()[idx].sum(axis=1) / g["count"].to_numpy()[idx].sum(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return {"cluster_ci_low": low, "cluster_ci_high": high}


def add_outcomes(f, stock, benchmark):
    result = f.copy()
    dates = stock.index
    p = result.row_pos.to_numpy(int)
    ep = stock.Open.shift(-1).to_numpy()[p]
    bv = benchmark.reindex(dates)
    bep = bv.Open.shift(-1).to_numpy()[p]
    ep = np.where(ep > 0, ep, np.nan)
    bep = np.where(bep > 0, bep, np.nan)
    result["entry_date"] = pd.Series(dates, index=dates).shift(-1).to_numpy()[p]
    result["entry_price"] = ep
    result["entry_gap"] = ep / result.Close.to_numpy() - 1
    for h in HORIZONS:
        xp = stock.Close.shift(-h).to_numpy()[p]
        bp = bv.Close.shift(-h).to_numpy()[p]
        result[f"NET{h}"] = xp / ep - 1 - .004
        result[f"BENCH{h}"] = bp / bep - 1
        result[f"EXCESS{h}"] = result[f"NET{h}"] - result[f"BENCH{h}"]
        result[f"exit_date{h}"] = pd.Series(dates, index=dates).shift(-h).to_numpy()[p]
        result[f"exit_price{h}"] = xp
        invalid = ((stock.Open.shift(-1).to_numpy()[p] <= 0)
                   | (stock.Volume.shift(-1).to_numpy()[p] <= 0)
                   | (stock.Volume.shift(-h).to_numpy()[p] <= 0) | (xp <= 0)
                   | (bep <= 0) | (bp <= 0))
        result.loc[invalid, [f"NET{h}", f"EXCESS{h}"]] = np.nan
    result["MAE20"] = stock.Low.where(stock.Low > 0).shift(-1).rolling(20, min_periods=1).min().shift(-19).to_numpy()[p] / ep - 1
    result["MFE20"] = stock.High.where(stock.High > 0).shift(-1).rolling(20, min_periods=1).max().shift(-19).to_numpy()[p] / ep - 1
    # No fabricated outcome: entry must be the next observed market session.
    calpos = benchmark.index.get_indexer(result.date)
    expected = pd.Series(benchmark.index).shift(-1).to_numpy()[calpos]
    result.loc[result.entry_date.to_numpy() != expected, [f"NET{h}" for h in HORIZONS]] = np.nan
    return result.replace([np.inf, -np.inf], np.nan)


def replay(f, benchmark, prices, start, end, asset="stock", core=False, keep=None, cost=.004, h=20):
    """Close exits cannot release cash for an earlier open on the same day."""
    work = f.sort_values(["entry_date", "ticker"])
    grouped = {d: g for d, g in work.groupby("entry_date")}
    calendar = benchmark.loc[start:end].index
    side, cash, units, active = cost / 2, 1., 0., []
    if core:
        units = 1 / (float(benchmark.loc[calendar[0], "Open"]) * (1 + side))
        cash = 0.
    accepted, curve = [], []
    bopen = float(benchmark.loc[calendar[0], "Open"])
    def price(t, d, field):
        x = prices[t] if asset == "stock" else benchmark
        value = float(x.loc[d, field]) if d in x.index else np.nan
        if not np.isfinite(value) or value <= 0:
            # Marking only; no buy or sell is allowed at this stale mark.
            value = float(x.loc[:d, "Close"].replace(0, np.nan).ffill().iloc[-1])
        return value
    for d in calendar:
        bo, bc = float(benchmark.loc[d, "Open"]), float(benchmark.loc[d, "Close"])
        eq = cash + units * bo + sum(p["shares"] * price(p["ticker"], d, "Open") for p in active)
        for _, r in grouped.get(d, pd.DataFrame()).iterrows():
            key = (r.ticker, str(r.entry_date.date()))
            if keep is not None and key not in keep:
                continue
            target = eq / 10
            if len(active) >= 10:
                continue
            if core:
                target = min(target, units * bo * (1 - side))
            elif target > cash + 1e-12:
                if keep is None:
                    continue
                target = cash
            if target <= 1e-12:
                continue
            ep = float(r.entry_price) if asset == "stock" else bo
            if core:
                units -= target / (bo * (1 - side))
            else:
                cash -= target
            active.append({"ticker": r.ticker, "shares": target / (ep * (1 + side)),
                           "end": r[f"exit_date{h}"], "xp": float(r[f"exit_price{h}"])})
            accepted.append(key)
        remaining = []
        for p in active:
            if p["end"] == d:
                xp = p["xp"] if asset == "stock" else bc
                proceeds = p["shares"] * xp * (1 - side)
                if core:
                    units += proceeds / (bc * (1 + side))
                else:
                    cash += proceeds
            else:
                remaining.append(p)
        active = remaining
        marked = sum(p["shares"] * price(p["ticker"], d, "Close") for p in active)
        eq = cash + units * bc + marked
        curve.append({"date": d, "equity": eq, "cash": cash,
                      "exposure": marked / eq, "positions": len(active), "benchmark": bc / bopen})
        if cash < -1e-8 or units < -1e-8:
            raise AssertionError("Borrowed cash or index units")
    if active:
        raise AssertionError("Unclosed positions at end of replay")
    daily = pd.DataFrame(curve).set_index("date")
    if core:
        daily.loc[daily.index[-1], "equity"] *= 1 - side
    values = daily.equity.to_numpy()
    high = np.maximum.accumulate(np.r_[1., values])[1:]
    summary = {"accepted": len(accepted), "signals": len(f), "total_return": values[-1] - 1,
               "benchmark_return": daily.benchmark.iloc[-1] - 1,
               "mdd": (values / high - 1).min(), "avg_exposure": daily.exposure.mean()}
    return daily, set(accepted), summary


def annual_returns(daily, label):
    rows = []
    for year, g in daily.groupby(daily.index.year):
        prior = daily.loc[daily.index < g.index[0]].tail(1)
        base = 1. if prior.empty else prior.equity.iloc[0]
        bb = 1. if prior.empty else prior.benchmark.iloc[0]
        vals = np.r_[base, g.equity.to_numpy()]
        rows.append({"model": label, "year": year, "return": g.equity.iloc[-1] / base - 1,
                     "index_return": g.benchmark.iloc[-1] / bb - 1,
                     "avg_exposure": g.exposure.mean(),
                     "year_mdd": (vals / np.maximum.accumulate(vals) - 1).min()})
    return rows


def analyze(args):
    cache, out = Path(args.cache_dir), Path(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)
    coverage = pd.read_csv(cache / "coverage.csv").fillna("")
    coverage.to_csv(out / "coverage.csv", index=False)
    if coverage.error.ne("").any():
        raise RuntimeError("Incomplete downloads; resolve coverage errors first")
    u = pd.read_csv(cache / "universe.csv", dtype={"ticker": str})
    benchmark = pd.read_csv(cache / "KOSPI.csv.gz", index_col=0, parse_dates=True)
    benchmark = repair_index_bars(benchmark, cache)
    repair_file = cache / "index_repair.json"
    (out / "index_repair.json").write_text(repair_file.read_text() if repair_file.exists() else "[]")
    amount_cache = Path(args.amount_dir)
    amount_frames = [pd.read_parquet(p) for p in sorted(amount_cache.glob("*.parquet"))]
    amount = pd.concat(amount_frames, ignore_index=True)
    amount["Date"] = pd.to_datetime(amount.Date)
    liquidity = {t: g.set_index("Date").sort_index() for t, g in amount.groupby("Code")}
    del amount, amount_frames
    pd.read_csv(amount_cache / "coverage.csv").to_csv(out / "historical_universe_coverage.csv", index=False)
    prices, pieces, legacy_pieces, audit = {}, [], [], []
    for n, meta in enumerate(u.to_dict("records"), 1):
        ticker = meta["ticker"]
        stock = pd.read_csv(cache / f"{ticker}.csv.gz", index_col=0, parse_dates=True)
        prices[ticker] = stock
        raw = liquidity.get(ticker, pd.DataFrame(columns=["Close", "Volume", "Amount", "Market"])).reindex(stock.index)
        for corrected in (True, False):
            inputs = stock.copy()
            if corrected:
                inputs["Amount"], inputs["RawClose"] = raw.Amount, raw.Close
                # OHLC is split-adjusted, so compare volumes on the same share basis.
                inputs["Volume"] = raw.Volume * raw.Close / stock.Close
            x = features(inputs, benchmark)
            eligible = ((x.index >= pd.Timestamp(args.start)) & (x.index <= pd.Timestamp(args.end))
                        & (x.row_pos >= DEFAULT_CONFIG.min_history - 1)
                        & (x.ADV20 >= 2_000_000_000) & (x.filter_price >= 1000)
                        & x[["RS20", "RS60", "RS120", "RS_RATIO_SLOPE"]].notna().all(axis=1))
            if corrected:
                eligible &= raw.Market.eq("KOSPI")
                # Zero-price halt bars must not lower the three-day breakout level.
                current_valid = stock[["Open", "High", "Low", "Close", "Volume"]].gt(0).all(axis=1)
                eligible &= current_valid & current_valid.shift(1).rolling(3).sum().eq(3)
            fields = ["row_pos", "Close", "ADV20", "RS20", "RS60", "RS120", "RS_RATIO_SLOPE",
                      "setup", "trigger", "regime", "dist20", "dd60"]
            panel = x.loc[eligible, fields].copy()
            panel["date"], panel["ticker"], panel["name"] = panel.index, ticker, meta["name"]
            (pieces if corrected else legacy_pieces).append(panel.reset_index(drop=True))
        old_path = Path(args.legacy_cache_dir) / f"{ticker}.csv.gz"
        if old_path.exists():
            old = pd.read_csv(old_path, index_col=0, parse_dates=True)
            overlap = old.index.intersection(stock.index)
            error = (stock.loc[overlap, "Close"] / old.loc[overlap, "Close"] - 1).abs()
            audit.append({"ticker": ticker, "overlap_rows": len(overlap),
                          "max_close_relative_difference": error.max(),
                          "median_close_relative_difference": error.median()})
        if n % 100 == 0 or n == len(u):
            print(f"Features {n}/{len(u)}", flush=True)
    pd.DataFrame(audit).to_csv(out / "source_overlap_audit.csv", index=False)
    panel = pd.concat(pieces, ignore_index=True)
    legacy = pd.concat(legacy_pieces, ignore_index=True)
    del pieces, legacy_pieces
    for h in (20, 60, 120):
        panel[f"RS{h}_PCTL"] = panel.groupby("date")[f"RS{h}"].rank(pct=True)
        legacy[f"RS{h}_PCTL"] = legacy.groupby("date")[f"RS{h}"].rank(pct=True)
    rs = ((panel.RS20 > 0) & (panel.RS_RATIO_SLOPE > 0) & (panel.RS20_PCTL >= .5)
          & (panel.RS60_PCTL >= .6) & (panel.RS120_PCTL >= .7))
    masks = {"v6": rs & panel.setup & panel.trigger, "rs_only": rs,
             "rs_trigger": rs & panel.trigger, "rs_setup": rs & panel.setup}
    legacy_mask = ((legacy.RS20 > 0) & (legacy.RS_RATIO_SLOPE > 0) & (legacy.RS20_PCTL >= .5)
                   & (legacy.RS60_PCTL >= .6) & (legacy.RS120_PCTL >= .7) & legacy.setup & legacy.trigger)
    funnel = []
    for year, g in panel.groupby(panel.date.dt.year):
        idx = g.index
        funnel.append({"year": year, "eligible_stock_days": len(g), "eligible_tickers": g.ticker.nunique(),
                       **{k + "_stock_days": int(m.loc[idx].sum()) for k, m in masks.items()}})
    pd.DataFrame(funnel).to_csv(out / "signal_funnel.csv", index=False)
    # Reset cooldown at the original start to audit reproduction separately from
    # the continuous long-history account, which can carry December 2021 state.
    recent = v7._cooldown(legacy.loc[legacy_mask & legacy.date.ge(pd.Timestamp("2022-01-03"))].copy(), 20)
    recent = pd.concat([add_outcomes(g, prices[t], benchmark) for t, g in recent.groupby("ticker")], ignore_index=True)
    recent = recent[recent.NET20.notna() & recent.EXCESS20.notna()]
    old_signals_path = Path("../v8-verified-reports/fixed_v6_signals.csv")
    if old_signals_path.exists():
        old = pd.read_csv(old_signals_path, dtype={"ticker": str}, parse_dates=["date"])
        old = old[old.market.eq("KOSPI")]
        old_keys = set(zip(old.ticker, old.date))
        new_keys = set(zip(recent.ticker, recent.date))
        pd.DataFrame([{"old_signals": len(old_keys), "new_signals": len(new_keys),
                       "common": len(old_keys & new_keys), "old_only": len(old_keys - new_keys),
                       "new_only": len(new_keys - old_keys), "exact_entry_match": old_keys == new_keys}]).to_csv(out / "recent_reproduction.csv", index=False)
        differences = [{"ticker": t, "date": d, "side": side}
                       for side, keys in [("old_only", old_keys-new_keys), ("new_only", new_keys-old_keys)] for t, d in keys]
        pd.DataFrame(differences, columns=["ticker", "date", "side"]).to_csv(out / "recent_entry_differences.csv", index=False)
    all_signals, invalid = [], []
    for variant in (*masks, "v6_proxy"):
        selected = v7._cooldown((legacy.loc[legacy_mask] if variant == "v6_proxy" else panel.loc[masks[variant]]).copy(), 20)
        print(f"Entries {variant}: {len(selected)}", flush=True)
        for ticker, f in selected.groupby("ticker"):
            result = add_outcomes(f, prices[ticker], benchmark)
            result["variant"] = variant
            bad = result.NET20.isna() | result.EXCESS20.isna()
            if bad.any():
                invalid.append(result.loc[bad])
            all_signals.append(result.loc[~bad])
    signals = pd.concat(all_signals, ignore_index=True)
    signals.to_csv(out / "signals.csv.gz", index=False, compression="gzip")
    pd.concat(invalid, ignore_index=True).to_csv(out / "unavailable_outcomes.csv", index=False) if invalid else None
    rows, periods, regimes, gaps, horizons = [], [], [], [], []
    for variant, f in signals.groupby("variant"):
        for year, g in f.groupby(f.date.dt.year):
            rows.append({"variant": variant, "year": year, **stats(g)})
        for label, lo, hi in [("1996_1999", 1996, 1999), ("2000_2007", 2000, 2007), ("2008_2014", 2008, 2014),
                              ("2015_2021", 2015, 2021), ("2022_2024", 2022, 2024),
                              ("2025_2026_reused", 2025, 2026), ("all_pre2025", 1996, 2024)]:
            g = f[f.date.dt.year.between(lo, hi)]
            if hi < 2025:
                g = g[g.exit_date20.le(pd.Timestamp(f"{hi}-12-31"))]
            periods.append({"variant": variant, "period": label, **stats(g), **bootstrap(g)})
        for period, ff in [("all", f), ("pre2025", f[f.exit_date20.le(pd.Timestamp("2024-12-30"))])]:
            for regime, g in ff.groupby("regime"):
                regimes.append({"variant": variant, "period": period, "regime": regime, **stats(g)})
            for gap, g in ff.groupby(pd.cut(ff.entry_gap, [-np.inf, 0, .03, np.inf], labels=["nonpositive", "0_to_3pct", "above_3pct"]), observed=True):
                gaps.append({"variant": variant, "period": period, "gap": str(gap), **stats(g)})
        # Same entries and common mature horizon, no picking a best hold duration.
        common = f[f.exit_date60.le(pd.Timestamp("2024-12-30")) & f[[f"NET{h}" for h in HORIZONS]].notna().all(axis=1)]
        for h in HORIZONS:
            horizons.append({"variant": variant, "horizon": h, "signals": len(common),
                             "mean_net": common[f"NET{h}"].mean(),
                             "mean_excess": common[f"EXCESS{h}"].mean()})
    for name, records in [("trade_yearly", rows), ("trade_periods", periods), ("regime", regimes),
                          ("entry_gap", gaps), ("holding_horizons", horizons)]:
        pd.DataFrame(records).to_csv(out / f"{name}.csv", index=False)
    base = signals[signals.variant.eq("v6")].copy()
    base.to_csv(out / "v6_signals.csv", index=False)
    rolling = []
    for year in range(2000, 2025):
        f = base[base.date.dt.year.between(year-4, year) & base.exit_date20.le(pd.Timestamp(f"{year}-12-31"))]
        rolling.append({"first_year": year-4, "last_year": year, **stats(f)})
    pd.DataFrame(rolling).to_csv(out / "rolling_5y.csv", index=False)
    portfolio_rows, annual = [], []
    start, end = benchmark.loc[args.start:].index[0], base.exit_date20.max()
    daily, keys, summary = replay(base, benchmark, prices, start, end)
    curves = {"v6_cash": daily}
    portfolio_rows.append({"model": "v6_cash", **summary})
    for label, asset, core in [("same_schedule_index", "index", False), ("index_core_v6", "stock", True)]:
        daily, _, summary = replay(base, benchmark, prices, start, end, asset=asset, core=core, keep=keys)
        curves[label] = daily
        portfolio_rows.append({"model": label, **summary})
    for label, daily in curves.items():
        daily.to_csv(out / f"equity_{label}.csv", index_label="date")
        annual.extend(annual_returns(daily, label))
    pd.DataFrame(portfolio_rows).to_csv(out / "portfolio_summary.csv", index=False)
    pd.DataFrame(annual).to_csv(out / "portfolio_yearly.csv", index=False)
    # Fully separate pre-2025 account, purge all trades whose exits cross its boundary.
    pre = base[(base.date.dt.year <= 2024) & (base.exit_date20 <= pd.Timestamp("2024-12-30"))]
    pre_end = benchmark.loc[:"2024-12-30"].index[-1]
    pre_daily, pre_keys, pre_summary = replay(pre, benchmark, prices, start, pre_end)
    pre_results = [{"model": "v6_cash", **pre_summary}]
    for label, asset, core in [("same_schedule_index", "index", False), ("index_core_v6", "stock", True)]:
        d, _, s = replay(pre, benchmark, prices, start, pre_end, asset=asset, core=core, keep=pre_keys)
        pre_results.append({"model": label, **s})
    pd.DataFrame(pre_results).to_csv(out / "portfolio_pre2025.csv", index=False)
    # Fixed cost sensitivities are reported, never selected as models.
    pd.DataFrame([{"round_trip_cost": c, "pre2025_signals": len(pre),
                   "mean_net": pre.NET20.mean() - (c - .004),
                   "mean_excess": pre.EXCESS20.mean() - (c - .004)}
                  for c in (.004, .008, .012)]).to_csv(out / "cost_sensitivity.csv", index=False)
    manifest = {"args": vars(args), "universe_count": len(u), "entry_rules": "frozen v6 KOSPI",
                "universe_limitation": "current 2026 survivors; historical exchange membership applied; delisted absent",
                "liquidity_source": json.loads((amount_cache / "source.json").read_text()),
                "liquidity": "actual KRX daily Amount; original price floor; volume normalized to adjusted share basis",
                "selection": "none; all diagnostics retrospective; 2025_2026 reused",
                "prices": "NAVER adjusted OHLC, dividend excluded", "cost": .004,
                "zero_volume_outcomes": "reported unavailable; not assumed executable",
                "amount_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in amount_cache.glob("*.parquet")},
                "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.glob("*.csv.gz")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(pd.DataFrame(pre_results).to_string(index=False), flush=True)
    print(pd.DataFrame(periods).query("variant == 'v6'").to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["collect", "extend", "analyze"], default="collect")
    p.add_argument("--universe", default="research/kospi-long-results/universe.csv")
    p.add_argument("--cache-dir", default="data_kospi_long")
    p.add_argument("--output-dir", default="research/kospi-long-results")
    p.add_argument("--download-start", default="1989-01-01")
    p.add_argument("--download-end", default="2026-10-07")
    p.add_argument("--start", default="1996-01-01")
    p.add_argument("--amount-dir", default="data_kospi_amount")
    p.add_argument("--legacy-cache-dir", default="data_robustness_v8")
    p.add_argument("--end", default="2026-09-04")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.phase in ("collect", "extend"):
        collect(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
