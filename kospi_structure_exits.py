"""Frozen entries, support stops and moving-average exits without a time cap.

Research only. Intraday stops use a support level known before entry. Closing
signals execute at the next executable open, never retrospectively at that close.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kospi_long_history import annual_returns

COST = .004
POLICIES = ("fixed20", "swing_ma5", "trend_ma20", "trend_ma60", "trailing_low10", "ma20_only", "ma60_only")
MA_PERIOD = {"swing_ma5": 5, "trend_ma20": 20, "trend_ma60": 60, "ma20_only": 20, "ma60_only": 60}
SUPPORT_POLICIES = {"swing_ma5", "trend_ma20", "trend_ma60", "trailing_low10"}


@dataclass
class PricePath:
    frame: pd.DataFrame
    dates: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    valid: np.ndarray
    locked: np.ndarray
    averages: dict[int, np.ndarray]


def prepare(stock):
    stock = stock.sort_index()
    if stock.index.has_duplicates:
        raise ValueError("Duplicate stock date")
    arrays = {c: stock[c].to_numpy(float) for c in ("Open", "High", "Low", "Close")}
    valid = stock[["Open", "High", "Low", "Close", "Volume"]].gt(0).all(axis=1).to_numpy()
    # A daily OHLC bar cannot establish that an order filled on a one-price bar.
    locked = valid & np.isclose(arrays["Open"], arrays["High"], rtol=0, atol=1e-8)
    locked &= np.isclose(arrays["Open"], arrays["Low"], rtol=0, atol=1e-8)
    locked &= np.isclose(arrays["Open"], arrays["Close"], rtol=0, atol=1e-8)
    return PricePath(stock, stock.index, arrays["Open"], arrays["High"], arrays["Low"],
                     arrays["Close"], valid, locked,
                     {h: stock.Close.rolling(h).mean().to_numpy(float) for h in (5, 20, 60)})


def initial_support(path, signal_pos):
    """Signal day plus nine preceding observations, never future lows."""
    lows = path.low[max(0, signal_pos - 9):signal_pos + 1]
    observed = lows[np.isfinite(lows) & (lows > 0)]
    if not len(observed):
        raise ValueError("No known support at signal")
    return float(observed.min())


def simulate_trade(path, signal_date, policy, as_of, block_locked=True):
    if policy not in POLICIES:
        raise ValueError(policy)
    i = path.dates.get_loc(pd.Timestamp(signal_date))
    entry, last = i + 1, path.dates.searchsorted(pd.Timestamp(as_of), side="right") - 1
    support = initial_support(path, i)
    basic = {"policy": policy, "signal_date": pd.Timestamp(signal_date), "support": support}
    if entry > last or entry >= len(path.dates):
        return {**basic, "status": "entry_outside_window"}
    basic.update({"entry_date": path.dates[entry], "entry_price": float(path.open[entry])})
    if not path.valid[entry] or (block_locked and path.locked[entry]):
        return {**basic, "status": "entry_unfilled"}
    ep, pending, active_stop = float(path.open[entry]), None, support
    mae, mfe = 0., 0.
    def finish(j, price, phase, reason, realized):
        net = price / ep / (1 + COST/2) * (1 - COST/2 if realized else 1) - 1
        return {**basic, "status": "closed" if realized else "marked_open",
                "exit_date": path.dates[j], "exit_price": float(price), "exit_phase": phase,
                "exit_reason": reason, "hold_observations": j-entry,
                "gross_return": price/ep-1, "net_return": net,
                "initial_risk": max(0., 1-support/ep), "mae": min(mae, price/ep-1),
                "mfe": max(mfe, price/ep-1), "final_stop": active_stop if policy in SUPPORT_POLICIES else np.nan}
    for j in range(entry, last + 1):
        if policy == "trailing_low10":
            # Ratchet using only the ten preceding bars, excluding today's low.
            lows = path.low[max(0, j-10):j]
            lows = lows[np.isfinite(lows) & (lows > 0)]
            if len(lows):
                active_stop = max(active_stop, float(lows.min()))
        if not path.valid[j]:
            if policy == "fixed20" and j >= i + 20:
                pending = pending or "fixed20_delayed"
            continue
        executable = not (block_locked and path.locked[j])
        # The open is observed before that day's high/low and closing signal.
        op = float(path.open[j])
        mae, mfe = min(mae, op/ep-1), max(mfe, op/ep-1)
        if pending and executable:
            return finish(j, op, "open", pending, True)
        if policy in SUPPORT_POLICIES:
            if op <= active_stop:
                if executable:
                    return finish(j, op, "open", "support_gap", True)
                pending = pending or "support_locked_delayed"
            elif path.low[j] <= active_stop:
                if executable:
                    # Same-day extremes could occur after the stop; do not count them.
                    return finish(j, active_stop, "intraday", "support_break", True)
                pending = pending or "support_locked_delayed"
        mae = min(mae, float(path.low[j])/ep-1)
        mfe = max(mfe, float(path.high[j])/ep-1)
        if policy == "fixed20":
            if j >= i + 20:
                if executable:
                    return finish(j, path.close[j], "close", "fixed20", True)
                pending = pending or "fixed20_locked_delayed"
        elif policy in MA_PERIOD:
            h = MA_PERIOD[policy]
            ma = path.averages[h][j]
            if np.isfinite(ma) and path.close[j] < ma:
                pending = pending or f"close_below_ma{h}"
    # Study end is a valuation date, not a forced strategy sale.
    mark_positions = np.flatnonzero(np.isfinite(path.close[:last+1]) & (path.close[:last+1] > 0))
    if not len(mark_positions):
        raise ValueError("No observed price to mark open holding")
    j = int(mark_positions[-1])
    return finish(j, path.close[j], "mark", pending or "trend_still_active", False)


def add_benchmark(row, benchmark):
    if row["status"] not in ("closed", "marked_open"):
        return row
    d, ed = row["exit_date"], row["entry_date"]
    if ed not in benchmark.index or d not in benchmark.index:
        raise ValueError("Missing observed benchmark at entry or exit")
    field = "Open" if row["exit_phase"] == "open" else "Close"
    ep, xp = float(benchmark.loc[ed, "Open"]), float(benchmark.loc[d, field])
    if min(ep, xp) <= 0:
        raise ValueError("Nonpositive benchmark price")
    br = xp/ep - 1
    return {**row, "benchmark_return": br, "excess_return": row["net_return"]-br,
            "benchmark_exit_field": field,
            "intraday_benchmark_approximation": row["exit_phase"] == "intraday"}


def replay(trades, paths, benchmark, start, end, asset="stock", core=False, keep=None):
    """10% opening-equity slots, no leverage, no same-ticker pyramiding.

    Intraday stop proceeds are unavailable for morning orders. When using the
    index comparator, a stock's intraday stop maps to that day's index close.
    That is a daily proxy, not a fabricated intraday index fill.
    """
    eligible = trades[trades.status.isin(["closed", "marked_open"])].copy()
    grouped = {d: g.sort_values("ticker") for d, g in eligible.groupby("entry_date")}
    calendar = benchmark.loc[start:end].index
    side, cash, index_units, active = COST/2, 1., 0., []
    accepted, skipped, curve = [], [], []
    if core:
        index_units = 1/(float(benchmark.loc[calendar[0], "Open"])*(1+side))
        cash = 0.
    bo0 = float(benchmark.loc[calendar[0], "Open"])
    for d in calendar:
        bo, bc = float(benchmark.loc[d, "Open"]), float(benchmark.loc[d, "Close"])
        def mark(p, field):
            if asset == "index":
                return bo if field == "Open" else bc
            frame = paths[p["ticker"]].frame
            price = float(frame.loc[d, field]) if d in frame.index else np.nan
            if not np.isfinite(price) or price <= 0:
                # Previous closing mark at an unavailable open; never future-fill.
                prior = frame.loc[frame.index < d if field == "Open" else frame.index <= d, "Close"]
                price = float(prior.replace(0, np.nan).ffill().iloc[-1])
            return price
        def exit_phase(p):
            return "close" if asset == "index" and p["phase"] == "intraday" else p["phase"]
        def sell_phase(phase):
            nonlocal cash, index_units, active
            remaining = []
            for p in active:
                if p["end"] == d and exit_phase(p) == phase:
                    xp = p["price"] if asset == "stock" else (bo if phase == "open" else bc)
                    proceeds = p["shares"]*xp*(1-side)
                    if core:
                        # Intraday stop cash waits for the close to rebuy the index.
                        if phase == "open":
                            index_units += proceeds/(bo*(1+side))
                        else:
                            cash += proceeds
                    else:
                        cash += proceeds
                else:
                    remaining.append(p)
            active = remaining
        sell_phase("open")
        opening_equity = cash + index_units*bo + sum(p["shares"]*mark(p, "Open") for p in active)
        for _, r in grouped.get(d, pd.DataFrame()).iterrows():
            key = (r.ticker, str(r.entry_date.date()))
            if keep is not None and key not in keep:
                continue
            target = opening_equity/10
            reason = None
            if any(p["ticker"] == r.ticker for p in active):
                reason = "same_ticker_already_held"
            elif len(active) >= 10:
                reason = "position_cap"
            elif core:
                target = min(target, index_units*bo*(1-side))
            elif cash + 1e-12 < target:
                if keep is None:
                    reason = "cash_limit"
                else:
                    target = cash
            if reason or target <= 1e-12:
                skipped.append({"ticker": r.ticker, "entry_date": d, "reason": reason or "no_budget"})
                continue
            if core:
                index_units -= target/(bo*(1-side))
            else:
                cash -= target
            ep = float(r.entry_price) if asset == "stock" else bo
            active.append({"ticker": r.ticker, "shares": target/(ep*(1+side)),
                           "end": r.exit_date, "phase": r.exit_phase, "price": float(r.exit_price)})
            accepted.append(key)
        # Entry-day gap invalidation executes only after all opening allocations.
        sell_phase("open")
        sell_phase("intraday")
        sell_phase("close")
        if core and cash > 0:
            index_units += cash/(bc*(1+side))
            cash = 0.
        value = sum(p["shares"]*mark(p, "Close") for p in active)
        equity = cash + index_units*bc + value
        curve.append({"date": d, "equity": equity, "cash": cash, "exposure": value/equity,
                      "positions": len(active), "benchmark": bc/bo0})
        if cash < -1e-9 or index_units < -1e-9:
            raise AssertionError("Borrowed cash/index units")
    if any(p["phase"] != "mark" for p in active):
        raise AssertionError("Realized exit remains beyond evaluation")
    daily = pd.DataFrame(curve).set_index("date")
    value = daily.equity.to_numpy()
    high = np.maximum.accumulate(np.r_[1., value])[1:]
    years = (daily.index[-1]-daily.index[0]).days/365.25
    summary = {"accepted": len(accepted), "skipped": len(skipped), "still_open": len(active),
               "total_return": value[-1]-1, "cagr": value[-1]**(1/years)-1 if years > 0 else np.nan,
               "benchmark_return": daily.benchmark.iloc[-1]-1,
               "mdd": (value/high-1).min(), "avg_exposure": daily.exposure.mean()}
    return daily, set(accepted), pd.DataFrame(skipped), summary


def event_stats(trades):
    filled = trades[trades.status.isin(["closed", "marked_open"])]
    return {"signals": len(trades), "filled": len(filled),
            "closed": int(trades.status.eq("closed").sum()),
            "marked_open": int(trades.status.eq("marked_open").sum()),
            "entry_unfilled": int(trades.status.eq("entry_unfilled").sum()),
            "mean_net": filled.net_return.mean(), "mean_excess": filled.excess_return.mean(),
            "median_net": filled.net_return.median(), "win_rate": filled.net_return.gt(0).mean(),
            "mean_hold": filled.hold_observations.mean(), "median_hold": filled.hold_observations.median(),
            "max_hold": filled.hold_observations.max(), "p90_hold": filled.hold_observations.quantile(.9),
            "mean_mae": filled.mae.mean(), "intraday_exits": int(filled.exit_phase.eq("intraday").sum())}


def run(args):
    cache, out = Path(args.cache_dir), Path(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)
    base = pd.read_csv(args.signals, dtype={"ticker": str}, parse_dates=["date", "entry_date", "exit_date20"])
    benchmark = pd.read_csv(cache/"KOSPI.csv.gz", index_col=0, parse_dates=True)
    paths = {t: prepare(pd.read_csv(cache/f"{t}.csv.gz", index_col=0, parse_dates=True)) for t in base.ticker.unique()}
    windows = [("pre2025", base[base.exit_date20 <= pd.Timestamp("2024-12-30")], "1996-01-01", "2024-12-30"),
               ("all", base, "1996-01-01", "2026-10-07"),
               ("evaluation_2025_26_reused", base[base.date >= pd.Timestamp("2025-01-01")], "2025-01-01", "2026-10-07")]
    records, comparisons, events, yearly, reasons, skipped_rows, checks = [], [], [], [], [], [], []
    # Prove original fixed20 prices/timing before conservative execution changes.
    for _, r in base.iterrows():
        x = simulate_trade(paths[r.ticker], r.date, "fixed20", benchmark.index[-1], block_locked=False)
        if x["status"] != "closed" or x["exit_date"] != r.exit_date20 or abs(x["exit_price"]-r.exit_price20) > 1e-8:
            raise AssertionError("Original fixed20 exit reproduction failed")
    checks.append({"check": "original_fixed20_timing_and_prices", "trades": len(base), "passed": True})
    for label, cohort, lo, hi in windows:
        end = min(pd.Timestamp(hi), benchmark.index[-1])
        for policy in POLICIES:
            rows = []
            for _, r in cohort.iterrows():
                trade = simulate_trade(paths[r.ticker], r.date, policy, end, block_locked=True)
                trade = add_benchmark(trade, benchmark)
                rows.append({"ticker": r.ticker, "name": r["name"], "period": label, **trade})
            f = pd.DataFrame(rows)
            records.append(f)
            events.append({"period": label, "policy": policy, **event_stats(f)})
            for reason, g in f.groupby("exit_reason", dropna=False):
                reasons.append({"period": label, "policy": policy, "reason": reason, "trades": len(g)})
            daily, keys, skipped, summary = replay(f, paths, benchmark, lo, end)
            comparisons.append({"period": label, "policy": policy, "asset": "stock_cash", **summary})
            yearly.extend([{**r, "period": label, "policy": policy} for r in annual_returns(daily, policy)])
            daily.to_csv(out/f"equity_{label}_{policy}.csv", index_label="date")
            if not skipped.empty:
                skipped_rows.append(skipped.assign(period=label, policy=policy))
            for asset, core, name in [("index", False, "matched_index_cash"), ("stock", True, "index_core")]:
                _, matched_keys, _, s = replay(f, paths, benchmark, lo, end, asset=asset, core=core, keep=keys)
                if matched_keys != keys:
                    raise AssertionError("Matched-account accepted schedule differs")
                comparisons.append({"period": label, "policy": policy, "asset": name, **s})
            print(label, policy, event_stats(f), summary, flush=True)
    trades = pd.concat(records, ignore_index=True)
    trades.to_csv(out/"trades.csv", index=False)
    for filename, rows in [("events", events), ("portfolio", comparisons), ("yearly", yearly),
                            ("exit_reasons", reasons), ("reproduction", checks)]:
        pd.DataFrame(rows).to_csv(out/f"{filename}.csv", index=False)
    pd.concat(skipped_rows, ignore_index=True).to_csv(out/"skipped.csv", index=False) if skipped_rows else None
    manifest = {"args": vars(args), "entry_signals": len(base), "policies": POLICIES,
                "support": "minimum positive low of signal day and prior nine observations; trailing policy ratchets prior ten lows",
                "pure_ma_controls": "ma20_only and ma60_only have no support stop",
                "protocol_note": "Three pure/trailing controls added as component diagnostics after observing the initial four-policy run, not selected optimizations",
                "maximum_hold": "none; open holdings marked at study end without sell cost",
                "close_signal": "next executable open", "gap_stop": "actual open if below support",
                "one_price_bars": "conservatively no entry/fill; pending exit delayed",
                "cost": COST, "same_ticker_pyramiding": False, "model_selection": "none",
                "intraday_index": "daily closing proxy; no intraday index price is invented",
                "entry_universe": "frozen V6 current-survivor entries, not historical full universe",
                "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(args.signals), cache/"KOSPI.csv.gz"]}}
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--signals", default="research/kospi-long-results/v6_signals.csv")
    p.add_argument("--cache-dir", default="data_kospi_long")
    p.add_argument("--output-dir", default="research/kospi-structure-results")
    run(p.parse_args())
