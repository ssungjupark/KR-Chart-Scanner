"""Research only: fixed v6 entries, cash-aware sizing and profit-armed exits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from data_loader import align_benchmark_ohlc, load_benchmark_range, load_price_range
from universe import load_krx_universe_frame
import robustness_v7 as v7
from robustness_v7_exact import EXPECTED_V6_COUNTS, _entry_mask_exact

EXITS = ("fixed20", "profit10_dd5", "profit10_atr2")
FEATURES = {
    "dryup": "PULLBACK_DRYUP", "rvol": "TRIGGER_RVOL_OK",
    "bb_squeeze": "BB_SQUEEZE_OK", "ma_compress": "MA_COMPRESS_OK",
    "retest": "RETEST_RECLAIM_OK", "not_bear": "MARKET_NOT_BEAR",
}
SIZING = ("equal", *FEATURES, "quality2")
COST = 0.004


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_prices(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def cached_prices(symbol: str, start: str, end: str, cache: Path,
                  benchmark: bool = False) -> pd.DataFrame:
    path = cache / f"{symbol}.csv.gz"
    if path.exists():
        return read_prices(path)
    loader = load_benchmark_range if benchmark else load_price_range
    x = loader(symbol, start, end, warmup_days=650, forward_days=75)
    x.to_csv(path, compression="gzip", index_label="Date")
    return x


def process_symbol(meta: dict, benchmarks: dict, args: argparse.Namespace, cache: Path):
    ticker, market = str(meta["ticker"]), meta["market"]
    try:
        stock = cached_prices(ticker, args.start, args.end, cache)
        x = v7.base_feature_frame(stock, benchmarks[market])
        x = v7._add_v7_features(x, benchmarks[market])
        eligible = (
            (x.index >= pd.Timestamp(args.start)) & (x.index <= pd.Timestamp(args.end))
            & (x["row_pos"] >= DEFAULT_CONFIG.min_history - 1)
            & (x["ADV20"] >= args.min_adv) & (x["Close"] >= args.min_price)
            & x[["RS20", "RS60", "RS120", "RS_RATIO_SLOPE"]].notna().all(axis=1)
        )
        rank = x.loc[eligible, ["row_pos", "RS20", "RS60", "RS120"]].copy()
        rank["date"], rank["ticker"], rank["market"] = rank.index, ticker, market
        # These are the v7-exact mature-outcome gates, before cross-sectional RS/cooldown.
        bench = align_benchmark_ohlc(x, benchmarks[market])
        mature = x["RET20"].notna() & x["EXCESS20"].notna()
        mature &= (x["Open"].shift(-1) > 0) & (bench["Open"].shift(-1) > 0)
        mature &= (x["Close"].shift(-20) > 0) & (bench["Close"].shift(-20) > 0)
        mask = eligible & x["BASE_LEADER20"] & mature
        cols = ["row_pos", "RS20", "RS60", "RS120", "RS_RATIO_SLOPE",
                "TRIG_BREAK3", "TRIG_MA5", "QUALITY_SCORE", "TRIGGER_RVOL",
                "BB_WIDTH", "MA_SPREAD", "RETEST_RECLAIM_OK", "MARKET_NOT_BEAR",
                "PULLBACK_DRYUP", "TRIGGER_RVOL_OK", "BB_SQUEEZE_OK", "MA_COMPRESS_OK"]
        cand = x.loc[mask, cols].copy()
        cand["date"], cand["ticker"], cand["market"] = cand.index, ticker, market
        cand["name"] = str(meta["name"])
        return rank.reset_index(drop=True), cand.reset_index(drop=True), None
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), {
            "ticker": ticker, "market": market, "error": f"{type(exc).__name__}: {exc}"}


def choose_exit(x: pd.DataFrame, signal_pos: int, method: str) -> tuple[int, str]:
    """All methods fall back to the SAME fixed-horizon close, never horizon open.

    A profitable closing high arms the stop; only a later close can trigger it.
    The trigger executes next open. Intraday highs never arm a closing-price stop.
    """
    entry, terminal = signal_pos + 1, signal_pos + 20
    if method not in EXITS:
        raise ValueError(method)
    if terminal >= len(x):
        raise ValueError("Incomplete 20-session outcome")
    if method == "fixed20":
        return terminal, "close"
    entry_price = float(x["Open"].iloc[entry])
    peak, armed = entry_price, False
    for j in range(entry, terminal):
        close = float(x["Close"].iloc[j])
        # Cost-aware arming; +10% must remain +10% after assumed round-trip costs.
        if not armed:
            peak = max(peak, close)
            if peak / entry_price - 1 - COST >= 0.10:
                armed = True
            continue
        peak = max(peak, close)
        if method == "profit10_dd5":
            hit = close <= peak * 0.95
        else:
            atr = float(x["ATR"].iloc[j])
            hit = np.isfinite(atr) and close <= peak - 2 * atr
        if hit:
            return j + 1, "open"
    return terminal, "close"


def outcomes(x: pd.DataFrame, benchmark: pd.DataFrame, signal_pos: int) -> dict:
    bench = align_benchmark_ohlc(x, benchmark)
    entry = signal_pos + 1
    ep, bep = float(x["Open"].iloc[entry]), float(bench["Open"].iloc[entry])
    row = {"ENTRY_DATE": x.index[entry], "ENTRY_PRICE": ep}
    for method in EXITS:
        end, phase = choose_exit(x, signal_pos, method)
        col = "Open" if phase == "open" else "Close"
        xp, bxp = float(x[col].iloc[end]), float(bench[col].iloc[end])
        if not all(np.isfinite(p) and p > 0 for p in (ep, bep, xp, bxp)):
            raise ValueError("Invalid execution price")
        ret, br = xp / ep - 1 - COST, bxp / bep - 1
        # A next-open sale cannot observe that day's subsequent high or low.
        path = x.iloc[entry:end] if phase == "open" else x.iloc[entry:end + 1]
        mfe = max(float(path["High"].max()), xp) / ep - 1
        mae = min(float(path["Low"].min()), xp) / ep - 1
        row.update({f"RET_{method}": ret, f"BENCH_{method}": br,
                    f"EXCESS_{method}": ret - br, f"MFE_{method}": mfe,
                    f"MAE_{method}": mae, f"HOLD_{method}": end - entry,
                    f"EXIT_DATE_{method}": x.index[end], f"EXIT_PRICE_{method}": xp,
                    f"EXIT_PHASE_{method}": phase})
    return row


def weights(frame: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "equal":
        return pd.Series(1.0, index=frame.index)
    good = frame["QUALITY_SCORE"] >= 2 if rule == "quality2" else frame[FEATURES[rule]].astype(bool)
    return good.map({True: 1.0, False: 0.5}).astype(float)


def metrics(frame: pd.DataFrame, rule: str, method: str) -> dict:
    if frame.empty:
        return {"signals": 0, "avg_budget_ret": np.nan, "avg_horizon_excess": np.nan}
    w, ret = weights(frame, rule), frame[f"RET_{method}"]
    # Undeployed cash earns zero. Full-slot benchmark stays invested to fixed20.
    budget = w * ret
    return {
        "signals": len(frame), "avg_weight": w.mean(), "avg_budget_ret": budget.mean(),
        "avg_horizon_excess": (budget - frame["BENCH_fixed20"]).mean(),
        "avg_trade_ret": ret.mean(), "avg_matched_excess": frame[f"EXCESS_{method}"].mean(),
        "deployed_weighted_ret": (w * ret).sum() / w.sum(),
        "win_rate": (ret > 0).mean(), "avg_trade_mae": frame[f"MAE_{method}"].mean(),
        "avg_slot_mae": (w * frame[f"MAE_{method}"]).mean(),
        "avg_hold": frame[f"HOLD_{method}"].mean(),
    }


def split(frame: pd.DataFrame, train_end: pd.Timestamp, validation_end: pd.Timestamp):
    # Purge outcomes crossing a model-selection boundary, regardless of exit method.
    terminal = pd.to_datetime(frame["EXIT_DATE_fixed20"])
    train = frame[(frame["date"] <= train_end) & (terminal <= train_end)]
    validation = frame[(frame["date"] > train_end) & (frame["date"] <= validation_end)
                       & (terminal <= validation_end)]
    evaluation = frame[frame["date"] > validation_end]
    return train, validation, evaluation


def select_model(frame: pd.DataFrame, train_end: pd.Timestamp, validation_end: pd.Timestamp):
    train, validation, _ = split(frame, train_end, validation_end)
    baseline_t, baseline_v = metrics(train, "equal", "fixed20"), metrics(validation, "equal", "fixed20")
    rows = []
    for rule in SIZING:
        for method in EXITS:
            t, v = metrics(train, rule, method), metrics(validation, rule, method)
            enough = t["signals"] >= 30 and v["signals"] >= 15
            absolute_positive = all(s[k] > 0 for s in (t, v)
                                    for k in ("avg_budget_ret", "avg_horizon_excess"))
            improves = all(s[k] >= b[k] for s, b in ((t, baseline_t), (v, baseline_v))
                           for k in ("avg_budget_ret", "avg_horizon_excess"))
            improvement = min(t["avg_budget_ret"] - baseline_t["avg_budget_ret"],
                              v["avg_budget_ret"] - baseline_v["avg_budget_ret"])
            eligible = bool(enough and absolute_positive and improves and improvement > 0)
            rows.append({"sizing": rule, "exit": method, "eligible": eligible,
                         "min_return_improvement": improvement,
                         **{f"train_{k}": value for k, value in t.items()},
                         **{f"validation_{k}": value for k, value in v.items()}})
    grid = pd.DataFrame(rows)
    valid = grid[grid["eligible"]].sort_values(
        ["min_return_improvement", "validation_avg_horizon_excess"], ascending=False)
    if valid.empty:
        return {"sizing": "equal", "exit": "fixed20", "status": "fallback_v6"}, grid
    best = valid.iloc[0]
    return {"sizing": best["sizing"], "exit": best["exit"],
            "status": "selected_train_validation"}, grid


def feature_decomposition(frame: pd.DataFrame, market: str, train_end, validation_end):
    rows = []
    for period, subset in zip(("train_purged", "validation_purged", "evaluation_2025_26"),
                              split(frame, train_end, validation_end)):
        for feature in (*FEATURES, "quality2"):
            flag = subset["QUALITY_SCORE"] >= 2 if feature == "quality2" else subset[FEATURES[feature]].astype(bool)
            for state in (False, True):
                rows.append({"market": market, "period": period, "feature": feature,
                             "satisfied": state, **metrics(subset[flag == state], "equal", "fixed20")})
    return rows


def portfolio(frame: pd.DataFrame, rule: str, method: str, benchmark: pd.DataFrame,
              prices: dict[str, pd.DataFrame], max_positions: int = 10):
    """Daily marked equity, shared cash, no leverage, 10% slot before scaling.

    At an open, only open-phase exits release cash. Close-phase exits remain
    occupied until close, including on days another signal tries to enter.
    """
    if frame.empty:
        return pd.DataFrame(), {"accepted": 0, "skipped": 0}
    work = frame.sort_values(["ENTRY_DATE", "ticker"]).copy()
    work["WEIGHT"] = weights(work, rule)
    first, last = work["ENTRY_DATE"].min(), work["EXIT_DATE_fixed20"].max()
    calendar = benchmark.loc[first:last].index
    grouped = {d: g for d, g in work.groupby("ENTRY_DATE")}
    cash, active, accepted, skipped, curve = 1.0, [], 0, 0, []
    first_bench_open = float(benchmark.loc[calendar[0], "Open"])
    for date in calendar:
        # Sell at the open before buying at the same open, pay half the cost each side.
        remaining = []
        for p in active:
            if p["end"] == date and p["phase"] == "open":
                cash += p["shares"] * p["exit_price"] * (1 - COST / 2)
            else:
                remaining.append(p)
        active = remaining
        open_equity = cash + sum(p["shares"] * float(prices[p["ticker"]].loc[date, "Open"])
                                 for p in active)
        for _, r in grouped.get(date, pd.DataFrame()).iterrows():
            target = open_equity / max_positions * float(r["WEIGHT"])
            if len(active) >= max_positions or cash + 1e-12 < target or target <= 0:
                skipped += 1
                continue
            shares = target / (float(r["ENTRY_PRICE"]) * (1 + COST / 2))
            cash -= target
            active.append({"ticker": r["ticker"], "shares": shares,
                           "end": r[f"EXIT_DATE_{method}"], "phase": r[f"EXIT_PHASE_{method}"],
                           "exit_price": float(r[f"EXIT_PRICE_{method}"])})
            accepted += 1
        remaining = []
        for p in active:
            if p["end"] == date and p["phase"] == "close":
                cash += p["shares"] * p["exit_price"] * (1 - COST / 2)
            else:
                remaining.append(p)
        active = remaining
        value = sum(p["shares"] * float(prices[p["ticker"]].loc[date, "Close"]) for p in active)
        equity = cash + value
        curve.append({"date": date, "equity": equity, "cash": cash,
                      "exposure": value / equity, "positions": len(active),
                      "benchmark": float(benchmark.loc[date, "Close"]) / first_bench_open})
        if cash < -1e-9:
            raise AssertionError("Portfolio borrowed cash")
    daily = pd.DataFrame(curve)
    if active:
        raise AssertionError("Positions remain beyond evaluation calendar")
    # Initial equity=1 participates in the high-water mark, including day-one losses.
    high_water = np.maximum.accumulate(np.r_[1.0, daily["equity"].to_numpy()])[1:]
    drawdown = daily["equity"].to_numpy() / high_water - 1
    return daily, {"accepted": accepted, "skipped": skipped,
                   "total_return": daily["equity"].iloc[-1] - 1,
                   "benchmark_return": daily["benchmark"].iloc[-1] - 1,
                   "full_index_excess": daily["equity"].iloc[-1] - daily["benchmark"].iloc[-1],
                   "max_drawdown": drawdown.min(), "avg_exposure": daily["exposure"].mean()}


def run(args: argparse.Namespace):
    out, cache = Path(args.output_dir), Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    settings = {k: getattr(args, k) for k in ("start", "end", "min_adv", "min_price", "cooldown")}
    settings_path = cache / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Cache settings differ; use a different cache directory")
    settings_path.write_text(json.dumps(settings, indent=2))
    universe_path = cache / "universe.csv"
    if universe_path.exists():
        universe = pd.read_csv(universe_path, dtype={"ticker": str})
    else:
        universe = load_krx_universe_frame()
        universe = universe[universe["ticker"].astype(str).str.fullmatch(r"\d{6}")].copy()
        write_csv(universe, universe_path)
    benchmarks = {m: cached_prices(s, args.start, args.end, cache, True)
                  for m, s in (("KOSPI", "KS11"), ("KOSDAQ", "KQ11"))}
    ranks, candidates, failures = [], [], []
    print(f"Numeric-code universe: {len(universe)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_symbol, m, benchmarks, args, cache)
                   for m in universe.to_dict("records")]
        for n, future in enumerate(as_completed(futures), 1):
            rank, cand, failure = future.result()
            if not rank.empty:
                ranks.append(rank)
            if not cand.empty:
                candidates.append(cand)
            if failure:
                failures.append(failure)
            if n % 250 == 0 or n == len(futures):
                print(f"Processed {n}/{len(futures)}; failures={len(failures)}", flush=True)
    write_csv(pd.DataFrame(failures, columns=["ticker", "market", "error"]), out / "failures.csv")
    if failures or not ranks or not candidates:
        raise RuntimeError("Incomplete universe: refusing model selection; see failures.csv")
    panel, cand = pd.concat(ranks, ignore_index=True), pd.concat(candidates, ignore_index=True)
    for h in (20, 60, 120):
        panel[f"RS{h}_PCTL"] = panel.groupby(["date", "market"])[f"RS{h}"].rank(pct=True)
    cand = cand.merge(panel[["date", "ticker", "market", "RS20_PCTL", "RS60_PCTL", "RS120_PCTL"]],
                      on=["date", "ticker", "market"], validate="many_to_one")
    fixed = pd.concat([v7._cooldown(cand.loc[(cand["market"] == market)
                       & _entry_mask_exact(cand, market, "baseline", "none")].copy(), args.cooldown)
                       for market in ("KOSPI", "KOSDAQ")], ignore_index=True)
    # Rebuild future paths only for fixed v6 entries; feature filters cannot replace signals.
    enriched, prices = [], {}
    for ticker, group in fixed.groupby("ticker"):
        stock = read_prices(cache / f"{ticker}.csv.gz")
        market = group["market"].iloc[0]
        prices[ticker] = stock
        x = v7.base_feature_frame(stock, benchmarks[market])
        for _, r in group.iterrows():
            enriched.append({**r.to_dict(), **outcomes(x, benchmarks[market], int(r["row_pos"]))})
    fixed = pd.DataFrame(enriched).sort_values(["date", "ticker"]).reset_index(drop=True)
    write_csv(fixed, out / "fixed_v6_signals.csv")
    te, ve = pd.Timestamp(args.train_end), pd.Timestamp(args.validation_end)
    reproduction = []
    for market, counts in EXPECTED_V6_COUNTS.items():
        f = fixed[fixed["market"] == market]
        actual = {"train": int((f["date"] <= te).sum()),
                  "validation": int(((f["date"] > te) & (f["date"] <= ve)).sum()),
                  "holdout": int((f["date"] > ve).sum())}
        reproduction.append({"market": market, "exact_match": actual == counts,
                             **{f"expected_{k}": v for k, v in counts.items()},
                             **{f"actual_{k}": v for k, v in actual.items()}})
    checks = pd.DataFrame(reproduction)
    write_csv(checks, out / "v6_reproduction_check.csv")
    print(checks.to_string(index=False), flush=True)
    if not checks["exact_match"].all():
        raise AssertionError("v6 baseline drifted; no v8 model will be selected")
    grids, selections, comparison, decomposition, exit_rows, sizing_rows = [], [], [], [], [], []
    portfolio_rows, attribution, yearly = [], [], []
    for market in ("KOSPI", "KOSDAQ"):
        f = fixed[fixed["market"] == market].copy()
        train, validation, evaluation = split(f, te, ve)
        selected, grid = select_model(f, te, ve)
        grid.insert(0, "market", market)
        grids.append(grid)
        selections.append({"market": market, **selected})
        decomposition.extend(feature_decomposition(f, market, te, ve))
        for period, subset in zip(("train_purged", "validation_purged", "evaluation_2025_26"),
                                  (train, validation, evaluation)):
            for method in EXITS:
                exit_rows.append({"market": market, "period": period, "exit": method,
                                  **metrics(subset, "equal", method)})
            for rule in SIZING:
                sizing_rows.append({"market": market, "period": period, "sizing": rule,
                                    **metrics(subset, rule, "fixed20")})
        for label, rule, method in (("v6_baseline", "equal", "fixed20"),
                                    ("v8_selected", selected["sizing"], selected["exit"])):
            comparison.append({"market": market, "model": label, "sizing": rule,
                               "exit": method, **metrics(evaluation, rule, method)})
            for year, subset in evaluation.groupby(evaluation["date"].dt.year):
                yearly.append({"market": market, "model": label, "year": year,
                               **metrics(subset, rule, method)})
            for period, subset in zip(("train_purged", "validation_purged", "evaluation_2025_26"),
                                      (train, validation, evaluation)):
                daily, pm = portfolio(subset, rule, method, benchmarks[market], prices)
                portfolio_rows.append({"market": market, "model": label, "period": period, **pm})
                write_csv(daily, out / f"equity_{market}_{label}_{period}.csv")
        a = evaluation[["date", "ticker", "name", "market", "QUALITY_SCORE"]].copy()
        a["weight"] = weights(evaluation, selected["sizing"])
        a["v6_ret"] = evaluation["RET_fixed20"]
        a["v8_ret"] = evaluation[f"RET_{selected['exit']}"]
        a["entry_budget_delta"] = a["weight"] * a["v8_ret"] - a["v6_ret"]
        a["exit_only_delta"] = a["v8_ret"] - a["v6_ret"]
        attribution.append(a)
        print(f"{market}: {selected}", flush=True)
    for name, data in (("selection_grid", pd.concat(grids)), ("selected_models", pd.DataFrame(selections)),
                       ("holdout_comparison", pd.DataFrame(comparison)),
                       ("feature_decomposition", pd.DataFrame(decomposition)),
                       ("exit_ablation", pd.DataFrame(exit_rows)), ("sizing_ablation", pd.DataFrame(sizing_rows)),
                       ("portfolio_comparison", pd.DataFrame(portfolio_rows)),
                       ("signal_attribution", pd.concat(attribution)), ("yearly_comparison", pd.DataFrame(yearly))):
        write_csv(data, out / f"{name}.csv")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(cache.glob("*.csv*"))}
    manifest = {"settings": settings, "train_end": args.train_end, "validation_end": args.validation_end,
                "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                "cost": COST, "universe_count": len(universe), "data_sha256": hashes,
                "evaluation_status": "2025-26 reused evaluation; not untouched holdout",
                "limitations": ["Current-listed universe retains survivorship bias",
                                "No dividends, taxes beyond fixed 40bp assumption or limit-lock fills",
                                "Portfolio uses actual open/close costs; trade table uses 40bp approximation",
                                "Sparse signal portfolio is compared with a fully invested index"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    print(pd.DataFrame(portfolio_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2022-01-03")
    p.add_argument("--end", default="2026-09-04")
    p.add_argument("--train-end", default="2023-12-28")
    p.add_argument("--validation-end", default="2024-12-30")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--output-dir", default="output_robustness_v8")
    p.add_argument("--cache-dir", default="data_robustness_v8")
    run(p.parse_args())
