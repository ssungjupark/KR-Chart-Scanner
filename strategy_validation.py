from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from data_loader import align_benchmark_ohlc, load_benchmark_range, load_price_range
from indicators import add_indicators
from market_validation_v2 import _confirmed_higher_low_state, _future_max, _future_min
from universe import load_krx_universe_frame


HORIZONS = (5, 10, 20)
SETUPS = ("reversal", "pullback60", "combined")
TRIGGERS = ("ma5", "reclaim", "break3")
RS20_GRID = (0.60, 0.70, 0.80, 0.90)
RS60_GRID = (0.50, 0.60, 0.70)


def _feature_frame(stock: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    cfg = DEFAULT_CONFIG
    bench = align_benchmark_ohlc(stock, benchmark)
    x = add_indicators(stock, bench["Close"], cfg).copy()

    x["row_pos"] = np.arange(len(x), dtype=np.int32)
    x["ADV20"] = (x["Close"] * x["Volume"]).rolling(20).mean()
    x["HL"] = _confirmed_higher_low_state(x, cfg.swing_span)
    x["RS60"] = x["Close"].pct_change(60) - bench["Close"].pct_change(60)

    x["DIST20"] = x["Close"] / x["SMA20"] - 1.0
    x["DIST60"] = x["Close"] / x["SMA60"] - 1.0
    x["MA_GAP"] = (x["SMA20"] / x["SMA60"] - 1.0).abs()
    prior_high60 = x["High"].shift(1).rolling(60).max()
    x["DD60"] = x["Close"] / prior_high60 - 1.0

    prev_vol20 = x["Volume"].shift(1).rolling(20).mean()
    x["VOL_RATIO"] = x["Volume"] / prev_vol20.replace(0, np.nan)
    correction_vol = (
        x["DOWN_VOL20"].notna()
        & x["UP_VOL20"].notna()
        & (x["DOWN_VOL20"] <= x["UP_VOL20"] * 1.10)
    )

    # Setup base only describes the location. Relative-strength rank and the
    # actual re-acceleration trigger are applied later, cross-sectionally.
    x["BASE_REVERSAL"] = (
        x["HL"]
        & (x["SMA20_SLOPE"] > 0)
        & (x["SMA60_SLOPE"] >= -0.01)
        & (x["MA_GAP"] <= 0.04)
        & x["DIST20"].between(-0.03, 0.03)
        & x["DIST60"].between(-0.03, 0.03)
        & x["RSI"].between(42, 68)
    )

    x["BASE_PULLBACK60"] = (
        (x["Close"] > x["SMA120"] * 0.98)
        & (x["SMA60"] > x["SMA120"])
        & (x["SMA60_SLOPE"] > 0)
        & x["DIST60"].between(-0.035, 0.035)
        & x["DD60"].between(-0.22, -0.05)
        & x["RSI"].between(35, 62)
        & correction_vol
    )

    sma5 = x["SMA5"]
    day_range = (x["High"] - x["Low"]).replace(0, np.nan)
    close_location = (x["Close"] - x["Low"]) / day_range

    # Three increasingly strict confirmations that the pullback/reversal has
    # stopped falling and price has begun to re-accelerate.
    x["TRIG_MA5"] = (
        (x["Close"] > sma5)
        & (sma5 > sma5.shift(1))
        & (x["Close"] > x["Close"].shift(1))
        & (close_location >= 0.55)
    )
    x["TRIG_RECLAIM"] = (
        (x["Close"] > x["High"].shift(1))
        & (x["Close"] > sma5)
        & (sma5 >= sma5.shift(1))
    )
    x["TRIG_BREAK3"] = (
        (x["Close"] > x["High"].shift(1).rolling(3).max())
        & (x["Close"] > sma5)
        & (sma5 > sma5.shift(1))
    )

    bench_sma20 = bench["Close"].rolling(20).mean()
    bench_sma60 = bench["Close"].rolling(60).mean()
    bench_sma120 = bench["Close"].rolling(120).mean()
    x["MARKET_BULL"] = (
        (bench["Close"] > bench_sma120)
        & (bench_sma20 > bench_sma60)
        & (bench_sma20 > bench_sma20.shift(5))
    )

    # Signal is known after today's close. All returns enter at next session open.
    entry = x["Open"].shift(-1).where(x["Open"].shift(-1) > 0)
    bench_entry = bench["Open"].shift(-1).where(bench["Open"].shift(-1) > 0)
    x["ENTRY"] = entry
    for h in HORIZONS:
        stock_ret = x["Close"].shift(-h) / entry - 1.0
        bench_ret = bench["Close"].shift(-h) / bench_entry - 1.0
        x[f"RET{h}"] = stock_ret.replace([np.inf, -np.inf], np.nan)
        x[f"EXCESS{h}"] = (stock_ret - bench_ret).replace([np.inf, -np.inf], np.nan)
    x["MFE20"] = (_future_max(x["High"], 20) / entry - 1.0).replace([np.inf, -np.inf], np.nan)
    x["MAE20"] = (_future_min(x["Low"], 20) / entry - 1.0).replace([np.inf, -np.inf], np.nan)
    return x


def _process_symbol(meta: dict, benchmarks: dict[str, pd.DataFrame], start: str, end: str,
                    min_adv: float, min_price: float) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    ticker, name, market = str(meta["ticker"]), str(meta["name"]), str(meta["market"])
    try:
        stock = load_price_range(ticker, start, end, warmup_days=550, forward_days=75)
        x = _feature_frame(stock, benchmarks[market])
        in_period = (x.index >= pd.Timestamp(start)) & (x.index <= pd.Timestamp(end))
        eligible_now = (
            in_period
            & (x["row_pos"] >= DEFAULT_CONFIG.min_history - 1)
            & (x["ADV20"] >= min_adv)
            & (x["Close"] >= min_price)
            & x["RS20"].notna()
            & x["RS60"].notna()
        )

        # Compact panel used only to compute same-day cross-sectional RS percentiles.
        rank = x.loc[eligible_now, ["row_pos", "RS20", "RS60"]].copy()
        rank.insert(0, "date", rank.index)
        rank.insert(1, "ticker", int(ticker))
        rank.insert(2, "market", market)
        rank["RS20"] = rank["RS20"].astype("float32")
        rank["RS60"] = rank["RS60"].astype("float32")
        rank = rank.reset_index(drop=True)

        candidate_mask = eligible_now & (x["BASE_REVERSAL"] | x["BASE_PULLBACK60"])
        candidate_mask &= x["RET20"].notna() & x["EXCESS20"].notna()
        cols = [
            "row_pos", "Close", "ADV20", "HL", "RSI", "RS20", "RS60", "RS_RATIO_SLOPE",
            "DIST20", "DIST60", "DD60", "VOL_RATIO", "MARKET_BULL",
            "BASE_REVERSAL", "BASE_PULLBACK60", "TRIG_MA5", "TRIG_RECLAIM", "TRIG_BREAK3",
            "RET5", "RET10", "RET20", "EXCESS5", "EXCESS10", "EXCESS20", "MFE20", "MAE20",
        ]
        cand = x.loc[candidate_mask, cols].copy()
        cand.insert(0, "date", cand.index)
        cand.insert(1, "ticker", int(ticker))
        cand.insert(2, "name", name)
        cand.insert(3, "market", market)
        cand = cand.reset_index(drop=True)
        return rank, cand, None
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"{ticker},{name},{market},{type(exc).__name__}: {exc}"


def _stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "signals": 0, "avg_5d": np.nan, "avg_10d": np.nan, "avg_20d": np.nan,
            "median_20d": np.nan, "win_rate_20d": np.nan, "avg_excess_20d": np.nan,
            "median_excess_20d": np.nan, "excess_win_rate_20d": np.nan,
            "avg_mfe_20d": np.nan, "avg_mae_20d": np.nan,
        }
    return {
        "signals": len(frame),
        "avg_5d": frame["RET5"].mean(),
        "avg_10d": frame["RET10"].mean(),
        "avg_20d": frame["RET20"].mean(),
        "median_20d": frame["RET20"].median(),
        "win_rate_20d": (frame["RET20"] > 0).mean(),
        "avg_excess_20d": frame["EXCESS20"].mean(),
        "median_excess_20d": frame["EXCESS20"].median(),
        "excess_win_rate_20d": (frame["EXCESS20"] > 0).mean(),
        "avg_mfe_20d": frame["MFE20"].mean(),
        "avg_mae_20d": frame["MAE20"].mean(),
    }


def _cooldown(frame: pd.DataFrame, sessions: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    out_idx: list[int] = []
    for _, g in frame.sort_values(["ticker", "row_pos"]).groupby("ticker", sort=False):
        last = -100000
        for idx, pos in zip(g.index.to_numpy(), g["row_pos"].astype(int).to_numpy()):
            if pos - last >= sessions:
                out_idx.append(int(idx))
                last = pos
    return frame.loc[out_idx].sort_values(["date", "ticker"]).copy()


def _base_subset(candidates: pd.DataFrame, setup: str, trigger: str, require_hl: bool,
                 cooldown: int) -> pd.DataFrame:
    if setup == "reversal":
        base = candidates["BASE_REVERSAL"]
    elif setup == "pullback60":
        base = candidates["BASE_PULLBACK60"]
    else:
        base = candidates["BASE_REVERSAL"] | candidates["BASE_PULLBACK60"]

    trig_col = {"ma5": "TRIG_MA5", "reclaim": "TRIG_RECLAIM", "break3": "TRIG_BREAK3"}[trigger]
    mask = base & candidates[trig_col]
    # These two conditions are deliberately applied before the cross-sectional
    # percentile filter so a top-ranked stock is also improving versus its index.
    mask &= (candidates["RS20"] > 0) & (candidates["RS_RATIO_SLOPE"] > 0)
    if require_hl and setup != "reversal":
        mask &= candidates["HL"]
    return _cooldown(candidates.loc[mask].copy(), cooldown)


def _model_frame(base: pd.DataFrame, rs20_min: float, rs60_min: float, bull_only: bool) -> pd.DataFrame:
    mask = (base["RS20_PCTL"] >= rs20_min) & (base["RS60_PCTL"] >= rs60_min)
    if bull_only:
        mask &= base["MARKET_BULL"]
    return base.loc[mask].copy()


def _utility(stats: dict) -> float:
    if not np.isfinite(stats.get("avg_excess_20d", np.nan)):
        return -999.0
    # Alpha is the main target. Median alpha reduces dependence on a few huge winners,
    # and excess-win rate rewards consistency. MAE receives a small penalty only.
    return float(
        stats["avg_excess_20d"]
        + 0.50 * stats["median_excess_20d"]
        + 0.02 * (stats["excess_win_rate_20d"] - 0.50)
        - 0.03 * max(0.0, abs(stats["avg_mae_20d"]) - 0.08)
    )


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

    ranks: list[pd.DataFrame] = []
    candidates: list[pd.DataFrame] = []
    failures: list[str] = []
    records = universe.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_symbol, meta, benchmarks, start, end, min_adv, min_price): meta
            for meta in records
        }
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            rank, cand, failure = fut.result()
            if not rank.empty:
                ranks.append(rank)
            if not cand.empty:
                candidates.append(cand)
            if failure:
                failures.append(failure)
            if i % 100 == 0 or i == total:
                print(f"Processed {i:,}/{total:,}; failures={len(failures):,}")

    if not ranks or not candidates:
        raise RuntimeError("No eligible observations/candidates")

    rank_panel = pd.concat(ranks, ignore_index=True)
    cand = pd.concat(candidates, ignore_index=True)
    for frame in (rank_panel, cand):
        frame["date"] = pd.to_datetime(frame["date"])

    print(f"Rank observations: {len(rank_panel):,}")
    print(f"Base candidates  : {len(cand):,}")

    rank_panel["RS20_PCTL"] = rank_panel.groupby(["date", "market"], sort=False)["RS20"].rank(pct=True)
    rank_panel["RS60_PCTL"] = rank_panel.groupby(["date", "market"], sort=False)["RS60"].rank(pct=True)
    rank_key = rank_panel[["date", "ticker", "market", "RS20_PCTL", "RS60_PCTL"]]
    cand = cand.merge(rank_key, on=["date", "ticker", "market"], how="left", validate="many_to_one")
    cand = cand[cand["RS20_PCTL"].notna() & cand["RS60_PCTL"].notna()].copy()

    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    train_mask = cand["date"] <= train_end_ts
    val_mask = (cand["date"] > train_end_ts) & (cand["date"] <= validation_end_ts)
    test_mask = cand["date"] > validation_end_ts

    # Pre-cache structural/trigger subsets. Percentile thresholds and market regime
    # are applied after this, so model selection remains fast and transparent.
    cache: dict[tuple[str, str, bool], pd.DataFrame] = {}
    for setup, trigger, require_hl in product(SETUPS, TRIGGERS, (False, True)):
        if setup == "reversal" and require_hl:
            continue
        cache[(setup, trigger, require_hl)] = _base_subset(cand, setup, trigger, require_hl, cooldown)

    model_rows: list[dict] = []
    model_id = 0
    for setup, trigger, require_hl, rs20_min, rs60_min, bull_only in product(
        SETUPS, TRIGGERS, (False, True), RS20_GRID, RS60_GRID, (False, True)
    ):
        if setup == "reversal" and require_hl:
            continue
        base = cache[(setup, trigger, require_hl)]
        chosen = _model_frame(base, rs20_min, rs60_min, bull_only)
        tr = chosen[chosen["date"] <= train_end_ts]
        va = chosen[(chosen["date"] > train_end_ts) & (chosen["date"] <= validation_end_ts)]
        tr_stats = _stats(tr)
        va_stats = _stats(va)
        model_id += 1
        row = {
            "model_id": model_id, "setup": setup, "trigger": trigger,
            "require_hl": require_hl, "rs20_min": rs20_min, "rs60_min": rs60_min,
            "bull_only": bull_only,
            **{f"train_{k}": v for k, v in tr_stats.items()},
            **{f"validation_{k}": v for k, v in va_stats.items()},
        }
        row["validation_utility"] = _utility(va_stats)
        row["eligible"] = bool(
            tr_stats["signals"] >= 120
            and va_stats["signals"] >= 60
            and tr_stats["avg_excess_20d"] > 0
            and tr_stats["excess_win_rate_20d"] >= 0.50
        )
        model_rows.append(row)

    models = pd.DataFrame(model_rows)
    eligible_models = models[models["eligible"]].copy()
    if eligible_models.empty:
        # Fallback still uses validation only; holdout remains untouched.
        eligible_models = models[(models["train_signals"] >= 120) & (models["validation_signals"] >= 60)].copy()
        print("WARNING: no model passed positive-alpha train gate; using count-qualified validation ranking")
    eligible_models = eligible_models.sort_values(
        ["validation_utility", "validation_avg_excess_20d", "validation_excess_win_rate_20d"],
        ascending=False,
    )
    best = eligible_models.iloc[0]

    key = (str(best["setup"]), str(best["trigger"]), bool(best["require_hl"]))
    selected_base = cache[key]
    selected = _model_frame(
        selected_base, float(best["rs20_min"]), float(best["rs60_min"]), bool(best["bull_only"])
    )
    test = selected[selected["date"] > validation_end_ts].copy()
    test_stats = _stats(test)

    selected_summary = {
        "setup": best["setup"], "trigger": best["trigger"], "require_hl": bool(best["require_hl"]),
        "rs20_min": float(best["rs20_min"]), "rs60_min": float(best["rs60_min"]),
        "bull_only": bool(best["bull_only"]),
        "train_signals": int(best["train_signals"]),
        "train_avg_excess_20d": float(best["train_avg_excess_20d"]),
        "validation_signals": int(best["validation_signals"]),
        "validation_avg_excess_20d": float(best["validation_avg_excess_20d"]),
        "validation_excess_win_rate_20d": float(best["validation_excess_win_rate_20d"]),
        "validation_utility": float(best["validation_utility"]),
        **{f"test_{k}": v for k, v in test_stats.items()},
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models.sort_values("validation_utility", ascending=False).to_csv(out / "model_grid.csv", index=False, encoding="utf-8-sig")
    eligible_models.head(30).to_csv(out / "top_validation_models.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([selected_summary]).to_csv(out / "selected_model_summary.csv", index=False, encoding="utf-8-sig")
    test.to_csv(out / "selected_holdout_signals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"failure": failures}).to_csv(out / "failures.csv", index=False, encoding="utf-8-sig")

    print("=" * 112)
    print("KR-Chart-Scanner | Cross-sectional RS + entry-trigger validation v4")
    print(f"Period       : {start} ~ {end}")
    print(f"Train        : through {train_end}")
    print(f"Validation   : through {validation_end}")
    print(f"Holdout test : after {validation_end}")
    print(f"ADV20 filter : >= {min_adv/1e8:.0f}억원")
    print(f"Models tested: {len(models):,}")
    print("=" * 112)
    print("SELECTED USING TRAIN + VALIDATION ONLY")
    print(
        f"setup={selected_summary['setup']} trigger={selected_summary['trigger']} "
        f"HL={selected_summary['require_hl']} RS20pct>={selected_summary['rs20_min']:.0%} "
        f"RS60pct>={selected_summary['rs60_min']:.0%} bull_only={selected_summary['bull_only']}"
    )
    print(
        f"Train      n={selected_summary['train_signals']:,} "
        f"excess20={selected_summary['train_avg_excess_20d']:+.2%}"
    )
    print(
        f"Validation n={selected_summary['validation_signals']:,} "
        f"excess20={selected_summary['validation_avg_excess_20d']:+.2%} "
        f"excess-win={selected_summary['validation_excess_win_rate_20d']:.2%}"
    )
    print("HOLDOUT RESULT — NOT USED FOR SELECTION")
    print(
        f"Test n={test_stats['signals']:,} avg20={test_stats['avg_20d']:+.2%} "
        f"median20={test_stats['median_20d']:+.2%} win20={test_stats['win_rate_20d']:.2%} "
        f"excess20={test_stats['avg_excess_20d']:+.2%} "
        f"excess-win={test_stats['excess_win_rate_20d']:.2%} "
        f"MFE={test_stats['avg_mfe_20d']:+.2%} MAE={test_stats['avg_mae_20d']:+.2%}"
    )
    print("=" * 112)
    print("Top validation models")
    show_cols = [
        "setup", "trigger", "require_hl", "rs20_min", "rs60_min", "bull_only",
        "train_signals", "train_avg_excess_20d", "validation_signals",
        "validation_avg_excess_20d", "validation_excess_win_rate_20d", "validation_utility",
    ]
    print(eligible_models[show_cols].head(12).to_string(index=False))
    print(f"Saved to {out}/")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-01-03")
    p.add_argument("--end", default="2026-09-04")
    p.add_argument("--train-end", default="2023-12-28")
    p.add_argument("--validation-end", default="2024-12-30")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1_000)
    p.add_argument("--cooldown", type=int, default=20)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--output-dir", default="output_strategy")
    a = p.parse_args()
    run(
        a.start, a.end, a.train_end, a.validation_end, a.min_adv, a.min_price,
        a.cooldown, a.workers, (a.max_symbols or None), a.output_dir,
    )


if __name__ == "__main__":
    main()
