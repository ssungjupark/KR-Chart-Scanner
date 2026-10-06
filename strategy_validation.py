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
SETUPS = ("leader20", "leader60", "leader_combo", "reversal")
TRIGGERS = ("ma5", "reclaim", "break3")
RS20_GRID = (0.50, 0.60, 0.70, 0.80)
RS60_GRID = (0.60, 0.70, 0.80)
RS120_GRID = (0.60, 0.70, 0.80)
SECTOR_GRID = (0.00, 0.50, 0.70)


def _feature_frame(stock: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    cfg = DEFAULT_CONFIG
    bench = align_benchmark_ohlc(stock, benchmark)
    x = add_indicators(stock, bench["Close"], cfg).copy()

    x["row_pos"] = np.arange(len(x), dtype=np.int32)
    x["ADV20"] = (x["Close"] * x["Volume"]).rolling(20).mean()
    x["HL"] = _confirmed_higher_low_state(x, cfg.swing_span)
    x["RS60"] = x["Close"].pct_change(60) - bench["Close"].pct_change(60)
    x["RS120"] = x["Close"].pct_change(120) - bench["Close"].pct_change(120)

    x["DIST20"] = x["Close"] / x["SMA20"] - 1.0
    x["DIST60"] = x["Close"] / x["SMA60"] - 1.0
    x["MA_GAP"] = (x["SMA20"] / x["SMA60"] - 1.0).abs()
    prior_high60 = x["High"].shift(1).rolling(60).max()
    prior_high120 = x["High"].shift(1).rolling(120).max()
    x["DD60"] = x["Close"] / prior_high60 - 1.0
    x["DD120"] = x["Close"] / prior_high120 - 1.0

    prev_vol20 = x["Volume"].shift(1).rolling(20).mean()
    x["VOL_RATIO"] = x["Volume"] / prev_vol20.replace(0, np.nan)
    correction_vol = (
        x["DOWN_VOL20"].notna()
        & x["UP_VOL20"].notna()
        & (x["DOWN_VOL20"] <= x["UP_VOL20"] * 1.10)
    )

    # Mature market leaders. Cross-sectional RS percentile gates are applied later.
    x["BASE_LEADER20"] = (
        (x["Close"] > x["SMA120"])
        & (x["SMA20"] > x["SMA60"])
        & (x["SMA60"] > x["SMA120"])
        & (x["SMA20_SLOPE"] > 0)
        & (x["SMA60_SLOPE"] > 0)
        & x["DIST20"].between(-0.03, 0.04)
        & x["DD60"].between(-0.16, -0.02)
        & x["RSI"].between(40, 72)
        & correction_vol
    )

    x["BASE_LEADER60"] = (
        (x["Close"] > x["SMA120"] * 0.98)
        & (x["SMA60"] > x["SMA120"])
        & (x["SMA60_SLOPE"] > 0)
        & x["DIST60"].between(-0.035, 0.035)
        & x["DD120"].between(-0.26, -0.05)
        & x["RSI"].between(35, 64)
        & correction_vol
    )

    # Early reversal remains in the research grid, but it is no longer assumed to
    # be the preferred live setup. It must earn its place out-of-sample.
    x["BASE_REVERSAL"] = (
        x["HL"]
        & (x["SMA20_SLOPE"] > 0)
        & (x["SMA60_SLOPE"] >= -0.01)
        & (x["MA_GAP"] <= 0.04)
        & x["DIST20"].between(-0.03, 0.03)
        & x["DIST60"].between(-0.03, 0.03)
        & x["RSI"].between(42, 68)
    )

    sma5 = x["SMA5"]
    day_range = (x["High"] - x["Low"]).replace(0, np.nan)
    close_location = (x["Close"] - x["Low"]) / day_range
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
    ticker = str(meta["ticker"])
    name = str(meta["name"])
    market = str(meta["market"])
    sector = str(meta.get("sector", "UNKNOWN") or "UNKNOWN")
    try:
        stock = load_price_range(ticker, start, end, warmup_days=650, forward_days=75)
        x = _feature_frame(stock, benchmarks[market])
        in_period = (x.index >= pd.Timestamp(start)) & (x.index <= pd.Timestamp(end))
        eligible_now = (
            in_period
            & (x["row_pos"] >= DEFAULT_CONFIG.min_history - 1)
            & (x["ADV20"] >= min_adv)
            & (x["Close"] >= min_price)
            & x["RS20"].notna() & x["RS60"].notna() & x["RS120"].notna()
        )

        rank = x.loc[eligible_now, ["row_pos", "RS20", "RS60", "RS120"]].copy()
        rank.insert(0, "date", rank.index)
        rank.insert(1, "ticker", ticker)
        rank.insert(2, "market", market)
        rank.insert(3, "sector", sector)
        for c in ("RS20", "RS60", "RS120"):
            rank[c] = rank[c].astype("float32")
        rank = rank.reset_index(drop=True)

        candidate_mask = eligible_now & (x["BASE_LEADER20"] | x["BASE_LEADER60"] | x["BASE_REVERSAL"])
        candidate_mask &= x["RET20"].notna() & x["EXCESS20"].notna()
        cols = [
            "row_pos", "Close", "ADV20", "HL", "RSI", "RS20", "RS60", "RS120", "RS_RATIO_SLOPE",
            "DIST20", "DIST60", "DD60", "DD120", "VOL_RATIO", "MARKET_BULL",
            "BASE_LEADER20", "BASE_LEADER60", "BASE_REVERSAL",
            "TRIG_MA5", "TRIG_RECLAIM", "TRIG_BREAK3",
            "RET5", "RET10", "RET20", "EXCESS5", "EXCESS10", "EXCESS20", "MFE20", "MAE20",
        ]
        cand = x.loc[candidate_mask, cols].copy()
        cand.insert(0, "date", cand.index)
        cand.insert(1, "ticker", ticker)
        cand.insert(2, "name", name)
        cand.insert(3, "market", market)
        cand.insert(4, "sector", sector)
        return rank, cand.reset_index(drop=True), None
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


def _base_subset(candidates: pd.DataFrame, setup: str, trigger: str, cooldown: int) -> pd.DataFrame:
    if setup == "leader20":
        base = candidates["BASE_LEADER20"]
    elif setup == "leader60":
        base = candidates["BASE_LEADER60"]
    elif setup == "leader_combo":
        base = candidates["BASE_LEADER20"] | candidates["BASE_LEADER60"]
    else:
        base = candidates["BASE_REVERSAL"]

    trig_col = {"ma5": "TRIG_MA5", "reclaim": "TRIG_RECLAIM", "break3": "TRIG_BREAK3"}[trigger]
    mask = base & candidates[trig_col]
    mask &= (candidates["RS20"] > 0) & (candidates["RS_RATIO_SLOPE"] > 0)
    return _cooldown(candidates.loc[mask].copy(), cooldown)


def _model_frame(base: pd.DataFrame, rs20_min: float, rs60_min: float, rs120_min: float,
                 sector_min: float, bull_only: bool) -> pd.DataFrame:
    mask = (
        (base["RS20_PCTL"] >= rs20_min)
        & (base["RS60_PCTL"] >= rs60_min)
        & (base["RS120_PCTL"] >= rs120_min)
    )
    if sector_min > 0:
        mask &= (base["SECTOR_N"] >= 3) & (base["SECTOR_RS60_PCTL"] >= sector_min)
    if bull_only:
        mask &= base["MARKET_BULL"]
    return base.loc[mask].copy()


def _utility(stats: dict) -> float:
    if not np.isfinite(stats.get("avg_excess_20d", np.nan)):
        return -999.0
    return float(
        stats["avg_excess_20d"]
        + 0.50 * stats["median_excess_20d"]
        + 0.02 * (stats["excess_win_rate_20d"] - 0.50)
        - 0.03 * max(0.0, abs(stats["avg_mae_20d"]) - 0.08)
    )


def _select_market_models(cand: pd.DataFrame, market: str, train_end: pd.Timestamp,
                          validation_end: pd.Timestamp, cooldown: int) -> tuple[pd.DataFrame, dict | None, pd.DataFrame]:
    m = cand[cand["market"] == market].copy()
    cache = {(s, t): _base_subset(m, s, t, cooldown) for s, t in product(SETUPS, TRIGGERS)}
    rows: list[dict] = []
    model_id = 0
    for setup, trigger, rs20_min, rs60_min, rs120_min, sector_min, bull_only in product(
        SETUPS, TRIGGERS, RS20_GRID, RS60_GRID, RS120_GRID, SECTOR_GRID, (False, True)
    ):
        chosen = _model_frame(cache[(setup, trigger)], rs20_min, rs60_min, rs120_min, sector_min, bull_only)
        tr = chosen[chosen["date"] <= train_end]
        va = chosen[(chosen["date"] > train_end) & (chosen["date"] <= validation_end)]
        ts = _stats(tr)
        vs = _stats(va)
        model_id += 1
        row = {
            "market": market, "model_id": model_id, "setup": setup, "trigger": trigger,
            "rs20_min": rs20_min, "rs60_min": rs60_min, "rs120_min": rs120_min,
            "sector_min": sector_min, "bull_only": bull_only,
            **{f"train_{k}": v for k, v in ts.items()},
            **{f"validation_{k}": v for k, v in vs.items()},
        }
        row["validation_utility"] = _utility(vs)
        row["eligible"] = bool(
            ts["signals"] >= 60
            and vs["signals"] >= 25
            and ts["avg_excess_20d"] > 0
            and ts["excess_win_rate_20d"] >= 0.50
            and vs["avg_excess_20d"] > 0
            and vs["excess_win_rate_20d"] >= 0.50
        )
        rows.append(row)

    grid = pd.DataFrame(rows)
    eligible = grid[grid["eligible"]].sort_values(
        ["validation_utility", "validation_avg_excess_20d", "validation_excess_win_rate_20d"],
        ascending=False,
    )
    if eligible.empty:
        return grid, None, pd.DataFrame()

    best = eligible.iloc[0]
    base = cache[(str(best["setup"]), str(best["trigger"]))]
    selected = _model_frame(
        base, float(best["rs20_min"]), float(best["rs60_min"]), float(best["rs120_min"]),
        float(best["sector_min"]), bool(best["bull_only"]),
    )
    info = best.to_dict()
    return grid, info, selected


def run(start: str, end: str, train_end: str, validation_end: str,
        min_adv: float, min_price: float, cooldown: int, workers: int,
        max_symbols: int | None, output_dir: str) -> None:
    universe = load_krx_universe_frame(max_symbols=max_symbols)
    print(f"Universe: {len(universe):,} currently listed KOSPI/KOSDAQ common stocks")
    known_sector = (universe.get("sector", pd.Series("UNKNOWN", index=universe.index)) != "UNKNOWN").mean()
    print(f"Sector metadata coverage: {known_sector:.1%}")
    print("NOTE: current-listing universe has survivorship bias because delisted listing data is unavailable in this runtime.")

    benchmarks = {
        "KOSPI": load_benchmark_range("KS11", start, end, warmup_days=650, forward_days=75),
        "KOSDAQ": load_benchmark_range("KQ11", start, end, warmup_days=650, forward_days=75),
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
            rank, c, failure = fut.result()
            if not rank.empty:
                ranks.append(rank)
            if not c.empty:
                candidates.append(c)
            if failure:
                failures.append(failure)
            if i % 100 == 0 or i == total:
                print(f"Processed {i:,}/{total:,}; failures={len(failures):,}")

    if not ranks or not candidates:
        raise RuntimeError("No eligible observations/candidates")

    rank_panel = pd.concat(ranks, ignore_index=True)
    cand = pd.concat(candidates, ignore_index=True)
    rank_panel["date"] = pd.to_datetime(rank_panel["date"])
    cand["date"] = pd.to_datetime(cand["date"])
    print(f"Rank observations: {len(rank_panel):,}")
    print(f"Base candidates  : {len(cand):,}")

    for horizon in (20, 60, 120):
        rank_panel[f"RS{horizon}_PCTL"] = rank_panel.groupby(["date", "market"], sort=False)[f"RS{horizon}"].rank(pct=True)

    sector_stats = (
        rank_panel[rank_panel["sector"] != "UNKNOWN"]
        .groupby(["date", "market", "sector"], sort=False)
        .agg(SECTOR_RS60=("RS60", "median"), SECTOR_N=("ticker", "nunique"))
        .reset_index()
    )
    if not sector_stats.empty:
        sector_stats["SECTOR_RS60_PCTL"] = sector_stats.groupby(["date", "market"], sort=False)["SECTOR_RS60"].rank(pct=True)
    else:
        sector_stats = pd.DataFrame(columns=["date", "market", "sector", "SECTOR_RS60", "SECTOR_N", "SECTOR_RS60_PCTL"])

    rank_key = rank_panel[["date", "ticker", "market", "RS20_PCTL", "RS60_PCTL", "RS120_PCTL"]]
    cand = cand.merge(rank_key, on=["date", "ticker", "market"], how="left", validate="many_to_one")
    cand = cand.merge(sector_stats, on=["date", "market", "sector"], how="left", validate="many_to_one")
    cand["SECTOR_N"] = cand["SECTOR_N"].fillna(0)
    cand["SECTOR_RS60_PCTL"] = cand["SECTOR_RS60_PCTL"].fillna(0.0)
    cand = cand[cand[["RS20_PCTL", "RS60_PCTL", "RS120_PCTL"]].notna().all(axis=1)].copy()

    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)

    grids: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    selected_rows: list[dict] = []
    for market in ("KOSPI", "KOSDAQ"):
        grid, info, selected = _select_market_models(cand, market, train_end_ts, validation_end_ts, cooldown)
        grids.append(grid)
        if info is None:
            print(f"{market}: DISABLED — no model showed positive alpha in both train and validation with minimum sample size")
            fallback = grid[(grid["train_signals"] >= 60) & (grid["validation_signals"] >= 25)].sort_values(
                "validation_utility", ascending=False
            ).head(1)
            if not fallback.empty:
                r = fallback.iloc[0]
                print(
                    f"  best fallback: {r.setup}/{r.trigger}, train excess={r.train_avg_excess_20d:+.2%}, "
                    f"validation excess={r.validation_avg_excess_20d:+.2%}"
                )
            continue
        test = selected[selected["date"] > validation_end_ts].copy()
        test_stats = _stats(test)
        info.update({f"test_{k}": v for k, v in test_stats.items()})
        selected_rows.append(info)
        selected_frames.append(test.assign(selected_market_model=market))
        print(
            f"{market}: ENABLED {info['setup']}/{info['trigger']} "
            f"RS20>={info['rs20_min']:.0%} RS60>={info['rs60_min']:.0%} RS120>={info['rs120_min']:.0%} "
            f"sector>={info['sector_min']:.0%} bull={info['bull_only']}"
        )
        print(
            f"  train excess={info['train_avg_excess_20d']:+.2%} n={int(info['train_signals'])}, "
            f"validation excess={info['validation_avg_excess_20d']:+.2%} n={int(info['validation_signals'])}"
        )
        print(
            f"  HOLDOUT: avg20={test_stats['avg_20d']:+.2%}, excess20={test_stats['avg_excess_20d']:+.2%}, "
            f"excess-win={test_stats['excess_win_rate_20d']:.2%}, n={test_stats['signals']}"
        )

    all_grid = pd.concat(grids, ignore_index=True)
    holdout = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    combined_stats = _stats(holdout) if not holdout.empty else _stats(pd.DataFrame())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_grid.to_csv(out / "model_grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(out / "selected_market_models.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(out / "selected_holdout_signals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([combined_stats]).to_csv(out / "combined_holdout_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"failure": failures}).to_csv(out / "failures.csv", index=False, encoding="utf-8-sig")

    print("=" * 116)
    print("KR-Chart-Scanner | Sector + long-term leader validation v5")
    print(f"Period       : {start} ~ {end}")
    print(f"Train        : through {train_end}")
    print(f"Validation   : through {validation_end}")
    print(f"Holdout test : after {validation_end} (never used to select a model)")
    print(f"Models tested: {len(all_grid):,}")
    print(f"Enabled markets: {', '.join([r['market'] for r in selected_rows]) if selected_rows else 'NONE'}")
    if not holdout.empty:
        print(
            f"COMBINED HOLDOUT n={combined_stats['signals']:,} avg20={combined_stats['avg_20d']:+.2%} "
            f"median20={combined_stats['median_20d']:+.2%} win20={combined_stats['win_rate_20d']:.2%} "
            f"excess20={combined_stats['avg_excess_20d']:+.2%} "
            f"excess-win={combined_stats['excess_win_rate_20d']:.2%} "
            f"MFE={combined_stats['avg_mfe_20d']:+.2%} MAE={combined_stats['avg_mae_20d']:+.2%}"
        )
    print(f"Failures: {len(failures)}")
    print(f"Saved to {out}/")
    print("=" * 116)


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
