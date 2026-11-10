from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from data_loader import align_benchmark_ohlc, load_benchmark_range, load_price_range
from strategy_validation import _feature_frame as base_feature_frame
from universe import load_krx_universe_frame


EXIT_METHODS = ("fixed20", "ma10", "ma20", "atr2", "protect10_5", "combo")
ENTRY_VARIANTS = ("baseline", "dryup", "rvol", "bb_squeeze", "ma_compress", "retest", "quality2", "quality3")
MARKET_GATES = ("none", "not_bear", "bull")
ROUND_TRIP_COST = 0.004

V6_MODEL = {
    "KOSPI": {
        "setup": "leader20",
        "trigger": "break3",
        "rs20": 0.50,
        "rs60": 0.60,
        "rs120": 0.70,
        "min_train": 60,
        "min_validation": 25,
    },
    "KOSDAQ": {
        "setup": "leader20",
        "trigger": "ma5",
        "rs20": 0.70,
        "rs60": 0.80,
        "rs120": 0.80,
        "min_train": 35,
        "min_validation": 15,
    },
}


def _add_v7_features(x: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()

    prev_vol20 = out["Volume"].shift(1).rolling(20).mean()
    prev_vol5 = out["Volume"].shift(1).rolling(5).mean()
    down = out["Close"].diff() < 0
    up = out["Close"].diff() > 0
    down_vol10 = out["Volume"].where(down).shift(1).rolling(10, min_periods=3).mean()
    up_vol20 = out["Volume"].where(up).shift(1).rolling(20, min_periods=5).mean()

    out["TRIGGER_RVOL"] = out["Volume"] / prev_vol20.replace(0, np.nan)
    out["PULLBACK_DRYUP"] = (
        (prev_vol5 <= prev_vol20 * 0.85)
        & (down_vol10 <= up_vol20 * 0.90)
    )
    out["TRIGGER_RVOL_OK"] = out["TRIGGER_RVOL"] >= 1.20

    sma20 = out["SMA20"]
    std20 = out["Close"].rolling(20).std()
    out["BB_WIDTH"] = (4.0 * std20 / sma20.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    bb_q30 = out["BB_WIDTH"].rolling(120, min_periods=60).quantile(0.30)
    out["BB_SQUEEZE_OK"] = out["BB_WIDTH"].shift(1) <= bb_q30.shift(1)

    ma_stack = pd.concat([out["SMA5"], out["SMA20"], out["SMA60"]], axis=1)
    out["MA_SPREAD"] = (
        (ma_stack.max(axis=1) - ma_stack.min(axis=1))
        / out["Close"].replace(0, np.nan)
    )
    ma_q40 = out["MA_SPREAD"].rolling(120, min_periods=60).quantile(0.40)
    out["MA_COMPRESS_OK"] = (
        (out["MA_SPREAD"].shift(1) <= ma_q40.shift(1))
        & (out["MA_SPREAD"] > out["MA_SPREAD"].shift(1))
        & (out["SMA20_SLOPE"] > 0)
    )

    day_range = (out["High"] - out["Low"]).replace(0, np.nan)
    out["CLOSE_LOCATION"] = (out["Close"] - out["Low"]) / day_range
    out["CLOSE_NEAR_HIGH"] = out["CLOSE_LOCATION"] >= 0.70

    prior_high60 = out["High"].shift(1).rolling(60).max()
    breakout_event = (
        (out["Close"] > prior_high60)
        & (out["TRIGGER_RVOL"] >= 1.20)
    )

    level = np.full(len(out), np.nan)
    age = np.full(len(out), np.nan)
    retest_ok = np.zeros(len(out), dtype=bool)

    last_level = np.nan
    last_event = -1
    retest_seen = False
    failed = False
    prior_break3 = out["High"].shift(1).rolling(3).max().to_numpy(float)
    lows = out["Low"].to_numpy(float)
    highs = out["High"].to_numpy(float)
    closes = out["Close"].to_numpy(float)
    bo = breakout_event.fillna(False).to_numpy(bool)

    for i in range(len(out)):
        if bo[i] and np.isfinite(prior_high60.iloc[i]):
            last_level = float(prior_high60.iloc[i])
            last_event = i
            retest_seen = False
            failed = False
        elif last_event >= 0 and i - last_event > 20:
            last_level = np.nan
            last_event = -1
            retest_seen = False
            failed = False

        if last_event >= 0:
            a = i - last_event
            level[i] = last_level
            age[i] = a
            if 1 <= a <= 20:
                if lows[i] <= last_level * 1.03 and highs[i] >= last_level * 0.97:
                    retest_seen = True
                if closes[i] < last_level * 0.97:
                    failed = True

                retest_ok[i] = bool(
                    a >= 2
                    and retest_seen
                    and not failed
                    and closes[i] >= last_level * 0.99
                    and np.isfinite(prior_break3[i])
                    and closes[i] > prior_break3[i]
                )

    out["BREAKOUT_LEVEL"] = level
    out["BREAKOUT_AGE"] = age
    out["RETEST_RECLAIM_OK"] = retest_ok

    quality_cols = [
        "PULLBACK_DRYUP",
        "TRIGGER_RVOL_OK",
        "BB_SQUEEZE_OK",
        "MA_COMPRESS_OK",
        "RETEST_RECLAIM_OK",
    ]
    out["QUALITY_SCORE"] = sum(out[c].fillna(False).astype(int) for c in quality_cols)

    bench = align_benchmark_ohlc(out, benchmark)
    b20 = bench["Close"].rolling(20).mean()
    b60 = bench["Close"].rolling(60).mean()
    b120 = bench["Close"].rolling(120).mean()
    out["MARKET_NOT_BEAR"] = ~(
        (bench["Close"] < b120)
        & (b20 < b60)
        & (b20 < b20.shift(5))
    )
    out["MARKET_BULL_V7"] = (
        (bench["Close"] > b120)
        & (b20 > b60)
        & (b20 > b20.shift(5))
    )
    out["SMA10"] = out["Close"].rolling(10).mean()
    return out


def _choose_exit_index(x: pd.DataFrame, i: int, method: str, horizon: int = 20) -> int | None:
    n = len(x)
    entry_idx = i + 1
    fixed_idx = i + horizon
    if entry_idx >= n or fixed_idx >= n:
        return None
    if method == "fixed20":
        return fixed_idx

    entry = float(x["Open"].iloc[entry_idx])
    if not np.isfinite(entry) or entry <= 0:
        return None

    peak_close = entry
    for j in range(entry_idx, fixed_idx):
        close = float(x["Close"].iloc[j])
        atr = float(x["ATR"].iloc[j]) if np.isfinite(x["ATR"].iloc[j]) else np.nan
        peak_close = max(peak_close, close)

        if j < entry_idx + 1:
            continue

        hit = False
        if method == "ma10":
            ma10 = x["SMA10"].iloc[j]
            hit = bool(np.isfinite(ma10) and close < ma10)
        elif method == "ma20":
            ma20 = x["SMA20"].iloc[j]
            hit = bool(np.isfinite(ma20) and close < ma20)
        elif method == "atr2":
            hit = bool(np.isfinite(atr) and close < peak_close - 2.0 * atr)
        elif method == "protect10_5":
            hit = bool(peak_close / entry - 1.0 >= 0.10 and close <= peak_close * 0.95)
        elif method == "combo":
            ma10 = x["SMA10"].iloc[j]
            ma_hit = bool(np.isfinite(ma10) and close < ma10)
            atr_hit = bool(np.isfinite(atr) and close < peak_close - 2.0 * atr)
            protect_hit = bool(peak_close / entry - 1.0 >= 0.10 and close <= peak_close * 0.95)
            hit = ma_hit or atr_hit or protect_hit
        else:
            raise ValueError(f"Unknown exit method: {method}")

        if hit:
            return min(j + 1, fixed_idx)
    return fixed_idx


def _add_exit_outcomes(
    x: pd.DataFrame,
    benchmark: pd.DataFrame,
    candidate_positions: np.ndarray,
    cost: float = ROUND_TRIP_COST,
) -> pd.DataFrame:
    bench = align_benchmark_ohlc(x, benchmark)
    rows: list[dict] = []
    for i in candidate_positions.astype(int):
        entry_idx = i + 1
        if entry_idx >= len(x):
            continue
        entry = float(x["Open"].iloc[entry_idx])
        bench_entry = float(bench["Open"].iloc[entry_idx])
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(bench_entry) or bench_entry <= 0:
            continue

        row: dict = {
            "row_pos": i,
            "ENTRY_DATE": x.index[entry_idx].strftime("%Y-%m-%d"),
        }

        for method in EXIT_METHODS:
            exit_idx = _choose_exit_index(x, i, method)
            if exit_idx is None or exit_idx >= len(x):
                continue

            if method == "fixed20":
                exit_price = float(x["Close"].iloc[exit_idx])
                bench_exit = float(bench["Close"].iloc[exit_idx])
            else:
                exit_price = float(x["Open"].iloc[exit_idx])
                bench_exit = float(bench["Open"].iloc[exit_idx])

            if not np.isfinite(exit_price) or exit_price <= 0 or not np.isfinite(bench_exit) or bench_exit <= 0:
                continue

            ret = exit_price / entry - 1.0 - cost
            bench_ret = bench_exit / bench_entry - 1.0
            path = x.iloc[entry_idx : exit_idx + 1]
            mfe = float(path["High"].max() / entry - 1.0)
            mae = float(path["Low"].min() / entry - 1.0)
            capture = ret / mfe if np.isfinite(mfe) and mfe > 0.01 else np.nan

            row[f"RET_{method}"] = ret
            row[f"EXCESS_{method}"] = ret - bench_ret
            row[f"MFE_{method}"] = mfe
            row[f"MAE_{method}"] = mae
            row[f"HOLD_{method}"] = int(exit_idx - entry_idx)
            row[f"CAPTURE_{method}"] = capture
            row[f"EXIT_DATE_{method}"] = x.index[exit_idx].strftime("%Y-%m-%d")

        rows.append(row)
    return pd.DataFrame(rows)


def _process_symbol(meta: dict, benchmarks: dict[str, pd.DataFrame], start: str, end: str,
                    min_adv: float, min_price: float) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    ticker = str(meta["ticker"])
    name = str(meta["name"])
    market = str(meta["market"])
    try:
        stock = load_price_range(ticker, start, end, warmup_days=650, forward_days=75)
        benchmark = benchmarks[market]
        x = base_feature_frame(stock, benchmark)
        x = _add_v7_features(x, benchmark)

        in_period = (x.index >= pd.Timestamp(start)) & (x.index <= pd.Timestamp(end))
        eligible = (
            in_period
            & (x["row_pos"] >= DEFAULT_CONFIG.min_history - 1)
            & (x["ADV20"] >= min_adv)
            & (x["Close"] >= min_price)
            & x["RS20"].notna()
            & x["RS60"].notna()
            & x["RS120"].notna()
        )

        rank = x.loc[eligible, ["row_pos", "RS20", "RS60", "RS120"]].copy()
        rank.insert(0, "date", rank.index)
        rank.insert(1, "ticker", ticker)
        rank.insert(2, "market", market)
        rank = rank.reset_index(drop=True)

        candidate_mask = eligible & x["BASE_LEADER20"]
        positions = x.loc[candidate_mask, "row_pos"].astype(int).to_numpy()
        if len(positions) == 0:
            return rank, pd.DataFrame(), None

        outcomes = _add_exit_outcomes(x, benchmark, positions)
        cols = [
            "row_pos", "Close", "ADV20", "RS20", "RS60", "RS120",
            "BASE_LEADER20", "TRIG_MA5", "TRIG_BREAK3",
            "PULLBACK_DRYUP", "TRIGGER_RVOL", "TRIGGER_RVOL_OK",
            "BB_WIDTH", "BB_SQUEEZE_OK", "MA_SPREAD", "MA_COMPRESS_OK",
            "BREAKOUT_LEVEL", "BREAKOUT_AGE", "RETEST_RECLAIM_OK",
            "CLOSE_LOCATION", "CLOSE_NEAR_HIGH", "QUALITY_SCORE",
            "MARKET_NOT_BEAR", "MARKET_BULL_V7",
        ]
        cand = x.loc[candidate_mask, cols].copy()
        cand.insert(0, "date", cand.index)
        cand.insert(1, "ticker", ticker)
        cand.insert(2, "name", name)
        cand.insert(3, "market", market)
        cand = cand.reset_index(drop=True)
        cand = cand.merge(outcomes, on="row_pos", how="left", validate="one_to_one")
        cand = cand[cand["RET_fixed20"].notna() & cand["EXCESS_fixed20"].notna()].copy()
        return rank, cand, None
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"{ticker},{name},{market},{type(exc).__name__}: {exc}"


def _cooldown(frame: pd.DataFrame, sessions: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    keep: list[int] = []
    for _, g in frame.sort_values(["ticker", "row_pos"]).groupby("ticker", sort=False):
        last = -100_000
        for idx, pos in zip(g.index.to_numpy(), g["row_pos"].astype(int).to_numpy()):
            if pos - last >= sessions:
                keep.append(int(idx))
                last = pos
    return frame.loc[keep].sort_values(["date", "ticker"]).copy()


def _stats(frame: pd.DataFrame, method: str) -> dict:
    ret = pd.to_numeric(frame.get(f"RET_{method}"), errors="coerce")
    ex = pd.to_numeric(frame.get(f"EXCESS_{method}"), errors="coerce")
    mfe = pd.to_numeric(frame.get(f"MFE_{method}"), errors="coerce")
    mae = pd.to_numeric(frame.get(f"MAE_{method}"), errors="coerce")
    hold = pd.to_numeric(frame.get(f"HOLD_{method}"), errors="coerce")
    capture = pd.to_numeric(frame.get(f"CAPTURE_{method}"), errors="coerce")
    valid = ret.notna() & ex.notna()
    if valid.sum() == 0:
        return {
            "signals": 0, "avg_ret": np.nan, "median_ret": np.nan, "win_rate": np.nan,
            "avg_excess": np.nan, "excess_win_rate": np.nan, "avg_mfe": np.nan,
            "avg_mae": np.nan, "avg_hold": np.nan, "median_capture": np.nan,
        }
    return {
        "signals": int(valid.sum()),
        "avg_ret": float(ret[valid].mean()),
        "median_ret": float(ret[valid].median()),
        "win_rate": float((ret[valid] > 0).mean()),
        "avg_excess": float(ex[valid].mean()),
        "excess_win_rate": float((ex[valid] > 0).mean()),
        "avg_mfe": float(mfe[valid].mean()),
        "avg_mae": float(mae[valid].mean()),
        "avg_hold": float(hold[valid].mean()),
        "median_capture": float(capture[valid].replace([np.inf, -np.inf], np.nan).median()),
    }


def _entry_mask(frame: pd.DataFrame, market: str, variant: str, gate: str) -> pd.Series:
    cfg = V6_MODEL[market]
    trigger_col = "TRIG_BREAK3" if cfg["trigger"] == "break3" else "TRIG_MA5"
    mask = (
        frame[trigger_col].astype(bool)
        & (frame["RS20_PCTL"] >= cfg["rs20"])
        & (frame["RS60_PCTL"] >= cfg["rs60"])
        & (frame["RS120_PCTL"] >= cfg["rs120"])
    )

    if variant == "dryup":
        mask &= frame["PULLBACK_DRYUP"]
    elif variant == "rvol":
        mask &= frame["TRIGGER_RVOL_OK"]
    elif variant == "bb_squeeze":
        mask &= frame["BB_SQUEEZE_OK"]
    elif variant == "ma_compress":
        mask &= frame["MA_COMPRESS_OK"]
    elif variant == "retest":
        mask &= frame["RETEST_RECLAIM_OK"]
    elif variant == "quality2":
        mask &= frame["QUALITY_SCORE"] >= 2
    elif variant == "quality3":
        mask &= frame["QUALITY_SCORE"] >= 3
    elif variant != "baseline":
        raise ValueError(variant)

    if gate == "not_bear":
        mask &= frame["MARKET_NOT_BEAR"]
    elif gate == "bull":
        mask &= frame["MARKET_BULL_V7"]
    elif gate != "none":
        raise ValueError(gate)
    return mask.fillna(False)


def _robust_score(ts: dict, vs: dict) -> float:
    vals = [ts["avg_ret"], vs["avg_ret"], ts["avg_excess"], vs["avg_excess"]]
    if not all(np.isfinite(v) for v in vals):
        return -999.0
    return float(
        min(ts["avg_ret"], vs["avg_ret"])
        + min(ts["avg_excess"], vs["avg_excess"])
        + 0.25 * sum(vals)
        + 0.20 * (ts["avg_mae"] + vs["avg_mae"])
    )


def _select_entry_model(frame: pd.DataFrame, market: str, train_end: pd.Timestamp,
                        validation_end: pd.Timestamp, cooldown: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cfg = V6_MODEL[market]
    rows = []
    chosen_cache: dict[tuple[str, str], pd.DataFrame] = {}

    for variant in ENTRY_VARIANTS:
        for gate in MARKET_GATES:
            chosen = _cooldown(frame.loc[_entry_mask(frame, market, variant, gate)].copy(), cooldown)
            chosen_cache[(variant, gate)] = chosen
            tr = chosen[chosen["date"] <= train_end]
            va = chosen[(chosen["date"] > train_end) & (chosen["date"] <= validation_end)]
            ts = _stats(tr, "fixed20")
            vs = _stats(va, "fixed20")
            eligible = bool(
                ts["signals"] >= cfg["min_train"]
                and vs["signals"] >= cfg["min_validation"]
                and ts["avg_ret"] > 0
                and vs["avg_ret"] > 0
                and ts["avg_excess"] > 0
                and vs["avg_excess"] > 0
            )
            rows.append({
                "market": market, "variant": variant, "market_gate": gate,
                **{f"train_{k}": v for k, v in ts.items()},
                **{f"validation_{k}": v for k, v in vs.items()},
                "eligible": eligible,
                "robust_score": _robust_score(ts, vs),
            })

    grid = pd.DataFrame(rows)
    eligible = grid[grid["eligible"]].sort_values(
        ["robust_score", "validation_avg_excess", "validation_avg_ret"],
        ascending=False,
    )
    if eligible.empty:
        baseline = grid[(grid["variant"] == "baseline") & (grid["market_gate"] == "none")].iloc[0]
        best = baseline.to_dict()
        best["selection_status"] = "fallback_to_v6_baseline"
    else:
        best = eligible.iloc[0].to_dict()
        best["selection_status"] = "selected_on_train_validation"

    selected = chosen_cache[(best["variant"], best["market_gate"])]
    return best, selected, grid


def _select_exit_method(selected: pd.DataFrame, train_end: pd.Timestamp,
                        validation_end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    rows = []
    for method in EXIT_METHODS:
        tr = selected[selected["date"] <= train_end]
        va = selected[(selected["date"] > train_end) & (selected["date"] <= validation_end)]
        ts = _stats(tr, method)
        vs = _stats(va, method)
        eligible = bool(
            ts["signals"] > 0
            and vs["signals"] > 0
            and ts["avg_ret"] > 0
            and vs["avg_ret"] > 0
            and ts["avg_excess"] > 0
            and vs["avg_excess"] > 0
        )
        rows.append({
            "method": method,
            **{f"train_{k}": v for k, v in ts.items()},
            **{f"validation_{k}": v for k, v in vs.items()},
            "eligible": eligible,
            "robust_score": _robust_score(ts, vs),
        })
    grid = pd.DataFrame(rows)
    eligible = grid[grid["eligible"]].sort_values(
        ["robust_score", "validation_avg_excess", "validation_avg_ret"],
        ascending=False,
    )
    if eligible.empty:
        best = grid[grid["method"] == "fixed20"].iloc[0].to_dict()
        best["selection_status"] = "fallback_fixed20"
    else:
        best = eligible.iloc[0].to_dict()
        best["selection_status"] = "selected_on_train_validation"
    return best, grid


def _concurrency_cap(frame: pd.DataFrame, method: str, max_positions: int = 10) -> tuple[pd.DataFrame, dict]:
    if frame.empty:
        return frame.copy(), {"accepted": 0, "skipped": 0, "avg_ret": np.nan, "avg_excess": np.nan}
    work = frame.copy()
    work["ENTRY_DATE_TS"] = pd.to_datetime(work["ENTRY_DATE"])
    work["EXIT_DATE_TS"] = pd.to_datetime(work[f"EXIT_DATE_{method}"])
    work = work.sort_values(["ENTRY_DATE_TS", "ticker"])
    active: list[pd.Timestamp] = []
    accepted: list[int] = []
    for idx, row in work.iterrows():
        entry_date = row["ENTRY_DATE_TS"]
        active = [d for d in active if d > entry_date]
        if len(active) >= max_positions:
            continue
        exit_date = row["EXIT_DATE_TS"]
        if pd.isna(exit_date):
            continue
        accepted.append(idx)
        active.append(exit_date)
    chosen = work.loc[accepted].copy()
    return chosen, {
        "accepted": len(chosen),
        "skipped": len(work) - len(chosen),
        "avg_ret": chosen[f"RET_{method}"].mean() if len(chosen) else np.nan,
        "avg_excess": chosen[f"EXCESS_{method}"].mean() if len(chosen) else np.nan,
        "win_rate": (chosen[f"RET_{method}"] > 0).mean() if len(chosen) else np.nan,
    }


def run(start: str, end: str, train_end: str, validation_end: str, min_adv: float,
        min_price: float, cooldown: int, workers: int, output_dir: str) -> None:
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)

    universe = load_krx_universe_frame()
    universe = universe[universe["ticker"].astype(str).str.fullmatch(r"\d{6}")].copy()
    print(f"Numeric-code universe: {len(universe):,}")

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
        for i, future in enumerate(as_completed(futures), 1):
            rank, cand, failure = future.result()
            if not rank.empty:
                ranks.append(rank)
            if not cand.empty:
                candidates.append(cand)
            if failure:
                failures.append(failure)
            if i % 250 == 0 or i == total:
                print(f"Processed {i:,}/{total:,}; failures={len(failures)}")

    rank_panel = pd.concat(ranks, ignore_index=True)
    cand = pd.concat(candidates, ignore_index=True)
    rank_panel["date"] = pd.to_datetime(rank_panel["date"])
    cand["date"] = pd.to_datetime(cand["date"])

    for h in (20, 60, 120):
        rank_panel[f"RS{h}_PCTL"] = rank_panel.groupby(["date", "market"], sort=False)[f"RS{h}"].rank(pct=True)
    key = rank_panel[["date", "ticker", "market", "RS20_PCTL", "RS60_PCTL", "RS120_PCTL"]]
    cand = cand.merge(key, on=["date", "ticker", "market"], how="left", validate="many_to_one")
    cand = cand[cand[["RS20_PCTL", "RS60_PCTL", "RS120_PCTL"]].notna().all(axis=1)].copy()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_ablation = []
    selected_rows = []
    all_exit_grid = []
    yearly_rows = []
    holdout_compare = []
    holdout_signals = []
    concurrency_rows = []

    for market in ("KOSPI", "KOSDAQ"):
        m = cand[cand["market"] == market].copy()
        best_entry, selected, grid = _select_entry_model(
            m, market, train_end_ts, validation_end_ts, cooldown
        )
        all_ablation.append(grid)

        best_exit, exit_grid = _select_exit_method(selected, train_end_ts, validation_end_ts)
        exit_grid.insert(0, "market", market)
        all_exit_grid.append(exit_grid)

        selected_row = {
            "market": market,
            "entry_variant": best_entry["variant"],
            "market_gate": best_entry["market_gate"],
            "entry_selection_status": best_entry["selection_status"],
            "exit_method": best_exit["method"],
            "exit_selection_status": best_exit["selection_status"],
        }
        selected_rows.append(selected_row)

        baseline = _cooldown(
            m.loc[_entry_mask(m, market, "baseline", "none")].copy(), cooldown
        )

        for year in range(2022, 2027):
            y = selected[selected["date"].dt.year == year]
            ys = _stats(y, best_exit["method"])
            yearly_rows.append({
                "market": market,
                "year": year,
                "entry_variant": best_entry["variant"],
                "market_gate": best_entry["market_gate"],
                "exit_method": best_exit["method"],
                **ys,
            })

        holdout_sel = selected[selected["date"] > validation_end_ts].copy()
        holdout_base = baseline[baseline["date"] > validation_end_ts].copy()
        hs = _stats(holdout_sel, best_exit["method"])
        hb = _stats(holdout_base, "fixed20")
        holdout_compare.extend([
            {"market": market, "model": "v6_baseline_net_cost", "method": "fixed20", **hb},
            {
                "market": market,
                "model": "v7_selected",
                "method": best_exit["method"],
                "entry_variant": best_entry["variant"],
                "market_gate": best_entry["market_gate"],
                **hs,
            },
        ])

        if not holdout_sel.empty:
            tmp = holdout_sel.copy()
            tmp["selected_exit"] = best_exit["method"]
            tmp["selected_entry_variant"] = best_entry["variant"]
            tmp["selected_market_gate"] = best_entry["market_gate"]
            holdout_signals.append(tmp)

        _, cstats = _concurrency_cap(holdout_sel, best_exit["method"], max_positions=10)
        concurrency_rows.append({
            "market": market,
            "method": best_exit["method"],
            "max_positions": 10,
            **cstats,
        })

        print("=" * 100)
        print(
            f"{market} V7 selected entry={best_entry['variant']} gate={best_entry['market_gate']} "
            f"exit={best_exit['method']}"
        )
        print(
            f"Holdout baseline net={hb['avg_ret']:+.2%} alpha={hb['avg_excess']:+.2%} "
            f"MAE={hb['avg_mae']:+.2%} n={hb['signals']}"
        )
        print(
            f"Holdout V7      net={hs['avg_ret']:+.2%} alpha={hs['avg_excess']:+.2%} "
            f"MAE={hs['avg_mae']:+.2%} n={hs['signals']}"
        )

    ablation = pd.concat(all_ablation, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    exit_df = pd.concat(all_exit_grid, ignore_index=True)
    yearly_df = pd.DataFrame(yearly_rows)
    holdout_df = pd.DataFrame(holdout_compare)
    concurrency_df = pd.DataFrame(concurrency_rows)
    signals_df = pd.concat(holdout_signals, ignore_index=True) if holdout_signals else pd.DataFrame()

    ablation.to_csv(out / "entry_ablation.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(out / "selected_models.csv", index=False, encoding="utf-8-sig")
    exit_df.to_csv(out / "exit_grid.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(out / "yearly_selected.csv", index=False, encoding="utf-8-sig")
    holdout_df.to_csv(out / "holdout_comparison.csv", index=False, encoding="utf-8-sig")
    concurrency_df.to_csv(out / "concurrency_cap10.csv", index=False, encoding="utf-8-sig")
    signals_df.to_csv(out / "holdout_signals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"failure": failures}).to_csv(out / "failures.csv", index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("SELECTED MODELS")
    print(selected_df.to_string(index=False))
    print("=" * 100)
    print("HOLDOUT COMPARISON")
    print(holdout_df.to_string(index=False))
    print("=" * 100)
    print("YEARLY SELECTED")
    print(yearly_df.to_string(index=False))
    print("=" * 100)
    print("CONCURRENCY-CAPPED HOLDOUT DIAGNOSTIC")
    print(concurrency_df.to_string(index=False))
    print(f"Failures: {len(failures)}")
    print(f"Saved to {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KR Chart Scanner robustness v7")
    p.add_argument("--start", default="2022-01-03")
    p.add_argument("--end", default="2026-09-04")
    p.add_argument("--train-end", default="2023-12-28")
    p.add_argument("--validation-end", default="2024-12-30")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--output-dir", default="output_robustness_v7")
    a = p.parse_args()
    run(
        start=a.start,
        end=a.end,
        train_end=a.train_end,
        validation_end=a.validation_end,
        min_adv=a.min_adv,
        min_price=a.min_price,
        cooldown=a.cooldown,
        workers=a.workers,
        output_dir=a.output_dir,
    )
