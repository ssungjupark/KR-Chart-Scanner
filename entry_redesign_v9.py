from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from kospi_long_history import (
    add_outcomes,
    bootstrap,
    features as v6_features,
    repair_index_bars,
    replay,
    stats,
)
import robustness_v7 as v7


ROUND_TRIP_COST = 0.004

# The order is intentional: diagnostics first, then progressively more structural changes.
MODEL_SPECS: dict[str, dict] = {
    "v6_baseline": {
        "rs": "v6",
        "trigger": "break3",
        "slope120": False,
        "dryup": False,
        "contraction": False,
        "not_bear": False,
        "gap_max": None,
    },
    "diag_setup_no_trigger": {
        "rs": "v6",
        "trigger": "none",
        "slope120": False,
        "dryup": False,
        "contraction": False,
        "not_bear": False,
        "gap_max": None,
    },
    "diag_prev_high": {
        "rs": "v6",
        "trigger": "prev_high",
        "slope120": False,
        "dryup": False,
        "contraction": False,
        "not_bear": False,
        "gap_max": None,
    },
    "diag_ma5_reclaim": {
        "rs": "v6",
        "trigger": "ma5_reclaim",
        "slope120": False,
        "dryup": False,
        "contraction": False,
        "not_bear": False,
        "gap_max": None,
    },
    "diag_ma20_reclaim": {
        "rs": "v6",
        "trigger": "ma20_reclaim",
        "slope120": False,
        "dryup": False,
        "contraction": False,
        "not_bear": False,
        "gap_max": None,
    },
    "diag_longrs_ma5": {
        "rs": "long",
        "trigger": "ma5_reclaim",
        "slope120": False,
        "dryup": False,
        "contraction": False,
        "not_bear": False,
        "gap_max": None,
    },
    "diag_reaccel_ma5": {
        "rs": "reaccel",
        "trigger": "ma5_reclaim",
        "slope120": False,
        "dryup": False,
        "contraction": False,
        "not_bear": False,
        "gap_max": None,
    },
    # Primary redesign: earlier reclaim entry, long-horizon leadership,
    # improving RS, rising 120d trend, avoid bear regimes and chase gaps.
    "v9_core": {
        "rs": "reaccel",
        "trigger": "ma5_reclaim",
        "slope120": True,
        "dryup": False,
        "contraction": False,
        "not_bear": True,
        "gap_max": 0.03,
    },
    "v9_core_dryup": {
        "rs": "reaccel",
        "trigger": "ma5_reclaim",
        "slope120": True,
        "dryup": True,
        "contraction": False,
        "not_bear": True,
        "gap_max": 0.03,
    },
    "v9_core_contraction": {
        "rs": "reaccel",
        "trigger": "ma5_reclaim",
        "slope120": True,
        "dryup": False,
        "contraction": True,
        "not_bear": True,
        "gap_max": 0.03,
    },
    "v9_full_quality": {
        "rs": "reaccel",
        "trigger": "ma5_reclaim",
        "slope120": True,
        "dryup_or_contraction": True,
        "dryup": False,
        "contraction": False,
        "not_bear": True,
        "gap_max": 0.03,
    },
}


def add_v9_features(inputs: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Extend frozen v6 features without changing the v6 baseline definition."""
    x = v6_features(inputs, benchmark)

    prev_close = x["Close"].shift(1)
    prev_sma5 = x["SMA5"].shift(1)
    prev_sma20 = x["SMA20"].shift(1)

    # Baseline trigger retained exactly for reproduction.
    x["TRIG_BREAK3"] = x["trigger"].fillna(False)

    # Earlier re-entry triggers. Each uses only signal-day and older information.
    x["TRIG_PREV_HIGH"] = (
        (x["Close"] > x["High"].shift(1))
        & (x["Close"] > x["SMA5"])
        & (x["SMA5"] > prev_sma5)
    )
    x["TRIG_MA5_RECLAIM"] = (
        (x["Close"] > x["SMA5"])
        & (prev_close <= prev_sma5)
        & (x["SMA5"] > prev_sma5)
    )
    x["TRIG_MA20_RECLAIM"] = (
        (x["Close"] > x["SMA20"])
        & (prev_close <= prev_sma20)
        & (x["SMA20_SLOPE"] > 0)
    )

    # Long trend quality: use a 20-session slope, not the noisy 5-session slope.
    x["SMA120_SLOPE20"] = x["SMA120"] / x["SMA120"].shift(20) - 1.0
    x["SMA120_UP20"] = x["SMA120_SLOPE20"] > 0

    # Pullback volume dry-up. This intentionally excludes signal-day volume.
    prev_vol20 = x["Volume"].shift(1).rolling(20).mean()
    prev_vol5 = x["Volume"].shift(1).rolling(5).mean()
    down = x["Close"].diff() < 0
    up = x["Close"].diff() > 0
    down_vol10 = x["Volume"].where(down).shift(1).rolling(10, min_periods=3).mean()
    up_vol20 = x["Volume"].where(up).shift(1).rolling(20, min_periods=5).mean()
    x["PULLBACK_DRYUP"] = (
        (prev_vol5 <= prev_vol20 * 0.85)
        & (down_vol10 <= up_vol20 * 0.90)
    )

    # Volatility contraction: Bollinger width and MA compression are measured
    # before the signal day to avoid calling the trigger bar itself a squeeze.
    std20 = x["Close"].rolling(20).std()
    x["BB_WIDTH"] = 4.0 * std20 / x["SMA20"].replace(0, np.nan)
    bb_q30 = x["BB_WIDTH"].rolling(120, min_periods=60).quantile(0.30)
    x["BB_SQUEEZE_OK"] = x["BB_WIDTH"].shift(1) <= bb_q30.shift(1)

    ma_stack = pd.concat([x["SMA5"], x["SMA20"], x["SMA60"]], axis=1)
    x["MA_SPREAD"] = (
        (ma_stack.max(axis=1) - ma_stack.min(axis=1))
        / x["Close"].replace(0, np.nan)
    )
    ma_q40 = x["MA_SPREAD"].rolling(120, min_periods=60).quantile(0.40)
    x["MA_COMPRESS_OK"] = x["MA_SPREAD"].shift(1) <= ma_q40.shift(1)
    x["CONTRACTION_OK"] = x["BB_SQUEEZE_OK"] | x["MA_COMPRESS_OK"]

    # RS reacceleration: preserve strong 60/120d leadership while replacing the
    # v6 hard RS20 percentile requirement with an improving 5d ratio trend.
    ratio = x["RS_RATIO"]
    x["RS_RATIO_SLOPE_PREV5"] = ratio.shift(5) / ratio.shift(10) - 1.0
    x["RS_REACCEL"] = (
        (x["RS_RATIO_SLOPE"] > 0)
        & (x["RS_RATIO_SLOPE"] > x["RS_RATIO_SLOPE_PREV5"])
    )
    x["RS_REACCEL_STRICT"] = (
        (x["RS_RATIO_SLOPE"] > 0)
        & (x["RS_RATIO_SLOPE_PREV5"] <= 0)
    )

    x["MARKET_NOT_BEAR"] = x["regime"].ne("bear")

    # Gap is future relative to the signal close, but it is only an execution rule
    # observed at the next open. It never changes whether the signal existed.
    x["NEXT_OPEN"] = x["Open"].shift(-1)
    x["NEXT_ENTRY_GAP"] = x["NEXT_OPEN"] / x["Close"] - 1.0
    return x


def _rs_mask(panel: pd.DataFrame, name: str) -> pd.Series:
    if name == "v6":
        return (
            (panel["RS20"] > 0)
            & (panel["RS_RATIO_SLOPE"] > 0)
            & (panel["RS20_PCTL"] >= 0.50)
            & (panel["RS60_PCTL"] >= 0.60)
            & (panel["RS120_PCTL"] >= 0.70)
        )
    if name == "long":
        return (
            (panel["RS20"] > 0)
            & (panel["RS_RATIO_SLOPE"] > 0)
            & (panel["RS60_PCTL"] >= 0.60)
            & (panel["RS120_PCTL"] >= 0.70)
        )
    if name == "reaccel":
        return (
            (panel["RS20"] > 0)
            & (panel["RS60_PCTL"] >= 0.60)
            & (panel["RS120_PCTL"] >= 0.70)
            & panel["RS_REACCEL"].astype(bool)
        )
    raise ValueError(f"Unknown RS model: {name}")


def _trigger_mask(panel: pd.DataFrame, name: str) -> pd.Series:
    mapping = {
        "none": pd.Series(True, index=panel.index),
        "break3": panel["TRIG_BREAK3"],
        "prev_high": panel["TRIG_PREV_HIGH"],
        "ma5_reclaim": panel["TRIG_MA5_RECLAIM"],
        "ma20_reclaim": panel["TRIG_MA20_RECLAIM"],
    }
    if name not in mapping:
        raise ValueError(f"Unknown trigger: {name}")
    return mapping[name].astype(bool)


def entry_mask(panel: pd.DataFrame, spec: dict) -> pd.Series:
    mask = panel["setup"].astype(bool)
    mask &= _rs_mask(panel, spec["rs"])
    mask &= _trigger_mask(panel, spec["trigger"])

    if spec.get("slope120"):
        mask &= panel["SMA120_UP20"].astype(bool)
    if spec.get("dryup"):
        mask &= panel["PULLBACK_DRYUP"].astype(bool)
    if spec.get("contraction"):
        mask &= panel["CONTRACTION_OK"].astype(bool)
    if spec.get("dryup_or_contraction"):
        mask &= panel["PULLBACK_DRYUP"].astype(bool) | panel["CONTRACTION_OK"].astype(bool)
    if spec.get("not_bear"):
        mask &= panel["MARKET_NOT_BEAR"].astype(bool)

    # Execution-time no-chase rule is deliberately applied before cooldown:
    # a skipped +3% gap does not consume the 20-session re-entry window.
    gap_max = spec.get("gap_max")
    if gap_max is not None:
        mask &= panel["NEXT_ENTRY_GAP"].notna()
        mask &= panel["NEXT_ENTRY_GAP"] <= float(gap_max)
    return mask.fillna(False)


def _period_rows(model: str, f: pd.DataFrame) -> list[dict]:
    periods = [
        ("1996_1999", 1996, 1999),
        ("2000_2007", 2000, 2007),
        ("2008_2014", 2008, 2014),
        ("2015_2021", 2015, 2021),
        ("2022_2024", 2022, 2024),
        ("all_pre2025", 1996, 2024),
        ("2025_2026_reused", 2025, 2026),
    ]
    rows = []
    for label, lo, hi in periods:
        g = f[f["date"].dt.year.between(lo, hi)]
        if hi < 2025:
            g = g[g["exit_date20"] <= pd.Timestamp(f"{hi}-12-31")]
        rows.append({"model": model, "period": label, **stats(g), **bootstrap(g)})
    return rows


def _research_windows(model: str, f: pd.DataFrame) -> list[dict]:
    # These are retrospective diagnostic windows, not untouched holdouts.
    windows = [
        ("development_1996_2012", pd.Timestamp("1996-01-01"), pd.Timestamp("2012-12-31")),
        ("validation_2013_2019", pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31")),
        ("recent_pre2025_2020_2024", pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-30")),
    ]
    rows = []
    for label, lo, hi in windows:
        g = f[(f["date"] >= lo) & (f["date"] <= hi) & (f["exit_date20"] <= hi)]
        rows.append({"model": model, "window": label, **stats(g)})
    return rows


def _adoption_gate(scorecard: pd.DataFrame, model: str = "v9_core") -> dict:
    """Conservative gate; failure means keep v6 as the production reference."""
    g = scorecard[scorecard["model"].eq(model)].set_index("window")
    required = [
        "development_1996_2012",
        "validation_2013_2019",
        "recent_pre2025_2020_2024",
    ]
    if any(w not in g.index for w in required):
        return {"model": model, "adopt": False, "reason": "missing research window"}

    checks = {
        "development_excess_positive": g.loc[required[0], "mean_excess"] > 0,
        "validation_net_positive": g.loc[required[1], "mean_net"] > 0,
        "validation_excess_positive": g.loc[required[1], "mean_excess"] > 0,
        "recent_net_positive": g.loc[required[2], "mean_net"] > 0,
        "recent_excess_positive": g.loc[required[2], "mean_excess"] > 0,
        "validation_min_50": g.loc[required[1], "signals"] >= 50,
        "recent_min_40": g.loc[required[2], "signals"] >= 40,
    }
    return {
        "model": model,
        "adopt": bool(all(checks.values())),
        **{k: bool(v) for k, v in checks.items()},
        "reason": "all prespecified gates passed" if all(checks.values()) else "retain v6; one or more gates failed",
    }


def analyze(args: argparse.Namespace) -> None:
    cache = Path(args.cache_dir)
    amount_cache = Path(args.amount_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    coverage = pd.read_csv(cache / "coverage.csv").fillna("")
    if coverage["error"].ne("").any():
        raise RuntimeError("Incomplete long-history price cache")
    universe = pd.read_csv(cache / "universe.csv", dtype={"ticker": str})

    benchmark = pd.read_csv(cache / "KOSPI.csv.gz", index_col=0, parse_dates=True)
    benchmark = repair_index_bars(benchmark, cache)

    amount_frames = [pd.read_parquet(p) for p in sorted(amount_cache.glob("*.parquet"))]
    if not amount_frames:
        raise RuntimeError("Historical KRX amount data is missing")
    amount = pd.concat(amount_frames, ignore_index=True)
    amount["Date"] = pd.to_datetime(amount["Date"])
    liquidity = {t: g.set_index("Date").sort_index() for t, g in amount.groupby("Code")}
    del amount, amount_frames

    prices: dict[str, pd.DataFrame] = {}
    pieces: list[pd.DataFrame] = []

    for n, meta in enumerate(universe.to_dict("records"), start=1):
        ticker = meta["ticker"]
        stock = pd.read_csv(cache / f"{ticker}.csv.gz", index_col=0, parse_dates=True)
        prices[ticker] = stock

        raw = liquidity.get(
            ticker,
            pd.DataFrame(columns=["Close", "Volume", "Amount", "Market"]),
        ).reindex(stock.index)

        inputs = stock.copy()
        inputs["Amount"] = raw["Amount"]
        inputs["RawClose"] = raw["Close"]
        inputs["Volume"] = raw["Volume"] * raw["Close"] / stock["Close"]

        x = add_v9_features(inputs, benchmark)
        eligible = (
            (x.index >= pd.Timestamp(args.start))
            & (x.index <= pd.Timestamp(args.end))
            & (x["row_pos"] >= DEFAULT_CONFIG.min_history - 1)
            & (x["ADV20"] >= args.min_adv)
            & (x["filter_price"] >= args.min_price)
            & x[["RS20", "RS60", "RS120", "RS_RATIO_SLOPE", "RS_RATIO_SLOPE_PREV5"]].notna().all(axis=1)
            & raw["Market"].eq("KOSPI")
        )
        valid = stock[["Open", "High", "Low", "Close", "Volume"]].gt(0).all(axis=1)
        eligible &= valid & valid.shift(1).rolling(3).sum().eq(3)

        fields = [
            "row_pos", "Close", "ADV20", "RS20", "RS60", "RS120", "RS_RATIO_SLOPE",
            "RS_RATIO_SLOPE_PREV5", "RS_REACCEL", "RS_REACCEL_STRICT", "setup", "regime",
            "dist20", "dd60", "TRIG_BREAK3", "TRIG_PREV_HIGH", "TRIG_MA5_RECLAIM",
            "TRIG_MA20_RECLAIM", "SMA120_SLOPE20", "SMA120_UP20", "PULLBACK_DRYUP",
            "BB_WIDTH", "BB_SQUEEZE_OK", "MA_SPREAD", "MA_COMPRESS_OK", "CONTRACTION_OK",
            "MARKET_NOT_BEAR", "NEXT_OPEN", "NEXT_ENTRY_GAP",
        ]
        panel = x.loc[eligible, fields].copy()
        panel["date"] = panel.index
        panel["ticker"] = ticker
        panel["name"] = meta["name"]
        pieces.append(panel.reset_index(drop=True))

        if n % 100 == 0 or n == len(universe):
            print(f"V9 features {n}/{len(universe)}", flush=True)

    panel = pd.concat(pieces, ignore_index=True)
    for h in (20, 60, 120):
        panel[f"RS{h}_PCTL"] = panel.groupby("date")[f"RS{h}"].rank(pct=True)

    all_signals: list[pd.DataFrame] = []
    funnel: list[dict] = []
    for model, spec in MODEL_SPECS.items():
        raw_mask = entry_mask(panel, spec)
        candidates = panel.loc[raw_mask].copy()
        selected = v7._cooldown(candidates, args.cooldown)
        print(f"{model}: candidates={len(candidates)} after_cooldown={len(selected)}", flush=True)

        model_parts: list[pd.DataFrame] = []
        for ticker, f in selected.groupby("ticker"):
            result = add_outcomes(f, prices[ticker], benchmark)
            result["model"] = model
            model_parts.append(result)
        if model_parts:
            signals = pd.concat(model_parts, ignore_index=True)
            signals = signals[signals["NET20"].notna() & signals["EXCESS20"].notna()].copy()
        else:
            signals = pd.DataFrame()
        all_signals.append(signals)

        funnel.append({
            "model": model,
            "candidate_stock_days": int(raw_mask.sum()),
            "signals_after_cooldown": len(selected),
            "completed_20d_trades": len(signals),
        })

    signals = pd.concat([x for x in all_signals if not x.empty], ignore_index=True)
    signals.to_csv(out / "v9_all_trades.csv.gz", index=False, compression="gzip")
    pd.DataFrame(funnel).to_csv(out / "v9_signal_funnel.csv", index=False)

    period_rows: list[dict] = []
    window_rows: list[dict] = []
    yearly_rows: list[dict] = []
    trigger_rows: list[dict] = []

    for model, f in signals.groupby("model"):
        period_rows.extend(_period_rows(model, f))
        window_rows.extend(_research_windows(model, f))
        for year, g in f.groupby(f["date"].dt.year):
            yearly_rows.append({"model": model, "year": int(year), **stats(g)})
        pre = f[f["exit_date20"] <= pd.Timestamp("2024-12-30")]
        trigger_rows.append({
            "model": model,
            **stats(pre),
        })

    periods = pd.DataFrame(period_rows)
    windows = pd.DataFrame(window_rows)
    yearly = pd.DataFrame(yearly_rows)
    comparison = pd.DataFrame(trigger_rows)

    periods.to_csv(out / "v9_periods.csv", index=False)
    windows.to_csv(out / "v9_research_windows.csv", index=False)
    yearly.to_csv(out / "v9_yearly.csv", index=False)
    comparison.to_csv(out / "v9_pre2025_comparison.csv", index=False)

    gate = _adoption_gate(windows, "v9_core")
    pd.DataFrame([gate]).to_csv(out / "v9_adoption_gate.csv", index=False)

    # Portfolio-level check for baseline and primary candidates.
    portfolio_rows = []
    start = benchmark.loc[args.start:].index[0]
    pre_end = benchmark.loc[:"2024-12-30"].index[-1]
    for model in ("v6_baseline", "v9_core", "v9_full_quality"):
        f = signals[(signals["model"] == model) & (signals["exit_date20"] <= pre_end)].copy()
        if f.empty:
            continue
        daily, keys, s = replay(f, benchmark, prices, start, pre_end)
        daily.to_csv(out / f"equity_{model}_cash.csv", index_label="date")
        portfolio_rows.append({"model": model, "portfolio": "stock_cash", **s})

        d_index, _, si = replay(
            f, benchmark, prices, start, pre_end, asset="index", core=False, keep=keys
        )
        d_index.to_csv(out / f"equity_{model}_same_schedule_index.csv", index_label="date")
        portfolio_rows.append({"model": model, "portfolio": "same_schedule_index", **si})

    pd.DataFrame(portfolio_rows).to_csv(out / "v9_portfolio_pre2025.csv", index=False)

    manifest = {
        "research_only": True,
        "baseline": "frozen v6 KOSPI setup + break3",
        "primary_candidate": "v9_core",
        "transaction_cost": ROUND_TRIP_COST,
        "cooldown": args.cooldown,
        "start": args.start,
        "end": args.end,
        "survivorship_bias": "current 2026 survivors only; historical KOSPI membership applied",
        "selection_note": "All pre-2025 history has now been observed. Research windows are diagnostics, not untouched holdouts.",
        "gap_rule": "skip next-open entry above +3%; skipped order does not consume cooldown",
        "model_specs": MODEL_SPECS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print()
    print("Pre-2025 event-level comparison")
    show_cols = ["model", "signals", "mean_net", "mean_excess", "excess_win_rate", "mean_mae"]
    print(comparison[show_cols].sort_values("mean_excess", ascending=False).to_string(index=False))
    print()
    print("V9 core adoption gate")
    print(pd.DataFrame([gate]).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="V9 KOSPI entry redesign on the frozen long-history data.")
    p.add_argument("--cache-dir", default="data_kospi_long")
    p.add_argument("--amount-dir", default="data_kospi_amount")
    p.add_argument("--output-dir", default="research/v9-entry-results")
    p.add_argument("--start", default="1996-01-01")
    p.add_argument("--end", default="2026-09-04")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
