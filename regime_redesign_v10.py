from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from entry_redesign_v9 import add_v9_features
from kospi_long_history import add_outcomes, bootstrap, repair_index_bars, replay, stats
import robustness_v7 as v7


ROUND_TRIP_COST = 0.004

WINDOWS = [
    ("1996_2007", pd.Timestamp("1996-01-01"), pd.Timestamp("2007-12-31")),
    ("2008_2014", pd.Timestamp("2008-01-01"), pd.Timestamp("2014-12-31")),
    ("2015_2021", pd.Timestamp("2015-01-01"), pd.Timestamp("2021-12-31")),
    ("2022_2024", pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-30")),
    ("2025_2026_reused", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31")),
]

MODEL_SPECS = {
    "v6_baseline": {"family": "v6"},
    "trend_break20": {"family": "trend_break20"},
    "near_high120_break20": {"family": "near_high120_break20"},
    "breakout60": {"family": "breakout60"},
    "breakout60_volume": {"family": "breakout60_volume"},
    "breakout120": {"family": "breakout120"},
    "fresh_trend_break20": {"family": "fresh_trend_break20"},
}


def extend_features(inputs: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    x = add_v9_features(inputs, benchmark)

    x["PRIOR_HIGH20"] = x["High"].shift(1).rolling(20).max()
    x["PRIOR_HIGH60"] = x["High"].shift(1).rolling(60).max()
    x["PRIOR_HIGH120"] = x["High"].shift(1).rolling(120).max()

    x["BREAK20"] = x["Close"] > x["PRIOR_HIGH20"]
    x["BREAK60"] = x["Close"] > x["PRIOR_HIGH60"]
    x["BREAK120"] = x["Close"] > x["PRIOR_HIGH120"]

    x["HIGH120_DIST"] = x["Close"] / x["PRIOR_HIGH120"] - 1.0
    x["EXT20"] = x["Close"] / x["SMA20"] - 1.0

    prev_vol20 = x["Volume"].shift(1).rolling(20).mean()
    x["VOL_RATIO20"] = x["Volume"] / prev_vol20.replace(0, np.nan)

    cross20_60 = (x["SMA20"] > x["SMA60"]) & (x["SMA20"].shift(1) <= x["SMA60"].shift(1))
    x["CROSS20_60_RECENT20"] = cross20_60.rolling(20).max().fillna(0).astype(bool)

    x["TREND_STACK"] = (
        (x["Close"] > x["SMA20"])
        & (x["SMA20"] > x["SMA60"])
        & (x["SMA60"] > x["SMA120"])
        & (x["SMA20_SLOPE"] > 0)
        & (x["SMA60_SLOPE"] > 0)
        & x["SMA120_UP20"].fillna(False)
    )

    b = benchmark.reindex(x.index)
    x["KOSPI_RET20"] = b["Close"] / b["Close"].shift(20) - 1.0
    x["KOSPI_RET60"] = b["Close"] / b["Close"].shift(60) - 1.0
    bma120 = b["Close"].rolling(120).mean()
    x["KOSPI_ABOVE120"] = b["Close"] > bma120

    x["ABOVE20"] = x["Close"] > x["SMA20"]
    x["ABOVE60"] = x["Close"] > x["SMA60"]
    x["ABOVE120"] = x["Close"] > x["SMA120"]
    return x


def leader_rs(panel: pd.DataFrame, loose: bool = False) -> pd.Series:
    floor = 0.60 if loose else 0.70
    return (
        (panel["RS20"] > 0)
        & (panel["RS60_PCTL"] >= floor)
        & (panel["RS120_PCTL"] >= floor)
        & (panel["RS_RATIO_SLOPE"] > 0)
    )


def model_mask(panel: pd.DataFrame, spec: dict) -> pd.Series:
    family = spec["family"]
    if family == "v6":
        return (
            panel["setup"].astype(bool)
            & panel["trigger"].astype(bool)
            & (panel["RS20"] > 0)
            & (panel["RS_RATIO_SLOPE"] > 0)
            & (panel["RS20_PCTL"] >= 0.50)
            & (panel["RS60_PCTL"] >= 0.60)
            & (panel["RS120_PCTL"] >= 0.70)
        ).fillna(False)

    trend = panel["TREND_STACK"].astype(bool)
    rs = leader_rs(panel)
    ext_ok = panel["EXT20"].between(-0.02, 0.12)

    if family == "trend_break20":
        mask = trend & rs & panel["BREAK20"].astype(bool) & ext_ok
    elif family == "near_high120_break20":
        mask = (
            trend
            & rs
            & panel["BREAK20"].astype(bool)
            & (panel["HIGH120_DIST"] >= -0.03)
            & ext_ok
        )
    elif family == "breakout60":
        mask = trend & rs & panel["BREAK60"].astype(bool) & ext_ok
    elif family == "breakout60_volume":
        mask = (
            trend
            & rs
            & panel["BREAK60"].astype(bool)
            & ext_ok
            & (panel["VOL_RATIO20"] >= 1.20)
        )
    elif family == "breakout120":
        mask = trend & rs & panel["BREAK120"].astype(bool) & ext_ok
    elif family == "fresh_trend_break20":
        mask = (
            trend
            & leader_rs(panel, loose=True)
            & panel["CROSS20_60_RECENT20"].astype(bool)
            & panel["BREAK20"].astype(bool)
            & panel["EXT20"].between(-0.02, 0.08)
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    return mask.fillna(False)


def period_label(d: pd.Timestamp) -> str:
    y = d.year
    if y <= 2007:
        return "1996_2007"
    if y <= 2014:
        return "2008_2014"
    if y <= 2021:
        return "2015_2021"
    if y <= 2024:
        return "2022_2024"
    return "2025_2026_reused"


def window_rows(model: str, f: pd.DataFrame) -> list[dict]:
    out = []
    for label, lo, hi in WINDOWS:
        g = f[(f["date"] >= lo) & (f["date"] <= hi)]
        if hi.year < 2025:
            g = g[g["exit_date20"] <= hi]
        out.append({"model": model, "window": label, **stats(g), **bootstrap(g)})
    pre = f[f["exit_date20"] <= pd.Timestamp("2024-12-30")]
    out.append({"model": model, "window": "all_pre2025", **stats(pre), **bootstrap(pre)})
    return out


def adoption_rows(scorecard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pre_windows = ["1996_2007", "2008_2014", "2015_2021", "2022_2024"]
    for model in MODEL_SPECS:
        g = scorecard[scorecard["model"].eq(model)].set_index("window")
        if "all_pre2025" not in g.index:
            continue
        overall = g.loc["all_pre2025"]
        positives = sum(
            bool((w in g.index) and (g.loc[w, "mean_excess"] > 0))
            for w in pre_windows
        )
        checks = {
            "overall_net_positive": overall["mean_net"] > 0,
            "overall_excess_positive": overall["mean_excess"] > 0,
            "positive_pre2025_windows_ge3": positives >= 3,
            "recent_2022_2024_positive": (
                "2022_2024" in g.index and g.loc["2022_2024", "mean_excess"] > 0
            ),
            "min_100_pre2025_signals": overall["signals"] >= 100,
            "top3_removed_excess_positive": overall["top3_removed_excess"] > 0,
        }
        rows.append({
            "model": model,
            "adopt": bool(all(checks.values())) and model != "v6_baseline",
            "positive_pre2025_windows": positives,
            **{k: bool(v) for k, v in checks.items()},
        })
    return pd.DataFrame(rows)


def _bin_summary(base: pd.DataFrame, factor: str, labels: pd.Series) -> pd.DataFrame:
    x = base.copy()
    x["bucket"] = labels.astype("object")
    x = x[x["bucket"].notna()].copy()
    rows = []
    for (period, bucket), g in x.groupby(["period_group", "bucket"], observed=True):
        rows.append({
            "factor": factor,
            "period_group": period,
            "bucket": str(bucket),
            **stats(g),
        })
    return pd.DataFrame(rows)


def diagnostics(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = base.copy()
    base["period_group"] = base["date"].map(period_label)

    features = [
        "RS20_PCTL", "RS60_PCTL", "RS120_PCTL", "RS_RATIO_SLOPE",
        "dist20", "dd60", "HIGH120_DIST", "EXT20", "VOL_RATIO20",
        "SMA120_SLOPE20", "BREADTH20", "BREADTH60", "BREADTH120",
        "KOSPI_RET20", "KOSPI_RET60", "SIGNALS_TODAY",
    ]
    rows = []
    for period, g in base.groupby("period_group"):
        r = {
            "period_group": period,
            "signals": len(g),
            "mean_net": g["NET20"].mean(),
            "mean_excess": g["EXCESS20"].mean(),
            "excess_win_rate": (g["EXCESS20"] > 0).mean(),
        }
        for col in features:
            r[f"{col}_mean"] = g[col].mean()
            r[f"{col}_median"] = g[col].median()
        rows.append(r)
    shift = pd.DataFrame(rows)

    parts = []
    parts.append(_bin_summary(base, "regime", base["regime"]))
    parts.append(_bin_summary(
        base,
        "breadth60",
        pd.cut(base["BREADTH60"], [-np.inf, .35, .50, .65, np.inf],
               labels=["<35%", "35-50%", "50-65%", ">=65%"]),
    ))
    parts.append(_bin_summary(
        base,
        "kospi_ret60",
        pd.cut(base["KOSPI_RET60"], [-np.inf, -.10, 0, .10, np.inf],
               labels=["<=-10%", "-10~0%", "0~10%", ">10%"]),
    ))
    parts.append(_bin_summary(
        base,
        "rs20_pctl",
        pd.cut(base["RS20_PCTL"], [.50, .60, .75, .90, 1.000001],
               labels=["50-60", "60-75", "75-90", "90-100"], include_lowest=True),
    ))
    parts.append(_bin_summary(
        base,
        "high120_dist",
        pd.cut(base["HIGH120_DIST"], [-np.inf, -.15, -.08, -.03, np.inf],
               labels=["<=-15%", "-15~-8%", "-8~-3%", ">-3%"]),
    ))
    parts.append(_bin_summary(
        base,
        "dist20",
        pd.cut(base["dist20"], [-np.inf, 0, .02, np.inf],
               labels=["below20", "0~2%", ">2%"]),
    ))
    parts.append(_bin_summary(
        base,
        "signal_crowding",
        pd.cut(base["SIGNALS_TODAY"], [0, 1, 3, np.inf],
               labels=["1", "2-3", "4+"], include_lowest=True),
    ))
    conditional = pd.concat(parts, ignore_index=True)
    return shift, conditional


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

        x = extend_features(inputs, benchmark)
        eligible = (
            (x.index >= pd.Timestamp(args.start))
            & (x.index <= pd.Timestamp(args.end))
            & (x["row_pos"] >= DEFAULT_CONFIG.min_history - 1)
            & (x["ADV20"] >= args.min_adv)
            & (x["filter_price"] >= args.min_price)
            & x[["RS20", "RS60", "RS120", "RS_RATIO_SLOPE"]].notna().all(axis=1)
            & raw["Market"].eq("KOSPI")
        )
        valid = stock[["Open", "High", "Low", "Close", "Volume"]].gt(0).all(axis=1)
        eligible &= valid & valid.shift(1).rolling(3).sum().eq(3)

        fields = [
            "row_pos", "Close", "SMA20", "SMA60", "SMA120", "SMA20_SLOPE",
            "SMA60_SLOPE", "ADV20", "RS20", "RS60", "RS120", "RS_RATIO_SLOPE",
            "setup", "trigger", "regime", "dist20", "dd60", "SMA120_SLOPE20",
            "SMA120_UP20", "NEXT_ENTRY_GAP", "PRIOR_HIGH20", "PRIOR_HIGH60",
            "PRIOR_HIGH120", "BREAK20", "BREAK60", "BREAK120", "HIGH120_DIST",
            "EXT20", "VOL_RATIO20", "CROSS20_60_RECENT20", "TREND_STACK",
            "KOSPI_RET20", "KOSPI_RET60", "KOSPI_ABOVE120", "ABOVE20",
            "ABOVE60", "ABOVE120",
        ]
        panel = x.loc[eligible, fields].copy()
        panel["date"] = panel.index
        panel["ticker"] = ticker
        panel["name"] = meta["name"]
        pieces.append(panel.reset_index(drop=True))

        if n % 100 == 0 or n == len(universe):
            print(f"V10 features {n}/{len(universe)}", flush=True)

    panel = pd.concat(pieces, ignore_index=True)
    for h in (20, 60, 120):
        panel[f"RS{h}_PCTL"] = panel.groupby("date")[f"RS{h}"].rank(pct=True)

    breadth = panel.groupby("date")[["ABOVE20", "ABOVE60", "ABOVE120"]].mean()
    breadth.columns = ["BREADTH20", "BREADTH60", "BREADTH120"]
    panel = panel.merge(breadth, left_on="date", right_index=True, how="left")

    all_signals = []
    funnel = []
    for model, spec in MODEL_SPECS.items():
        raw_mask = model_mask(panel, spec)
        candidates = panel.loc[raw_mask].copy()
        selected = v7._cooldown(candidates, args.cooldown)
        print(f"{model}: candidates={len(candidates)} after_cooldown={len(selected)}", flush=True)

        parts = []
        for ticker, f in selected.groupby("ticker"):
            r = add_outcomes(f, prices[ticker], benchmark)
            r["model"] = model
            parts.append(r)
        signals = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not signals.empty:
            signals = signals[signals["NET20"].notna() & signals["EXCESS20"].notna()].copy()
        all_signals.append(signals)
        funnel.append({
            "model": model,
            "candidate_stock_days": int(raw_mask.sum()),
            "signals_after_cooldown": len(selected),
            "completed_20d_trades": len(signals),
        })

    signals = pd.concat([x for x in all_signals if not x.empty], ignore_index=True)
    signals["SIGNALS_TODAY"] = signals.groupby(["model", "date"])["ticker"].transform("count")
    signals.to_csv(out / "v10_all_trades.csv.gz", index=False, compression="gzip")
    pd.DataFrame(funnel).to_csv(out / "v10_signal_funnel.csv", index=False)

    score_rows = []
    yearly_rows = []
    for model, f in signals.groupby("model"):
        score_rows.extend(window_rows(model, f))
        for year, g in f.groupby(f["date"].dt.year):
            yearly_rows.append({"model": model, "year": int(year), **stats(g)})

    score = pd.DataFrame(score_rows)
    yearly = pd.DataFrame(yearly_rows)
    gates = adoption_rows(score)
    score.to_csv(out / "v10_windows.csv", index=False)
    yearly.to_csv(out / "v10_yearly.csv", index=False)
    gates.to_csv(out / "v10_adoption_gate.csv", index=False)

    base = signals[signals["model"].eq("v6_baseline")].copy()
    shift, conditional = diagnostics(base)
    shift.to_csv(out / "v10_v6_regime_feature_shift.csv", index=False)
    conditional.to_csv(out / "v10_v6_conditional_performance.csv", index=False)

    pre = score[score["window"].eq("all_pre2025")].copy()
    pre = pre.sort_values("mean_excess", ascending=False)
    pre.to_csv(out / "v10_pre2025_ranking.csv", index=False)

    portfolio_rows = []
    start = benchmark.loc[args.start:].index[0]
    pre_end = benchmark.loc[:"2024-12-30"].index[-1]
    replay_models = pre.head(3)["model"].tolist()
    if "v6_baseline" not in replay_models:
        replay_models.append("v6_baseline")
    for model in replay_models:
        f = signals[(signals["model"].eq(model)) & (signals["exit_date20"] <= pre_end)].copy()
        if f.empty:
            continue
        daily, keys, s = replay(f, benchmark, prices, start, pre_end)
        portfolio_rows.append({"model": model, "portfolio": "stock_cash", **s})
        d_index, _, si = replay(
            f, benchmark, prices, start, pre_end, asset="index", core=False, keep=keys
        )
        portfolio_rows.append({"model": model, "portfolio": "same_schedule_index", **si})
    pd.DataFrame(portfolio_rows).to_csv(out / "v10_portfolio_pre2025.csv", index=False)

    manifest = {
        "research_only": True,
        "purpose": "Explain why V6 improves in 2025-2026 and compare structural strategy families.",
        "transaction_cost": ROUND_TRIP_COST,
        "selection_rule": "2025-2026 is descriptive only and is not used by the adoption gate.",
        "adoption_gate": "positive net and excess pre-2025, positive excess in >=3/4 pre-2025 windows, 2022-2024 positive, >=100 signals, top-3 winners removed still positive.",
        "survivorship_bias": "current 2026 survivors only; historical KOSPI membership applied",
        "models": MODEL_SPECS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("\n=== V10 PRE-2025 RANKING ===")
    cols = ["model", "signals", "mean_net", "mean_excess", "excess_win_rate", "top3_removed_excess"]
    print(pre[cols].to_string(index=False))
    print("\n=== V10 ADOPTION GATE ===")
    print(gates.to_string(index=False))
    print("\n=== V6 PERIOD FEATURE SHIFT ===")
    print(shift.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="V10 KOSPI regime decomposition and structural redesign.")
    p.add_argument("--cache-dir", default="data_kospi_long")
    p.add_argument("--amount-dir", default="data_kospi_amount")
    p.add_argument("--output-dir", default="research/v10-regime-results")
    p.add_argument("--start", default="1996-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    args = p.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
