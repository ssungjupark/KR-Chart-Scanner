from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from kospi_long_history import add_outcomes, bootstrap, repair_index_bars, stats
from regime_redesign_v10 import extend_features, model_mask
import robustness_v7 as v7


WINDOWS = [
    ("1996_2007", pd.Timestamp("1996-01-01"), pd.Timestamp("2007-12-31")),
    ("2008_2014", pd.Timestamp("2008-01-01"), pd.Timestamp("2014-12-31")),
    ("2015_2021", pd.Timestamp("2015-01-01"), pd.Timestamp("2021-12-31")),
    ("2022_2024", pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-30")),
    ("2025_2026_reused", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31")),
]

# The 50-65% liquid-universe breadth band was discovered retrospectively in V10.
# V11 treats it as a research hypothesis, never as an untouched validation result.
SPECS = {
    "v6_baseline": {"base": "v6", "breadth": None, "ret60": None, "breadth_col": "LIQ_BREADTH60"},
    "v6_core_b50_65_r0": {"base": "v6", "breadth": (0.50, 0.65), "ret60": 0.00, "breadth_col": "LIQ_BREADTH60"},
    "v6_b50_65_r2p5": {"base": "v6", "breadth": (0.50, 0.65), "ret60": 0.025, "breadth_col": "LIQ_BREADTH60"},
    "v6_b50_65_r5": {"base": "v6", "breadth": (0.50, 0.65), "ret60": 0.05, "breadth_col": "LIQ_BREADTH60"},
    "v6_b45_65_r0": {"base": "v6", "breadth": (0.45, 0.65), "ret60": 0.00, "breadth_col": "LIQ_BREADTH60"},
    "v6_b50_70_r0": {"base": "v6", "breadth": (0.50, 0.70), "ret60": 0.00, "breadth_col": "LIQ_BREADTH60"},
    "v6_b45_70_r0": {"base": "v6", "breadth": (0.45, 0.70), "ret60": 0.00, "breadth_col": "LIQ_BREADTH60"},
    "v6_true_b50_65_r0": {"base": "v6", "breadth": (0.50, 0.65), "ret60": 0.00, "breadth_col": "ALL_BREADTH60"},
    "v6_true_b45_70_r0": {"base": "v6", "breadth": (0.45, 0.70), "ret60": 0.00, "breadth_col": "ALL_BREADTH60"},
    "near_high_core_b50_65_r0": {"base": "near_high", "breadth": (0.50, 0.65), "ret60": 0.00, "breadth_col": "LIQ_BREADTH60"},
}


def apply_spec(panel: pd.DataFrame, spec: dict) -> pd.Series:
    if spec["base"] == "v6":
        mask = model_mask(panel, {"family": "v6"})
    elif spec["base"] == "near_high":
        mask = model_mask(panel, {"family": "near_high120_break20"})
    else:
        raise ValueError(spec["base"])

    if spec["ret60"] is not None:
        mask &= panel["KOSPI_RET60"] > float(spec["ret60"])
    if spec["breadth"] is not None:
        lo, hi = spec["breadth"]
        mask &= panel[spec["breadth_col"]].between(lo, hi, inclusive="both")
    return mask.fillna(False)


def top_removed_excess(g: pd.DataFrame, n: int = 3) -> float:
    if len(g) <= n:
        return np.nan
    return g.nsmallest(len(g) - n, "EXCESS20")["EXCESS20"].mean()


def period_rows(model: str, f: pd.DataFrame) -> list[dict]:
    rows = []
    for label, lo, hi in WINDOWS:
        g = f[(f["date"] >= lo) & (f["date"] <= hi)].copy()
        if hi.year < 2025:
            g = g[g["exit_date20"] <= hi]
        rows.append({"model": model, "window": label, **stats(g), **bootstrap(g)})
    pre = f[f["exit_date20"] <= pd.Timestamp("2024-12-30")].copy()
    rows.append({"model": model, "window": "all_pre2025", **stats(pre), **bootstrap(pre)})
    return rows


def robustness_gate(periods: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eras = ["1996_2007", "2008_2014", "2015_2021", "2022_2024"]
    for model in SPECS:
        g = periods[periods.model.eq(model)].set_index("window")
        if "all_pre2025" not in g.index:
            continue
        pre = g.loc["all_pre2025"]
        era_ex = [float(g.loc[e, "mean_excess"]) if e in g.index else np.nan for e in eras]
        era_net = [float(g.loc[e, "mean_net"]) if e in g.index else np.nan for e in eras]
        era_n = [int(g.loc[e, "signals"]) if e in g.index else 0 for e in eras]
        checks = {
            "pre_net_positive": bool(pre["mean_net"] > 0),
            "pre_excess_positive": bool(pre["mean_excess"] > 0),
            "all_eras_excess_positive": bool(all(x > 0 for x in era_ex)),
            "all_eras_net_positive": bool(all(x > 0 for x in era_net)),
            "min_20_each_era": bool(all(x >= 20 for x in era_n)),
            "top3_removed_positive": bool(pre["top3_removed_excess"] > 0),
            "cluster_ci_low_positive": bool(pre["cluster_ci_low"] > 0),
        }
        rows.append({
            "model": model,
            "research_pass": bool(all(v for k, v in checks.items() if k != "cluster_ci_low_positive")),
            "strict_stat_pass": bool(all(checks.values())),
            "min_era_excess": np.nanmin(era_ex),
            "min_era_net": np.nanmin(era_net),
            "min_era_signals": min(era_n),
            **checks,
        })
    return pd.DataFrame(rows)


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
    all_breadth_num = pd.Series(0.0, index=benchmark.index)
    all_breadth_den = pd.Series(0.0, index=benchmark.index)

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

        valid = stock[["Open", "High", "Low", "Close", "Volume"]].gt(0).all(axis=1)
        historical_member = (
            (x["row_pos"] >= DEFAULT_CONFIG.min_history - 1)
            & raw["Market"].eq("KOSPI")
            & valid
        )
        idx = x.index[historical_member]
        all_breadth_num.loc[idx] += x.loc[idx, "ABOVE60"].astype(float)
        all_breadth_den.loc[idx] += 1.0

        eligible = (
            (x.index >= pd.Timestamp(args.start))
            & (x.index <= pd.Timestamp(args.end))
            & (x["row_pos"] >= DEFAULT_CONFIG.min_history - 1)
            & (x["ADV20"] >= args.min_adv)
            & (x["filter_price"] >= args.min_price)
            & x[["RS20", "RS60", "RS120", "RS_RATIO_SLOPE"]].notna().all(axis=1)
            & raw["Market"].eq("KOSPI")
        )
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
        p = x.loc[eligible, fields].copy()
        p["date"] = p.index
        p["ticker"] = ticker
        p["name"] = meta["name"]
        pieces.append(p.reset_index(drop=True))

        if n % 100 == 0 or n == len(universe):
            print(f"V11 features {n}/{len(universe)}", flush=True)

    panel = pd.concat(pieces, ignore_index=True)
    for h in (20, 60, 120):
        panel[f"RS{h}_PCTL"] = panel.groupby("date")[f"RS{h}"].rank(pct=True)

    liq_breadth = panel.groupby("date")["ABOVE60"].mean().rename("LIQ_BREADTH60")
    all_breadth = (all_breadth_num / all_breadth_den.replace(0, np.nan)).rename("ALL_BREADTH60")
    panel = panel.merge(liq_breadth, left_on="date", right_index=True, how="left")
    panel = panel.merge(all_breadth, left_on="date", right_index=True, how="left")

    all_signals = []
    funnel = []
    for model, spec in SPECS.items():
        raw_mask = apply_spec(panel, spec)
        candidates = panel.loc[raw_mask].copy()
        # Market-state filters are applied BEFORE cooldown. A rejected market day does not consume cooldown.
        selected = v7._cooldown(candidates, args.cooldown)
        print(f"{model}: candidates={len(candidates)} after_cooldown={len(selected)}", flush=True)
        parts = []
        for ticker, f in selected.groupby("ticker"):
            r = add_outcomes(f, prices[ticker], benchmark)
            r["model"] = model
            parts.append(r)
        sig = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not sig.empty:
            sig = sig[sig["NET20"].notna() & sig["EXCESS20"].notna()].copy()
        all_signals.append(sig)
        funnel.append({
            "model": model,
            "candidates_before_cooldown": int(raw_mask.sum()),
            "after_cooldown": len(selected),
            "completed": len(sig),
        })

    signals = pd.concat([x for x in all_signals if not x.empty], ignore_index=True)
    signals.to_csv(out / "v11_all_trades.csv.gz", index=False, compression="gzip")
    pd.DataFrame(funnel).to_csv(out / "v11_signal_funnel.csv", index=False)

    rows = []
    for model, f in signals.groupby("model"):
        rows.extend(period_rows(model, f))
    periods = pd.DataFrame(rows)
    periods.to_csv(out / "v11_periods.csv", index=False)

    gates = robustness_gate(periods)
    gates.to_csv(out / "v11_robustness_gate.csv", index=False)

    pre = periods[periods.window.eq("all_pre2025")].sort_values("mean_excess", ascending=False)
    pre.to_csv(out / "v11_pre2025_ranking.csv", index=False)

    # Breadth audit: how different is the liquid eligible breadth from the wider historical-member breadth?
    audit = panel.groupby("date")[["LIQ_BREADTH60", "ALL_BREADTH60", "KOSPI_RET60"]].first().dropna()
    audit["breadth_gap"] = audit["LIQ_BREADTH60"] - audit["ALL_BREADTH60"]
    audit.describe(percentiles=[.05, .25, .5, .75, .95]).T.to_csv(out / "v11_breadth_audit.csv")

    core = signals[signals.model.eq("v6_core_b50_65_r0")].copy()
    core["year"] = core["date"].dt.year
    yearly = []
    for year, g in core.groupby("year"):
        yearly.append({"year": int(year), **stats(g)})
    pd.DataFrame(yearly).to_csv(out / "v11_core_yearly.csv", index=False)

    manifest = {
        "research_only": True,
        "important": "The core 50-65% breadth hypothesis was discovered from V10 pre-2025 diagnostics. V11 is robustness analysis, not untouched validation.",
        "entry": "next trading-day open",
        "exit": "20th trading-day close",
        "round_trip_cost": 0.004,
        "cooldown": args.cooldown,
        "survivorship_bias": "current 2026 survivor universe remains; historical KOSPI membership is applied when available",
        "specs": SPECS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("\n=== V11 PRE-2025 RANKING ===")
    cols = ["model", "signals", "mean_net", "mean_excess", "excess_win_rate", "top3_removed_excess", "cluster_ci_low", "cluster_ci_high"]
    print(pre[cols].to_string(index=False))
    print("\n=== V11 ROBUSTNESS GATE ===")
    print(gates.to_string(index=False))
    print("\n=== CORE PERIODS ===")
    print(periods[periods.model.eq("v6_core_b50_65_r0")].to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="V11 market-state robustness for V6 KOSPI pullback entries")
    p.add_argument("--cache-dir", default="data_kospi_long")
    p.add_argument("--amount-dir", default="data_kospi_amount")
    p.add_argument("--output-dir", default="research/v11-market-state-results")
    p.add_argument("--start", default="1996-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
