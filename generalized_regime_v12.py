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

# V12 does not tune the V11 50-65% band. It asks whether a broader,
# economically interpretable state -- positive index momentum, non-extreme
# participation, and improving breadth -- generalizes across eras.
SPECS = {
    "v6_baseline": {"kind": "baseline"},
    "v11_fixed_b50_65_r2p5": {"kind": "fixed_v11"},
    "broadening_mid": {"kind": "broadening_mid"},
    "early_expansion": {"kind": "early_expansion"},
    "dynamic_middle": {"kind": "dynamic_middle"},
    "dynamic_middle_all": {"kind": "dynamic_middle_all"},
    "trend_participation": {"kind": "trend_participation"},
}


def rolling_last_percentile(s: pd.Series, window: int = 756, min_periods: int = 252) -> pd.Series:
    def pct(x: np.ndarray) -> float:
        if len(x) == 0 or not np.isfinite(x[-1]):
            return np.nan
        a = x[np.isfinite(x)]
        if len(a) == 0:
            return np.nan
        return float(np.mean(a <= x[-1]))

    return s.rolling(window, min_periods=min_periods).apply(pct, raw=True)


def apply_spec(panel: pd.DataFrame, spec: dict) -> pd.Series:
    mask = model_mask(panel, {"family": "v6"})
    kind = spec["kind"]

    if kind == "baseline":
        return mask.fillna(False)

    if kind == "fixed_v11":
        mask &= panel["KOSPI_RET60"] > 0.025
        mask &= panel["LIQ_BREADTH60"].between(0.50, 0.65, inclusive="both")

    elif kind == "broadening_mid":
        # Strong index, medium participation, and clear short/medium-term broadening.
        mask &= panel["KOSPI_RET60"] > 0.025
        mask &= panel["KOSPI_ABOVE120"].astype(bool)
        mask &= panel["LIQ_BREADTH60"].between(0.40, 0.70, inclusive="both")
        mask &= panel["LIQ_BREADTH20"] > panel["LIQ_BREADTH60"]
        mask &= panel["LIQ_BREADTH60_D10"] > 0

    elif kind == "early_expansion":
        # A more explicit early-cycle breadth expansion test; no narrow 50-65 band.
        mask &= panel["KOSPI_RET60"] > 0.025
        mask &= panel["KOSPI_ABOVE120"].astype(bool)
        mask &= panel["LIQ_BREADTH60"].between(0.35, 0.70, inclusive="both")
        mask &= panel["LIQ_BREADTH20"] > panel["LIQ_BREADTH60"]
        mask &= panel["LIQ_BREADTH60_D20"] >= 0.05

    elif kind == "dynamic_middle":
        # Replace an absolute breadth band with the middle of its own trailing 3-year distribution.
        mask &= panel["KOSPI_RET60"] > 0.025
        mask &= panel["KOSPI_ABOVE120"].astype(bool)
        mask &= panel["LIQ_BREADTH60_PCTL756"].between(0.35, 0.75, inclusive="both")
        mask &= panel["LIQ_BREADTH60_D20"] > 0

    elif kind == "dynamic_middle_all":
        # Same idea using the wider historical-member breadth, testing universe-definition robustness.
        mask &= panel["KOSPI_RET60"] > 0.025
        mask &= panel["KOSPI_ABOVE120"].astype(bool)
        mask &= panel["ALL_BREADTH60_PCTL756"].between(0.35, 0.75, inclusive="both")
        mask &= panel["ALL_BREADTH60_D20"] > 0

    elif kind == "trend_participation":
        # Minimal structural rule: trend is up, participation is not saturated, and is improving.
        mask &= panel["KOSPI_RET60"] > 0
        mask &= panel["KOSPI_ABOVE120"].astype(bool)
        mask &= panel["LIQ_BREADTH60"] < 0.75
        mask &= panel["LIQ_BREADTH20"] > panel["LIQ_BREADTH60"]
        mask &= panel["LIQ_BREADTH60_D20"] > 0

    else:
        raise ValueError(kind)

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


def rolling_rows(model: str, f: pd.DataFrame) -> list[dict]:
    rows = []
    for start_year in range(1996, 2021):
        lo = pd.Timestamp(f"{start_year}-01-01")
        hi = pd.Timestamp(f"{start_year + 4}-12-31")
        g = f[(f["date"] >= lo) & (f["date"] <= hi) & (f["exit_date20"] <= hi)].copy()
        if len(g) == 0:
            continue
        rows.append({
            "model": model,
            "start_year": start_year,
            "end_year": start_year + 4,
            **stats(g),
        })
    return rows


def gate(periods: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    eras = ["1996_2007", "2008_2014", "2015_2021", "2022_2024"]
    rows = []
    for model in SPECS:
        g = periods[periods.model.eq(model)].set_index("window")
        if "all_pre2025" not in g.index:
            continue
        pre = g.loc["all_pre2025"]
        era_ex = [float(g.loc[e, "mean_excess"]) if e in g.index else np.nan for e in eras]
        era_net = [float(g.loc[e, "mean_net"]) if e in g.index else np.nan for e in eras]
        era_n = [int(g.loc[e, "signals"]) if e in g.index else 0 for e in eras]
        rw = rolling[rolling.model.eq(model)].copy()
        eligible_rw = rw[rw.signals >= 15]
        rolling_positive_share = float((eligible_rw.mean_excess > 0).mean()) if len(eligible_rw) else np.nan
        rolling_worst_excess = float(eligible_rw.mean_excess.min()) if len(eligible_rw) else np.nan
        checks = {
            "pre_net_positive": bool(pre["mean_net"] > 0),
            "pre_excess_positive": bool(pre["mean_excess"] > 0),
            "all_eras_excess_positive": bool(all(x > 0 for x in era_ex)),
            "all_eras_net_positive": bool(all(x > 0 for x in era_net)),
            "min_20_each_era": bool(all(x >= 20 for x in era_n)),
            "top3_removed_positive": bool(pre["top3_removed_excess"] > 0),
            "rolling_positive_ge70pct": bool(np.isfinite(rolling_positive_share) and rolling_positive_share >= 0.70),
            "rolling_worst_above_minus1pct": bool(np.isfinite(rolling_worst_excess) and rolling_worst_excess > -0.01),
            "cluster_ci_low_positive": bool(pre["cluster_ci_low"] > 0),
        }
        research_keys = [k for k in checks if k != "cluster_ci_low_positive"]
        rows.append({
            "model": model,
            "research_pass": bool(all(checks[k] for k in research_keys)),
            "strict_stat_pass": bool(all(checks.values())),
            "min_era_excess": np.nanmin(era_ex),
            "min_era_net": np.nanmin(era_net),
            "min_era_signals": min(era_n),
            "rolling_windows": int(len(eligible_rw)),
            "rolling_positive_share": rolling_positive_share,
            "rolling_worst_excess": rolling_worst_excess,
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
            "SMA120_UP20", "NEXT_ENTRY_GAP", "KOSPI_RET20", "KOSPI_RET60",
            "KOSPI_ABOVE120", "ABOVE20", "ABOVE60", "ABOVE120",
        ]
        p = x.loc[eligible, fields].copy()
        p["date"] = p.index
        p["ticker"] = ticker
        p["name"] = meta["name"]
        pieces.append(p.reset_index(drop=True))

        if n % 100 == 0 or n == len(universe):
            print(f"V12 features {n}/{len(universe)}", flush=True)

    panel = pd.concat(pieces, ignore_index=True)
    for h in (20, 60, 120):
        panel[f"RS{h}_PCTL"] = panel.groupby("date")[f"RS{h}"].rank(pct=True)

    daily = panel.groupby("date")[["ABOVE20", "ABOVE60", "ABOVE120"]].mean().rename(
        columns={
            "ABOVE20": "LIQ_BREADTH20",
            "ABOVE60": "LIQ_BREADTH60",
            "ABOVE120": "LIQ_BREADTH120",
        }
    )
    daily["ALL_BREADTH60"] = all_breadth_num / all_breadth_den.replace(0, np.nan)
    daily = daily.sort_index()
    for col in ["LIQ_BREADTH60", "ALL_BREADTH60"]:
        daily[f"{col}_D10"] = daily[col] - daily[col].shift(10)
        daily[f"{col}_D20"] = daily[col] - daily[col].shift(20)
        daily[f"{col}_PCTL756"] = rolling_last_percentile(daily[col])

    panel = panel.merge(daily, left_on="date", right_index=True, how="left")

    all_signals = []
    funnel = []
    for model, spec in SPECS.items():
        raw_mask = apply_spec(panel, spec)
        candidates = panel.loc[raw_mask].copy()
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
    signals.to_csv(out / "v12_all_trades.csv.gz", index=False, compression="gzip")
    pd.DataFrame(funnel).to_csv(out / "v12_signal_funnel.csv", index=False)

    period_data = []
    rolling_data = []
    yearly_data = []
    for model, f in signals.groupby("model"):
        period_data.extend(period_rows(model, f))
        rolling_data.extend(rolling_rows(model, f))
        pre = f[f["exit_date20"] <= pd.Timestamp("2024-12-30")].copy()
        pre["year"] = pre["date"].dt.year
        for year, g in pre.groupby("year"):
            yearly_data.append({"model": model, "year": int(year), **stats(g)})

    periods = pd.DataFrame(period_data)
    rolling = pd.DataFrame(rolling_data)
    yearly = pd.DataFrame(yearly_data)
    periods.to_csv(out / "v12_periods.csv", index=False)
    rolling.to_csv(out / "v12_rolling5y.csv", index=False)
    yearly.to_csv(out / "v12_yearly.csv", index=False)

    gates = gate(periods, rolling)
    gates.to_csv(out / "v12_gate.csv", index=False)
    pre = periods[periods.window.eq("all_pre2025")].sort_values("mean_excess", ascending=False)
    pre.to_csv(out / "v12_pre2025_ranking.csv", index=False)

    daily.to_csv(out / "v12_market_state_daily.csv.gz", compression="gzip")
    manifest = {
        "research_only": True,
        "important": "V12 generalizes the V11 market-state hypothesis. 1996-2024 is already research-exposed and is not untouched OOS data.",
        "entry": "next trading-day open",
        "exit": "20th trading-day close",
        "round_trip_cost": 0.004,
        "cooldown": args.cooldown,
        "survivorship_bias": "current 2026 survivor universe remains; historical KOSPI membership is applied when available",
        "specs": SPECS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("\n=== V12 PRE-2025 RANKING ===")
    cols = ["model", "signals", "mean_net", "mean_excess", "excess_win_rate", "top3_removed_excess", "cluster_ci_low", "cluster_ci_high"]
    print(pre[cols].to_string(index=False))
    print("\n=== V12 GATE ===")
    print(gates.to_string(index=False))
    print("\n=== 2025-2026 REUSED ===")
    print(periods[periods.window.eq("2025_2026_reused")][cols].sort_values("mean_excess", ascending=False).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="V12 generalized market-state research for V6 KOSPI pullback entries")
    p.add_argument("--cache-dir", default="data_kospi_long")
    p.add_argument("--amount-dir", default="data_kospi_amount")
    p.add_argument("--output-dir", default="research/v12-generalized-regime-results")
    p.add_argument("--start", default="1996-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
