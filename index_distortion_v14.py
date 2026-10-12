from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from kospi_long_history import add_outcomes, bootstrap, repair_index_bars, stats
from regime_redesign_v10 import extend_features, model_mask
import generalized_regime_v12 as v12
import robustness_v7 as v7


MEGA_TICKERS = {"005930", "000660"}  # Samsung Electronics, SK hynix
WINDOWS = v12.WINDOWS

# V14 is diagnostic. Thresholds are prespecified and are not grid-searched.
SPECS = {
    "v6_baseline": {"kind": "baseline"},
    "v13_original": {"kind": "v13"},
    "v13_exmega_trades": {"kind": "v13_exmega_trades"},
    "broad_confirmation": {"kind": "broad_confirmation"},
    "capweight_unconfirmed": {"kind": "capweight_unconfirmed"},
    "ew_market_strong": {"kind": "ew_market_strong"},
    "median_market_strong": {"kind": "median_market_strong"},
}


def rolling_compound_return(s: pd.Series, window: int = 60) -> pd.Series:
    return (1.0 + s).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0


def add_market_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Create causal market proxies from the liquid KOSPI cross-section.

    The equal-weight proxies are deliberately independent of KOSPI index weights.
    Samsung Electronics and SK hynix are also removed in EXMEGA variants so that
    a cap-weighted megacap rally cannot mechanically define the market state.
    """
    daily = panel.groupby("date")[["ABOVE20", "ABOVE60", "ABOVE120"]].mean().rename(
        columns={
            "ABOVE20": "LIQ_BREADTH20",
            "ABOVE60": "LIQ_BREADTH60",
            "ABOVE120": "LIQ_BREADTH120",
        }
    )
    daily["KOSPI_RET60"] = panel.groupby("date")["KOSPI_RET60"].first()
    daily["KOSPI_ABOVE120"] = panel.groupby("date")["KOSPI_ABOVE120"].first()
    daily["LIQ_EW_RET1"] = panel.groupby("date")["RET1"].mean()
    daily["LIQ_MEDIAN_RET60"] = panel.groupby("date")["STOCK_RET60"].median()

    ex = panel.loc[~panel["is_mega"]].copy()
    daily["EXMEGA_EW_RET1"] = ex.groupby("date")["RET1"].mean()
    daily["EXMEGA_MEDIAN_RET60"] = ex.groupby("date")["STOCK_RET60"].median()
    daily["EXMEGA_BREADTH60"] = ex.groupby("date")["ABOVE60"].mean()

    daily = daily.sort_index()
    daily["LIQ_EW_RET60"] = rolling_compound_return(daily["LIQ_EW_RET1"], 60)
    daily["EXMEGA_EW_RET60"] = rolling_compound_return(daily["EXMEGA_EW_RET1"], 60)

    pct_cols = [
        "KOSPI_RET60",
        "LIQ_BREADTH60",
        "EXMEGA_BREADTH60",
        "LIQ_EW_RET60",
        "EXMEGA_EW_RET60",
        "LIQ_MEDIAN_RET60",
        "EXMEGA_MEDIAN_RET60",
    ]
    for col in pct_cols:
        daily[f"{col}_PCTL756"] = v12.rolling_last_percentile(
            daily[col], window=756, min_periods=252
        )

    daily["CAPWEIGHT_MINUS_EXMEGA_EW60"] = daily["KOSPI_RET60"] - daily["EXMEGA_EW_RET60"]
    daily["KOSPI_MINUS_TYPICAL60"] = daily["KOSPI_RET60"] - daily["EXMEGA_MEDIAN_RET60"]
    return daily


def v13_state(panel: pd.DataFrame) -> pd.Series:
    return (
        panel["KOSPI_ABOVE120"].astype(bool)
        & (panel["KOSPI_RET60_PCTL756"] >= 0.65)
        & panel["LIQ_BREADTH60_PCTL756"].between(0.30, 0.70, inclusive="both")
    ).fillna(False)


def apply_spec(panel: pd.DataFrame, spec: dict) -> pd.Series:
    base = model_mask(panel, {"family": "v6"})
    kind = spec["kind"]

    if kind == "baseline":
        return base.fillna(False)

    regime = v13_state(panel)
    if kind == "v13":
        mask = base & regime
    elif kind == "v13_exmega_trades":
        mask = base & regime & (~panel["is_mega"])
    elif kind == "broad_confirmation":
        # V13 state plus a strong non-megacap equal-weight market.
        mask = (
            base
            & regime
            & (panel["EXMEGA_EW_RET60"] > 0)
            & (panel["EXMEGA_EW_RET60_PCTL756"] >= 0.65)
        )
    elif kind == "capweight_unconfirmed":
        # Complementary V13 state: the cap-weighted KOSPI is strong, but the
        # non-megacap equal-weight market is not in its top 35% state.
        mask = base & regime & (panel["EXMEGA_EW_RET60_PCTL756"] < 0.65)
    elif kind == "ew_market_strong":
        # Remove cap-weighted KOSPI momentum from the state definition entirely.
        mask = (
            base
            & (panel["EXMEGA_EW_RET60"] > 0)
            & (panel["EXMEGA_EW_RET60_PCTL756"] >= 0.65)
            & panel["EXMEGA_BREADTH60_PCTL756"].between(0.30, 0.70, inclusive="both")
        )
    elif kind == "median_market_strong":
        # Typical-stock version: cross-sectional median 60-day return is strong.
        mask = (
            base
            & (panel["EXMEGA_MEDIAN_RET60"] > 0)
            & (panel["EXMEGA_MEDIAN_RET60_PCTL756"] >= 0.65)
            & panel["EXMEGA_BREADTH60_PCTL756"].between(0.30, 0.70, inclusive="both")
        )
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
        if len(g):
            rows.append({
                "model": model,
                "start_year": start_year,
                "end_year": start_year + 4,
                **stats(g),
            })
    return rows


def practical_screen(periods: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    """Frozen practical tolerance discussed before V14 results were observed."""
    eras = ["1996_2007", "2008_2014", "2015_2021", "2022_2024"]
    rows = []
    for model in SPECS:
        g = periods[periods.model.eq(model)].set_index("window")
        if "all_pre2025" not in g.index:
            continue
        pre = g.loc["all_pre2025"]
        era_ex = [float(g.loc[e, "mean_excess"]) if e in g.index else np.nan for e in eras]
        rw = rolling[(rolling.model == model) & (rolling.signals >= 15)].copy()
        pos_share = float((rw.mean_excess > 0).mean()) if len(rw) else np.nan
        worst = float(rw.mean_excess.min()) if len(rw) else np.nan
        checks = {
            "pre_net_positive": bool(pre["mean_net"] > 0),
            "pre_excess_ge_0p7pct": bool(pre["mean_excess"] >= 0.007),
            "eras_positive_ge3of4": bool(sum(x > 0 for x in era_ex if np.isfinite(x)) >= 3),
            "top3_removed_nonnegative": bool(pre["top3_removed_excess"] >= 0),
            "rolling_positive_ge65pct": bool(np.isfinite(pos_share) and pos_share >= 0.65),
            "rolling_worst_above_minus1p5pct": bool(np.isfinite(worst) and worst >= -0.015),
            "bootstrap_low_above_minus0p7pct": bool(pre["cluster_ci_low"] >= -0.007),
        }
        rows.append({
            "model": model,
            "practical_pass": bool(all(checks.values())),
            "positive_eras": int(sum(x > 0 for x in era_ex if np.isfinite(x))),
            "rolling_windows": int(len(rw)),
            "rolling_positive_share": pos_share,
            "rolling_worst_excess": worst,
            **checks,
        })
    return pd.DataFrame(rows)


def build_panel(args: argparse.Namespace):
    cache = Path(args.cache_dir)
    amount_cache = Path(args.amount_dir)

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
        x["RET1"] = x["Close"].pct_change(fill_method=None)
        x["STOCK_RET60"] = x["Close"] / x["Close"].shift(60) - 1.0

        valid = stock[["Open", "High", "Low", "Close", "Volume"]].gt(0).all(axis=1)
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
            "KOSPI_ABOVE120", "ABOVE20", "ABOVE60", "ABOVE120", "RET1",
            "STOCK_RET60",
        ]
        p = x.loc[eligible, fields].copy()
        p["date"] = p.index
        p["ticker"] = ticker
        p["name"] = meta["name"]
        p["is_mega"] = ticker in MEGA_TICKERS
        pieces.append(p.reset_index(drop=True))

        if n % 100 == 0 or n == len(universe):
            print(f"V14 features {n}/{len(universe)}", flush=True)

    panel = pd.concat(pieces, ignore_index=True)
    for h in (20, 60, 120):
        panel[f"RS{h}_PCTL"] = panel.groupby("date")[f"RS{h}"].rank(pct=True)

    daily = add_market_state(panel)
    panel = panel.merge(daily, left_on="date", right_index=True, how="left", suffixes=("", "_STATE"))
    # KOSPI fields already exist stock-by-stock; prefer the single daily-state versions if duplicated.
    for col in ["KOSPI_RET60", "KOSPI_ABOVE120"]:
        state_col = f"{col}_STATE"
        if state_col in panel.columns:
            panel[col] = panel[state_col]
            panel.drop(columns=[state_col], inplace=True)
    return panel, daily, prices, benchmark


def analyze(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel, daily, prices, benchmark = build_panel(args)

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
    signals.to_csv(out / "v14_all_trades.csv.gz", index=False, compression="gzip")
    pd.DataFrame(funnel).to_csv(out / "v14_signal_funnel.csv", index=False)

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
    periods.to_csv(out / "v14_periods.csv", index=False)
    rolling.to_csv(out / "v14_rolling5y.csv", index=False)
    yearly.to_csv(out / "v14_yearly.csv", index=False)

    screen = practical_screen(periods, rolling)
    screen.to_csv(out / "v14_practical_screen.csv", index=False)
    pre = periods[periods.window.eq("all_pre2025")].sort_values("mean_excess", ascending=False)
    pre.to_csv(out / "v14_pre2025_ranking.csv", index=False)

    # Direct attribution of the original V13 state.
    v13 = signals[signals.model.eq("v13_original")].copy()
    decomposition = []
    if len(v13):
        v13["confirmation_state"] = np.where(
            v13["EXMEGA_EW_RET60_PCTL756"] >= 0.65,
            "broad_confirmed",
            "capweight_unconfirmed",
        )
        for label, g in v13.groupby("confirmation_state"):
            pre_g = g[g["exit_date20"] <= pd.Timestamp("2024-12-30")]
            decomposition.append({"slice": label, **stats(pre_g), **bootstrap(pre_g)})
        for label, g in v13.groupby(v13["ticker"].isin(MEGA_TICKERS)):
            pre_g = g[g["exit_date20"] <= pd.Timestamp("2024-12-30")]
            decomposition.append({
                "slice": "mega_trades" if label else "nonmega_trades",
                **stats(pre_g), **bootstrap(pre_g),
            })
    pd.DataFrame(decomposition).to_csv(out / "v14_v13_decomposition.csv", index=False)

    daily.to_csv(out / "v14_market_state_daily.csv.gz", compression="gzip")
    manifest = {
        "research_only": True,
        "important": "V14 tests whether V13 is merely a cap-weighted KOSPI/Samsung-SK hynix distortion. Research history is exposed, not untouched OOS.",
        "entry": "next trading-day open",
        "exit": "20th trading-day close",
        "round_trip_cost": 0.004,
        "cooldown": args.cooldown,
        "mega_tickers_removed_from_exmega_state": sorted(MEGA_TICKERS),
        "equal_weight_definition": "daily mean return of the contemporaneously liquid, historically KOSPI-member current-survivor universe; 60-session daily-rebalanced compound return",
        "survivorship_bias": "current 2026 survivor universe remains; V14 does not solve survivorship bias",
        "practical_screen_frozen_before_results": {
            "pre_mean_excess": ">= 0.7%p/trade",
            "positive_eras": ">= 3 of 4",
            "top3_removed_excess": ">= 0",
            "rolling_5y_positive_share": ">= 65% for windows with >=15 trades",
            "rolling_5y_worst_excess": ">= -1.5%p",
            "bootstrap_ci_low": ">= -0.7%p",
        },
        "specs": SPECS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    cols = ["model", "signals", "mean_net", "mean_excess", "excess_win_rate", "top3_removed_excess", "cluster_ci_low", "cluster_ci_high"]
    print("\n=== V14 PRE-2025 RANKING ===")
    print(pre[cols].to_string(index=False))
    print("\n=== V14 PRACTICAL SCREEN ===")
    print(screen.to_string(index=False))
    print("\n=== V14 V13 DECOMPOSITION ===")
    print(pd.DataFrame(decomposition).to_string(index=False))
    print("\n=== V14 2025-2026 REUSED ===")
    print(periods[periods.window.eq("2025_2026_reused")][cols].sort_values("mean_excess", ascending=False).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="V14 KOSPI cap-weight distortion diagnostics")
    p.add_argument("--cache-dir", default="data_kospi_long")
    p.add_argument("--amount-dir", default="data_kospi_amount")
    p.add_argument("--output-dir", default="research/v14-index-distortion-results")
    p.add_argument("--start", default="1996-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
