from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import index_distortion_v14 as v14
from kospi_long_history import add_outcomes, bootstrap, replay, stats
import robustness_v7 as v7


CANDIDATE = "v13_exmega_trades"
PRE_END = pd.Timestamp("2024-12-30")
ERAS = ["1996_2007", "2008_2014", "2015_2021", "2022_2024"]


def select_candidate(panel: pd.DataFrame, cooldown: int) -> pd.DataFrame:
    mask = v14.apply_spec(panel, v14.SPECS[CANDIDATE])
    return v7._cooldown(panel.loc[mask].copy(), cooldown)


def completed_trades(selected, prices, benchmark):
    parts = []
    for ticker, f in selected.groupby("ticker"):
        r = add_outcomes(f, prices[ticker], benchmark)
        parts.append(r)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(out):
        out = out[out["NET20"].notna() & out["EXCESS20"].notna()].copy()
    return out


def rolling_table(pre: pd.DataFrame, years: int, min_trades: int = 15) -> pd.DataFrame:
    rows = []
    last_start = 2024 - years + 1
    for start in range(1996, last_start + 1):
        end = start + years - 1
        hi = pd.Timestamp(f"{end}-12-31")
        g = pre[(pre.date.dt.year.between(start, end)) & (pre.exit_date20 <= hi)].copy()
        if len(g) >= min_trades:
            rows.append({"start_year": start, "end_year": end, **stats(g)})
    return pd.DataFrame(rows)


def era_table(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, lo, hi in v14.WINDOWS:
        if label == "2025_2026_reused":
            continue
        g = trades[(trades.date >= lo) & (trades.date <= hi) & (trades.exit_date20 <= hi)].copy()
        rows.append({"window": label, **stats(g), **bootstrap(g)})
    return pd.DataFrame(rows)


def cost_table(pre: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost in (0.004, 0.008, 0.012):
        delta = cost - 0.004
        rows.append({
            "round_trip_cost": cost,
            "signals": len(pre),
            "mean_net": float(pre.NET20.mean() - delta),
            "mean_excess": float(pre.EXCESS20.mean() - delta),
        })
    return pd.DataFrame(rows)


def winner_trim_table(pre: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in (0, 1, 3, 5, 10):
        g = pre.sort_values("EXCESS20")
        if n:
            g = g.iloc[:-n] if len(g) > n else g.iloc[0:0]
        rows.append({
            "top_winners_removed": n,
            "signals": len(g),
            "mean_net": float(g.NET20.mean()) if len(g) else np.nan,
            "mean_excess": float(g.EXCESS20.mean()) if len(g) else np.nan,
            "median_excess": float(g.EXCESS20.median()) if len(g) else np.nan,
        })
    return pd.DataFrame(rows)


def horizon_table(pre: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in (5, 10, 20, 40, 60):
        net, ex, exitcol = f"NET{h}", f"EXCESS{h}", f"exit_date{h}"
        g = pre[pre[net].notna() & pre[ex].notna() & (pd.to_datetime(pre[exitcol]) <= PRE_END)]
        rows.append({
            "horizon": h,
            "signals": len(g),
            "mean_net": float(g[net].mean()) if len(g) else np.nan,
            "mean_excess": float(g[ex].mean()) if len(g) else np.nan,
            "median_excess": float(g[ex].median()) if len(g) else np.nan,
        })
    return pd.DataFrame(rows)


def concentration_table(pre: pd.DataFrame) -> pd.DataFrame:
    by_ticker = pre.groupby(["ticker", "name"], as_index=False).agg(
        signals=("EXCESS20", "size"),
        total_excess=("EXCESS20", "sum"),
        mean_excess=("EXCESS20", "mean"),
    )
    return by_ticker.sort_values("total_excess", ascending=False)


def final_gate(pre, eras, rolling5, costs, trim) -> dict:
    ci = bootstrap(pre)
    positive_eras = int((eras.mean_excess > 0).sum())
    rolling_share = float((rolling5.mean_excess > 0).mean()) if len(rolling5) else np.nan
    rolling_worst = float(rolling5.mean_excess.min()) if len(rolling5) else np.nan
    cost08 = float(costs.loc[np.isclose(costs.round_trip_cost, 0.008), "mean_excess"].iloc[0])
    top3 = float(trim.loc[trim.top_winners_removed.eq(3), "mean_excess"].iloc[0])
    checks = {
        "pre_net_positive": bool(pre.NET20.mean() > 0),
        "pre_excess_ge_0p7pct": bool(pre.EXCESS20.mean() >= 0.007),
        "positive_eras_ge3of4": bool(positive_eras >= 3),
        "top3_removed_nonnegative": bool(top3 >= 0),
        "cost_0p8pct_excess_positive": bool(cost08 > 0),
        "rolling5_positive_ge65pct": bool(np.isfinite(rolling_share) and rolling_share >= 0.65),
        "rolling5_worst_above_minus1p5pct": bool(np.isfinite(rolling_worst) and rolling_worst >= -0.015),
        "bootstrap_low_above_minus0p7pct": bool(ci["cluster_ci_low"] >= -0.007),
    }
    return {
        "candidate": CANDIDATE,
        "final_pass": bool(all(checks.values())),
        "verdict": "PRACTICAL_PASS" if all(checks.values()) else "RESEARCH_ONLY",
        "signals_pre2025": int(len(pre)),
        "mean_net_pre2025": float(pre.NET20.mean()),
        "mean_excess_pre2025": float(pre.EXCESS20.mean()),
        "positive_eras": positive_eras,
        "rolling5_windows": int(len(rolling5)),
        "rolling5_positive_share": rolling_share,
        "rolling5_worst_excess": rolling_worst,
        "cluster_ci_low": float(ci["cluster_ci_low"]),
        "cluster_ci_high": float(ci["cluster_ci_high"]),
        **checks,
    }


def analyze(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    panel, daily_state, prices, benchmark = v14.build_panel(args)
    selected = select_candidate(panel, args.cooldown)
    trades = completed_trades(selected, prices, benchmark)
    trades.to_csv(out / "v15_candidate_trades.csv.gz", index=False, compression="gzip")

    pre = trades[trades.exit_date20 <= PRE_END].copy()
    reused = trades[trades.date >= pd.Timestamp("2025-01-01")].copy()

    eras = era_table(trades)
    rolling5 = rolling_table(pre, 5)
    rolling7 = rolling_table(pre, 7)
    costs = cost_table(pre)
    trim = winner_trim_table(pre)
    horizons = horizon_table(pre)
    concentration = concentration_table(pre)
    gate = final_gate(pre, eras, rolling5, costs, trim)

    eras.to_csv(out / "v15_eras.csv", index=False)
    rolling5.to_csv(out / "v15_rolling5y.csv", index=False)
    rolling7.to_csv(out / "v15_rolling7y.csv", index=False)
    costs.to_csv(out / "v15_cost_sensitivity.csv", index=False)
    trim.to_csv(out / "v15_winner_trim.csv", index=False)
    horizons.to_csv(out / "v15_horizons.csv", index=False)
    concentration.to_csv(out / "v15_ticker_concentration.csv", index=False)
    pd.DataFrame([gate]).to_csv(out / "v15_final_gate.csv", index=False)
    pd.DataFrame([{"window": "all_pre2025", **stats(pre), **bootstrap(pre)},
                  {"window": "2025_2026_reused", **stats(reused), **bootstrap(reused)}]).to_csv(
        out / "v15_periods.csv", index=False
    )

    # Fixed 10-slot portfolio diagnostics; no parameter selection from these results.
    start = benchmark.loc[pd.Timestamp(args.start):].index[0]
    pre_end = benchmark.loc[:PRE_END].index[-1]
    portfolio_rows = []
    eq, accepted, summary = replay(pre, benchmark, prices, start, pre_end, cost=0.004)
    eq.to_csv(out / "v15_equity_candidate.csv", index_label="date")
    portfolio_rows.append({"model": "candidate_cash", **summary})

    idx_eq, _, idx_summary = replay(pre, benchmark, prices, start, pre_end, asset="index", keep=accepted, cost=0.004)
    idx_eq.to_csv(out / "v15_equity_same_schedule_index.csv", index_label="date")
    portfolio_rows.append({"model": "same_schedule_index", **idx_summary})

    core_eq, _, core_summary = replay(pre, benchmark, prices, start, pre_end, core=True, keep=accepted, cost=0.004)
    core_eq.to_csv(out / "v15_equity_index_core.csv", index_label="date")
    portfolio_rows.append({"model": "index_core_candidate", **core_summary})
    pd.DataFrame(portfolio_rows).to_csv(out / "v15_portfolio.csv", index=False)

    daily_state.to_csv(out / "v15_market_state_daily.csv.gz", compression="gzip")
    manifest = {
        "research_only": True,
        "version": "V15 final frozen validation",
        "candidate": "V13 generalized market state with Samsung Electronics and SK hynix excluded as direct trade candidates",
        "candidate_logic": {
            "stock_signal": "unchanged V6 setup + trigger + RS",
            "market_state": "KOSPI above SMA120; KOSPI 60d return trailing-756 percentile >=65%; liquid breadth60 trailing-756 percentile 30-70%",
            "excluded_trade_tickers": sorted(v14.MEGA_TICKERS),
        },
        "selection": "none in V15; candidate frozen from V14 before final stress results",
        "round_trip_cost_base": 0.004,
        "survivorship_bias": "current 2026 survivor universe remains; historical exchange membership applied when available",
        "final_gate": gate,
        "important": "1996-2024 and 2025-2026 are research-exposed. V15 is a stress-validation closeout, not a new untouched holdout.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("\n=== V15 FINAL GATE ===")
    print(pd.DataFrame([gate]).to_string(index=False))
    print("\n=== ERAS ===")
    print(eras.to_string(index=False))
    print("\n=== COST ===")
    print(costs.to_string(index=False))
    print("\n=== WINNER TRIM ===")
    print(trim.to_string(index=False))
    print("\n=== ROLLING 5Y WORST ===")
    print(rolling5.nsmallest(8, "mean_excess").to_string(index=False))
    print("\n=== PORTFOLIO ===")
    print(pd.DataFrame(portfolio_rows).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="V15 final frozen validation")
    p.add_argument("--cache-dir", default="data_kospi_long")
    p.add_argument("--amount-dir", default="data_kospi_amount")
    p.add_argument("--output-dir", default="research/v15-final-validation-results")
    p.add_argument("--start", default="1996-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
