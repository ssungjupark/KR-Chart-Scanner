from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import final_validation_v15 as v15
import index_distortion_v14 as v14


PRE_END = pd.Timestamp("2024-12-30")
SIGNAL_WEIGHT = 0.10
COSTS = (0.004, 0.008, 0.012)
INDUSTRY_URL = "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/master/data/listing/desc/2026-03-23.csv"


def px(frame: pd.DataFrame, date: pd.Timestamp, field: str) -> float:
    if date in frame.index:
        value = float(frame.loc[date, field])
        if np.isfinite(value) and value > 0:
            return value
    prior = frame.loc[:date, "Close"].replace(0, np.nan).ffill()
    if prior.empty or not np.isfinite(prior.iloc[-1]):
        return np.nan
    return float(prior.iloc[-1])


def trade_key(ticker, entry_date) -> tuple[str, str]:
    return str(ticker).zfill(6), str(pd.Timestamp(entry_date).date())


def index_buy_hold(benchmark: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cost: float) -> pd.DataFrame:
    side = cost / 2
    calendar = benchmark.loc[start:end].index
    first_open = float(benchmark.loc[calendar[0], "Open"])
    units = 1.0 / (first_open * (1 + side))
    rows = []
    for d in calendar:
        close = float(benchmark.loc[d, "Close"])
        rows.append({
            "date": d,
            "equity": units * close,
            "stock_exposure": 0.0,
            "index_exposure": 1.0,
            "equity_exposure": 1.0,
            "positions": 0,
        })
    out = pd.DataFrame(rows).set_index("date")
    out.loc[out.index[-1], "equity"] *= 1 - side
    return out


def summarize_curve(curve: pd.DataFrame) -> dict:
    vals = curve.equity.astype(float)
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    total = float(vals.iloc[-1] - 1)
    cagr = float(vals.iloc[-1] ** (1 / years) - 1) if years > 0 and vals.iloc[-1] > 0 else np.nan
    high = np.maximum.accumulate(np.r_[1.0, vals.to_numpy()])[1:]
    mdd = float((vals.to_numpy() / high - 1).min())
    daily = vals.pct_change().dropna()
    vol = float(daily.std(ddof=1) * np.sqrt(252)) if len(daily) > 1 else np.nan
    return {"total_return": total, "cagr": cagr, "mdd": mdd, "ann_vol": vol}


def replay_full_equity(
    trades: pd.DataFrame,
    benchmark: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost: float = 0.004,
    signal_weight: float = SIGNAL_WEIGHT,
    allowed_keys: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict, set[tuple[str, str]]]:
    """KOSPI core + fixed 10% signal substitutions, always 100% equity exposed.

    allowed_keys freezes the accepted-signal set for contribution stress tests. If supplied,
    signals outside that set are ignored rather than backfilled, so removing a winner/industry
    cannot make room for a formerly capacity-skipped signal.
    """
    side = cost / 2
    work = trades.sort_values(["entry_date", "ticker"]).copy()
    if allowed_keys is not None:
        keys = [trade_key(t, d) for t, d in zip(work.ticker, work.entry_date)]
        work = work.loc[[k in allowed_keys for k in keys]].copy()
    grouped = {d: g for d, g in work.groupby("entry_date")}
    calendar = benchmark.loc[start:end].index
    first_open = float(benchmark.loc[calendar[0], "Open"])
    index_units = 1.0 / (first_open * (1 + side))
    active: list[dict] = []
    accepted: set[tuple[str, str]] = set()
    skipped_capacity = 0
    rows = []

    for d in calendar:
        bo = float(benchmark.loc[d, "Open"])
        bc = float(benchmark.loc[d, "Close"])
        stock_open = sum(p["shares"] * px(prices[p["ticker"]], d, "Open") for p in active)
        equity_open = index_units * bo + stock_open

        for _, r in grouped.get(d, pd.DataFrame()).iterrows():
            target = equity_open * signal_weight
            available_net_index = index_units * bo * (1 - side)
            if available_net_index + 1e-12 < target:
                skipped_capacity += 1
                continue
            ep = float(r.entry_price)
            if not np.isfinite(ep) or ep <= 0:
                continue
            index_units -= target / (bo * (1 - side))
            if index_units < -1e-10:
                raise AssertionError("index core went negative")
            index_units = max(index_units, 0.0)
            shares = target / (ep * (1 + side))
            active.append({
                "ticker": str(r.ticker).zfill(6),
                "shares": shares,
                "end": pd.Timestamp(r.exit_date20),
                "xp": float(r.exit_price20),
            })
            accepted.add(trade_key(r.ticker, r.entry_date))

        remaining = []
        for p in active:
            if p["end"] == d:
                proceeds = p["shares"] * p["xp"] * (1 - side)
                index_units += proceeds / (bc * (1 + side))
            else:
                remaining.append(p)
        active = remaining

        stock_value = sum(p["shares"] * px(prices[p["ticker"]], d, "Close") for p in active)
        index_value = index_units * bc
        equity = index_value + stock_value
        if equity <= 0:
            raise AssertionError("non-positive equity")
        stock_exp = stock_value / equity
        index_exp = index_value / equity
        total_exp = stock_exp + index_exp
        if abs(total_exp - 1.0) > 1e-8:
            raise AssertionError(f"equity exposure drifted: {total_exp}")
        rows.append({
            "date": d,
            "equity": equity,
            "stock_exposure": stock_exp,
            "index_exposure": index_exp,
            "equity_exposure": total_exp,
            "positions": len(active),
        })

    if active:
        raise AssertionError("unclosed signal positions at replay end")
    out = pd.DataFrame(rows).set_index("date")
    out.loc[out.index[-1], "equity"] *= 1 - side
    summary = summarize_curve(out)
    summary.update({
        "cost": cost,
        "signal_weight": signal_weight,
        "accepted_signals": len(accepted),
        "input_signals": len(trades),
        "skipped_capacity": skipped_capacity,
        "avg_signal_stock_exposure": float(out.stock_exposure.mean()),
        "avg_total_equity_exposure": float(out.equity_exposure.mean()),
        "max_signal_stock_exposure": float(out.stock_exposure.max()),
    })
    return out, summary, accepted


def subperiod_return(curve: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[float, float]:
    g = curve[(curve.index >= lo) & (curve.index <= hi)]
    if g.empty:
        return np.nan, np.nan
    prior = curve[curve.index < g.index[0]].tail(1)
    base = 1.0 if prior.empty else float(prior.equity.iloc[0])
    ret = float(g.equity.iloc[-1] / base - 1)
    years = (g.index[-1] - g.index[0]).days / 365.25
    cagr = float((1 + ret) ** (1 / years) - 1) if years > 0 and ret > -1 else np.nan
    return ret, cagr


def rolling_compare(model: pd.DataFrame, base: pd.DataFrame, years: int) -> pd.DataFrame:
    rows = []
    first_year, last_year = int(model.index.min().year), int(model.index.max().year)
    for start_year in range(first_year, last_year - years + 2):
        end_year = start_year + years - 1
        lo, hi = pd.Timestamp(f"{start_year}-01-01"), pd.Timestamp(f"{end_year}-12-31")
        mr, mc = subperiod_return(model, lo, hi)
        br, bc = subperiod_return(base, lo, hi)
        if np.isfinite(mr) and np.isfinite(br):
            rows.append({
                "start_year": start_year,
                "end_year": end_year,
                "model_return": mr,
                "kospi_return": br,
                "excess_return": mr - br,
                "model_cagr": mc,
                "kospi_cagr": bc,
                "cagr_excess": mc - bc,
            })
    return pd.DataFrame(rows)


def accepted_trade_frame(trades: pd.DataFrame, accepted: set[tuple[str, str]]) -> pd.DataFrame:
    x = trades.copy()
    x["_key"] = [trade_key(t, d) for t, d in zip(x.ticker, x.entry_date)]
    return x[x._key.isin(accepted)].copy()


def load_industries() -> pd.DataFrame:
    desc = pd.read_csv(INDUSTRY_URL, dtype={"Code": str})
    desc["Code"] = desc.Code.astype(str).str.zfill(6)
    desc["Industry"] = desc.Industry.fillna("UNKNOWN").replace("", "UNKNOWN")
    return desc[["Code", "Industry"]].drop_duplicates("Code")


def stress_gate(costs: pd.DataFrame, r5: pd.DataFrame, r7: pd.DataFrame, trim: pd.DataFrame, sectors: pd.DataFrame) -> dict:
    c08 = float(costs.loc[np.isclose(costs.cost, 0.008), "cagr_excess"].iloc[0])
    c12 = float(costs.loc[np.isclose(costs.cost, 0.012), "cagr_excess"].iloc[0])
    r5_share = float((r5.cagr_excess > 0).mean())
    r7_share = float((r7.cagr_excess > 0).mean())
    worst5 = float(r5.cagr_excess.min())
    top3 = float(trim.loc[trim.removed_top_n.eq(3), "cagr_excess"].iloc[0])
    sector_positive = int((sectors.cagr_excess > 0).sum()) if len(sectors) else 0
    sector_tests = int(len(sectors))
    checks = {
        "cost_0p8_positive": c08 > 0,
        "cost_1p2_nonnegative": c12 >= 0,
        "rolling5_positive_ge55pct": r5_share >= 0.55,
        "rolling7_positive_ge65pct": r7_share >= 0.65,
        "worst5_cagr_excess_above_minus2pct": worst5 >= -0.02,
        "top3_removed_still_beats_kospi": top3 > 0,
        "sector_leaveout_4of5_positive": sector_tests >= 5 and sector_positive >= 4,
    }
    return {
        "stress_pass": bool(all(checks.values())),
        "verdict": "ROBUST_OVERLAY_CANDIDATE" if all(checks.values()) else "RESEARCH_OVERLAY_ONLY",
        "rolling5_positive_share": r5_share,
        "rolling7_positive_share": r7_share,
        "worst5_cagr_excess": worst5,
        "sector_positive_tests": sector_positive,
        "sector_tests": sector_tests,
        **checks,
    }


def analyze(args):
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    panel, daily_state, prices, benchmark = v14.build_panel(args)
    selected = v15.select_candidate(panel, args.cooldown)
    trades = v15.completed_trades(selected, prices, benchmark)
    pre = trades[trades.exit_date20 <= PRE_END].copy()

    start = benchmark.loc[pd.Timestamp(args.start):].index[0]
    end = benchmark.loc[:PRE_END].index[-1]

    base04 = index_buy_hold(benchmark, start, end, 0.004)
    model04, model04_summary, accepted = replay_full_equity(pre, benchmark, prices, start, end, cost=0.004)
    base04_summary = summarize_curve(base04)
    model04_summary["cagr_excess"] = model04_summary["cagr"] - base04_summary["cagr"]
    model04_summary["total_return_excess"] = model04_summary["total_return"] - base04_summary["total_return"]

    model04.to_csv(outdir / "equity_overlay_10pct_base.csv", index_label="date")
    base04.to_csv(outdir / "equity_kospi_base.csv", index_label="date")

    rolling5 = rolling_compare(model04, base04, 5)
    rolling7 = rolling_compare(model04, base04, 7)
    rolling5.to_csv(outdir / "rolling5_portfolio.csv", index=False)
    rolling7.to_csv(outdir / "rolling7_portfolio.csv", index=False)

    cost_rows = []
    for cost in COSTS:
        b = index_buy_hold(benchmark, start, end, cost)
        m, s, _ = replay_full_equity(pre, benchmark, prices, start, end, cost=cost, allowed_keys=accepted)
        bs = summarize_curve(b)
        cost_rows.append({
            "cost": cost,
            "model_total_return": s["total_return"],
            "kospi_total_return": bs["total_return"],
            "total_return_excess": s["total_return"] - bs["total_return"],
            "model_cagr": s["cagr"],
            "kospi_cagr": bs["cagr"],
            "cagr_excess": s["cagr"] - bs["cagr"],
            "model_mdd": s["mdd"],
            "model_ann_vol": s["ann_vol"],
        })
    costs = pd.DataFrame(cost_rows)
    costs.to_csv(outdir / "cost_stress.csv", index=False)

    accepted_trades = accepted_trade_frame(pre, accepted)
    accepted_trades.to_csv(outdir / "accepted_trades.csv.gz", index=False, compression="gzip")
    ranked = accepted_trades.sort_values("EXCESS20", ascending=False)
    trim_rows = []
    for n in (0, 1, 3, 5, 10):
        drop_keys = set(ranked.head(n)._key) if n else set()
        allowed = accepted - drop_keys
        curve, s, _ = replay_full_equity(pre, benchmark, prices, start, end, cost=0.004, allowed_keys=allowed)
        trim_rows.append({
            "removed_top_n": n,
            "remaining_signals": len(allowed),
            "model_total_return": s["total_return"],
            "kospi_total_return": base04_summary["total_return"],
            "total_return_excess": s["total_return"] - base04_summary["total_return"],
            "model_cagr": s["cagr"],
            "kospi_cagr": base04_summary["cagr"],
            "cagr_excess": s["cagr"] - base04_summary["cagr"],
        })
    trim = pd.DataFrame(trim_rows)
    trim.to_csv(outdir / "winner_removal_stress.csv", index=False)

    industries = load_industries()
    accepted_ind = accepted_trades.merge(industries, left_on="ticker", right_on="Code", how="left")
    accepted_ind["Industry"] = accepted_ind.Industry.fillna("UNKNOWN")
    industry_rank = (accepted_ind.groupby("Industry", as_index=False)
                     .agg(signals=("EXCESS20", "size"), total_trade_excess=("EXCESS20", "sum"), mean_trade_excess=("EXCESS20", "mean"))
                     .sort_values("total_trade_excess", ascending=False))
    industry_rank.to_csv(outdir / "industry_contribution_rank.csv", index=False)

    sector_rows = []
    for industry in industry_rank[industry_rank.Industry.ne("UNKNOWN")].head(5).Industry:
        removed = set(accepted_ind.loc[accepted_ind.Industry.eq(industry), "_key"])
        allowed = accepted - removed
        curve, s, _ = replay_full_equity(pre, benchmark, prices, start, end, cost=0.004, allowed_keys=allowed)
        sector_rows.append({
            "removed_industry": industry,
            "removed_signals": len(removed),
            "remaining_signals": len(allowed),
            "model_total_return": s["total_return"],
            "kospi_total_return": base04_summary["total_return"],
            "total_return_excess": s["total_return"] - base04_summary["total_return"],
            "model_cagr": s["cagr"],
            "kospi_cagr": base04_summary["cagr"],
            "cagr_excess": s["cagr"] - base04_summary["cagr"],
        })
    sectors = pd.DataFrame(sector_rows)
    sectors.to_csv(outdir / "industry_leaveout_stress.csv", index=False)

    # Reused 2025-current diagnostic. This history has been seen during model development and is not OOS.
    reused = trades[trades.entry_date >= pd.Timestamp("2025-01-01")].copy()
    reused_rows = []
    if len(reused):
        rstart = benchmark.loc[pd.Timestamp("2025-01-01"):].index[0]
        rend = benchmark.index[-1]
        rb = index_buy_hold(benchmark, rstart, rend, 0.004)
        rm, rs, _ = replay_full_equity(reused, benchmark, prices, rstart, rend, cost=0.004)
        rbs = summarize_curve(rb)
        reused_rows.append({
            "signals": len(reused),
            "model_total_return": rs["total_return"],
            "kospi_total_return": rbs["total_return"],
            "total_return_excess": rs["total_return"] - rbs["total_return"],
            "model_cagr": rs["cagr"],
            "kospi_cagr": rbs["cagr"],
            "cagr_excess": rs["cagr"] - rbs["cagr"],
        })
    pd.DataFrame(reused_rows).to_csv(outdir / "reused_2025_current.csv", index=False)

    gate = stress_gate(costs, rolling5, rolling7, trim, sectors)
    pd.DataFrame([{**model04_summary, **gate}]).to_csv(outdir / "stress_gate.csv", index=False)

    manifest = {
        "version": "V15 frozen 10% full-equity stress validation",
        "signal_logic_changed": False,
        "signal_weight_changed": False,
        "signal_weight": SIGNAL_WEIGHT,
        "base_cost": 0.004,
        "stress_costs": list(COSTS),
        "winner_removal": "freeze baseline accepted keys, then replace removed winners with KOSPI rather than backfilling skipped signals",
        "industry_snapshot": INDUSTRY_URL,
        "industry_warning": "2026-03 current industry classification is applied retrospectively; leave-one-industry-out is exploratory, not point-in-time sector validation",
        "research_warning": "1996-2024 and 2025-current have been exposed during research; this is stress validation, not untouched OOS",
        "gate": gate,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("\n=== BASE 10% FULL-EQUITY OVERLAY ===")
    print(pd.DataFrame([model04_summary]).to_string(index=False))
    print("\n=== COST STRESS ===")
    print(costs.to_string(index=False))
    print("\n=== ROLLING 5Y ===")
    print(pd.DataFrame([{
        "windows": len(rolling5),
        "positive_share": float((rolling5.cagr_excess > 0).mean()),
        "worst_cagr_excess": float(rolling5.cagr_excess.min()),
        "median_cagr_excess": float(rolling5.cagr_excess.median()),
    }]).to_string(index=False))
    print("\n=== ROLLING 7Y ===")
    print(pd.DataFrame([{
        "windows": len(rolling7),
        "positive_share": float((rolling7.cagr_excess > 0).mean()),
        "worst_cagr_excess": float(rolling7.cagr_excess.min()),
        "median_cagr_excess": float(rolling7.cagr_excess.median()),
    }]).to_string(index=False))
    print("\n=== WINNER REMOVAL ===")
    print(trim.to_string(index=False))
    print("\n=== INDUSTRY LEAVE-OUT ===")
    print(sectors.to_string(index=False))
    print("\n=== STRESS GATE ===")
    print(pd.DataFrame([gate]).to_string(index=False))
    if reused_rows:
        print("\n=== 2025-CURRENT REUSED ===")
        print(pd.DataFrame(reused_rows).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="V15 frozen 10% full-equity stress validation")
    p.add_argument("--cache-dir", default="data_kospi_long")
    p.add_argument("--amount-dir", default="data_kospi_amount")
    p.add_argument("--output-dir", default="research/v15-full-equity-stress-results")
    p.add_argument("--start", default="1996-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
