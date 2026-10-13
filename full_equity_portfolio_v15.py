from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import final_validation_v15 as v15
import index_distortion_v14 as v14


PRE_END = pd.Timestamp("2024-12-30")
WEIGHTS = (0.10, 0.20, 0.25)
ROUND_TRIP_COST = 0.004
SIDE = ROUND_TRIP_COST / 2


def px(frame: pd.DataFrame, date: pd.Timestamp, field: str) -> float:
    if date in frame.index:
        value = float(frame.loc[date, field])
        if np.isfinite(value) and value > 0:
            return value
    prior = frame.loc[:date, "Close"].replace(0, np.nan).ffill()
    if prior.empty or not np.isfinite(prior.iloc[-1]):
        return np.nan
    return float(prior.iloc[-1])


def index_buy_hold(benchmark: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    calendar = benchmark.loc[start:end].index
    first_open = float(benchmark.loc[calendar[0], "Open"])
    units = 1.0 / (first_open * (1 + SIDE))
    rows = []
    for d in calendar:
        close = float(benchmark.loc[d, "Close"])
        equity = units * close
        rows.append({
            "date": d,
            "equity": equity,
            "stock_exposure": 0.0,
            "index_exposure": 1.0,
            "equity_exposure": 1.0,
            "positions": 0,
        })
    out = pd.DataFrame(rows).set_index("date")
    out.loc[out.index[-1], "equity"] *= 1 - SIDE
    return out


def replay_full_equity(
    trades: pd.DataFrame,
    benchmark: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    signal_weight: float,
) -> tuple[pd.DataFrame, dict]:
    """Always keep capital in equities: KOSPI core plus fixed-weight signal substitutions.

    Entries execute at next open. A signal position is funded by selling KOSPI at the same open.
    Exits execute at the signal's fixed 20-session close and proceeds are immediately reinvested
    into KOSPI at that close. No cash sleeve, borrowing, or leverage is permitted.
    """
    work = trades.sort_values(["entry_date", "ticker"]).copy()
    grouped = {d: g for d, g in work.groupby("entry_date")}
    calendar = benchmark.loc[start:end].index
    first_open = float(benchmark.loc[calendar[0], "Open"])
    index_units = 1.0 / (first_open * (1 + SIDE))
    active: list[dict] = []
    accepted: list[tuple[str, str]] = []
    skipped_capacity = 0
    rows = []

    for d in calendar:
        bo = float(benchmark.loc[d, "Open"])
        bc = float(benchmark.loc[d, "Close"])
        stock_open = sum(p["shares"] * px(prices[p["ticker"]], d, "Open") for p in active)
        equity_open = index_units * bo + stock_open

        # Open entries cannot use proceeds from close exits later the same day.
        for _, r in grouped.get(d, pd.DataFrame()).iterrows():
            target = equity_open * signal_weight
            available_net_index = index_units * bo * (1 - SIDE)
            if available_net_index + 1e-12 < target:
                skipped_capacity += 1
                continue
            ep = float(r.entry_price)
            if not np.isfinite(ep) or ep <= 0:
                continue
            index_units -= target / (bo * (1 - SIDE))
            if index_units < -1e-10:
                raise AssertionError("index core went negative")
            index_units = max(index_units, 0.0)
            shares = target / (ep * (1 + SIDE))
            active.append({
                "ticker": str(r.ticker),
                "shares": shares,
                "end": pd.Timestamp(r.exit_date20),
                "xp": float(r.exit_price20),
            })
            accepted.append((str(r.ticker), str(pd.Timestamp(r.entry_date).date())))

        remaining = []
        for p in active:
            if p["end"] == d:
                proceeds = p["shares"] * p["xp"] * (1 - SIDE)
                index_units += proceeds / (bc * (1 + SIDE))
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
    # Liquidate the remaining KOSPI core at the terminal close for an apples-to-apples net result.
    out.loc[out.index[-1], "equity"] *= 1 - SIDE
    summary = summarize_curve(out)
    summary.update({
        "signal_weight": signal_weight,
        "accepted_signals": len(accepted),
        "input_signals": len(trades),
        "skipped_capacity": skipped_capacity,
        "avg_signal_stock_exposure": float(out.stock_exposure.mean()),
        "avg_total_equity_exposure": float(out.equity_exposure.mean()),
        "max_signal_stock_exposure": float(out.stock_exposure.max()),
        "avg_positions": float(out.positions.mean()),
    })
    return out, summary


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


def period_returns(curve: pd.DataFrame, windows: list[tuple[str, pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    rows = []
    for label, lo, hi in windows:
        g = curve[(curve.index >= lo) & (curve.index <= hi)]
        if g.empty:
            continue
        prior = curve[curve.index < g.index[0]].tail(1)
        base = 1.0 if prior.empty else float(prior.equity.iloc[0])
        rows.append({"window": label, "return": float(g.equity.iloc[-1] / base - 1)})
    return pd.DataFrame(rows)


def yearly_returns(curve: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for year, g in curve.groupby(curve.index.year):
        prior = curve[curve.index < g.index[0]].tail(1)
        base = 1.0 if prior.empty else float(prior.equity.iloc[0])
        rows.append({"model": model, "year": int(year), "return": float(g.equity.iloc[-1] / base - 1)})
    return pd.DataFrame(rows)


def analyze(args):
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    panel, daily_state, prices, benchmark = v14.build_panel(args)
    selected = v15.select_candidate(panel, args.cooldown)
    trades = v15.completed_trades(selected, prices, benchmark)
    pre = trades[trades.exit_date20 <= PRE_END].copy()

    start = benchmark.loc[pd.Timestamp(args.start):].index[0]
    end = benchmark.loc[:PRE_END].index[-1]
    baseline = index_buy_hold(benchmark, start, end)
    baseline_summary = summarize_curve(baseline)
    baseline_summary.update({
        "model": "kospi_buy_hold",
        "signal_weight": 0.0,
        "accepted_signals": 0,
        "input_signals": len(pre),
        "skipped_capacity": 0,
        "avg_signal_stock_exposure": 0.0,
        "avg_total_equity_exposure": 1.0,
        "max_signal_stock_exposure": 0.0,
        "avg_positions": 0.0,
    })
    baseline.to_csv(outdir / "equity_kospi_buy_hold.csv", index_label="date")

    summaries = [baseline_summary]
    curves = {"kospi_buy_hold": baseline}
    for weight in WEIGHTS:
        curve, summary = replay_full_equity(pre, benchmark, prices, start, end, weight)
        label = f"signal_overlay_{int(weight * 100)}pct"
        summary["model"] = label
        curves[label] = curve
        summaries.append(summary)
        curve.to_csv(outdir / f"equity_{label}.csv", index_label="date")

    summary_df = pd.DataFrame(summaries)
    base_total = float(summary_df.loc[summary_df.model.eq("kospi_buy_hold"), "total_return"].iloc[0])
    base_cagr = float(summary_df.loc[summary_df.model.eq("kospi_buy_hold"), "cagr"].iloc[0])
    summary_df["total_return_minus_kospi"] = summary_df.total_return - base_total
    summary_df["cagr_minus_kospi"] = summary_df.cagr - base_cagr
    summary_df.to_csv(outdir / "full_equity_summary.csv", index=False)

    era_windows = [(label, lo, hi) for label, lo, hi in v14.WINDOWS if label != "2025_2026_reused"]
    era_frames = []
    for model, curve in curves.items():
        x = period_returns(curve, era_windows)
        x["model"] = model
        era_frames.append(x)
    eras = pd.concat(era_frames, ignore_index=True)
    base_eras = eras[eras.model.eq("kospi_buy_hold")][["window", "return"]].rename(columns={"return": "kospi_return"})
    eras = eras.merge(base_eras, on="window", how="left")
    eras["excess_return"] = eras["return"] - eras["kospi_return"]
    eras.to_csv(outdir / "full_equity_eras.csv", index=False)

    annual = pd.concat([yearly_returns(curve, model) for model, curve in curves.items()], ignore_index=True)
    base_year = annual[annual.model.eq("kospi_buy_hold")][["year", "return"]].rename(columns={"return": "kospi_return"})
    annual = annual.merge(base_year, on="year", how="left")
    annual["excess_return"] = annual["return"] - annual["kospi_return"]
    annual.to_csv(outdir / "full_equity_yearly.csv", index=False)

    yearly_screen = []
    for model, g in annual[~annual.model.eq("kospi_buy_hold")].groupby("model"):
        yearly_screen.append({
            "model": model,
            "years": len(g),
            "years_outperform": int((g.excess_return > 0).sum()),
            "outperform_share": float((g.excess_return > 0).mean()),
            "mean_annual_excess": float(g.excess_return.mean()),
            "worst_annual_excess": float(g.excess_return.min()),
            "best_annual_excess": float(g.excess_return.max()),
        })
    pd.DataFrame(yearly_screen).to_csv(outdir / "full_equity_yearly_screen.csv", index=False)

    manifest = {
        "version": "V15 full-equity portfolio correction",
        "signal_candidate": v15.CANDIDATE,
        "signal_logic_changed": False,
        "portfolio_question": "Compare V15 signals with KOSPI at approximately 100% equity exposure instead of a mostly-cash standalone account.",
        "baseline": "100% KOSPI buy-and-hold, same 0.4% round-trip cost assumption at initial buy/final liquidation",
        "overlay_models": [f"{int(w*100)}% of then-current equity per accepted signal, funded by selling KOSPI; proceeds return to KOSPI on fixed 20-session close exit" for w in WEIGHTS],
        "cash_policy": "no strategic cash; capital is held in KOSPI core and/or signal stocks at all times",
        "leverage": False,
        "selection": "weights 10/20/25% are reported together; none is selected ex post",
        "research_warning": "same research-exposed 1996-2024 history; this is portfolio accounting correction, not untouched OOS validation",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("\n=== FULL EQUITY SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\n=== ERA EXCESS VS KOSPI ===")
    print(eras.to_string(index=False))
    print("\n=== YEARLY OUTPERFORMANCE ===")
    print(pd.DataFrame(yearly_screen).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="V15 fully-invested equity portfolio comparison")
    p.add_argument("--cache-dir", default="data_kospi_long")
    p.add_argument("--amount-dir", default="data_kospi_amount")
    p.add_argument("--output-dir", default="research/v15-full-equity-results")
    p.add_argument("--start", default="1996-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
