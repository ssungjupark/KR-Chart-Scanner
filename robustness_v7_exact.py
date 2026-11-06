from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import robustness_v7 as v7
from config import DEFAULT_CONFIG
from data_loader import load_price_range


EXPECTED_V6_COUNTS = {
    "KOSPI": {"train": 129, "validation": 75, "holdout": 60},
    "KOSDAQ": {"train": 68, "validation": 33, "holdout": 34},
}


def _process_symbol_exact(meta: dict, benchmarks: dict[str, pd.DataFrame], start: str, end: str,
                          min_adv: float, min_price: float) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Same v7 feature pipeline, but keep the v6 absolute-RS fields needed for exact reproduction."""
    ticker = str(meta["ticker"])
    name = str(meta["name"])
    market = str(meta["market"])
    try:
        stock = load_price_range(ticker, start, end, warmup_days=650, forward_days=75)
        benchmark = benchmarks[market]
        x = v7.base_feature_frame(stock, benchmark)
        x = v7._add_v7_features(x, benchmark)

        in_period = (x.index >= pd.Timestamp(start)) & (x.index <= pd.Timestamp(end))
        eligible = (
            in_period
            & (x["row_pos"] >= DEFAULT_CONFIG.min_history - 1)
            & (x["ADV20"] >= min_adv)
            & (x["Close"] >= min_price)
            & x["RS20"].notna()
            & x["RS60"].notna()
            & x["RS120"].notna()
            & x["RS_RATIO_SLOPE"].notna()
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

        outcomes = v7._add_exit_outcomes(x, benchmark, positions)
        cols = [
            "row_pos", "Close", "ADV20", "RS20", "RS60", "RS120", "RS_RATIO_SLOPE",
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


def _entry_mask_exact(frame: pd.DataFrame, market: str, variant: str, gate: str) -> pd.Series:
    """Reproduce the v6 signal order exactly, then layer one v7 condition at a time."""
    cfg = v7.V6_MODEL[market]
    trigger_col = "TRIG_BREAK3" if cfg["trigger"] == "break3" else "TRIG_MA5"
    mask = (
        frame[trigger_col].astype(bool)
        & (frame["RS20"] > 0)
        & (frame["RS_RATIO_SLOPE"] > 0)
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


def _write_reproduction_check(output_dir: str) -> None:
    out = Path(output_dir)
    ablation = pd.read_csv(out / "entry_ablation.csv")
    holdout = pd.read_csv(out / "holdout_comparison.csv")
    rows = []

    for market, expected in EXPECTED_V6_COUNTS.items():
        base = ablation[
            (ablation["market"] == market)
            & (ablation["variant"] == "baseline")
            & (ablation["market_gate"] == "none")
        ].iloc[0]
        h = holdout[
            (holdout["market"] == market)
            & (holdout["model"] == "v6_baseline_net_cost")
        ].iloc[0]
        actual = {
            "train": int(base["train_signals"]),
            "validation": int(base["validation_signals"]),
            "holdout": int(h["signals"]),
        }
        rows.append({
            "market": market,
            "expected_train": expected["train"],
            "actual_train": actual["train"],
            "expected_validation": expected["validation"],
            "actual_validation": actual["validation"],
            "expected_holdout": expected["holdout"],
            "actual_holdout": actual["holdout"],
            "exact_match": actual == expected,
        })

    check = pd.DataFrame(rows)
    check.to_csv(out / "v6_reproduction_check.csv", index=False, encoding="utf-8-sig")
    print("=" * 100)
    print("V6 REPRODUCTION CHECK")
    print(check.to_string(index=False))


# Patch only the two places that caused the apples-to-apples mismatch.
v7._process_symbol = _process_symbol_exact
v7._entry_mask = _entry_mask_exact


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KR Chart Scanner v7 with exact v6 baseline reproduction")
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

    v7.run(
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
    _write_reproduction_check(a.output_dir)
