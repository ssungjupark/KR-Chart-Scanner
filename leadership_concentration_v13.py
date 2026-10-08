from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import generalized_regime_v12 as v12
from regime_redesign_v10 import model_mask


SPECS = {
    "v6_baseline": {"kind": "baseline"},
    "v11_fixed_b50_65_r2p5": {"kind": "fixed_v11"},
    "strong_index_mid_breadth": {"kind": "strong_index_mid_breadth"},
    "leadership_divergence_15": {"kind": "leadership_divergence_15"},
    "leadership_divergence_25": {"kind": "leadership_divergence_25"},
    "concentrated_bull": {"kind": "concentrated_bull"},
}


def _ensure_state(panel: pd.DataFrame) -> None:
    if "KOSPI_RET60_PCTL756" not in panel.columns:
        daily_ret = panel.groupby("date")["KOSPI_RET60"].first().sort_index()
        pct = v12.rolling_last_percentile(daily_ret, window=756, min_periods=252)
        panel["KOSPI_RET60_PCTL756"] = panel["date"].map(pct)
    if "LEADERSHIP_SPREAD" not in panel.columns:
        panel["LEADERSHIP_SPREAD"] = panel["KOSPI_RET60_PCTL756"] - panel["LIQ_BREADTH60_PCTL756"]


def apply_spec(panel: pd.DataFrame, spec: dict) -> pd.Series:
    _ensure_state(panel)
    mask = model_mask(panel, {"family": "v6"})
    kind = spec["kind"]

    if kind == "baseline":
        return mask.fillna(False)

    if kind == "fixed_v11":
        mask &= panel["KOSPI_RET60"] > 0.025
        mask &= panel["LIQ_BREADTH60"].between(0.50, 0.65, inclusive="both")

    elif kind == "strong_index_mid_breadth":
        # Relative definition of the V11 idea: index momentum is strong versus its own
        # recent history while participation sits in the middle of its own distribution.
        mask &= panel["KOSPI_ABOVE120"].astype(bool)
        mask &= panel["KOSPI_RET60_PCTL756"] >= 0.65
        mask &= panel["LIQ_BREADTH60_PCTL756"].between(0.30, 0.70, inclusive="both")

    elif kind == "leadership_divergence_15":
        # Strong index + materially weaker breadth percentile = concentrated leadership.
        mask &= panel["KOSPI_ABOVE120"].astype(bool)
        mask &= panel["KOSPI_RET60_PCTL756"] >= 0.65
        mask &= panel["LIQ_BREADTH60_PCTL756"].between(0.25, 0.75, inclusive="both")
        mask &= panel["LEADERSHIP_SPREAD"] >= 0.15

    elif kind == "leadership_divergence_25":
        # Stricter concentration check, prespecified rather than grid-searched.
        mask &= panel["KOSPI_ABOVE120"].astype(bool)
        mask &= panel["KOSPI_RET60_PCTL756"] >= 0.70
        mask &= panel["LIQ_BREADTH60_PCTL756"].between(0.20, 0.70, inclusive="both")
        mask &= panel["LEADERSHIP_SPREAD"] >= 0.25

    elif kind == "concentrated_bull":
        # Hybrid absolute/relative definition. Avoid very weak and fully broad markets,
        # while requiring the index to be stronger than breadth by a meaningful margin.
        mask &= panel["KOSPI_RET60"] > 0
        mask &= panel["KOSPI_ABOVE120"].astype(bool)
        mask &= panel["LIQ_BREADTH60"].between(0.35, 0.70, inclusive="both")
        mask &= panel["LEADERSHIP_SPREAD"] >= 0.15

    else:
        raise ValueError(kind)

    return mask.fillna(False)


def rename_outputs(out: Path) -> None:
    for p in list(out.glob("v12_*")):
        p.rename(out / p.name.replace("v12_", "v13_", 1))
    manifest = out / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        data["important"] = (
            "V13 tests whether the V11 fixed breadth band is really a concentrated-leadership regime. "
            "1996-2024 and 2025-2026 are research-exposed, not untouched OOS data."
        )
        data["specs"] = SPECS
        data["derived_state"] = {
            "KOSPI_RET60_PCTL756": "causal trailing 756-session percentile of KOSPI 60-day return",
            "LIQ_BREADTH60_PCTL756": "causal trailing 756-session percentile of liquid-universe breadth60",
            "LEADERSHIP_SPREAD": "KOSPI_RET60_PCTL756 - LIQ_BREADTH60_PCTL756",
        }
        manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="V13 concentrated-leadership market-state research")
    p.add_argument("--cache-dir", default="data_kospi_long")
    p.add_argument("--amount-dir", default="data_kospi_amount")
    p.add_argument("--output-dir", default="research/v13-leadership-concentration-results")
    p.add_argument("--start", default="1996-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--min-adv", type=float, default=2_000_000_000)
    p.add_argument("--min-price", type=float, default=1000)
    p.add_argument("--cooldown", type=int, default=20)
    args = p.parse_args()

    v12.SPECS = SPECS
    v12.apply_spec = apply_spec
    v12.analyze(args)
    rename_outputs(Path(args.output_dir))


if __name__ == "__main__":
    main()
