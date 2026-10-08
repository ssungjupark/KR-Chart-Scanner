# v8 fixed-entry research

v7 reduced risk but also reduced returns. This experiment freezes v6's entry
signals and 20-session cooldown. Quality variables affect position size only;
they cannot suppress an entry or allow a replacement signal.

## Prespecified comparisons

- Equal size vs half-size when dry-up, RVOL, BB contraction, MA compression,
  retest, non-bear regime or quality >=2 is absent. Each rule is tested separately.
- Fixed20 vs a closing-profit-armed 5% drawdown stop or 2ATR stop. Arming requires
  at least 10% closing-price gain after the 40bp cost assumption. Losing trades
  retain the same fixed20 close unless they previously armed the profit stop.
- All fallback exits execute at the same fixed20 close. Triggered exits execute
  next open. Open exits exclude subsequent exit-day highs/lows from MAE/MFE.

## Selection and verification

1. Fail on any data failure or baseline count mismatch. Expected KOSPI counts:
   129/75/60; KOSDAQ: 68/33/34 for train/validation/evaluation.
2. Purge trades whose fixed20 outcomes cross the 2023 or 2024 selection boundary.
3. Select on 2022-23 and 2024 only. Require positive cash-budget returns and
   full-slot fixed-horizon index excess in both periods, minimum sample counts,
   and improvement over v6 in both periods. Otherwise retain v6.
4. Report 2025-26 as reused evaluation, since previous versions already examined
   these years. It is no longer an untouched holdout. Do not retune using results.
5. Preserve CSV price snapshots, universe snapshot, dependencies and SHA256 hashes.

`avg_budget_ret` is weight * net trade return with unused cash earning zero.
`avg_horizon_excess` subtracts the full-slot index return to the original fixed20
date, even for early exits. `avg_matched_excess` compares only the executed trade's
holding interval; it is a supplementary diagnostic. Neither is portfolio return.

The portfolio reports daily marked equity, cumulative return, full-index excess,
maximum drawdown, exposure and accepted/skipped signals. It starts with cash,
uses 10 slots and 10% of opening equity per full-size trade, caps leverage at zero,
and releases close-exit cash only after that day's opening orders. Half the cost
is paid per side. A separate portfolio is simulated for each market and split.

The current-listed universe still has survivorship bias. Dividends and execution
limits are not modeled. Sparse entries leave cash idle, so a strategy can show
positive average trade alpha yet underperform a continuously invested index.
Live scanner files are unaffected. This PR is stacked on the v7 research branch.

## Outputs

`v6_reproduction_check.csv`, `fixed_v6_signals.csv`, `feature_decomposition.csv`,
`sizing_ablation.csv`, `exit_ablation.csv`, `selection_grid.csv`,
`selected_models.csv`, `holdout_comparison.csv` (reused evaluation),
`yearly_comparison.csv`, `signal_attribution.csv`, `portfolio_comparison.csv`,
daily equity CSVs, `manifest.json`, `dependencies.txt`, `failures.csv`.

Run `python -m unittest discover -s tests -v` then `python robustness_v8.py`.
The GitHub workflow saves both reports and the exact downloaded inputs.
