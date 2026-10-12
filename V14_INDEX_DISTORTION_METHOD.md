# V14 Index Distortion Test

## Question

V13 improved V6 when the cap-weighted KOSPI was strong while breadth was only mid-range. Korea's index is unusually concentrated in Samsung Electronics and SK hynix, so V14 asks whether that result is simply a cap-weighted megacap artifact.

## Fixed stock signal

The V6 stock-level setup, trigger, RS thresholds, 20-session cooldown, next-open entry, 20-session close exit, and 0.4% round-trip cost are unchanged.

## Megacap-neutral market proxies

`005930` and `000660` are excluded from the `EXMEGA` market-state variables.

The market proxies are built from the contemporaneously liquid KOSPI-member rows used for cross-sectional ranking:

- `EXMEGA_EW_RET1`: daily equal-weight mean stock return excluding Samsung Electronics and SK hynix.
- `EXMEGA_EW_RET60`: 60-session daily-rebalanced compound return of `EXMEGA_EW_RET1`.
- `EXMEGA_MEDIAN_RET60`: cross-sectional median individual-stock 60-session return excluding the two megacaps.
- `EXMEGA_BREADTH60`: fraction of non-megacap liquid names above their own 60-day average.
- trailing 756-session percentiles are calculated causally with a 252-session minimum history.

This is not a reconstructed official equal-weight KOSPI index. The current-survivor universe limitation remains, so V14 is a distortion diagnostic rather than a clean historical index study.

## Prespecified models

1. `v6_baseline`: unchanged V6.
2. `v13_original`: exact V13 generalized market state.
3. `v13_exmega_trades`: same V13 state but Samsung Electronics and SK hynix cannot themselves be traded.
4. `broad_confirmation`: V13 state plus non-megacap equal-weight 60-day return in its trailing top 35% and positive.
5. `capweight_unconfirmed`: V13 state where the non-megacap equal-weight proxy fails that top-35% confirmation. This directly partitions the V13 hypothesis.
6. `ew_market_strong`: removes KOSPI cap-weighted momentum from the regime definition and uses only non-megacap equal-weight momentum plus non-megacap breadth.
7. `median_market_strong`: same idea using the typical stock's 60-day return instead of an equal-weight portfolio.

No threshold grid is searched.

## Frozen practical tolerance

Before V14 results are inspected, a candidate is considered practical-screen eligible only if all conditions hold:

- pre-2025 mean net return > 0;
- pre-2025 mean excess return >= +0.7%p per trade;
- at least 3 of 4 long eras have positive mean excess return;
- removing the top 3 excess-return trades leaves mean excess return >= 0;
- among rolling 5-year windows with at least 15 trades, at least 65% have positive mean excess return;
- worst eligible rolling 5-year mean excess return >= -1.5%p;
- date-cluster bootstrap 95% lower bound >= -0.7%p.

These tolerances intentionally allow some regime failure rather than demanding positive alpha in every five-year window.

## Reproduction checks

The workflow must reproduce:

- V6 pre-2025 completed trades = 1,129;
- V13 original pre-2025 completed trades = 210.

If either fails, V14 results are not trusted.

## Interpretation rule

- If `broad_confirmation` or the megacap-neutral `ew_market_strong` / `median_market_strong` rules retain the V13 edge, the effect is not just Samsung Electronics / SK hynix index concentration.
- If only `capweight_unconfirmed` works, the prior result is primarily a cap-weighted leadership artifact and should not be treated as a general market regime.
- If excluding the two megacaps as trade candidates destroys results, direct winner concentration is also a problem.

2025-2026 remains reused evaluation and is never used to choose the V14 structure.
