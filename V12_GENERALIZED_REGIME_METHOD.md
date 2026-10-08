# V12 generalized market-state research

## Question

V11 found that the fixed condition `KOSPI 60d return > 2.5%` plus liquid-universe 60-day breadth between 50% and 65% improved V6 materially. That band was found after looking at historical results, so V12 does **not** optimize the band further. Instead it tests whether the broader market structure generalizes:

- the index is in a positive medium-term trend,
- participation is not fully saturated,
- breadth is improving or broadening,
- V6 stock selection is otherwise unchanged.

1996-2024 has already been used for research and is not an untouched holdout. 2025-2026 is also a reused evaluation period.

## Frozen components

V12 freezes the long-history data corrections, historical KOSPI membership logic, liquidity and price filters, V6 stock-level RS/setup/trigger rules, 20-session per-ticker cooldown, next-session-open entry, fixed 20-session close exit, and 0.4% round-trip cost.

The existing V6 baseline must reproduce 1,129 completed pre-2025 trades before V12 results are accepted.

## Market-state definitions

The liquid breadth series is computed each date from the historically eligible/liquid KOSPI panel:

- `LIQ_BREADTH20`: fraction above SMA20
- `LIQ_BREADTH60`: fraction above SMA60
- `LIQ_BREADTH120`: fraction above SMA120

The wider `ALL_BREADTH60` series uses historical KOSPI members with mature/valid price history but does not require the V6 liquidity and price filters.

All breadth changes and rolling percentiles use current and past observations only.

### Benchmarks

- `v6_baseline`: frozen V6, no market-state filter.
- `v11_fixed_b50_65_r2p5`: V11 fixed hypothesis, KOSPI 60-day return > 2.5% and liquid breadth60 in 50-65%.

### Generalized candidates

- `broadening_mid`: KOSPI 60-day return > 2.5%, index above SMA120, breadth60 40-70%, breadth20 > breadth60, breadth60 higher than 10 sessions ago.
- `early_expansion`: same positive index structure, breadth60 35-70%, breadth20 > breadth60, breadth60 improves at least 5 percentage points over 20 sessions.
- `dynamic_middle`: replace a fixed breadth level with the 35th-75th percentile of the trailing 756-session breadth distribution, require breadth60 to be rising over 20 sessions.
- `dynamic_middle_all`: same dynamic-percentile test using the wider historical-member breadth definition.
- `trend_participation`: minimal structural rule; positive KOSPI 60-day return, index above SMA120, breadth60 below 75%, breadth20 > breadth60, breadth60 rising over 20 sessions.

No candidate is selected by 2025-2026 performance.

## Evaluation

For every model V12 reports:

- 1996-2007, 2008-2014, 2015-2021, 2022-2024 separately
- reused 2025-2026 separately
- all pre-2025 trades
- mean net return and KOSPI excess return
- excess win rate, MAE/MFE, top-3-winner-removed excess return
- date-cluster bootstrap interval
- rolling five-year windows from 1996-2000 through 2020-2024
- yearly results

A five-year window enters the robustness score only when it has at least 15 completed trades.

## Research gate

A generalized candidate passes the research gate only when all of the following are true pre-2025:

1. overall mean net return > 0
2. overall mean KOSPI excess return > 0
3. mean excess return > 0 in all four long eras
4. mean net return > 0 in all four long eras
5. at least 20 trades in each long era
6. top three winners removed still leaves positive excess return
7. at least 70% of eligible rolling five-year windows have positive excess return
8. worst eligible rolling five-year mean excess return is better than -1.0%

The stricter statistical gate additionally requires the lower date-cluster bootstrap bound to be above zero.

Passing this research gate still does not make the model production-ready because the current-survivor universe creates survivorship bias and all historical periods have now been research-exposed.
