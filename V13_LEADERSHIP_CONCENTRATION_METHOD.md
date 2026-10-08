# V13 leadership concentration hypothesis

## Why this exists

V12 showed that the V11 `breadth60 50-65%` effect does not generalize as a simple breadth-expansion regime. The V11 fixed sample contains many falling-breadth observations, and explicit breadth-rising rules became too sparse and unstable.

The next hypothesis is different: V11 may be selecting a **concentrated bull market**. The KOSPI is strong, but only a moderate share of stocks are above their 60-day moving average. In that environment, a cross-sectional relative-strength strategy can potentially benefit from unusually clear leadership.

## Frozen components

V6 stock selection, historical KOSPI membership, liquidity and price filters, next-open entry, fixed 20-session close exit, 20-session ticker cooldown, 0.4% round-trip cost, and all data corrections remain unchanged.

V11 fixed `KOSPI 60d return > 2.5% + liquid breadth60 50-65%` remains a benchmark.

## Causal market-state variables

For every trading day, using current and past data only:

- `KOSPI_RET60_PCTL756`: percentile of current KOSPI 60-day return in the trailing 756 sessions, minimum 252 observations.
- `LIQ_BREADTH60_PCTL756`: percentile of current liquid-universe breadth60 in the trailing 756 sessions.
- `LEADERSHIP_SPREAD = KOSPI_RET60_PCTL756 - LIQ_BREADTH60_PCTL756`.

A positive leadership spread means index momentum is stronger relative to its own history than market participation is relative to its history.

## Prespecified candidates

- `strong_index_mid_breadth`: KOSPI above SMA120, return percentile >= 65%, breadth percentile 30-70%.
- `leadership_divergence_15`: KOSPI above SMA120, return percentile >= 65%, breadth percentile 25-75%, leadership spread >= 15 percentage points.
- `leadership_divergence_25`: stricter version with return percentile >= 70%, breadth percentile 20-70%, spread >= 25 points.
- `concentrated_bull`: KOSPI 60-day return > 0, KOSPI above SMA120, raw breadth60 35-70%, leadership spread >= 15 points.

These are not grid-searched. 2025-2026 is not used to select among them.

## Evaluation and gate

V13 reuses the V12 gate:

1. positive pre-2025 mean net return
2. positive pre-2025 mean KOSPI excess return
3. positive excess return in all four long eras
4. positive net return in all four long eras
5. at least 20 trades in every long era
6. positive excess return after removing the top three winners
7. at least 70% of eligible rolling five-year windows positive
8. worst eligible rolling five-year excess return > -1.0%
9. strict statistical pass additionally requires the lower date-cluster bootstrap bound > 0

No result should be treated as untouched out-of-sample evidence because historical periods have already been research-exposed and the stock universe still has current-survivor bias.
