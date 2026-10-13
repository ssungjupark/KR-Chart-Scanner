# V15 Full-Equity Portfolio Comparison

## Purpose
Correct the earlier portfolio comparison that left most capital in cash. The V15 signal logic is unchanged. This test keeps capital fully invested in equities at all times by holding a KOSPI core and replacing part of that core with V15 signal stocks.

## Fixed accounting
- Period: 1996 through 2024-12-30
- Baseline: 100% KOSPI buy-and-hold
- Initial/final KOSPI trade cost included
- Round-trip cost assumption: 0.4%
- Signal entry: next open
- Signal exit: fixed 20-session close
- On entry, KOSPI is sold to fund the signal stock
- On exit, proceeds are immediately reinvested into KOSPI
- Strategic cash: 0%
- Leverage: 0%
- Total equity exposure check: 100% every day
- Signal rules, market state, cooldown and direct Samsung Electronics/SK hynix exclusion are unchanged from V15

## Full-period results

| Model | Total return | CAGR | CAGR vs KOSPI | MDD | Ann. vol | Accepted signals | Avg signal-stock sleeve |
|---|---:|---:|---:|---:|---:|---:|---:|
| KOSPI buy & hold | +170.88% | 3.50% | - | -71.63% | 25.23% | - | 0.00% |
| 10% per signal overlay | +215.29% | 4.04% | +0.54%p | -71.63% | 25.49% | 200 / 208 | 5.22% |
| 20% per signal overlay | +299.28% | 4.89% | +1.39%p | -71.63% | 26.26% | 169 / 208 | 8.80% |
| 25% per signal overlay | +295.16% | 4.85% | +1.36%p | -71.63% | 26.68% | 151 / 208 | 9.83% |

The 20% and 25% variants skip more signals because simultaneous positions can exhaust the KOSPI core. Therefore they are not pure linear rescalings of the 10% model.

## Era comparison vs KOSPI

| Era | 10% overlay excess | 20% overlay excess | 25% overlay excess |
|---|---:|---:|---:|
| 1996-2007 | +0.07% | -11.91% | -25.27% |
| 2008-2014 | +2.49% | +18.45% | +30.24% |
| 2015-2021 | +10.97% | +20.52% | +30.19% |
| 2022-2024 | +4.88% | +13.32% | +5.27% |

The 10% overlay is the only tested sizing that avoids a negative era, although its 1996-2007 advantage is essentially flat. The 20% and 25% variants produce higher full-sample wealth but fail the early-era robustness check.

## Year-by-year consistency

| Model | Years outperforming KOSPI | Share | Mean annual excess | Worst annual excess |
|---|---:|---:|---:|---:|
| 10% overlay | 12 / 29 | 41.4% | +0.34%p | -12.17%p |
| 20% overlay | 11 / 29 | 37.9% | +1.20%p | -20.30%p |
| 25% overlay | 13 / 29 | 44.8% | +1.10%p | -30.02%p |

Outperformance is not frequent year by year. The long-run improvement comes from episodic periods of large positive stock-selection contribution rather than steady annual alpha.

## Interpretation
The earlier `candidate_cash +60.7% vs KOSPI +172%` comparison was not an appropriate primary benchmark because average stock exposure was only about 5%. When unused capital is instead kept in KOSPI, the same 10% signal substitution raises net terminal wealth from +170.9% to +215.3% with nearly unchanged full-period volatility and MDD.

This supports treating V15 as an **equity overlay / stock-selection model**, not as a standalone cash-timing strategy.

However, the result does not justify selecting 20% or 25% ex post. Those larger weights have worse early-era robustness and lower signal acceptance because of capacity constraints. The cleanest prespecified architecture remains the existing 10-slot / 10%-per-signal overlay, now with the unused capital invested in KOSPI rather than cash.

## Research status
This is still research-exposed history and still uses the current-survivor universe. It corrects portfolio accounting and benchmark comparability; it does not create a new untouched out-of-sample test.
