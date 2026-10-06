# Robustness v6 results

Run: 2026-10-06, GitHub Actions run 37454819774

## Method

- Universe: 2,525 currently listed KOSPI/KOSDAQ common stocks with normal six-digit numeric tickers.
- Period: 2022-01-03 through 2026-09-04.
- Train: through 2023-12-28.
- Validation: 2023-12-29 through 2024-12-30.
- Holdout: after 2024-12-30, never used for model selection.
- Liquidity: ADV20 >= KRW 2bn, price >= KRW 1,000.
- Cross-sectional RS percentiles computed separately within KOSPI and KOSDAQ for 20, 60 and 120 trading-day relative strength.
- Entry-pattern filters are applied before the 20-session signal cooldown.
- Model selection requires positive average 20-day excess return in both train and validation, plus minimum sample counts. Excess-return hit rate is reported but not used as a hard gate.
- Important limitation: the universe contains stocks currently listed today, so historical results still have survivorship bias because the FinanceDataReader delisted-list endpoint returned no usable data in this runtime.

## Selected KOSPI model

Setup: leader20 / break3

- RS20 percentile >= 50%
- RS60 percentile >= 60%
- RS120 percentile >= 70%
- Market bull gate: off

| Period | Signals | Avg 20d return | Avg 20d excess | Excess win rate | Avg MAE 20d |
|---|---:|---:|---:|---:|---:|
| Train 2022-2023 | 129 | n/a | +0.45% | n/a | n/a |
| Validation 2024 | 75 | +2.40% | +2.31% | 60.0% | -9.67% |
| Holdout 2025-2026 | 60 | n/a | +2.64% | n/a | n/a |

Annual decomposition of the exact selected model:

| Year | Signals | Avg 20d return | Avg 20d excess | Excess win rate | Avg MAE 20d |
|---|---:|---:|---:|---:|---:|
| 2022 | 67 | -1.09% | +0.93% | 47.8% | -10.50% |
| 2023 | 62 | +0.31% | -0.06% | 46.8% | -10.59% |
| 2024 | 75 | +2.40% | +2.31% | 60.0% | -9.67% |
| 2025 | 49 | +7.68% | +3.07% | 42.9% | -9.73% |
| 2026 | 11 | -4.26% | +0.73% | 36.4% | -14.43% |

## Selected KOSDAQ model

Setup: leader20 / ma5

- RS20 percentile >= 70%
- RS60 percentile >= 80%
- RS120 percentile >= 80%
- Market bull gate: off

| Period | Signals | Avg 20d excess |
|---|---:|---:|
| Train 2022-2023 | 68 | +2.79% |
| Validation 2024 | 33 | +1.93% |
| Holdout 2025-2026 | 34 | +1.27% |

Annual decomposition of the exact selected model:

| Year | Signals | Avg 20d return | Avg 20d excess | Excess win rate | Avg MAE 20d |
|---|---:|---:|---:|---:|---:|
| 2022 | 38 | -4.08% | -1.62% | 42.1% | -13.39% |
| 2023 | 30 | +10.83% | +8.37% | 50.0% | -12.10% |
| 2024 | 33 | -1.63% | +1.93% | 54.5% | -16.48% |
| 2025 | 25 | +5.55% | +1.65% | 48.0% | -12.80% |
| 2026 | 9 | -5.00% | +0.22% | 44.4% | -24.08% |

## RS sensitivity

Using the selected chart setup/trigger and imposing the same RS percentile floor on RS20, RS60 and RS120:

| Market | RS floor | Total signals | Mean annual excess | Worst annual excess | Positive years |
|---|---:|---:|---:|---:|---:|
| KOSPI | 50% | 405 | +0.14% | -1.39% | 3/5 |
| KOSPI | 60% | 303 | +0.19% | -2.40% | 3/5 |
| KOSPI | 70% | 162 | +1.68% | -5.49% | 4/5 |
| KOSPI | 80% | 52 | -2.06% | -11.41% | 3/5 |
| KOSDAQ | 50% | 718 | +1.16% | -2.77% | 2/5 |
| KOSDAQ | 60% | 455 | +1.07% | -0.95% | 2/5 |
| KOSDAQ | 70% | 232 | +1.40% | -1.67% | 2/5 |
| KOSDAQ | 80% | 82 | -0.21% | -2.94% | 2/5 |

## Interpretation

The v4 result was overstated by mixing KOSPI and KOSDAQ and by using a model that had negative train/validation alpha. v6 is more credible because model selection is market-specific, uses 20/60/120-day cross-sectional relative strength, applies the signal cooldown after model filters, and requires positive mean alpha in both train and validation.

KOSPI is materially more stable than the earlier v4 model: four of five annual buckets have positive excess return and 2023 is approximately flat. KOSDAQ is less stable, with a negative 2022 and much larger adverse excursion, especially in 2026.

The models still are not production-ready. In 2026 both selected models have negative absolute 20-day returns despite positive benchmark-relative alpha, and drawdown/MAE remains large. The next research step should optimize for both positive absolute return and positive excess return, penalize MAE, add transaction costs/slippage, and run portfolio-level overlapping-position simulations. A historical-universe solution is also needed to remove survivorship bias.
