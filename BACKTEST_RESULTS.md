# Backtest Results

## Run 1: reversal score panel test

- Period: 2024-01-02 to 2026-09-04
- Universe: 20-stock KOSPI research panel
- Benchmark: KOSPI (`KS11`)
- Minimum recorded score: 60
- Cooldown: 20 trading sessions per stock
- Total signals: 522
- Forward windows: 5, 10 and 20 trading sessions
- Signal generation uses only data available through each historical evaluation date.

### Performance by grade

| Grade | Signals | Avg 5D | Avg 10D | Avg 20D | Median 20D | 20D win rate | Avg 20D excess vs KOSPI | Avg 20D MFE | Avg 20D MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 60 | +0.53% | +0.81% | +2.71% | +1.44% | 59.32% | +2.46% | +13.37% | -10.42% |
| B | 151 | +0.18% | +0.38% | +2.56% | +0.82% | 52.98% | -1.25% | +13.00% | -9.41% |
| C | 311 | +1.46% | +3.71% | +5.75% | +3.42% | 58.84% | +1.89% | +16.24% | -7.97% |

### Performance by score band

| Score band | Signals | Avg 5D | Avg 10D | Avg 20D | Median 20D | 20D win rate | Avg 20D excess vs KOSPI | Avg 20D MFE | Avg 20D MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60-69 | 311 | +1.46% | +3.71% | +5.75% | +3.42% | 58.84% | +1.89% | +16.24% | -7.97% |
| 70-79 | 151 | +0.18% | +0.38% | +2.56% | +0.82% | 52.98% | -1.25% | +13.00% | -9.41% |
| 80-89 | 55 | +0.60% | +0.96% | +2.57% | +1.39% | 59.26% | +2.15% | +13.37% | -10.51% |
| 90-100 | 5 | -0.29% | -0.85% | +4.28% | +2.36% | 60.00% | +5.73% | +13.44% | -9.45% |

## Interpretation

The first test does not show a monotonic relationship between the current score and subsequent raw return. In particular, 60-69 signals outperformed 70-79 and 80-89 signals on average 20-day raw return. This means the initial weights should be treated as hypotheses rather than calibrated probabilities.

A plausible mechanism is that some lower-score signals occur earlier in a reversal, while higher scores require confirmation that arrives after part of the move has already occurred. This must be tested rather than assumed.

Excess return versus the benchmark gives a somewhat different picture: A-grade signals produced +2.46% average 20-day excess return, while B-grade signals were -1.25%. The 90-100 bucket is too small at only five observations to draw a conclusion.

## Important limitations

This is a research-panel test, not a production-grade market backtest. The 20 stocks are a current hand-selected KOSPI panel, so survivorship and selection bias remain. Signals across stocks can also overlap in the same market regime and are not statistically independent.

The next validation step should use a much broader KRX universe with liquidity filters, separate in-sample and out-of-sample periods, and factor-level attribution before changing the scoring weights.
