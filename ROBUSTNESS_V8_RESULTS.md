# Robustness v8 results

Completed 2026-10-08. GitHub Actions [run 37739042122](https://github.com/ssungjupark/KR-Chart-Scanner/actions/runs/37739042122) passed, including 13 execution/accounting tests. Local and remote result tables agree. Research PR: [#2](https://github.com/ssungjupark/KR-Chart-Scanner/pull/2).

## Result

Neither market produced a sizing/exit policy satisfying the prespecified positive absolute return, positive full-slot excess and improvement requirements in both purged training and validation. Both retain equal sizing and fixed20. This does not prove every new rule is useless; it means none earned adoption under this experiment's criteria.

| Reused 2025-26 evaluation | KOSPI | KOSDAQ |
|---|---:|---:|
| Fixed v6 entry signals | 60 | 34 |
| v6 / selected v8 average net trade return | 5.09% | 2.35% |
| Average same-horizon index excess | 2.17% | 0.87% |
| Average trade MAE | -10.59% | -15.79% |
| Selected sizing / exit | equal / fixed20 | equal / fixed20 |

Exact entry counts reproduce: KOSPI 129/75/60 and KOSDAQ 68/33/34 for training, validation and evaluation. All 2,524 frozen current-universe stocks processed without errors. Purged selection uses 122/69 KOSPI and 67/33 KOSDAQ trades in training/validation.

## Exit-only evaluation on identical entries

| Exit | KOSPI net trade return | KOSPI horizon excess | KOSDAQ net trade return | KOSDAQ horizon excess |
|---|---:|---:|---:|---:|
| Fixed20 | 5.09% | 2.17% | 2.35% | 0.87% |
| Net +10% arms, then 5% closing drawdown | 4.87% | 1.95% | 0.64% | -0.84% |
| Net +10% arms, then 2ATR closing drawdown | 4.53% | 1.61% | 2.47% | 0.98% |

Profit-armed stops did not improve KOSPI return. KOSDAQ's 2ATR increase is small and does not satisfy training/validation requirements. Losing trades that never arm remain exposed through fixed20, so MAE improves only slightly.

## Sizing-only evaluation

Budget returns include unused half-slot cash earning zero and retain every entry.

| Rule | KOSPI budget return | KOSPI horizon excess | KOSDAQ budget return | KOSDAQ horizon excess |
|---|---:|---:|---:|---:|
| Equal | 5.09% | 2.17% | 2.35% | 0.87% |
| Half size without dry-up | 4.61% | 1.69% | 3.83% | 2.35% |
| Half size below quality2 | 5.01% | 2.09% | 2.95% | 1.47% |
| Half size in bear regime | 4.93% | 2.01% | 2.42% | 0.94% |

The KOSDAQ dry-up result is exploratory reused-evaluation evidence, not a selected model. It fails the training/validation adoption criteria. Reduced slot MAE from scaling is partly mechanical lower exposure, not a change in the underlying stocks' adverse paths.

## Actual 10-slot portfolios

These cumulative returns are separate from average trade returns. Each market starts with cash, opens full-size positions at 10% of opening equity, pays 20bp per side, uses no leverage and releases close-exit cash after opening orders.

| Market / actual evaluation calendar | Accepted / skipped | Portfolio return | Fully invested index | Difference | Portfolio MDD | Average exposure |
|---|---:|---:|---:|---:|---:|---:|
| KOSPI, 2025-01-06 to 2026-10-02 | 58 / 2 | 32.66% | 185.48% | -152.82pp | -8.87% | 26.13% |
| KOSDAQ, 2025-01-03 to 2026-07-09 | 34 / 0 | 6.81% | 15.40% | -8.59pp | -13.74% | 17.89% |

The chosen v8 portfolios equal v6 because both fall back. Positive event-level excess does not establish portfolio outperformance. Sparse entries and substantial cash exposure miss much of a sustained market rise. The index comparator remains fully invested; these portfolios have a different risk/exposure profile.

## Benchmark data correction

The upstream KRX index cache stopped on 2026-09-17. Its terminal candle was an intraday snapshot; the preceding overlap through 2026-09-16 exactly matches NAVER. Completed NAVER candles replace the terminal bar and extend the series, without changing the entry window ending 2026-09-04.

One KOSPI fixed20 outcome runs through 2026-10-02. Earlier code filled the September index value forward to that date. Correcting this changes the prior KOSPI average excess from 2.24% to 2.17%; net stock return stays 5.09%. Outcome windows now reject missing index prices instead of filling them forward. For 2026 alone, net KOSPI return is -4.66% and corrected excess is -0.06%; net KOSDAQ return is -5.40% and excess is -0.18%.

## Scope and reproducibility

2025-26 is reused evaluation, not untouched holdout. Current-universe survivorship bias remains. Dividends, limit-lock execution and a variable tax/slippage schedule are not modeled. Full feature, sizing, exit, attribution and portfolio CSVs accompany the workflow artifact; frozen input data, dependencies and SHA256 hashes are preserved. The live scanner is outside this research PR.
