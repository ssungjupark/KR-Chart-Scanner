# V15 Frozen 10% Full-Equity Stress Results

## Verdict
`RESEARCH_OVERLAY_ONLY` — the frozen stress gate failed.

No V15 signal threshold or 10% sizing parameter was changed after the gate was specified.

## Base portfolio, 1996-2024
- KOSPI core + 10% per accepted V15 signal
- total equity exposure: 100% every day
- accepted signals: 200 / 208
- average signal-stock sleeve: 5.22%
- round-trip cost: 0.4%
- total return: +215.29%
- KOSPI total return: +170.88%
- CAGR: 4.04% vs KOSPI 3.50% (+0.54%p)
- MDD: -71.63%
- annualized volatility: 25.49%

## Cost stress
| Round-trip cost | Overlay total return | KOSPI total return | Overlay CAGR | KOSPI CAGR | CAGR excess |
|---|---:|---:|---:|---:|---:|
| 0.4% | +215.29% | +170.88% | 4.04% | 3.50% | +0.54%p |
| 0.8% | +167.43% | +169.79% | 3.45% | 3.48% | -0.03%p |
| 1.2% | +126.88% | +168.72% | 2.87% | 3.47% | -0.60%p |

The full-equity overlay loses its KOSPI edge once the assumed round-trip cost reaches 0.8%.

## Rolling consistency
| Window | Number | Positive CAGR-excess share | Median CAGR excess | Worst CAGR excess |
|---|---:|---:|---:|---:|
| 5-year | 25 | 48.0% | -0.52%p | -1.89%p |
| 7-year | 23 | 65.2% | +0.50%p | -1.43%p |

The 7-year requirement passed, but the 5-year requirement failed. The edge is episodic rather than consistently present across medium-length market cycles.

## Winner-dependence stress
The baseline accepted-signal set was frozen. Removed winners were replaced by KOSPI exposure; formerly capacity-skipped signals were not backfilled.

| Top winners removed | Overlay total return | KOSPI | CAGR excess |
|---|---:|---:|---:|
| 0 | +215.29% | +170.88% | +0.54%p |
| 1 | +196.43% | +170.88% | +0.32%p |
| 3 | +166.49% | +170.88% | -0.06%p |
| 5 | +145.06% | +170.88% | -0.36%p |
| 10 | +107.27% | +170.88% | -0.95%p |

Removing only the top three accepted winners eliminates the full-period KOSPI edge. This is the most important robustness failure.

## Industry leave-one-out
The five highest-contributing current-industry groups were removed one at a time without backfilling. All five variants retained positive CAGR excess versus KOSPI:

| Removed current industry | Removed signals | CAGR excess |
|---|---:|---:|
| 절연선 및 케이블 제조업 | 5 | +0.13%p |
| 의약품 제조업 | 9 | +0.35%p |
| 1차 철강 제조업 | 11 | +0.38%p |
| 특수 목적용 기계 제조업 | 2 | +0.39%p |
| 1차 비철금속 제조업 | 5 | +0.39%p |

So the edge is not explained by one industry group. However this test uses a 2026-03 industry snapshot retrospectively, so it is exploratory rather than point-in-time sector validation.

## Reused 2025-current diagnostic
- completed signals: 20
- overlay total return: +188.63%
- KOSPI total return: +182.26%
- total-return difference: +6.37%p
- CAGR excess: +2.30%p

This period has already influenced research and is not out-of-sample evidence.

## Frozen gate
Pass:
- rolling 7-year positive share >=65%
- worst rolling 5-year CAGR excess >= -2.0%p
- industry leave-one-out 5/5 positive

Fail:
- 0.8% cost CAGR excess >0
- 1.2% cost CAGR excess >=0
- rolling 5-year positive share >=55%
- top-3 winners removed still beats KOSPI

Final status: **RESEARCH_OVERLAY_ONLY**.

## Interpretation
The corrected fully-invested comparison confirms that the V15 stock-selection overlay adds value at the base 0.4% transaction-cost assumption. It is therefore materially better than the earlier mostly-cash interpretation.

But the edge is not robust enough for production auto-trading: it is thin relative to realistic cost stress, dependent on a few large winners, and absent in more than half of rolling 5-year windows. No further threshold tuning should be performed on the same 1996-2026 research-exposed sample.
