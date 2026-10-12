# V15 Final Validation Results

## Final verdict

**RESEARCH_ONLY — stop threshold tuning on the same historical sample.**

The frozen candidate was reproduced exactly:

- pre-2025 completed trades: **208**
- 2025-2026 reused trades: **20**

The candidate is the V13 market regime with Samsung Electronics (`005930`) and SK hynix (`000660`) excluded as direct trade candidates.

## Pre-2025 trade-level result

- mean net 20-session return: **+2.35%**
- mean KOSPI excess return: **+1.18%p**
- median excess return: **-0.32%p**
- excess win rate: **49.0%**
- top-3 winner removed mean excess: **+0.31%p**
- date-cluster bootstrap 95% interval: **-0.89%p to +3.42%p**

## Era robustness

| Era | Trades | Mean net | Mean excess |
|---|---:|---:|---:|
| 1996-2007 | 36 | +5.00% | +0.67%p |
| 2008-2014 | 65 | +0.53% | +0.50%p |
| 2015-2021 | 71 | +2.67% | +1.48%p |
| 2022-2024 | 36 | +2.32% | +2.34%p |

All four eras have positive mean net and positive mean excess returns.

## Cost stress

| Round-trip cost | Mean net | Mean excess |
|---|---:|---:|
| 0.4% | +2.35% | +1.18%p |
| 0.8% | +1.95% | +0.78%p |
| 1.2% | +1.55% | +0.38%p |

Transaction costs are not the reason the candidate fails the final gate.

## Winner dependence

| Top excess winners removed | Mean excess |
|---:|---:|
| 0 | +1.18%p |
| 1 | +0.87%p |
| 3 | +0.31%p |
| 5 | -0.12%p |
| 10 | -1.01%p |

The candidate survives the prespecified top-3 test, but its edge is still meaningfully right-tail dependent.

## Rolling five-year stability

There were **22** rolling five-year windows with at least 15 trades.

- positive mean excess windows: **54.5%**
- frozen gate requirement: **>=65%**
- worst five-year mean excess: **-1.49%p**
- frozen worst-window tolerance: **>=-1.50%p**

Worst windows included:

- 2005-2009: **-1.49%p**
- 2018-2022: **-1.44%p**
- 2017-2021: **-1.28%p**
- 2016-2020: **-1.18%p**

The rolling stability criterion is the first decisive failure.

Rolling seven-year windows were materially more stable (18/22 positive, about **81.8%**), which suggests the effect is persistent at a longer horizon but can underperform for multi-year stretches.

## Holding-horizon diagnostic

For the same completed signals:

- 5 sessions: mean excess **-0.47%p**
- 10 sessions: **-0.16%p**
- 20 sessions: **+1.18%p**
- 40 sessions: **+0.23%p**
- 60 sessions: **+0.14%p**

The edge is concentrated around the 20-session horizon rather than appearing immediately after entry. This remains a warning that the trigger itself is not a strong short-horizon timing signal.

## Portfolio-level result, 1996-2024

Using the fixed 10-slot implementation with 10% opening-equity position sizing and no leverage:

| Model | Total return | KOSPI benchmark | MDD | Avg exposure |
|---|---:|---:|---:|---:|
| Candidate stock + cash | +60.72% | +171.96% | -11.74% | 5.23% |
| Same-schedule index + cash | +17.45% | +171.96% | -8.27% | 5.22% |
| Index core + candidate substitutions | +215.29% | +171.96% | -71.63% | 5.22% candidate exposure |

The candidate clearly improves the asset selected on its sparse entry schedule, but a standalone stock+cash strategy has very low average exposure and therefore large cash drag. It is better interpreted as a selective overlay than as a complete portfolio strategy.

The very large MDD reported for the index-core substitution simulation should not be treated as an adoption metric without a separate audit of that portfolio construction; V15 did not select or tune on it.

## 2025-2026 reused evaluation

- trades: **20**
- mean net: **+9.92%**
- mean excess: **+2.47%p**
- median excess: **-3.92%p**
- top-3 winner removed excess: **-5.58%p**

Recent performance is strongly right-tail dependent and cannot rescue the historical gate.

## Frozen gate outcome

Passed:

- pre-2025 mean net > 0
- pre-2025 mean excess >= +0.7%p
- at least 3/4 eras positive (actual 4/4)
- top-3 winner removed excess >= 0
- 0.8% round-trip cost excess > 0
- worst rolling 5-year excess >= -1.5%p

Failed:

- rolling 5-year positive share >=65% (**actual 54.5%**)
- date-cluster bootstrap lower bound >=-0.7%p (**actual -0.89%p**)

Therefore `final_pass = False` and `verdict = RESEARCH_ONLY`.

## Stop rule

Do **not** create V16 by changing the same market percentile or breadth thresholds on the same 1996-2026 history. That would convert the remaining uncertainty into data-mined backtest fit.

If research resumes, it must materially change the information set or validation set, for example:

1. point-in-time universe including delisted names to reduce survivorship bias;
2. sector-relative leadership (`market -> sector -> stock`) rather than another market threshold;
3. forward/paper signals accumulated after the V15 freeze date.

The production `scanner.py` / `main.py` remains unchanged.
