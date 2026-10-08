# V13 leadership concentration results

## Bottom line

V13 improves the V11 fixed raw-breadth hypothesis, but it does not pass the frozen robustness gate.

The best generalized candidate is `strong_index_mid_breadth`:

- KOSPI close above SMA120
- KOSPI 60-day return is at or above the 65th percentile of its trailing 756 sessions
- liquid-universe breadth60 is between the 30th and 70th percentiles of its trailing 756 sessions
- frozen V6 stock-level RS/setup/trigger rules
- next-session-open entry, fixed 20-session close exit, 0.4% round-trip cost

This removes the V11 fixed raw breadth 50-65% band and replaces it with relative, causal market-state percentiles.

## Pre-2025 comparison

| Model | Trades | Mean net | Mean KOSPI excess | Excess win rate | Top 3 winners removed excess | Cluster bootstrap 95% |
|---|---:|---:|---:|---:|---:|---:|
| V6 baseline | 1,129 | -0.50% | -0.45%p | 43.4% | -0.67%p | -1.20%p to +0.36%p |
| V11 fixed 50-65 / +2.5% | 231 | +1.93% | +1.20%p | 46.3% | +0.28%p | -0.88%p to +3.28%p |
| **V13 strong_index_mid_breadth** | **210** | **+2.24%** | **+1.10%p** | **48.6%** | **+0.23%p** | **-0.89%p to +3.39%p** |
| concentrated_bull | 251 | +1.49% | +0.73%p | 46.6% | -0.13%p | -1.07%p to +2.69%p |
| leadership_divergence_15 | 182 | +1.48% | +0.36%p | 46.7% | -0.66%p | -1.92%p to +2.70%p |
| leadership_divergence_25 | 104 | +1.73% | +0.09%p | 48.1% | -1.54%p | -3.15%p to +3.37%p |

The explicit leadership-spread filters do not improve robustness. The useful part is simpler: **strong index momentum plus middle, non-saturated breadth**.

## `strong_index_mid_breadth` by long era

| Era | Trades | Mean net | Mean KOSPI excess |
|---|---:|---:|---:|
| 1996-2007 | 37 | +4.80% | +0.42%p |
| 2008-2014 | 66 | +0.29% | +0.40%p |
| 2015-2021 | 71 | +2.67% | +1.48%p |
| 2022-2024 | 36 | +2.32% | +2.34%p |

This is the first generalized, relative market-state definition tested in this research sequence that has both positive mean net return and positive mean index excess in all four long eras with at least 20 trades in each era.

## Why it still fails the frozen gate

The main weakness is rolling stability.

Among the 22 rolling five-year windows with at least 15 completed trades:

- only 54.5% have positive mean excess return
- worst five-year mean excess return is -1.60%p
- the required thresholds were >=70% positive windows and worst excess > -1.0%p
- the date-cluster bootstrap lower bound remains negative (-0.89%p)

There are clear weak clusters around 2004-2013 and again 2016-2023. Therefore the broad-era averages hide meaningful multi-year periods where the edge disappears.

The pre-2025 median excess return is also still negative (-0.51%p), even though the mean is +1.10%p. The overall result is right-tail dependent, although removing the top three winners still leaves a small positive +0.23%p mean excess.

## Reused 2025-2026 evaluation

`strong_index_mid_breadth` has 20 completed reused trades:

- mean net return +9.92%
- mean KOSPI excess +2.47%p
- median excess -3.92%p
- top three winners removed excess -5.58%p

This recent result is strongly right-tail dependent and must not be used as evidence to tune the rule further.

## Research decision

Do not merge V13 into the production scanner and do not tune the 65% / 30-70% thresholds on the same historical sample.

The useful finding is structural rather than production-ready:

> V6 behaves materially better when KOSPI medium-term momentum is strong relative to its own history while market participation is neither very weak nor fully saturated.

This is more defensible than the raw V11 50-65% breadth band, but the edge is not stable enough across rolling multi-year windows to automate as a buy rule.

A next research step should change the information set rather than further optimize market-breadth thresholds. Good candidates are sector-relative leadership, market-cap concentration / index-contribution measures, or historical-universe data that reduces survivor bias.
