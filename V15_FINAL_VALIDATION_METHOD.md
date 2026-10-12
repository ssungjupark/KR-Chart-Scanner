# V15 Final Frozen Validation

V15 is the stopping version for this research sequence. It does not search for a new entry rule or market threshold.

## Frozen candidate

The candidate is fixed from V14 before V15 stress results are inspected:

- stock entry = unchanged V6 setup + trigger + RS rules;
- market regime = KOSPI above SMA120, KOSPI 60-day return trailing-756-session percentile >= 65%, liquid breadth60 trailing-756-session percentile between 30% and 70%;
- Samsung Electronics (`005930`) and SK hynix (`000660`) are excluded as direct trade candidates;
- 20-session cooldown;
- entry at next session open;
- base exit at signal-date +20th session close;
- base round-trip cost = 0.4%.

The V14 result showed that excluding these two direct trades did not weaken the strategy, while a KOSPI-only strong state without non-megacap confirmation was not robust. V15 therefore freezes the simpler V13 regime rather than adding another market filter.

## Final stress diagnostics

V15 reports, without selecting among them:

- four long eras;
- rolling 5-year and rolling 7-year windows;
- 0.4%, 0.8%, and 1.2% round-trip costs;
- top 1/3/5/10 winner removal;
- fixed 5/10/20/40/60-session horizons;
- ticker contribution concentration;
- date-cluster bootstrap;
- 10-slot stock+cash portfolio;
- same-schedule index portfolio;
- index-core portfolio with candidate substitutions;
- 2025-2026 reused evaluation reported separately.

## Frozen practical gate

The candidate passes only if all of the following hold:

1. pre-2025 mean net return > 0;
2. pre-2025 mean excess return >= +0.7%p per trade;
3. at least 3 of 4 long eras have positive mean excess return;
4. removing the top 3 excess-return trades leaves mean excess >= 0;
5. at 0.8% round-trip cost, mean excess remains > 0;
6. at least 65% of rolling 5-year windows with >=15 trades have positive mean excess;
7. worst eligible rolling 5-year mean excess >= -1.5%p;
8. date-cluster bootstrap 95% lower bound >= -0.7%p.

These criteria were set before V15 final results. V15 does not alter them after seeing the output.

## Interpretation

- `PRACTICAL_PASS`: candidate is strong enough for a separate forward/paper-trading phase, not immediate live auto-trading.
- `RESEARCH_ONLY`: preserve the result as evidence, but do not keep tuning the same historical sample. Any future work must change the information set materially (for example sector-relative leadership or survivorship-clean data) rather than adjust thresholds.

1996-2024 and 2025-2026 are already research-exposed. V15 is not an untouched out-of-sample test.
