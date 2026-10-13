# V15 Frozen Full-Equity Stress Protocol

This is not a new strategy version and does not change any V15 signal threshold.

Frozen portfolio:
- 100% equity exposure at all times
- KOSPI core when no signal is active
- 10% of then-current equity per accepted V15 signal, funded by selling KOSPI
- fixed 20-session holding period
- no leverage and no strategic cash

Predefined stress checks:
- round-trip costs: 0.4%, 0.8%, 1.2%
- rolling 5-year and 7-year KOSPI-relative CAGR
- remove top 1/3/5/10 accepted winners without backfilling skipped signals
- remove each of the five highest-contributing current-industry groups without backfilling
- reused 2025-current diagnostic

Frozen practical stress gate before seeing results:
- 0.8% cost CAGR excess > 0
- 1.2% cost CAGR excess >= 0
- at least 55% of rolling 5-year windows have positive CAGR excess
- at least 65% of rolling 7-year windows have positive CAGR excess
- worst rolling 5-year CAGR excess >= -2.0%p
- removing the top 3 accepted winners still leaves positive full-period CAGR excess
- at least 4 of 5 leave-one-industry-out tests retain positive CAGR excess

Industry caveat: the 2026-03-23 FinanceDataReader/KRX description snapshot is a current classification applied retrospectively. Industry leave-out is therefore exploratory and not a point-in-time sector test.

All 1996-2024 and 2025-current data remain research-exposed. Passing this stress gate is not equivalent to untouched out-of-sample validation.
