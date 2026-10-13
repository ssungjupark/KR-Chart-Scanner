# V15 Forward OOS Paper Tracker

This directory is the immutable forward ledger for the frozen V15 10% fully-invested overlay.

## Start

- Model freeze: 2026-10-13 KST, after the Korean market close.
- First eligible signal close: **2026-10-14**.
- 2026-10-13 signals are intentionally excluded so the forward sample starts strictly after implementation/freeze.

## Portfolio

The account is always 100% exposed to equities. With no active signal it holds the KOSPI core. Each accepted V15 signal substitutes 10% of then-current equity from the KOSPI core into the signal stock, up to 10 simultaneous positions. There is no strategic cash and no leverage.

A signal generated at a completed close enters at the next observed stock-session open. It exits at the close of the 20th observed stock session after the signal date and immediately returns to the KOSPI core. Base round-trip friction is frozen at 0.4%.

## Files written by the scheduled workflow

- `state.json`: operational state, pending orders and active positions.
- `signals.csv`: immutable signal ledger, including market-state inputs and data-source provenance.
- `trades.csv`: completed paper trades and same-interval KOSPI return.
- `portfolio.csv`: daily fully-invested overlay NAV vs KOSPI NAV.
- `latest.md`: human-readable current status.

## OOS integrity

The workflow never creates signals dated before 2026-10-14 and does not backfill a missed historical signal after a later market session has completed. Orders that were already generated before their entry session may still be filled from their predetermined next observed open if the workflow itself was delayed. Any threshold/model change requires a new model id, a new spec hash and a new forward start date.

The scheduled run is 18:30 KST on weekdays. `scanner.py` and `main.py` are not used or modified by this tracker.
