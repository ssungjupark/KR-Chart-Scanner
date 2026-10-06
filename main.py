from __future__ import annotations

import argparse

from config import DEFAULT_CONFIG
from data_loader import align_benchmark, load_benchmark_data, load_price_data
from indicators import add_indicators
from scanner import score_reversal


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def run(ticker: str, as_of: str, benchmark: str) -> None:
    cfg = DEFAULT_CONFIG

    stock = load_price_data(ticker, as_of)
    bench = load_benchmark_data(benchmark, as_of)
    benchmark_close = align_benchmark(stock, bench)
    enriched = add_indicators(stock, benchmark_close, cfg)

    result = score_reversal(ticker, enriched, cfg)

    m = result.metrics

    print("=" * 64)
    print(f"KR-Chart-Scanner | Reversal scan | {result.as_of}")
    print("=" * 64)
    print(f"Ticker : {result.ticker}")
    print(f"Score  : {result.score}/100 ({result.grade})")
    print()
    print(f"Close  : {m['close']:,.0f}")
    print(f"SMA20  : {m['sma20']:,.0f}")
    print(f"SMA60  : {m['sma60']:,.0f}")
    print(f"SMA120 : {m['sma120']:,.0f}")
    print(f"RSI14  : {m['rsi14']:.1f}")
    print(f"20d excess return vs benchmark : {pct(m['rs20_excess_return'])}")
    print(f"Room to prior 60d high         : {pct(m['space_to_prior_60d_high'])}")
    print()

    labels = {
        "higher_low": "Higher Low",
        "sma20_rising": "20d MA rising",
        "sma60_flat_or_rising": "60d MA flat/rising",
        "ma20_ma60_converged": "20d/60d MA converged",
        "price_near_ma20": "Price near 20d MA",
        "price_near_ma60": "Price near 60d MA",
        "rsi_normal": "RSI in 45-65",
        "rs20_positive": "20d relative strength positive",
        "rs_trend_positive": "Relative-strength trend positive",
        "correction_volume_ok": "Correction volume healthy",
        "resistance_space_ok": "Enough room to 60d resistance",
    }

    print("Conditions")
    for key, passed in result.checks.items():
        symbol = "PASS" if passed else "FAIL"
        print(f"[{symbol:4}] {labels[key]}")

    print()
    print("Diagnostic metrics")
    print(f"20d MA slope (5d) : {pct(m['sma20_slope_5d'])}")
    print(f"60d MA slope (5d) : {pct(m['sma60_slope_5d'])}")
    print(f"120d MA slope (5d): {pct(m['sma120_slope_5d'])}")
    print(f"20d/60d MA gap    : {pct(m['ma20_ma60_gap'])}")
    print(f"Price vs SMA20    : {pct(m['distance_to_sma20'])}")
    print(f"Price vs SMA60    : {pct(m['distance_to_sma60'])}")
    print(f"RS ratio slope    : {pct(m['rs_ratio_slope_5d'])}")
    print(
        "Swing lows          : "
        f"{m['previous_swing_low']} -> {m['latest_swing_low']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score a Korean stock using only data available at a historical as-of date."
    )
    parser.add_argument("--ticker", default="103590", help="KRX ticker, e.g. 103590")
    parser.add_argument("--as-of", default="2026-09-04", help="YYYY-MM-DD")
    parser.add_argument("--benchmark", default="KS11", help="Benchmark symbol, default KOSPI")
    args = parser.parse_args()

    run(args.ticker, args.as_of, args.benchmark)
