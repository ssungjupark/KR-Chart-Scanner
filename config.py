from dataclasses import dataclass


@dataclass(frozen=True)
class ScannerConfig:
    # Core moving averages
    ma_short: int = 5
    ma_fast: int = 20
    ma_mid: int = 60
    ma_long: int = 120

    # Momentum / volatility
    rsi_period: int = 14
    atr_period: int = 14

    # Slope lookback in trading days
    slope_lookback: int = 5

    # Swing-point definition: a pivot must be the lowest/highest point
    # inside a centered window of 2 * swing_span + 1 trading days.
    swing_span: int = 3

    # Reversal-screen thresholds
    ma_gap_max: float = 0.04          # 20d / 60d MA gap <= 4%
    ma_price_band: float = 0.03       # close may sit within +/- 3% of MA
    fast_slope_min: float = 0.002     # 20d MA: +0.2% or more over 5 days
    mid_slope_flat_min: float = -0.002  # 60d MA can be almost flat
    rsi_low: float = 45.0
    rsi_high: float = 65.0
    rs20_min: float = 0.0             # outperform benchmark over 20 days
    rs_slope_min: float = 0.0
    resistance_space_min: float = 0.08  # >= 8% room to prior 60d high

    # Minimum history required before scoring
    min_history: int = 150


DEFAULT_CONFIG = ScannerConfig()
