import unittest

import numpy as np
import pandas as pd

from index_distortion_v14 import add_market_state, rolling_compound_return


class V14MarketStateTests(unittest.TestCase):
    def _panel(self, n_days=320):
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        rows = []
        tickers = ["005930", "000660", "111111", "222222"]
        for i, d in enumerate(dates):
            for t in tickers:
                mega = t in {"005930", "000660"}
                rows.append({
                    "date": d,
                    "ticker": t,
                    "is_mega": mega,
                    "RET1": 0.10 if mega else 0.01,
                    "STOCK_RET60": 0.50 if mega else 0.05,
                    "ABOVE20": True,
                    "ABOVE60": True if mega else (i % 2 == 0),
                    "ABOVE120": True,
                    "KOSPI_RET60": 0.25 + i * 0.0001,
                    "KOSPI_ABOVE120": True,
                })
        return pd.DataFrame(rows)

    def test_exmega_equal_weight_really_excludes_megacaps(self):
        daily = add_market_state(self._panel())
        last = daily.iloc[-1]
        self.assertAlmostEqual(last["EXMEGA_EW_RET1"], 0.01, places=12)
        self.assertAlmostEqual(last["LIQ_EW_RET1"], 0.055, places=12)
        self.assertAlmostEqual(last["EXMEGA_MEDIAN_RET60"], 0.05, places=12)

    def test_rolling_compound_is_causal(self):
        s = pd.Series([0.01] * 80)
        a = rolling_compound_return(s, 60)
        b = s.copy()
        b.iloc[70:] = 0.50
        b = rolling_compound_return(b, 60)
        pd.testing.assert_series_equal(a.iloc[:70], b.iloc[:70])

    def test_percentiles_are_bounded(self):
        daily = add_market_state(self._panel())
        cols = [c for c in daily.columns if c.endswith("PCTL756")]
        for c in cols:
            x = daily[c].dropna()
            self.assertTrue(((x >= 0) & (x <= 1)).all(), c)


if __name__ == "__main__":
    unittest.main()
