import unittest

import numpy as np
import pandas as pd

import v15_full_equity_stress as s


class V15FullEquityStressTests(unittest.TestCase):
    def toy(self):
        dates = pd.bdate_range("2020-01-02", periods=30)
        bench = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1_000_000}, index=dates)
        prices = {
            "000001": pd.DataFrame({"Open": 100.0, "High": 102.0, "Low": 99.0, "Close": np.linspace(100, 120, len(dates)), "Volume": 1_000_000}, index=dates),
            "000002": pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": np.linspace(100, 110, len(dates)), "Volume": 1_000_000}, index=dates),
        }
        trades = pd.DataFrame([
            {"ticker": "000001", "entry_date": dates[1], "entry_price": 100.0, "exit_date20": dates[21], "exit_price20": 115.0, "EXCESS20": 0.146},
            {"ticker": "000002", "entry_date": dates[2], "entry_price": 100.0, "exit_date20": dates[22], "exit_price20": 108.0, "EXCESS20": 0.076},
        ])
        return dates, bench, prices, trades

    def test_replay_is_fully_in_equities(self):
        dates, bench, prices, trades = self.toy()
        curve, summary, accepted = s.replay_full_equity(trades, bench, prices, dates[0], dates[-1], cost=0.004)
        self.assertTrue((curve.equity_exposure.sub(1).abs() < 1e-9).all())
        self.assertEqual(len(accepted), 2)
        self.assertAlmostEqual(summary["avg_total_equity_exposure"], 1.0, places=10)

    def test_allowed_keys_does_not_backfill(self):
        dates, bench, prices, trades = self.toy()
        allowed = {s.trade_key("000001", dates[1])}
        _, summary, accepted = s.replay_full_equity(trades, bench, prices, dates[0], dates[-1], cost=0.004, allowed_keys=allowed)
        self.assertEqual(accepted, allowed)
        self.assertEqual(summary["accepted_signals"], 1)

    def test_rolling_compare_flat_curves_zero_excess(self):
        dates = pd.bdate_range("2010-01-04", "2020-12-31")
        vals = np.linspace(1, 2, len(dates))
        curve = pd.DataFrame({"equity": vals}, index=dates)
        x = s.rolling_compare(curve, curve.copy(), 5)
        self.assertTrue((x.excess_return.abs() < 1e-12).all())
        self.assertTrue((x.cagr_excess.abs() < 1e-12).all())


if __name__ == "__main__":
    unittest.main()
