import unittest

import numpy as np
import pandas as pd

import full_equity_portfolio_v15 as fe


class FullEquityPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.idx = pd.bdate_range("2024-01-02", periods=30)
        self.bench = pd.DataFrame({
            "Open": np.linspace(100, 110, len(self.idx)),
            "High": np.linspace(101, 111, len(self.idx)),
            "Low": np.linspace(99, 109, len(self.idx)),
            "Close": np.linspace(100.5, 110.5, len(self.idx)),
            "Volume": 1_000_000,
        }, index=self.idx)
        self.stock = pd.DataFrame({
            "Open": np.linspace(50, 60, len(self.idx)),
            "High": np.linspace(51, 61, len(self.idx)),
            "Low": np.linspace(49, 59, len(self.idx)),
            "Close": np.linspace(50.5, 60.5, len(self.idx)),
            "Volume": 1_000_000,
        }, index=self.idx)

    def test_portfolio_is_always_fully_in_equities(self):
        entry = self.idx[2]
        exit_ = self.idx[22]
        trades = pd.DataFrame([{
            "ticker": "TEST",
            "entry_date": entry,
            "entry_price": float(self.stock.loc[entry, "Open"]),
            "exit_date20": exit_,
            "exit_price20": float(self.stock.loc[exit_, "Close"]),
        }])
        curve, summary = fe.replay_full_equity(
            trades, self.bench, {"TEST": self.stock}, self.idx[0], self.idx[-1], 0.20
        )
        self.assertTrue(np.allclose(curve.equity_exposure.to_numpy(), 1.0))
        self.assertTrue((curve.index_exposure >= -1e-12).all())
        self.assertTrue((curve.stock_exposure >= -1e-12).all())
        self.assertEqual(summary["accepted_signals"], 1)
        self.assertAlmostEqual(summary["avg_total_equity_exposure"], 1.0, places=10)

    def test_signal_is_funded_by_reducing_index_core(self):
        entry = self.idx[2]
        exit_ = self.idx[22]
        trades = pd.DataFrame([{
            "ticker": "TEST",
            "entry_date": entry,
            "entry_price": float(self.stock.loc[entry, "Open"]),
            "exit_date20": exit_,
            "exit_price20": float(self.stock.loc[exit_, "Close"]),
        }])
        curve, _ = fe.replay_full_equity(
            trades, self.bench, {"TEST": self.stock}, self.idx[0], self.idx[-1], 0.25
        )
        during = curve.loc[entry:self.idx[21]]
        self.assertGreater(during.stock_exposure.max(), 0.20)
        self.assertLess(during.index_exposure.min(), 0.80)
        after = curve.loc[self.idx[23]:]
        self.assertTrue((after.stock_exposure < 1e-12).all())


if __name__ == "__main__":
    unittest.main()
