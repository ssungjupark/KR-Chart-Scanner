import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import pandas as pd

from kospi_long_history import add_outcomes, features, repair_index_bars, replay


def candles(n=180):
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(100, 130, n)
    return pd.DataFrame({"Open": close, "High": close + 2, "Low": close - 2,
                         "Close": close, "Volume": 1000.}, index=dates)


class LongHistoryTests(unittest.TestCase):
    def test_zero_index_open_uses_verified_observation_and_records_source(self):
        x = candles(1)
        x.loc[x.index[0], "Open"] = 0
        response = SimpleNamespace(text="Date,Open,High,Low,Close\n2020-01-01,99,102,98,100\n",
                                   raise_for_status=lambda: None)
        with tempfile.TemporaryDirectory() as folder, patch("kospi_long_history.requests.get", return_value=response):
            result = repair_index_bars(x, Path(folder))
            self.assertEqual(result.Open.iloc[0], 99)
            self.assertTrue((Path(folder) / "index_repair.json").exists())

    def test_index_source_disagreement_is_not_silently_repaired(self):
        x = candles(1)
        x.loc[x.index[0], "Open"] = 0
        response = SimpleNamespace(text="Date,Open,High,Low,Close\n2020-01-01,99,102,98,120\n",
                                   raise_for_status=lambda: None)
        with tempfile.TemporaryDirectory() as folder, patch("kospi_long_history.requests.get", return_value=response):
            with self.assertRaisesRegex(ValueError, "disagree"):
                repair_index_bars(x, Path(folder))

    def test_future_candles_cannot_change_entry_features(self):
        x, b = candles(), candles()
        cut = x.index[155]
        before = features(x.loc[:cut], b.loc[:cut])
        changed = x.copy()
        changed.loc[changed.index > cut, ["Close", "High", "Volume"]] *= 10
        after = features(changed, b).loc[:cut]
        pd.testing.assert_frame_equal(before, after)

    def test_actual_amount_and_original_price_override_adjusted_proxy(self):
        x = candles()
        x["Amount"], x["RawClose"] = 100_000_000_000., x.Close * 50
        f = features(x, candles())
        self.assertEqual(f.ADV20.iloc[-1], 100_000_000_000.)
        self.assertEqual(f.filter_price.iloc[-1], x.RawClose.iloc[-1])

    def test_outcomes_use_next_open_and_exclude_signal_day_extremes(self):
        x, b = candles(80), candles(80)
        x.loc[x.index[0], "Low"] = 1
        f = pd.DataFrame({"row_pos": [0], "Close": [x.Close.iloc[0]], "date": [x.index[0]]})
        o = add_outcomes(f, x, b).iloc[0]
        self.assertAlmostEqual(o.NET20, x.Close.iloc[20] / x.Open.iloc[1] - 1 - .004)
        self.assertAlmostEqual(o.MAE20, x.Low.iloc[1:21].min() / x.Open.iloc[1] - 1)
        self.assertEqual(o.exit_date20, x.index[20])

    def test_zero_volume_entry_is_not_an_executable_trade(self):
        x, b = candles(80), candles(80)
        x.loc[x.index[1], "Volume"] = 0
        f = pd.DataFrame({"row_pos": [0], "Close": [x.Close.iloc[0]], "date": [x.index[0]]})
        self.assertTrue(pd.isna(add_outcomes(f, x, b).NET20.iloc[0]))

    def test_flat_stock_and_matched_index_have_same_cost_and_cash(self):
        x = candles(3)
        x[["Open", "Close"]] = 100.
        f = pd.DataFrame({"ticker": ["a"], "entry_date": [x.index[0]], "entry_price": [100.],
                          "exit_date20": [x.index[2]], "exit_price20": [100.]})
        daily, keys, s = replay(f, x, {"a": x}, x.index[0], x.index[-1])
        _, _, matched = replay(f, x, {"a": x}, x.index[0], x.index[-1], asset="index", keep=keys)
        self.assertAlmostEqual(s["total_return"], .1 * ((1 - .002) / (1 + .002) - 1))
        self.assertAlmostEqual(s["total_return"], matched["total_return"])
        self.assertTrue((daily.cash >= 0).all())


if __name__ == "__main__":
    unittest.main()
