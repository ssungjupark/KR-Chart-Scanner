import unittest

import numpy as np
import pandas as pd

from generalized_regime_v12 import rolling_last_percentile


class V12RegimeTests(unittest.TestCase):
    def test_rolling_breadth_percentile_has_no_future_leak(self):
        s = pd.Series(np.linspace(0.2, 0.8, 400))
        a = rolling_last_percentile(s, window=100, min_periods=20)
        edited = s.copy()
        edited.iloc[250:] = edited.iloc[250:] * 0 + 0.01
        b = rolling_last_percentile(edited, window=100, min_periods=20)
        pd.testing.assert_series_equal(a.iloc[:250], b.iloc[:250])

    def test_rolling_breadth_percentile_is_bounded(self):
        s = pd.Series(np.sin(np.arange(500) / 20) * 0.2 + 0.5)
        p = rolling_last_percentile(s, window=120, min_periods=30).dropna()
        self.assertTrue(((p >= 0) & (p <= 1)).all())


if __name__ == "__main__":
    unittest.main()
