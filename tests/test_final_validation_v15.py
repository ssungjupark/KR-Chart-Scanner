import unittest

import pandas as pd

from final_validation_v15 import cost_table, winner_trim_table


class V15ValidationTests(unittest.TestCase):
    def test_cost_stress_is_linear_from_0p4_base(self):
        x = pd.DataFrame({"NET20": [0.10, 0.00], "EXCESS20": [0.08, -0.02]})
        t = cost_table(x).set_index("round_trip_cost")
        self.assertAlmostEqual(t.loc[0.004, "mean_excess"], 0.03)
        self.assertAlmostEqual(t.loc[0.008, "mean_excess"], 0.026)
        self.assertAlmostEqual(t.loc[0.012, "mean_excess"], 0.022)

    def test_winner_trim_removes_largest_excess(self):
        x = pd.DataFrame({
            "NET20": [0.01, 0.02, 0.03, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 2.00],
            "EXCESS20": [-0.10, -0.05, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 1.00],
        })
        t = winner_trim_table(x).set_index("top_winners_removed")
        self.assertLess(t.loc[1, "mean_excess"], t.loc[0, "mean_excess"])


if __name__ == "__main__":
    unittest.main()
